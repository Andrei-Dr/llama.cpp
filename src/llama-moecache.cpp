#include "llama-moecache.h"

#include "llama-impl.h"
#include "llama-model.h"

#include "ggml.h"
#include "ggml-backend.h"

#include <algorithm>
#include <atomic>
#include <cinttypes>
#include <condition_variable>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <map>
#include <mutex>
#include <thread>
#include <unordered_map>
#include <vector>

namespace {

struct layer_state {
    llama_moe_cache_layer pub;

    // LRU bookkeeping (host side; the tables mirror expert_slot)
    std::vector<int32_t>  slot_expert;      // slot -> expert id, -1 when empty
    std::vector<int32_t>  expert_slot;      // expert id -> slot, -1 when uncached
    std::vector<uint64_t> slot_last_use;    // slot -> lamport clock of last hit
    std::vector<bool>     slot_in_flight;   // slot has an upload pending
    std::vector<bool>     expert_in_flight; // expert has an upload pending
    const ggml_tensor *   scale_src[3] = { nullptr, nullptr, nullptr }; // host up/gate/down scales that need a device copy
    std::vector<int32_t>  pending;          // admitted uncached ids since the last step (dedup, obs order)
    int32_t               n_free = 0;       // slots that have never been assigned

    // gated admission: token clocks of the last `admit` misses of every expert
    std::vector<uint32_t> miss_ring; // [n_expert * admit]
    std::vector<uint8_t>  miss_pos;  // [n_expert]
    uint32_t              tok_clock = 0;

    // prompt warm-up: routing counts of the large batches since the last step()
    std::vector<uint32_t> warm_cnt; // [n_expert]
    bool                  warm_seen = false;
};

struct upload_job {
    size_t  layer_idx;
    int32_t expert;
    int32_t slot;
};

struct moe_cache {
    llama_moe_cache_params params;

    uint64_t clock   = 0;
    uint64_t n_steps = 0;

    uint64_t n_hit       = 0;
    uint64_t n_miss      = 0;
    uint64_t n_upload    = 0;
    uint64_t n_evict     = 0;
    uint64_t n_warm      = 0; // uploads scheduled by the prompt warm-up (included in n_upload)
    uint64_t upload_bytes = 0;

    std::mutex mtx; // guards layer bookkeeping (observe runs during graph exec)

    std::vector<layer_state> layers;
    std::vector<int>         il_to_idx;
    std::unordered_map<const ggml_tensor *, size_t> by_key;

    std::vector<ggml_context *>        ctxs;
    std::vector<ggml_backend_buffer_t> bufs;

    // async upload worker: slices are copied to the device off the decode thread; the new
    // table mapping is only published at a later step() once the upload has completed, so
    // a running graph never reads a torn slot
    std::thread             worker;
    std::mutex              wmtx;
    std::condition_variable wcv;
    std::condition_variable dcv;            // signaled when the worker finishes a job (deterministic mode)
    int                     n_busy = 0;     // jobs popped by the worker and not yet in `done` (under wmtx)
    bool                    sync_publish = false; // LLAMA_MOE_CACHE_SYNC=1: step() waits for every scheduled upload
    std::vector<ggml_backend_buffer_type_t> ctx_bufts; // buffer type of each context in `ctxs` (host tables: CPU)
    std::vector<std::vector<size_t>>        ctx_layers; // the layers whose tensors each context in `ctxs` holds
    std::vector<int32_t>                    ctx_slots;  // the slot count each device context's tensors are shaped for
    int32_t                 n_live    = 0;        // slots per layer now: params.n_slots, fewer after a resume that did not fit
    bool                    suspended = false;    // device slots released (prefill mode, memory pressure); lookups return nullptr
    bool                    pressure  = false;    // ... by suspend(pressure = true): a compute buffer needed the room
    const llama_model *     owner     = nullptr;  // the model whose host-resident experts this cache mirrors
    std::deque<upload_job>  todo;
    std::vector<upload_job> done;
    bool                    stop = false;
};

moe_cache * g_cache = nullptr;
std::atomic<uint64_t> g_generation{0}; // llama_moe_cache_generation(); read by any context
float       g_cache_bias = 0.0f; // set once in init, before any table entry is written
std::mutex  g_init_mtx;
bool        g_init_done = false;

// called by the CPU mul_mat_id (thread 0) for the first expert matmul of a cached layer
void moe_obs_cb(int32_t il, const struct ggml_tensor * ids, void * ud) {
    moe_cache * mc = (moe_cache *) ud;

    if (il < 0 || il >= (int32_t) mc->il_to_idx.size() || mc->il_to_idx[il] < 0) {
        return;
    }
    layer_state & ls = mc->layers[mc->il_to_idx[il]];

    const int64_t  n_ids    = ids->ne[0];
    const int64_t  n_tokens = ids->ne[1];
    const int32_t  admit    = mc->params.admit;
    const uint32_t window   = (uint32_t) mc->params.window;

    std::lock_guard<std::mutex> lock(mc->mtx);
    for (int64_t t = 0; t < n_tokens; ++t) {
        const uint32_t tok = ++ls.tok_clock;
        for (int64_t i = 0; i < n_ids; ++i) {
            const int32_t id = *(const int32_t *) ((const char *) ids->data + t*ids->nb[1] + i*ids->nb[0]);
            if (id < 0 || id >= (int32_t) ls.expert_slot.size()) {
                continue;
            }
            const int32_t slot = ls.expert_slot[id];
            if (slot >= 0) {
                mc->n_hit++;
                ls.slot_last_use[slot] = ++mc->clock;
                continue;
            }
            mc->n_miss++;
            if (ls.expert_in_flight[id]) {
                continue;
            }

            // free slots are filled on first miss (cold start); the gate only guards evictions
            bool admitted = admit <= 1 || (int32_t) ls.pending.size() < ls.n_free;
            if (!admitted) {
                uint32_t * ring = &ls.miss_ring[(size_t) id*admit];
                uint8_t  & pos  = ls.miss_pos[id];
                ring[pos] = tok;
                pos = (pos + 1) % admit;
                const uint32_t oldest = ring[pos]; // the miss `admit` misses ago, 0 = never
                admitted = oldest != 0 && tok - oldest < window;
            }
            if (!admitted) {
                continue;
            }

            bool dup = false;
            for (int32_t p : ls.pending) {
                if (p == id) { dup = true; break; }
            }
            if (!dup) {
                ls.pending.push_back(id);
            }
        }
    }
}

// custom op over the top-k ids of a large batch (runs on the CPU backend, one task)
void moe_warm_obs_op(ggml_tensor * dst, const ggml_tensor * a, int ith, int nth, void * ud) {
    GGML_UNUSED(dst);
    GGML_UNUSED(nth);
    moe_cache * mc = g_cache;
    const size_t idx = (size_t) (uintptr_t) ud;
    if (ith != 0 || !mc || idx >= mc->layers.size() || a->type != GGML_TYPE_I32 || !ggml_is_contiguous(a)) {
        return;
    }
    layer_state & ls = mc->layers[idx];

    const int32_t * ids = (const int32_t *) a->data;
    const int64_t   n   = ggml_nelements(a);

    std::lock_guard<std::mutex> lock(mc->mtx);
    for (int64_t i = 0; i < n; ++i) {
        if (ids[i] >= 0 && ids[i] < (int32_t) ls.warm_cnt.size()) {
            ls.warm_cnt[ids[i]]++;
        }
    }
    ls.warm_seen = true;
}

size_t upload_slice(ggml_tensor * dst_c, const ggml_tensor * src, int32_t expert, int32_t slot) {
    const size_t sz = src->nb[2];
    GGML_ASSERT(dst_c->nb[2] == sz);
    GGML_ASSERT((size_t) (slot + 1)*sz <= ggml_nbytes(dst_c) && (size_t) (expert + 1)*sz <= ggml_nbytes(src));
    ggml_backend_tensor_set(dst_c, (const char *) src->data + (size_t) expert*sz, (size_t) slot*sz, sz);
    return sz;
}

void set_table_entry(llama_moe_cache_layer & pub, int32_t expert, int32_t slot_or_dummy) {
    const int32_t v = slot_or_dummy;
    if (pub.sel_scale) {
        const float s = slot_or_dummy == pub.n_slots ? 1.0f : 1.0f + g_cache_bias;
        ggml_backend_tensor_set(pub.sel_scale, &s, (size_t) expert*sizeof(float), sizeof(float));
    }
    ggml_backend_tensor_set(pub.dev_table,  &v, (size_t) expert*sizeof(int32_t), sizeof(int32_t));
    ggml_backend_tensor_set(pub.host_table, &v, (size_t) expert*sizeof(int32_t), sizeof(int32_t));
}

bool on_host(const ggml_tensor * t) {
    return t && t->data && t->buffer && ggml_backend_buffer_is_host(t->buffer);
}

} // namespace

// every slot empty, every table entry -> the dummy slot, per-expert scale copies re-uploaded. Used at init and when the
// device slots come back after prefill mode (fresh buffers). keep_warm keeps the routing counts a prefill collected, so
// the next step() re-ranks the new slots by the prompt that was just processed.
static void reset_slots(moe_cache * mc, bool keep_warm) {
    std::lock_guard<std::mutex> lock(mc->mtx);
    const int32_t n_slots = mc->n_live;
    for (size_t idx = 0; idx < mc->layers.size(); ++idx) {
        layer_state & ls = mc->layers[idx];
        const int64_t n_expert = ls.pub.down_src->ne[2];
        ls.slot_expert.assign(n_slots, -1);
        ls.expert_slot.assign(n_expert, -1);
        ls.slot_last_use.assign(n_slots, 0);
        ls.slot_in_flight.assign(n_slots, false);
        ls.n_free = n_slots;
        ls.expert_in_flight.assign(n_expert, false);
        ls.miss_ring.assign((size_t) n_expert*mc->params.admit, 0);
        ls.miss_pos.assign(n_expert, 0);
        ls.pending.clear();
        if (!keep_warm) {
            ls.warm_cnt.assign(n_expert, 0);
            ls.warm_seen = false;
        }

        {
            ggml_tensor * const dst[3] = { ls.pub.up_s_c, ls.pub.gate_s_c, ls.pub.down_s_c };
            for (int i = 0; i < 3; ++i) {
                if (ls.scale_src[i]) {
                    ggml_backend_tensor_set(dst[i], ls.scale_src[i]->data, 0, ggml_nbytes(ls.scale_src[i]));
                }
            }
        }

        if (ls.pub.sel_scale) {
            const std::vector<float> ones(n_expert, 1.0f);
            ggml_backend_tensor_set(ls.pub.sel_scale, ones.data(), 0, n_expert*sizeof(float));
        }
        std::vector<int32_t> dummy(n_expert, n_slots);
        ggml_backend_tensor_set(ls.pub.dev_table,  dummy.data(), 0, n_expert*sizeof(int32_t));
        ggml_backend_tensor_set(ls.pub.host_table, dummy.data(), 0, n_expert*sizeof(int32_t));
    }
}

// (re)create the unallocated tensors of cache context ci for n_slots slots per layer. The host context (CPU buffer type) holds
// the host tables only; a device context holds its layers' slot tensors (ne[2] = n_slots + 1), per-expert scale copies,
// routing scales and device table. The previous tensors of the context must no longer be in use (no buffer, no graph).
static bool build_ctx(moe_cache * mc, size_t ci, int32_t n_slots) {
    const std::vector<size_t> & idxs = mc->ctx_layers[ci];
    const bool tables_only = mc->ctx_bufts[ci] == ggml_backend_cpu_buffer_type();
    ggml_init_params ip = {
        /*.mem_size  =*/ ggml_tensor_overhead()*(idxs.size()*10 + 8),
        /*.mem_buffer=*/ nullptr,
        /*.no_alloc  =*/ true,
    };
    ggml_context * ctx = ggml_init(ip);
    if (!ctx) {
        return false;
    }
    if (mc->ctxs[ci]) {
        ggml_free(mc->ctxs[ci]);
    }
    mc->ctxs[ci]      = ctx;
    mc->ctx_slots[ci] = n_slots;

    auto new_cache_tensor = [&](const ggml_tensor * src, const char * name, int il) {
        ggml_tensor * t = ggml_new_tensor_3d(ctx, src->type, src->ne[0], src->ne[1], n_slots + 1);
        ggml_format_name(t, "moe_cache_%s.%d", name, il);
        return t;
    };

    for (size_t idx : idxs) {
        llama_moe_cache_layer & pub = mc->layers[idx].pub;
        const int64_t n_expert = pub.down_src->ne[2];
        if (tables_only) {
            pub.host_table = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, 1, n_expert);
            ggml_format_name(pub.host_table, "moe_cache_htbl.%d", pub.il);
            continue;
        }
        pub.n_slots = n_slots;
        if (pub.gate_up_src) {
            pub.gate_up_c = new_cache_tensor(pub.gate_up_src, "gate_up", pub.il);
        } else {
            pub.up_c   = new_cache_tensor(pub.up_src,   "up",   pub.il);
            pub.gate_c = new_cache_tensor(pub.gate_src, "gate", pub.il);
        }
        pub.down_c = new_cache_tensor(pub.down_src, "down", pub.il);
        {
            ggml_tensor ** const dst[3] = { &pub.up_s_c, &pub.gate_s_c, &pub.down_s_c };
            for (int i = 0; i < 3; ++i) {
                if (const ggml_tensor * s = mc->layers[idx].scale_src[i]) {
                    *dst[i] = ggml_dup_tensor(ctx, s);
                    ggml_format_name(*dst[i], "moe_cache_scale%d.%d", i, pub.il);
                }
            }
        }
        if (mc->params.bias > 0.0f) {
            pub.sel_scale = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_expert);
            ggml_format_name(pub.sel_scale, "moe_cache_sel.%d", pub.il);
        }
        pub.dev_table = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, 1, n_expert);
        ggml_format_name(pub.dev_table, "moe_cache_tbl.%d", pub.il);
    }
    return true;
}

static bool alloc_ctx(moe_cache * mc, size_t ci) {
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors_from_buft(mc->ctxs[ci], mc->ctx_bufts[ci]);
    if (!buf) {
        return false;
    }
    ggml_backend_buffer_clear(buf, 0);
    mc->bufs[ci] = buf;
    return true;
}

void llama_moe_cache_init(const llama_model & model, const llama_moe_cache_params & params) {
    std::lock_guard<std::mutex> init_lock(g_init_mtx);
    if (g_init_done) {
        return;
    }
    if (params.n_slots <= 0) {
        g_init_done = true;
        return;
    }

    auto * mc = new moe_cache();
    mc->params = params;
    mc->owner  = &model;
    mc->params.max_inserts = std::max(1, params.max_inserts);
    mc->params.admit       = std::min(255, std::max(1, params.admit));
    mc->params.window      = std::max(1, params.window);
    mc->params.warm        = params.warm <= 0 ? 0 : std::max(5, params.warm); // 1-4 tokens belong to the cache chain
    // deterministic mode (test / paired-comparison use): uploads scheduled at step N are always published at step N+1, so
    // which experts hit the cache is a function of the token history alone, not of PCIe timing. Costs decode speed.
    {
        const char * e = getenv("LLAMA_MOE_CACHE_SYNC");
        mc->sync_publish = e && atoi(e) != 0;
    }
    mc->params.bias        = std::max(0.0f, params.bias);
    g_cache_bias           = mc->params.bias;
    const int32_t n_slots = params.n_slots;

    // collect the host-resident expert layers, grouped by the buffer type of the device that
    // runs the layer. (Not the router's buffer: bf16 routers land on CUDA_Host on GPUs without
    // bf16 support, which would silently disable the cache.)
    std::map<ggml_backend_buffer_type_t, std::vector<size_t>> groups;

    mc->il_to_idx.assign(model.layers.size(), -1);
    for (size_t il = 0; il < model.layers.size(); ++il) {
        const auto & l = model.layers[il];
        const bool fused = l.ffn_gate_up_exps != nullptr;
        if (!l.ffn_down_exps || (!fused && (!l.ffn_up_exps || !l.ffn_gate_exps))) {
            continue;
        }
        // weights not loaded (dry-run / memory estimation) or already on a device: nothing to cache
        if (!on_host(l.ffn_down_exps) || (fused ? !on_host(l.ffn_gate_up_exps) : (!on_host(l.ffn_up_exps) || !on_host(l.ffn_gate_exps)))) {
            continue;
        }
        ggml_backend_dev_t dev = model.dev_layer((int) il);
        if (!dev || ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_CPU) {
            continue; // no device home for the cache
        }

        layer_state ls;
        ls.pub.il          = (int) il;
        ls.pub.n_slots     = n_slots;
        ls.pub.gate_up_src = fused ? l.ffn_gate_up_exps : nullptr;
        ls.pub.up_src      = fused ? nullptr : l.ffn_up_exps;
        ls.pub.gate_src    = fused ? nullptr : l.ffn_gate_exps;
        ls.pub.down_src    = l.ffn_down_exps;
        {
            ggml_tensor * const   scales[3] = { l.ffn_up_exps_s, l.ffn_gate_exps_s, l.ffn_down_exps_s };
            ggml_tensor ** const  dst[3]    = { &ls.pub.up_s_c, &ls.pub.gate_s_c, &ls.pub.down_s_c };
            for (int i = 0; i < 3; ++i) {
                if (scales[i] && on_host(scales[i])) {
                    ls.scale_src[i] = scales[i]; // device copy allocated below
                } else {
                    *dst[i] = scales[i];
                }
            }
        }

        mc->il_to_idx[il] = (int) mc->layers.size();
        groups[ggml_backend_dev_buffer_type(dev)].push_back(mc->layers.size());
        mc->layers.push_back(std::move(ls));
    }

    if (mc->layers.empty()) {
        LLAMA_LOG_INFO("%s: --moe-expert-cache %d but no host-resident expert layers found - disabled\n", __func__, n_slots);
        delete mc;
        return; // not final: a later model (e.g. after a dry-run estimate) may qualify
    }

    mc->ctx_bufts.push_back(ggml_backend_cpu_buffer_type()); // context 0: the host tables of every layer
    mc->ctx_layers.emplace_back(mc->layers.size());
    for (size_t i = 0; i < mc->layers.size(); ++i) {
        mc->ctx_layers[0][i] = i;
    }
    for (auto & g : groups) {
        mc->ctx_bufts.push_back(g.first);
        mc->ctx_layers.push_back(g.second);
    }
    mc->ctxs.assign(mc->ctx_bufts.size(), nullptr);
    mc->bufs.assign(mc->ctx_bufts.size(), nullptr);
    mc->ctx_slots.assign(mc->ctx_bufts.size(), 0);
    bool ok = true;
    for (size_t ci = 0; ok && ci < mc->ctxs.size(); ++ci) {
        ok = build_ctx(mc, ci, n_slots) && alloc_ctx(mc, ci);
        if (!ok) {
            LLAMA_LOG_WARN("%s: failed to allocate MoE cache buffer on %s - cache disabled\n", __func__, ggml_backend_buft_name(mc->ctx_bufts[ci]));
        }
    }
    if (!ok) {
        for (auto * b : mc->bufs) { if (b) { ggml_backend_buffer_free(b); } }
        for (auto * c : mc->ctxs) { if (c) { ggml_free(c); } }
        delete mc;
        g_init_done = true; // a real model was seen and allocation failed: stay disabled
        return;
    }
    mc->n_live = n_slots;

    // init bookkeeping + tables (everything uncached -> dummy slot n_slots)
    reset_slots(mc, /*keep_warm=*/false);
    size_t vram = 0;
    for (size_t idx = 0; idx < mc->layers.size(); ++idx) {
        layer_state & ls = mc->layers[idx];
        mc->by_key[ls.pub.gate_up_src ? ls.pub.gate_up_src : ls.pub.up_src] = idx;
        for (const ggml_tensor * t : { ls.pub.up_c, ls.pub.gate_c, ls.pub.gate_up_c, ls.pub.down_c }) {
            vram += t ? ggml_nbytes(t) : 0;
        }
    }

    mc->worker = std::thread([mc]() {
        for (;;) {
            upload_job j;
            {
                std::unique_lock<std::mutex> lk(mc->wmtx);
                mc->wcv.wait(lk, [mc]() { return mc->stop || !mc->todo.empty(); });
                if (mc->stop) {
                    return;
                }
                j = mc->todo.front();
                mc->todo.pop_front();
                mc->n_busy++;
            }
            llama_moe_cache_layer & pub = mc->layers[j.layer_idx].pub;
            if (pub.gate_up_src) {
                upload_slice(pub.gate_up_c, pub.gate_up_src, j.expert, j.slot);
            } else {
                upload_slice(pub.up_c,   pub.up_src,   j.expert, j.slot);
                upload_slice(pub.gate_c, pub.gate_src, j.expert, j.slot);
            }
            upload_slice(pub.down_c, pub.down_src, j.expert, j.slot);
            {
                std::lock_guard<std::mutex> lk(mc->wmtx);
                mc->done.push_back(j);
                mc->n_busy--;
            }
            mc->dcv.notify_all();
        }
    });

    ggml_set_moe_obs_callback(moe_obs_cb, mc);
    g_cache     = mc;
    g_init_done = true;

    LLAMA_LOG_INFO("%s: MoE expert cache enabled: %zu layers x %d slots, %d inserts/step, admit %d misses / %d tokens, %.1f MiB device memory\n",
            __func__, mc->layers.size(), n_slots, mc->params.max_inserts, mc->params.admit, mc->params.window, vram/1024.0/1024.0);
    LLAMA_LOG_INFO("%s: MoE expert cache-aware routing: %s (selection prob x %.2f for cached experts)\n", __func__, mc->params.bias > 0.0f ? "ON - outputs differ from exact routing" : "off", 1.0 + mc->params.bias);
    LLAMA_LOG_INFO("%s: MoE expert cache prompt warm-up: %s (batches >= %d tokens)\n", __func__, mc->params.warm ? "on" : "off", mc->params.warm);
    if (mc->sync_publish) {
        LLAMA_LOG_INFO("%s: MoE expert cache DETERMINISTIC publish (LLAMA_MOE_CACHE_SYNC): uploads wait for the next step\n", __func__);
    }
}

bool llama_moe_cache_owned_by(const llama_model & model) {
    return g_cache != nullptr && g_cache->owner == &model;
}

// free the device slot buffers of the first n cache contexts (the host tables stay); returns the bytes freed
static size_t free_device_bufs(moe_cache * mc, size_t n) {
    size_t freed = 0;
    for (size_t i = 0; i < n; ++i) {
        if (mc->ctx_bufts[i] == ggml_backend_cpu_buffer_type()) {
            continue;
        }
        if (mc->bufs[i]) {
            freed += ggml_backend_buffer_get_size(mc->bufs[i]);
            ggml_backend_buffer_free(mc->bufs[i]);
            mc->bufs[i] = nullptr;
        }
        // detach the tensors from the freed buffer: ggml_backend_alloc_ctx_tensors_from_buft only allocates tensors whose data
        // is NULL, so a stale pointer would make resume() "allocate" nothing and fail
        for (ggml_tensor * t = ggml_get_first_tensor(mc->ctxs[i]); t; t = ggml_get_next_tensor(mc->ctxs[i], t)) {
            t->data   = nullptr;
            t->buffer = nullptr;
        }
    }
    return freed;
}

size_t llama_moe_cache_suspend(bool pressure) {
    moe_cache * mc = g_cache;
    if (!mc || mc->suspended) {
        return 0;
    }
    {
        // queued uploads are dropped; one already being copied must land before its buffer goes away
        std::unique_lock<std::mutex> wlk(mc->wmtx);
        mc->todo.clear();
        mc->dcv.wait(wlk, [mc]() { return mc->n_busy == 0; });
        mc->done.clear();
    }
    const size_t freed = free_device_bufs(mc, mc->ctxs.size());
    mc->suspended = true;
    mc->pressure  = pressure;
    g_generation++;
    LLAMA_LOG_INFO("moe-cache: %s: released %.1f MiB of device slots\n", pressure ? "memory pressure" : "prefill mode", freed/1024.0/1024.0);
    return freed;
}

// allocate every device context for n slots per layer (tensors re-created where the shape differs); all-or-nothing
static bool alloc_device_slots(moe_cache * mc, int32_t n) {
    for (size_t ci = 0; ci < mc->ctxs.size(); ++ci) {
        if (mc->ctx_bufts[ci] == ggml_backend_cpu_buffer_type()) {
            continue;
        }
        if ((mc->ctx_slots[ci] != n && !build_ctx(mc, ci, n)) || !alloc_ctx(mc, ci)) {
            free_device_bufs(mc, ci + 1); // ci too: a failed allocation may have placed some of its tensors before it gave up
            return false;
        }
    }
    return true;
}

// room left on the device after a resume that did not fit: the temporary pools and CUDA graphs grow back after it, so a smaller
// resume must not take the last byte. At least 64 MiB (the startup layout leaves them about this much); a recovery raises it
// to the pool size it had to release (llama_moe_cache_set_resume_margin)
static constexpr size_t k_resume_margin_min = 64u << 20;
static size_t           g_resume_margin     = k_resume_margin_min;
static bool             g_after_recovery    = false; // a recovery ran: every resume leaves the margin, even the full count

void llama_moe_cache_set_resume_margin(size_t bytes) {
    g_resume_margin  = std::max(g_resume_margin, bytes);
    g_after_recovery = true;
}

// the most slots per layer every device context could hold now, keeping the resume margin free
static int32_t slots_that_fit(moe_cache * mc) {
    int32_t n_fit = mc->params.n_slots;
    for (size_t ci = 0; ci < mc->ctxs.size(); ++ci) {
        if (mc->ctx_bufts[ci] == ggml_backend_cpu_buffer_type()) {
            continue;
        }
        ggml_backend_dev_t dev = ggml_backend_buft_get_device(mc->ctx_bufts[ci]);
        size_t free = 0, total = 0;
        if (dev) {
            ggml_backend_dev_memory(dev, &free, &total);
        }
        size_t per_slot = 0; // one slot of every layer in this context (the dummy slot and the small tables count as one more)
        for (size_t idx : mc->ctx_layers[ci]) {
            const llama_moe_cache_layer & pub = mc->layers[idx].pub;
            per_slot += pub.down_src->nb[2] + (pub.gate_up_src ? pub.gate_up_src->nb[2] : pub.up_src->nb[2] + pub.gate_src->nb[2]);
        }
        const int64_t n = free > g_resume_margin && per_slot > 0 ? (int64_t) ((free - g_resume_margin)/per_slot) - 2 : 0;
        n_fit = (int32_t) std::min<int64_t>(n_fit, std::max<int64_t>(n, 0));
    }
    return n_fit;
}

bool llama_moe_cache_resume() {
    moe_cache * mc = g_cache;
    if (!mc || !mc->suspended) {
        return true;
    }
    // all the configured slots when they fit (the prefill-mode round trip); otherwise as many as fit next to what grew in the
    // meantime (a compute buffer's recorded floor, a draft context's buffers), so a larger buffer costs slots, not the cache.
    // A later suspend / resume (the next prefill) tries the full count again. Once a recovery ran, every resume leaves the
    // margin free (at least 64 MiB, or the pools that recovery released), like the compute-buffer floors it goes with (both
    // stay for the process).
    int32_t n = mc->params.n_slots;
    if (g_after_recovery) {
        // the full count fitting is not enough once memory ran out: taken to the last MiB, the pools of the contexts would trade
        // places at every graph until one growth failed mid-graph (seen: 8 of 8 back with ~9 MiB free, 44 reclaims, an abort)
        n = std::min(n, slots_that_fit(mc));
    }
    bool ok = n >= 1 && alloc_device_slots(mc, n);
    if (!ok) {
        for (n = std::min(n - 1, slots_that_fit(mc)); n >= 1 && !(ok = alloc_device_slots(mc, n)); --n) {
        }
    }
    if (!ok) {
        LLAMA_LOG_WARN("moe-cache: could not re-allocate any device slots - cache stays off (outputs unchanged, decode slower)\n");
        mc->pressure = true; // the owning context retries, paced (llama_context::decode)
        return false;
    }
    size_t got = 0;
    for (size_t ci = 0; ci < mc->ctxs.size(); ++ci) {
        if (mc->ctx_bufts[ci] != ggml_backend_cpu_buffer_type()) {
            got += ggml_backend_buffer_get_size(mc->bufs[ci]);
        }
    }
    mc->n_live = n;
    reset_slots(mc, /*keep_warm=*/true);
    mc->suspended = false;
    mc->pressure  = false;
    g_generation++;
    LLAMA_LOG_INFO("moe-cache: decode mode: restored %.1f MiB of device slots, %d of %d per layer (re-ranked by the prompt at the next step)\n",
            got/1024.0/1024.0, n, mc->params.n_slots);
    return true;
}

bool llama_moe_cache_uses_buft(ggml_backend_buffer_type_t buft) {
    return g_cache != nullptr && buft != ggml_backend_cpu_buffer_type() &&
        std::find(g_cache->ctx_bufts.begin(), g_cache->ctx_bufts.end(), buft) != g_cache->ctx_bufts.end();
}

bool llama_moe_cache_pressure_suspended() {
    return g_cache != nullptr && g_cache->suspended && g_cache->pressure;
}

uint64_t llama_moe_cache_generation() {
    return g_generation.load(std::memory_order_relaxed);
}

bool llama_moe_cache_active() {
    return g_cache != nullptr;
}

const llama_moe_cache_layer * llama_moe_cache_lookup(const ggml_tensor * key) {
    if (!g_cache || !key || g_cache->suspended) {
        return nullptr;
    }
    auto it = g_cache->by_key.find(key);
    if (it == g_cache->by_key.end()) {
        return nullptr;
    }
    return &g_cache->layers[it->second].pub;
}

ggml_tensor * llama_moe_cache_build_warm_obs(ggml_context * ctx, const ggml_tensor * key, ggml_tensor * selected_experts) {
    moe_cache * mc = g_cache;
    if (!mc || !key || mc->params.warm <= 0 || selected_experts->ne[1] < mc->params.warm) {
        return nullptr;
    }
    auto it = mc->by_key.find(key);
    if (it == mc->by_key.end()) {
        return nullptr;
    }
    // the top-k ids are a strided view of the argsort
    ggml_tensor * ids = ggml_cont(ctx, selected_experts);
    return ggml_map_custom1(ctx, ids, moe_warm_obs_op, 1, (void *) (uintptr_t) it->second);
}

void llama_moe_cache_step() {
    moe_cache * mc = g_cache;
    if (!mc || mc->suspended) {
        return; // prefill mode: no slots to fill; the warm-up counts wait for resume()
    }

    // 1) publish completed uploads (sync point: no graph is executing)
    {
        std::unique_lock<std::mutex> wlk(mc->wmtx);
        if (mc->sync_publish) {
            mc->dcv.wait(wlk, [mc]() { return mc->todo.empty() && mc->n_busy == 0; });
        }
        std::lock_guard<std::mutex> lk(mc->mtx);
        for (const auto & j : mc->done) {
            layer_state & ls = mc->layers[j.layer_idx];
            ls.slot_expert[j.slot]        = j.expert;
            ls.expert_slot[j.expert]      = j.slot;
            ls.slot_last_use[j.slot]      = ++mc->clock;
            ls.slot_in_flight[j.slot]     = false;
            ls.expert_in_flight[j.expert] = false;
            set_table_entry(ls.pub, j.expert, j.slot);
        }
        mc->done.clear();
    }

    std::lock_guard<std::mutex> lock(mc->mtx);
    mc->n_steps++;

    // 2) schedule new uploads: evict at a sync point (clear the victim's table entry now),
    //    then hand the slice copies to the worker
    bool queued = false;

    // victim for `id`: an empty non-in-flight slot if any, else the LRU non-in-flight slot whose expert is not in
    // `keep`. Evicts, marks the upload in flight and queues it; false when no slot can be taken
    auto place = [&](size_t li, int32_t id, const std::vector<bool> * keep) -> bool {
        layer_state & ls = mc->layers[li];

        int32_t  slot = -1;
        uint64_t best = UINT64_MAX;
        for (int32_t s = 0; s < mc->n_live; ++s) {
            if (ls.slot_in_flight[s]) {
                continue;
            }
            if (ls.slot_expert[s] < 0) { slot = s; break; }
            if (keep && (*keep)[ls.slot_expert[s]]) {
                continue;
            }
            if (ls.slot_last_use[s] < best) { best = ls.slot_last_use[s]; slot = s; }
        }
        if (slot < 0) {
            return false;
        }

        const int32_t victim = ls.slot_expert[slot];
        if (victim < 0) {
            ls.n_free--;
        } else {
            ls.expert_slot[victim] = -1;
            ls.slot_expert[slot]   = -1;
            set_table_entry(ls.pub, victim, mc->n_live);
            mc->n_evict++;
        }
        ls.slot_in_flight[slot]  = true;
        ls.expert_in_flight[id]  = true;

        const llama_moe_cache_layer & pub = ls.pub;
        mc->n_upload++;
        mc->upload_bytes += pub.down_src->nb[2] + (pub.gate_up_src ? pub.gate_up_src->nb[2] : pub.up_src->nb[2] + pub.gate_src->nb[2]);

        std::lock_guard<std::mutex> wlk(mc->wmtx);
        mc->todo.push_back({li, id, slot});
        queued = true;
        return true;
    };

    // 2a) prompt warm-up: a large batch ran since the last step. Its most-routed experts take the slots, best
    //     first, displacing cached experts the batch used less (or never); no insert budget, the worker drains
    //     the queue while decoding starts, and a slot only counts once its upload is published
    uint64_t n_warm = 0;
    for (size_t li = 0; li < mc->layers.size(); ++li) {
        layer_state & ls = mc->layers[li];
        if (!ls.warm_seen) {
            continue;
        }
        const int32_t n_expert = (int32_t) ls.warm_cnt.size();
        const int32_t n_want   = std::min(mc->n_live, n_expert);

        std::vector<int32_t> order(n_expert);
        for (int32_t e = 0; e < n_expert; ++e) {
            order[e] = e;
        }
        std::partial_sort(order.begin(), order.begin() + n_want, order.end(), [&](int32_t a, int32_t b) {
            return ls.warm_cnt[a] != ls.warm_cnt[b] ? ls.warm_cnt[a] > ls.warm_cnt[b] : a < b;
        });

        std::vector<bool> keep(n_expert, false);
        for (int32_t i = 0; i < n_want && ls.warm_cnt[order[i]] > 0; ++i) {
            keep[order[i]] = true;
        }
        for (int32_t i = 0; i < n_want && ls.warm_cnt[order[i]] > 0; ++i) {
            const int32_t id = order[i];
            if (ls.expert_slot[id] >= 0 || ls.expert_in_flight[id]) {
                continue;
            }
            if (!place(li, id, &keep)) {
                break;
            }
            n_warm++;
        }

        std::fill(ls.warm_cnt.begin(), ls.warm_cnt.end(), 0);
        std::fill(ls.miss_ring.begin(), ls.miss_ring.end(), 0); // the admission history predates the new contents
        ls.warm_seen = false;
    }
    if (n_warm > 0) {
        mc->n_warm += n_warm;
        LLAMA_LOG_INFO("moe-cache: prompt warm-up: %" PRIu64 " uploads\n", n_warm);
    }

    // 2b) gated admissions observed by the cache chain
    for (size_t li = 0; li < mc->layers.size(); ++li) {
        layer_state & ls = mc->layers[li];

        int budget = mc->params.max_inserts;
        for (auto it = ls.pending.rbegin(); it != ls.pending.rend() && budget > 0; ++it) {
            const int32_t id = *it;
            if (ls.expert_slot[id] >= 0 || ls.expert_in_flight[id]) {
                continue;
            }
            if (!place(li, id, nullptr)) {
                break; // every slot is in flight; try again next step
            }
            --budget;
        }
        ls.pending.clear();
    }
    if (queued) {
        mc->wcv.notify_one();
    }

    if (mc->n_steps % 256 == 0) {
        const uint64_t n = mc->n_hit + mc->n_miss;
        LLAMA_LOG_INFO("moe-cache: steps %" PRIu64 " | hit rate %.1f%% | uploads %" PRIu64 " (%.1f MiB, %.2f MiB/step) | evictions %" PRIu64 " | warm-up uploads %" PRIu64 "\n",
                mc->n_steps, n ? 100.0*mc->n_hit/n : 0.0, mc->n_upload, mc->upload_bytes/1024.0/1024.0,
                mc->upload_bytes/1024.0/1024.0/mc->n_steps, mc->n_evict, mc->n_warm);
    }
}
