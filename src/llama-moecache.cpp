#include "llama-moecache.h"

#include "llama-impl.h"
#include "llama-model.h"

#include "ggml.h"
#include "ggml-backend.h"

#include <algorithm>
#include <chrono>
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

    bool                  pf_ok = false;    // prefetch may target this layer (slots on the registered backend's device, pinned sources)

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
    std::vector<ggml_backend_buffer_type_t> layer_buft; // buffer type of each layer's device slots (valid while suspended too)
    bool                    suspended = false;    // device slots released (prefill mode); lookups return nullptr
    const llama_model *     owner     = nullptr;  // the model whose host-resident experts this cache mirrors
    // pre-gated prefetch (see llama_moe_cache_build_prefetch)
    ggml_backend_t          pf_backend = nullptr;
    ggml_backend_buffer_t   pf_stage   = nullptr; // pinned mirror of every layer's table (async copies from pageable memory may sync)
    int                     pf_budget  = 0;
    int                     pf_topk    = 16;
    uint64_t                n_pf       = 0;      // prefetch uploads
    uint64_t                pf_bytes   = 0;
    // issue thread: the head op (first node of layer L's host split) only plans under mtx and hands the copies over; this
    // thread enqueues them on the compute stream while the host experts run; the tail op (last node of the same split)
    // waits until it is done, so every copy is in the stream before the scheduler enqueues layer L+1's kernels
    struct pf_copy { ggml_tensor * dst; const void * src; size_t offset; size_t size; };
    // window mode (LLAMA_MOE_PREFETCH_MODE=window, default): layer L's TAIL (the host experts just finished, DDR4 goes idle
    // while the device runs L's combine and L+1's attention) issues L+2's predicted misses on a SEPARATE copy stream and
    // records an event; layer L+1's tail makes the compute stream wait on it (ahead of L+1's combine, so ahead of L+2's cache
    // chain). step mode (=step): the head of layer L's host split uploads L+1's misses on the compute stream (pf1-pf3: the DMA
    // lands inside the DDR4-bound host window and costs more than it saves).
    int                     pf_mode         = 1;       // 0 = step, 1 = window
    ggml_backend_t          pf_copy_backend = nullptr; // window: second instance of the device backend = its own stream
    ggml_backend_event_t    pf_ev[2]        = { nullptr, nullptr };
    int                     pf_pending_il   = -1;      // window: target layer of the last issued, not yet waited copy
    int                     pf_pending_ev   = -1;
    ggml_tensor *           pf_build_pred   = nullptr; // graph build: prediction the next tail node consumes
    int                     pf_build_target = -1;
    ggml_backend_t          pf_job_backend  = nullptr; // helper job: stream to issue on, event to record after it (or null)
    ggml_backend_event_t    pf_job_event    = nullptr;
    ggml_backend_event_t    pf_fence        = nullptr; // window: compute-stream position the copy stream waits for before its writes
    ggml_backend_event_t    pf_job_fence    = nullptr;
    bool                    pf_inline  = false;  // LLAMA_MOE_PREFETCH_INLINE=1: issue on the head op's thread (A/B)
    bool                    pf_touch   = true;   // LLAMA_MOE_PREFETCH_TOUCH=0: do not refresh the LRU clock of predicted cached experts
    std::thread             pf_thread;
    std::mutex              pf_mtx;
    std::condition_variable pf_cv;
    std::vector<pf_copy>    pf_todo;
    bool                    pf_busy    = false;
    bool                    pf_stop    = false;
    uint64_t                pf_head_ns = 0, pf_tail_ns = 0, pf_calls = 0; // host time in the head / tail ops
    std::deque<upload_job>  todo;
    std::vector<upload_job> done;
    bool                    stop = false;
};

moe_cache * g_cache = nullptr;

// wait until the helper thread has enqueued the posted copies (and recorded their event)
void pf_join_helper(moe_cache * mc) {
    if (mc->pf_inline) {
        return;
    }
    std::unique_lock<std::mutex> plk(mc->pf_mtx);
    mc->pf_cv.wait(plk, [mc]() { return !mc->pf_busy || mc->pf_stop; });
}


// which layers the pre-gated prefetch may target: slots in dev_buft (the registered backend's device) and pinned host experts
int pf_qualify(moe_cache * mc, ggml_backend_buffer_type_t host_buft, ggml_backend_buffer_type_t dev_buft) {
    const size_t n_expert = mc->layers.empty() ? 0 : mc->layers[0].expert_slot.size();
    auto pinned = [&](const ggml_tensor * t) { return t == nullptr || (t->buffer && ggml_backend_buffer_get_type(t->buffer) == host_buft); };
    int n_ok = 0;
    for (size_t li = 0; li < mc->layers.size(); ++li) {
        layer_state & ls = mc->layers[li];
        const llama_moe_cache_layer & pub = ls.pub;
        const bool slots_ok = mc->suspended ? (li < mc->layer_buft.size() && mc->layer_buft[li] == dev_buft)
                                            : (pub.down_c && pub.down_c->buffer && ggml_backend_buffer_get_type(pub.down_c->buffer) == dev_buft);
        ls.pf_ok = slots_ok && ls.expert_slot.size() == n_expert &&
                   pinned(pub.down_src) && pinned(pub.gate_up_src) && pinned(pub.up_src) && pinned(pub.gate_src);
        n_ok += ls.pf_ok;
    }
    return n_ok;
}

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
    const int32_t n_slots = mc->params.n_slots;
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
    {
        const char * e = getenv("LLAMA_MOE_PREFETCH");
        mc->pf_budget = e ? std::max(0, atoi(e)) : 0;
        const char * tc = getenv("LLAMA_MOE_PREFETCH_TOUCH");
        mc->pf_touch = !(tc && atoi(tc) == 0);
        const char * md = getenv("LLAMA_MOE_PREFETCH_MODE");
        mc->pf_mode = (md && strcmp(md, "step") == 0) ? 0 : 1;
        const char * in = getenv("LLAMA_MOE_PREFETCH_INLINE");
        mc->pf_inline = in && atoi(in) != 0;
        const char * k = getenv("LLAMA_MOE_PREFETCH_TOPK");
        mc->pf_topk = k && atoi(k) > 0 ? atoi(k) : 16;
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

    auto new_cache_tensor = [&](ggml_context * ctx, const ggml_tensor * src, const char * name, int il) {
        ggml_tensor * t = ggml_new_tensor_3d(ctx, src->type, src->ne[0], src->ne[1], n_slots + 1);
        ggml_format_name(t, "moe_cache_%s.%d", name, il);
        return t;
    };

    auto alloc_group = [&](ggml_backend_buffer_type_t buft, const std::vector<size_t> & idxs, bool tables_only) -> bool {
        ggml_init_params ip = {
            /*.mem_size  =*/ ggml_tensor_overhead()*(idxs.size()*10 + 8),
            /*.mem_buffer=*/ nullptr,
            /*.no_alloc  =*/ true,
        };
        ggml_context * ctx = ggml_init(ip);
        if (!ctx) {
            return false;
        }
        mc->ctxs.push_back(ctx);
        mc->ctx_bufts.push_back(buft);

        for (size_t idx : idxs) {
            llama_moe_cache_layer & pub = mc->layers[idx].pub;
            const int64_t n_expert = pub.down_src->ne[2];
            if (tables_only) {
                pub.host_table = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, 1, n_expert);
                ggml_format_name(pub.host_table, "moe_cache_htbl.%d", pub.il);
                continue;
            }
            if (mc->layer_buft.size() < mc->layers.size()) {
                mc->layer_buft.resize(mc->layers.size(), nullptr);
            }
            mc->layer_buft[idx] = buft;
            if (pub.gate_up_src) {
                pub.gate_up_c = new_cache_tensor(ctx, pub.gate_up_src, "gate_up", pub.il);
            } else {
                pub.up_c   = new_cache_tensor(ctx, pub.up_src,   "up",   pub.il);
                pub.gate_c = new_cache_tensor(ctx, pub.gate_src, "gate", pub.il);
            }
            pub.down_c    = new_cache_tensor(ctx, pub.down_src, "down", pub.il);
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

        ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors_from_buft(ctx, buft);
        if (!buf) {
            LLAMA_LOG_WARN("%s: failed to allocate MoE cache buffer on %s - cache disabled\n", __func__, ggml_backend_buft_name(buft));
            return false;
        }
        ggml_backend_buffer_clear(buf, 0);
        mc->bufs.push_back(buf);
        return true;
    };

    std::vector<size_t> all(mc->layers.size());
    for (size_t i = 0; i < all.size(); ++i) {
        all[i] = i;
    }
    bool ok = alloc_group(ggml_backend_cpu_buffer_type(), all, /*tables_only=*/true);
    for (auto & g : groups) {
        ok = ok && alloc_group(g.first, g.second, /*tables_only=*/false);
    }
    if (!ok) {
        for (auto * b : mc->bufs) { ggml_backend_buffer_free(b); }
        for (auto * c : mc->ctxs) { ggml_free(c); }
        delete mc;
        g_init_done = true; // a real model was seen and allocation failed: stay disabled
        return;
    }

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

size_t llama_moe_cache_suspend() {
    moe_cache * mc = g_cache;
    if (!mc || mc->suspended) {
        return 0;
    }
    if (mc->pf_copy_backend) {
        pf_join_helper(mc);
        ggml_backend_synchronize(mc->pf_copy_backend); // no prefetch copy may still write the slots being freed
        mc->pf_pending_il = -1;
    }
    {
        // queued uploads are dropped; one already being copied must land before its buffer goes away
        std::unique_lock<std::mutex> wlk(mc->wmtx);
        mc->todo.clear();
        mc->dcv.wait(wlk, [mc]() { return mc->n_busy == 0; });
        mc->done.clear();
    }
    size_t freed = 0;
    for (size_t i = 0; i < mc->ctxs.size(); ++i) {
        if (mc->ctx_bufts[i] == ggml_backend_cpu_buffer_type() || !mc->bufs[i]) {
            continue; // the host tables stay
        }
        freed += ggml_backend_buffer_get_size(mc->bufs[i]);
        ggml_backend_buffer_free(mc->bufs[i]);
        mc->bufs[i] = nullptr;
        // detach the tensors from the freed buffer: ggml_backend_alloc_ctx_tensors_from_buft only allocates tensors whose data
        // is NULL, so a stale pointer would make resume() "allocate" nothing and fail
        for (ggml_tensor * t = ggml_get_first_tensor(mc->ctxs[i]); t; t = ggml_get_next_tensor(mc->ctxs[i], t)) {
            t->data   = nullptr;
            t->buffer = nullptr;
        }
    }
    mc->suspended = true;
    LLAMA_LOG_INFO("moe-cache: prefill mode: released %.1f MiB of device slots\n", freed/1024.0/1024.0);
    return freed;
}

bool llama_moe_cache_resume() {
    moe_cache * mc = g_cache;
    if (!mc || !mc->suspended) {
        return true;
    }
    size_t got = 0;
    for (size_t i = 0; i < mc->ctxs.size(); ++i) {
        if (mc->ctx_bufts[i] == ggml_backend_cpu_buffer_type() || mc->bufs[i]) {
            continue;
        }
        ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors_from_buft(mc->ctxs[i], mc->ctx_bufts[i]);
        if (!buf) {
            LLAMA_LOG_WARN("moe-cache: could not re-allocate the device slots on %s - cache stays off (outputs unchanged, decode slower)\n",
                    ggml_backend_buft_name(mc->ctx_bufts[i]));
            for (size_t j = 0; j < i; ++j) { // all-or-nothing: give back what this pass took
                if (mc->ctx_bufts[j] != ggml_backend_cpu_buffer_type() && mc->bufs[j]) {
                    ggml_backend_buffer_free(mc->bufs[j]);
                    mc->bufs[j] = nullptr;
                }
            }
            return false;
        }
        ggml_backend_buffer_clear(buf, 0);
        mc->bufs[i] = buf;
        got += ggml_backend_buffer_get_size(buf);
    }
    reset_slots(mc, /*keep_warm=*/true);
    mc->suspended = false;
    if (mc->pf_backend) {
        ggml_backend_dev_t dev = ggml_backend_get_device(mc->pf_backend);
        pf_qualify(mc, dev ? ggml_backend_dev_host_buffer_type(dev) : nullptr, ggml_backend_get_default_buffer_type(mc->pf_backend));
    }
    LLAMA_LOG_INFO("moe-cache: decode mode: restored %.1f MiB of device slots (re-ranked by the prompt at the next step)\n", got/1024.0/1024.0);
    return true;
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

bool llama_moe_cache_set_backend(const llama_model & model, ggml_backend_t backend) {
    moe_cache * mc = g_cache;
    if (!mc || mc->owner != &model) {
        return false;
    }
    if (backend != nullptr) {
        std::lock_guard<std::mutex> lock(mc->mtx);
        if (mc->pf_backend != nullptr && mc->pf_backend != backend) {
            // one registered backend: a second DEFAULT context of the same model would run its graphs on other streams
            LLAMA_LOG_WARN("moe-cache: pre-gated prefetch already bound to another context; not re-registering\n");
            return false;
        }
    }
    if (mc->pf_thread.joinable()) {
        {
            std::lock_guard<std::mutex> plk(mc->pf_mtx);
            mc->pf_stop = true;
        }
        mc->pf_cv.notify_all();
        mc->pf_thread.join();
        mc->pf_stop = false;
        // a job posted but never picked up (aborted graph) must not survive into the next registration
        std::lock_guard<std::mutex> plk(mc->pf_mtx);
        mc->pf_todo.clear();
        mc->pf_busy        = false;
        mc->pf_job_backend = nullptr;
        mc->pf_job_event   = nullptr;
        mc->pf_job_fence   = nullptr;
    }
    std::lock_guard<std::mutex> lock(mc->mtx);
    if (mc->pf_copy_backend) {
        ggml_backend_synchronize(mc->pf_copy_backend); // nothing may still read the stage or write the slots
    }
    for (auto & ev : mc->pf_ev) {
        if (ev) {
            ggml_backend_event_free(ev);
            ev = nullptr;
        }
    }
    if (mc->pf_fence) {
        ggml_backend_event_free(mc->pf_fence);
        mc->pf_fence = nullptr;
    }
    if (mc->pf_copy_backend) {
        ggml_backend_free(mc->pf_copy_backend);
        mc->pf_copy_backend = nullptr;
    }
    mc->pf_pending_il = -1;
    if (mc->pf_stage) {
        ggml_backend_buffer_free(mc->pf_stage);
        mc->pf_stage = nullptr;
    }
    mc->pf_backend = nullptr;
    if (!backend || mc->pf_budget <= 0 || mc->layers.empty()) {
        return false;
    }
    ggml_backend_dev_t          dev       = ggml_backend_get_device(backend);
    ggml_backend_buffer_type_t  host_buft = dev ? ggml_backend_dev_host_buffer_type(dev) : nullptr;
    ggml_backend_buffer_type_t  dev_buft  = ggml_backend_get_default_buffer_type(backend);
    const size_t n_expert = mc->layers[0].expert_slot.size();
    // a layer qualifies when its slots live on this backend's device (the copies are ordered on its stream) and its host
    // experts are pinned host memory of that device (true async DMA; a pageable source makes cudaMemcpyAsync sync the stream
    // first, which would serialize the host experts behind the cache hits)
    // (while suspended the slots are unallocated: judged by the buffer type the slots are created in, re-checked by resume())
    const int n_ok = host_buft ? pf_qualify(mc, host_buft, dev_buft) : 0;
    mc->pf_stage = n_ok > 0 ? ggml_backend_buft_alloc_buffer(host_buft, mc->layers.size()*n_expert*sizeof(int32_t)) : nullptr;
    if (!mc->pf_stage) {
        LLAMA_LOG_WARN("moe-cache: pre-gated prefetch disabled: no layer with device slots on %s and pinned host experts\n",
                ggml_backend_name(backend));
        return false;
    }
    if (mc->pf_mode == 1) {
        mc->pf_copy_backend = ggml_backend_dev_init(dev, nullptr);
        mc->pf_ev[0] = mc->pf_copy_backend ? ggml_backend_event_new(dev) : nullptr;
        mc->pf_ev[1] = mc->pf_copy_backend ? ggml_backend_event_new(dev) : nullptr;
        mc->pf_fence = mc->pf_copy_backend ? ggml_backend_event_new(dev) : nullptr;
        ggml_backend_dev_props props;
        ggml_backend_dev_get_props(dev, &props);
        if (!mc->pf_copy_backend || !mc->pf_ev[0] || !mc->pf_ev[1] || !mc->pf_fence || !props.caps.async || !props.caps.events) {
            LLAMA_LOG_WARN("moe-cache: pre-gated prefetch disabled: %s has no second stream / events for window mode\n",
                    ggml_backend_name(backend));
            for (auto & ev : mc->pf_ev) { if (ev) { ggml_backend_event_free(ev); ev = nullptr; } }
            if (mc->pf_fence) { ggml_backend_event_free(mc->pf_fence); mc->pf_fence = nullptr; }
            if (mc->pf_copy_backend) { ggml_backend_free(mc->pf_copy_backend); mc->pf_copy_backend = nullptr; }
            ggml_backend_buffer_free(mc->pf_stage);
            mc->pf_stage = nullptr;
            return false;
        }
    }
    mc->pf_backend = backend;
    // the issue thread calls the CUDA runtime without selecting a device: only the first GPU device qualifies (else inline)
    if (!mc->pf_inline && ggml_backend_dev_count() > 0) {
        ggml_backend_dev_t first_gpu = nullptr;
        for (size_t i = 0; i < ggml_backend_dev_count() && !first_gpu; ++i) {
            if (ggml_backend_dev_type(ggml_backend_dev_get(i)) == GGML_BACKEND_DEVICE_TYPE_GPU) {
                first_gpu = ggml_backend_dev_get(i);
            }
        }
        if (first_gpu != dev) {
            mc->pf_inline = true;
        }
    }
    if (!mc->pf_inline) {
        mc->pf_thread = std::thread([mc]() {
            std::unique_lock<std::mutex> plk(mc->pf_mtx);
            while (true) {
                mc->pf_cv.wait(plk, [mc]() { return mc->pf_stop || (mc->pf_busy && !mc->pf_todo.empty()); });
                if (mc->pf_stop) {
                    return;
                }
                std::vector<moe_cache::pf_copy> todo;
                todo.swap(mc->pf_todo);
                ggml_backend_t       be = mc->pf_job_backend;
                ggml_backend_event_t ev = mc->pf_job_event;
                ggml_backend_event_t fe = mc->pf_job_fence;
                plk.unlock();
                if (fe) {
                    ggml_backend_event_wait(be, fe); // behind every compute-stream reader enqueued before the tail
                }
                for (const auto & c : todo) {
                    ggml_backend_tensor_set_async(be, c.dst, c.src, c.offset, c.size);
                }
                if (ev) {
                    ggml_backend_event_record(ev, be);
                }
                plk.lock();
                mc->pf_busy = false;
                mc->pf_cv.notify_all();
            }
        });
    }
    if ((size_t) n_ok < mc->layers.size()) {
        LLAMA_LOG_WARN("moe-cache: pre-gated prefetch covers %d of %zu cached layers (others: slots on another device or pageable experts)\n",
                n_ok, mc->layers.size());
    }
    if (backend && mc->pf_budget > 0) {
        LLAMA_LOG_INFO("moe-cache: pre-gated prefetch on: %s mode, up to %d uploads per layer per step from the top-%d prediction (%s, %s issue)\n",
                mc->pf_mode == 1 ? "window (L+2, copy stream)" : "step (L+1, compute stream)",
                mc->pf_budget, mc->pf_topk, ggml_backend_name(backend), mc->pf_inline ? "inline" : "async");
    }
    return true;
}

void llama_moe_cache_release_backend(const llama_model & model, ggml_backend_t backend) {
    moe_cache * mc = g_cache;
    if (!mc || mc->owner != &model || backend == nullptr) {
        return;
    }
    {
        std::lock_guard<std::mutex> lock(mc->mtx);
        if (mc->pf_backend != backend) {
            return;
        }
    }
    llama_moe_cache_set_backend(model, nullptr);
}

int llama_moe_cache_prefetch_budget() {
    moe_cache * mc = g_cache;
    if (!mc || mc->suspended || !mc->pf_backend || mc->params.bias > 0.0f) {
        return 0;
    }
    return mc->pf_budget;
}

int llama_moe_cache_prefetch_topk() {
    return g_cache ? g_cache->pf_topk : 16;
}

bool llama_moe_cache_prefetch_ok(const ggml_tensor * key_next) {
    moe_cache * mc = g_cache;
    if (llama_moe_cache_prefetch_budget() <= 0 || !key_next) {
        return false;
    }
    auto it = mc->by_key.find(key_next);
    return it != mc->by_key.end() && mc->layers[it->second].pf_ok;
}

// plan a prefetch into layer idx from predicted ids a [k, n_tokens] (I32, sorted by router score per token, best first):
// pick LRU victims, rewrite the bookkeeping and the host table NOW, stage the device table, and return the copies (slot slices
// then the table) that must land before the layer's next device read. Called with nothing locked; takes mc->mtx.
static void pf_plan(moe_cache * mc, size_t idx, const ggml_tensor * a, std::vector<moe_cache::pf_copy> & copies) {
    layer_state & ls = mc->layers[idx];
    llama_moe_cache_layer & pub = ls.pub;
    const int32_t n_expert = (int32_t) ls.expert_slot.size();
    const int32_t n_slots  = mc->params.n_slots;

    // the union of the predicted sets over the batch's tokens, most confident first: the ids of each token are sorted by
    // router score, so walk rank-major (every token's best, then every token's second best, ...)
    std::vector<int32_t> want;
    std::vector<bool>    keep(n_expert, false);
    for (int64_t k = 0; k < a->ne[0]; ++k) {
        for (int64_t t = 0; t < a->ne[1]; ++t) {
            const int32_t id = *(const int32_t *) ((const char *) a->data + t*a->nb[1] + k*a->nb[0]);
            if (id >= 0 && id < n_expert && !keep[id]) {
                keep[id] = true;
                want.push_back(id);
            }
        }
    }

    std::unique_lock<std::mutex> lock(mc->mtx);
    int32_t * host_table = (int32_t *) pub.host_table->data;
    int budget = mc->pf_budget;
    bool changed = false;
    for (const int32_t id : want) {
        if (ls.expert_slot[id] >= 0) {
            if (mc->pf_touch) {
                ls.slot_last_use[ls.expert_slot[id]] = ++mc->clock; // predicted again: keep it warm
            }
            continue;
        }
        if (ls.expert_in_flight[id] || budget <= 0) {
            continue;
        }
        // victim: an empty slot, else the LRU slot whose expert is not predicted; never a slot with a worker upload in flight
        int32_t  slot = -1;
        uint64_t best = UINT64_MAX;
        for (int32_t s = 0; s < n_slots; ++s) {
            if (ls.slot_in_flight[s]) {
                continue;
            }
            if (ls.slot_expert[s] < 0) { slot = s; break; }
            if (keep[ls.slot_expert[s]]) {
                continue;
            }
            if (ls.slot_last_use[s] < best) { best = ls.slot_last_use[s]; slot = s; }
        }
        if (slot < 0) {
            break;
        }
        const int32_t victim = ls.slot_expert[slot];
        if (victim < 0) {
            ls.n_free--;
        } else {
            ls.expert_slot[victim] = -1;
            host_table[victim]     = n_slots;
            mc->n_evict++;
        }
        // stream-ordered: behind this step's kernels enqueued so far (incl. every earlier reader of the slot), ahead of the
        // target layer's kernels; the host experts live in pinned memory, so these are true async DMA copies
        size_t bytes = 0;
        auto put = [&](ggml_tensor * dst_c, const ggml_tensor * src) {
            const size_t sz = src->nb[2];
            copies.push_back({ dst_c, (const char *) src->data + (size_t) id*sz, (size_t) slot*sz, sz });
            bytes += sz;
        };
        if (pub.gate_up_c) {
            put(pub.gate_up_c, pub.gate_up_src);
        } else {
            put(pub.up_c, pub.up_src);
            put(pub.gate_c, pub.gate_src);
        }
        put(pub.down_c, pub.down_src);
        ls.slot_expert[slot]   = id;
        ls.expert_slot[id]     = slot;
        ls.slot_last_use[slot] = ++mc->clock;
        host_table[id]         = slot;
        mc->n_pf++;
        mc->pf_bytes += bytes;
        budget--;
        changed = true;
    }
    if (changed) {
        // the device table mirrors the host table: stage it in pinned memory and copy it once, ordered after the slot copies
        // above. This layer's region is rewritten only by the layer's next prefetch; the previous table copy has been consumed
        // by then: step mode issues on the compute stream (stream order), window mode waits for the previous copy at the next
        // layer's tail and fences its copy stream behind the compute stream, and step()/suspend() drain the copy stream.
        const size_t nb = ggml_nbytes(pub.dev_table);
        void * stage = (char *) ggml_backend_buffer_get_base(mc->pf_stage) + idx*n_expert*sizeof(int32_t);
        GGML_ASSERT(nb == (size_t) n_expert*sizeof(int32_t));
        memcpy(stage, host_table, nb);
        copies.push_back({ pub.dev_table, stage, 0, nb });
    }
}

// custom op (CPU, one task), step mode: first node of layer L's host-expert split; a = layer L+1's predicted ids; ud = its index
static void moe_prefetch_op(ggml_tensor * dst, const ggml_tensor * a, int ith, int nth, void * ud) {
    GGML_UNUSED(dst);
    GGML_UNUSED(nth);
    moe_cache * mc = g_cache;
    const size_t idx = (size_t) (uintptr_t) ud;
    if (ith != 0 || !mc || mc->suspended || !mc->pf_backend || !mc->pf_stage || idx >= mc->layers.size() || a->type != GGML_TYPE_I32 ||
            !mc->layers[idx].pf_ok) {
        return;
    }
    const auto t0 = std::chrono::steady_clock::now();
    std::vector<moe_cache::pf_copy> copies;
    pf_plan(mc, idx, a, copies);
    if (!copies.empty()) {
        if (mc->pf_inline) {
            for (const auto & c : copies) {
                ggml_backend_tensor_set_async(mc->pf_backend, c.dst, c.src, c.offset, c.size);
            }
        } else {
            {
                std::lock_guard<std::mutex> plk(mc->pf_mtx);
                GGML_ASSERT(!mc->pf_busy && "prefetch issue still running: the tail join of the previous layer was skipped");
                mc->pf_todo = std::move(copies);
                mc->pf_job_backend = mc->pf_backend;
                mc->pf_job_event   = nullptr;
                mc->pf_job_fence   = nullptr;
                mc->pf_busy = true;
            }
            mc->pf_cv.notify_all();
        }
    }
    mc->pf_head_ns += (uint64_t) std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - t0).count();
    mc->pf_calls++;
}

// custom op (CPU, one task), step mode: last node of the host split whose head posted a prefetch
static void moe_prefetch_join_op(ggml_tensor * dst, const ggml_tensor * a, int ith, int nth, void * ud) {
    GGML_UNUSED(dst); GGML_UNUSED(a); GGML_UNUSED(nth); GGML_UNUSED(ud);
    moe_cache * mc = g_cache;
    if (ith != 0 || !mc) {
        return;
    }
    const auto t0 = std::chrono::steady_clock::now();
    pf_join_helper(mc);
    mc->pf_tail_ns += (uint64_t) std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - t0).count();
}

// window mode, last node of layer L's host split (ud = L's il, and target il + 1 << 16, 0 = none; b = target's predicted ids):
//  1) the copies issued by the previous tail target L+1: make the compute stream wait for them (ahead of L's combine, i.e.
//     ahead of L+1's cache chain). A pending target <= L means a tail was skipped: the target's device reads were enqueued
//     unordered with its slot copies -> abort rather than compute with torn slots.
//  2) plan L+2's misses and hand them to the helper, which issues them on the copy stream and records the event.
static void moe_prefetch_tail(const ggml_tensor * b, uintptr_t ud) {
    moe_cache * mc = g_cache;
    if (!mc || !mc->pf_copy_backend) {
        return;
    }
    const int il     = (int) (ud & 0xffff);
    const int target = (int) (ud >> 16) - 1;
    const auto t0 = std::chrono::steady_clock::now();
    pf_join_helper(mc);
    if (mc->pf_pending_il >= 0) {
        if (mc->pf_pending_il == il + 1 || mc->pf_pending_il > il + 1) {
            // > il + 1: left behind by a graph that stopped early (abort / failure before step()); order it and drop it
            ggml_backend_event_wait(mc->pf_backend, mc->pf_ev[mc->pf_pending_ev]);
            mc->pf_pending_il = -1;
        } else if (mc->pf_pending_il <= il) {
            GGML_ABORT("moe-cache: prefetch into layer %d was not waited for before its device reads (tail of layer %d missing)",
                    mc->pf_pending_il, mc->pf_pending_il - 1);
        }
    }
    if (b != nullptr && target >= 0 && !mc->suspended && target < (int32_t) mc->il_to_idx.size() && mc->il_to_idx[target] >= 0 &&
            mc->layers[mc->il_to_idx[target]].pf_ok && b->type == GGML_TYPE_I32) {
        GGML_ASSERT(mc->pf_pending_il < 0 && "window prefetch: a previous copy is still unwaited");
        std::vector<moe_cache::pf_copy> copies;
        pf_plan(mc, (size_t) mc->il_to_idx[target], b, copies);
        if (!copies.empty()) {
            const int e = target & 1;
            // fence: the copy stream starts only behind everything enqueued on the compute stream so far (earlier graphs of
            // this decode may still have to read the target's slots, e.g. several ubatches between two step() calls)
            ggml_backend_event_record(mc->pf_fence, mc->pf_backend);
            if (mc->pf_inline) {
                ggml_backend_event_wait(mc->pf_copy_backend, mc->pf_fence);
                for (const auto & c : copies) {
                    ggml_backend_tensor_set_async(mc->pf_copy_backend, c.dst, c.src, c.offset, c.size);
                }
                ggml_backend_event_record(mc->pf_ev[e], mc->pf_copy_backend);
            } else {
                {
                    std::lock_guard<std::mutex> plk(mc->pf_mtx);
                    GGML_ASSERT(!mc->pf_busy);
                    mc->pf_todo        = std::move(copies);
                    mc->pf_job_backend = mc->pf_copy_backend;
                    mc->pf_job_event   = mc->pf_ev[e];
                    mc->pf_job_fence   = mc->pf_fence;
                    mc->pf_busy        = true;
                }
                mc->pf_cv.notify_all();
            }
            mc->pf_pending_il = target;
            mc->pf_pending_ev = e;
        }
        mc->pf_calls++;
    }
    mc->pf_tail_ns += (uint64_t) std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - t0).count();
}

static void moe_prefetch_tail_op2(ggml_tensor * dst, const ggml_tensor * a, const ggml_tensor * b, int ith, int nth, void * ud) {
    GGML_UNUSED(dst); GGML_UNUSED(a); GGML_UNUSED(nth);
    if (ith == 0) {
        moe_prefetch_tail(b, (uintptr_t) ud);
    }
}

static void moe_prefetch_tail_op1(ggml_tensor * dst, const ggml_tensor * a, int ith, int nth, void * ud) {
    GGML_UNUSED(dst); GGML_UNUSED(a); GGML_UNUSED(nth);
    if (ith == 0) {
        moe_prefetch_tail(nullptr, (uintptr_t) ud);
    }
}

int llama_moe_cache_prefetch_ahead() {
    return g_cache && g_cache->pf_mode == 1 ? 2 : 1;
}

ggml_tensor * llama_moe_cache_build_prefetch_join(ggml_context * ctx, ggml_tensor * after, int il, bool has_head) {
    moe_cache * mc = g_cache;
    if (llama_moe_cache_prefetch_budget() <= 0 || !after) {
        return nullptr;
    }
    if (mc->pf_mode == 0) {
        return has_head ? ggml_map_custom1(ctx, after, moe_prefetch_join_op, 1, nullptr) : nullptr;
    }
    // window: a tail node at EVERY cached decode layer (it carries the wait for the copies the previous tail issued)
    ggml_tensor * pred = mc->pf_build_pred;
    const int target   = mc->pf_build_target;
    mc->pf_build_pred   = nullptr;
    mc->pf_build_target = -1;
    const uintptr_t ud = (uintptr_t) (il & 0xffff) | ((uintptr_t) (pred ? target + 1 : 0) << 16);
    return pred ? ggml_map_custom2(ctx, after, pred, moe_prefetch_tail_op2, 1, (void *) ud)
                : ggml_map_custom1(ctx, after, moe_prefetch_tail_op1, 1, (void *) ud);
}

ggml_tensor * llama_moe_cache_build_prefetch(ggml_context * ctx, const ggml_tensor * key_next, ggml_tensor * pred_ids) {
    moe_cache * mc = g_cache;
    if (llama_moe_cache_prefetch_budget() <= 0 || !key_next || !pred_ids) {
        return nullptr;
    }
    auto it = mc->by_key.find(key_next);
    if (it == mc->by_key.end() || !mc->layers[it->second].pf_ok) {
        return nullptr;
    }
    if (mc->pf_mode == 1) {
        // consumed by this layer's tail node (built by llama_moe_cache_build_prefetch_join in the same build_moe_ffn call)
        mc->pf_build_pred   = pred_ids;
        mc->pf_build_target = mc->layers[it->second].pub.il;
        return nullptr;
    }
    return ggml_map_custom1(ctx, pred_ids, moe_prefetch_op, 1, (void *) (uintptr_t) it->second);
}

void llama_moe_cache_step() {
    moe_cache * mc = g_cache;
    if (!mc || mc->suspended) {
        return; // prefill mode: no slots to fill; the warm-up counts wait for resume()
    }

    // 0) window prefetch: everything issued this step has been waited for by the last tail; drain the copy stream anyway
    if (mc->pf_copy_backend) {
        pf_join_helper(mc);
        ggml_backend_synchronize(mc->pf_copy_backend);
        mc->pf_pending_il = -1;
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
        for (int32_t s = 0; s < mc->params.n_slots; ++s) {
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
            set_table_entry(ls.pub, victim, mc->params.n_slots);
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
        const int32_t n_want   = std::min(mc->params.n_slots, n_expert);

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
        LLAMA_LOG_INFO("moe-cache: steps %" PRIu64 " | hit rate %.1f%% | uploads %" PRIu64 " (%.1f MiB, %.2f MiB/step) | evictions %" PRIu64 " | warm-up uploads %" PRIu64 " | prefetch %" PRIu64 " (%.2f MiB/step, head %.1f us, join %.1f us per layer)\n",
                mc->n_steps, n ? 100.0*mc->n_hit/n : 0.0, mc->n_upload, mc->upload_bytes/1024.0/1024.0,
                mc->upload_bytes/1024.0/1024.0/mc->n_steps, mc->n_evict, mc->n_warm, mc->n_pf, mc->pf_bytes/1024.0/1024.0/mc->n_steps,
                mc->pf_calls ? mc->pf_head_ns/1e3/mc->pf_calls : 0.0, mc->pf_calls ? mc->pf_tail_ns/1e3/mc->pf_calls : 0.0);
    }
}
