#include "llama-moe-hybrid.h"

#include "llama-graph.h"
#include "llama-impl.h"
#include "llama-model.h"
#include "llama-moe-cache.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>

struct llama_moe_hybrid::layer_state {
    const llama_moe_hybrid_params * params = nullptr;

    // set by build_lookup for the graph being built
    int64_t         n_expert = 0;
    const int32_t * slots    = nullptr; // slot map of the MoE cache, nullptr if the layer has no cached experts
    int32_t         n_slots  = 0;

    // work buffers of the split op
    std::vector<int32_t> counts;
    std::vector<uint8_t> cached;
    std::vector<uint8_t> on_cpu;

    // cumulative, written by the split op
    size_t n_graphs     = 0;
    size_t n_cpu        = 0;
    size_t n_upload     = 0;
    size_t n_cached     = 0;
    size_t n_assign     = 0;
    size_t n_assign_cpu = 0;
};

llama_moe_hybrid_split llama_moe_hybrid::choose_split(const int32_t * counts, const uint8_t * cached, int64_t n_expert, const llama_moe_hybrid_params & params) {
    // the used experts outside the MoE cache, by the tokens routed to them
    int32_t max_count = 0;
    int64_t n_cached  = 0;
    for (int64_t e = 0; e < n_expert; ++e) {
        if (counts[e] == 0) {
            continue;
        }
        if (cached != nullptr && cached[e]) {
            n_cached++;
        } else {
            max_count = std::max(max_count, counts[e]);
        }
    }
    std::vector<int64_t> n_at(max_count + 1, 0);
    int64_t n_free = 0; // used experts outside the cache
    for (int64_t e = 0; e < n_expert; ++e) {
        if (counts[e] > 0 && !(cached != nullptr && cached[e])) {
            n_at[counts[e]]++;
            n_free++;
        }
    }

    const double t_cached = (double) params.us_gpu_cached*(double) n_cached;

    // c = 0: every used expert on the device
    llama_moe_hybrid_split best;
    best.n_cached = n_cached;
    best.n_upload = n_free;
    double best_t = t_cached + (double) params.us_gpu_expert*(double) n_free;

    int64_t n_cpu = 0;
    int64_t n_assign_cpu = 0;
    for (int32_t c = 1; c <= max_count; ++c) {
        n_cpu        += n_at[c];
        n_assign_cpu += (int64_t) c*n_at[c];
        const double t_cpu = (double) params.us_assign*(double) n_assign_cpu + (double) params.us_cpu_expert*(double) n_cpu;
        const double t_gpu = t_cached + (double) params.us_gpu_expert*(double) (n_free - n_cpu);
        const double t = std::max(t_cpu, t_gpu);
        if (t < best_t) {
            best_t            = t;
            best.c            = c;
            best.n_cpu        = n_cpu;
            best.n_upload     = n_free - n_cpu;
            best.n_assign_cpu = n_assign_cpu;
        }
    }
    return best;
}

void llama_moe_hybrid::fill_dev_ids(const int32_t * ids, int64_t n_used, int64_t n_tokens, const uint8_t * on_cpu, int64_t n_expert, int32_t * dev_ids) {
    // the rows of the experts on the CPU are computed by the device against another expert and masked to zero. The device
    // matmul must not see the same expert twice for one token, so the CPU experts of a token take distinct stand-ins it is
    // not routed to: the experts the device computes anyway come first (they are uploaded already), then any other expert
    // (an extra upload)
    std::vector<int32_t> standins;
    standins.reserve(n_expert);
    std::vector<uint8_t> used(n_expert, 0);
    for (int64_t i = 0; i < n_used*n_tokens; ++i) {
        used[ids[i]] = 1;
    }
    for (int64_t e = 0; e < n_expert; ++e) {
        if (used[e] && !on_cpu[e]) {
            standins.push_back((int32_t) e);
        }
    }
    for (int64_t e = 0; e < n_expert; ++e) {
        if (!(used[e] && !on_cpu[e])) {
            standins.push_back((int32_t) e);
        }
    }

    for (int64_t it = 0; it < n_tokens; ++it) {
        const int32_t * r_in  = ids     + it*n_used;
        int32_t       * r_out = dev_ids + it*n_used;
        for (int64_t k = 0; k < n_used; ++k) {
            r_out[k] = on_cpu[r_in[k]] ? -1 : r_in[k];
        }
        size_t next = 0;
        for (int64_t k = 0; k < n_used; ++k) {
            if (r_out[k] >= 0) {
                continue;
            }
            for (; next < standins.size(); ++next) {
                const int32_t s = standins[next];
                bool taken = false;
                for (int64_t j = 0; j < n_used; ++j) {
                    taken = taken || r_in[j] == s || r_out[j] == s;
                }
                if (!taken) {
                    break;
                }
            }
            GGML_ASSERT(next < standins.size());
            r_out[k] = standins[next++];
        }
    }
}

// the ops of the split, each a single task of the CPU backend

static void llama_moe_hybrid_dev_ids_op(ggml_tensor * dst, int ith, int nth, void * userdata) {
    GGML_UNUSED(nth);
    if (ith != 0) {
        return;
    }
    auto * l = static_cast<llama_moe_hybrid::layer_state *>(userdata);

    const ggml_tensor * ids = dst->src[0];
    const int64_t n_used   = ids->ne[0];
    const int64_t n_tokens = ids->ne[1];
    const int64_t n_expert = l->n_expert;
    const int32_t * in = (const int32_t *) ids->data;

    l->counts.assign(n_expert, 0);
    for (int64_t i = 0; i < n_used*n_tokens; ++i) {
        GGML_ASSERT(in[i] >= 0 && in[i] < n_expert);
        l->counts[in[i]]++;
    }
    l->cached.assign(n_expert, 0);
    if (l->slots != nullptr) {
        for (int64_t e = 0; e < n_expert; ++e) {
            l->cached[e] = l->slots[e] < l->n_slots;
        }
    }

    const llama_moe_hybrid_split split = llama_moe_hybrid::choose_split(l->counts.data(), l->cached.data(), n_expert, *l->params);

    l->on_cpu.assign(n_expert, 0);
    for (int64_t e = 0; e < n_expert; ++e) {
        l->on_cpu[e] = l->counts[e] > 0 && !l->cached[e] && l->counts[e] <= split.c;
    }
    llama_moe_hybrid::fill_dev_ids(in, n_used, n_tokens, l->on_cpu.data(), n_expert, (int32_t *) dst->data);

    l->n_graphs++;
    l->n_cpu        += split.n_cpu;
    l->n_upload     += split.n_upload;
    l->n_cached     += split.n_cached;
    l->n_assign     += n_used*n_tokens;
    l->n_assign_cpu += split.n_assign_cpu;
}

static void llama_moe_hybrid_cpu_ids_op(ggml_tensor * dst, int ith, int nth, void * userdata) {
    GGML_UNUSED(nth);
    GGML_UNUSED(userdata);
    if (ith != 0) {
        return;
    }
    const int32_t * ids = (const int32_t *) dst->src[0]->data;
    const int32_t * dev = (const int32_t *) dst->src[1]->data;
    int32_t       * out = (int32_t *) dst->data;
    const int64_t n = ggml_nelements(dst);
    for (int64_t i = 0; i < n; ++i) {
        out[i] = ids[i] == dev[i] ? -1 : ids[i];
    }
}

static void llama_moe_hybrid_mask_op(ggml_tensor * dst, int ith, int nth, void * userdata) {
    GGML_UNUSED(nth);
    GGML_UNUSED(userdata);
    if (ith != 0) {
        return;
    }
    const int32_t * ids = (const int32_t *) dst->src[0]->data;
    const int32_t * dev = (const int32_t *) dst->src[1]->data;
    float         * out = (float *) dst->data;
    const int64_t n = ggml_nelements(dst);
    for (int64_t i = 0; i < n; ++i) {
        out[i] = ids[i] == dev[i] ? 1.0f : 0.0f;
    }
}

std::unique_ptr<llama_moe_hybrid> llama_moe_hybrid::init_from_env(const llama_model & model) {
    const char * on = getenv("LLAMA_MOE_HYBRID");
    if (on == nullptr || atoi(on) == 0) {
        return nullptr;
    }
    if (model.hparams.n_expert == 0) {
        return nullptr;
    }
    llama_moe_hybrid_params p;
    if (const char * e = getenv("LLAMA_MOE_HYBRID_MIN")) {
        p.n_min = std::max<int64_t>(2, atoll(e));
    }
    if (const char * e = getenv("LLAMA_MOE_HYBRID_MAX")) {
        p.n_max = atoll(e);
    }
    if (const char * e = getenv("LLAMA_MOE_HYBRID_COSTS")) {
        float a = 0, c = 0, g = 0, h = 0;
        if (sscanf(e, "%f,%f,%f,%f", &a, &c, &g, &h) == 4 && a > 0 && c >= 0 && g > 0 && h >= 0) {
            p.us_assign     = a;
            p.us_cpu_expert = c;
            p.us_gpu_expert = g;
            p.us_gpu_cached = h;
        } else {
            LLAMA_LOG_WARN("%s: LLAMA_MOE_HYBRID_COSTS is not \"assign,cpu_expert,gpu_expert,gpu_cached\", the defaults are used\n", __func__);
        }
    }
    return std::make_unique<llama_moe_hybrid>(model, p);
}

llama_moe_hybrid::llama_moe_hybrid(const llama_model & model, const llama_moe_hybrid_params & params) :
        params(params), layers(model.layers.size()) {
    for (layer_state & l : layers) {
        l.params = &this->params;
    }
    LLAMA_LOG_INFO("llama_moe_hybrid: ubatches of %lld to %lld tokens split the host experts of a layer between the CPU and the device "
            "(us per layer: %.1f per routed token, %.1f per CPU expert, %.1f per uploaded expert, %.1f per cached expert)\n",
            (long long) params.n_min, (long long) params.n_max, params.us_assign, params.us_cpu_expert, params.us_gpu_expert, params.us_gpu_cached);
}

llama_moe_hybrid::~llama_moe_hybrid() {
    layer_state sum;
    for (const layer_state & l : layers) {
        sum.n_graphs     += l.n_graphs;
        sum.n_cpu        += l.n_cpu;
        sum.n_upload     += l.n_upload;
        sum.n_cached     += l.n_cached;
        sum.n_assign     += l.n_assign;
        sum.n_assign_cpu += l.n_assign_cpu;
    }
    if (sum.n_graphs > 0) {
        LLAMA_LOG_INFO("llama_moe_hybrid: %zu layer splits, experts: %zu on the CPU, %zu uploaded, %zu from the MoE cache, %.2f%% of the routed tokens on the CPU\n",
                sum.n_graphs, sum.n_cpu, sum.n_upload, sum.n_cached, 100.0*sum.n_assign_cpu/sum.n_assign);
    }
}

llama_moe_hybrid_lookup llama_moe_hybrid::build_lookup(const llm_graph_context & g, const llama_moe_cache_layer & layer, ggml_tensor * selected_experts) const {
    const int32_t il       = layer.il;
    const int64_t n_used   = selected_experts->ne[0];
    const int64_t n_tokens = selected_experts->ne[1];

    if (il < 0 || il >= (int32_t) layers.size() || n_tokens < params.n_min || n_tokens > params.n_max) {
        return {};
    }
    // the same limits as the CPU part of the MoE cache, and no observer of the graph: the stand-in ids of the device chain
    // are not routing (imatrix), and benchmark routing can repeat an expert for one token
    if (layer.weight_before_ffn || !g.loras->empty() || g.cparams.cb_eval != nullptr || g.cparams.moe_random_routing) {
        return {};
    }
    // not while the MoE cache is suspended: in MoE prefill mode the scheduler uploads whole expert tensors ahead
    if (g.moe_cache != nullptr && g.moe_cache->suspended()) {
        return {};
    }
    if (layer.down_exps == nullptr || (layer.gate_up_exps == nullptr && layer.up_exps == nullptr)) {
        return {};
    }
    // room for distinct stand-ins, and distinct ids per token (not the divided ids of GROVEMOE)
    const int64_t n_expert = layer.down_exps->ne[2];
    if (n_expert < 2*n_used || g.model.arch == LLM_ARCH_GROVEMOE) {
        return {};
    }

    // the device of the layer
    ggml_backend_t backend = nullptr;
    const int n_backends = ggml_backend_sched_get_n_backends(g.sched);
    for (int i = 0; i < n_backends && backend == nullptr; ++i) {
        ggml_backend_t b = ggml_backend_sched_get_backend(g.sched, i);
        ggml_backend_dev_t dev = ggml_backend_get_device(b);
        if (dev == nullptr || dev != g.model.dev_layer(il)) {
            continue;
        }
        const auto type = ggml_backend_dev_type(dev);
        if (type == GGML_BACKEND_DEVICE_TYPE_GPU || type == GGML_BACKEND_DEVICE_TYPE_IGPU) {
            backend = b;
        }
    }
    if (backend == nullptr) {
        return {};
    }

    // the experts are weights in plain host memory that the device cannot read in place, and not rotated
    for (ggml_tensor * w : { layer.up_exps, layer.gate_exps, layer.down_exps, layer.gate_up_exps }) {
        if (w == nullptr) {
            continue;
        }
        if (w->buffer == nullptr || ggml_backend_buffer_get_usage(w->buffer) != GGML_BACKEND_BUFFER_USAGE_WEIGHTS ||
                !ggml_backend_buffer_is_host(w->buffer) || ggml_backend_supports_buft(backend, ggml_backend_buffer_get_type(w->buffer)) ||
                w->ne[2] != n_expert || (g.hdmd != nullptr && g.hdmd->rot.count(w) > 0)) {
            return {};
        }
        // the device computes the matmul of the experts it is given
        ggml_tensor * b = ggml_new_tensor_3d(g.ctx0, GGML_TYPE_F32, w->ne[0], w == layer.down_exps ? n_used : 1, n_tokens);
        if (!ggml_backend_supports_op(backend, ggml_mul_mat_id(g.ctx0, w, b, selected_experts))) {
            return {};
        }
    }

    // a device that offloads such ubatches from host weights takes the whole layer, there is no CPU part to split from
    {
        ggml_tensor probe = {};
        probe.op    = GGML_OP_MUL_MAT_ID;
        probe.type  = GGML_TYPE_F32;
        probe.ne[0] = 1;
        probe.ne[1] = n_used;
        probe.ne[2] = n_tokens;
        probe.ne[3] = 1;
        for (int i = 0; i < n_backends; ++i) {
            if (ggml_backend_offload_op(ggml_backend_sched_get_backend(g.sched, i), &probe)) {
                return {};
            }
        }
    }

    layer_state & l = layers[il];
    l.n_expert = n_expert;
    l.slots    = g.moe_cache != nullptr ? g.moe_cache->resident_slots(il, l.n_slots) : nullptr;

    // the routing is computed on the device and read by the CPU split of the three ops below
    ggml_tensor * ids = selected_experts;
    if (!ggml_is_contiguous(ids)) {
        ids = ggml_cont(g.ctx0, ids);
        ggml_backend_sched_set_tensor_backend(g.sched, ids, backend);
    }

    llama_moe_hybrid_lookup res;
    res.backend = backend;
    {
        ggml_tensor * args[1] = { ids };
        res.dev_ids = ggml_custom_4d(g.ctx0, GGML_TYPE_I32, n_used, n_tokens, 1, 1, args, 1, llama_moe_hybrid_dev_ids_op, 1, &l);
        g.cb(res.dev_ids, "ffn_moe_hyb_dev_ids", il);
    }
    {
        ggml_tensor * args[2] = { ids, res.dev_ids };
        res.cpu_ids = ggml_custom_4d(g.ctx0, GGML_TYPE_I32, n_used, n_tokens, 1, 1, args, 2, llama_moe_hybrid_cpu_ids_op, 1, nullptr);
        g.cb(res.cpu_ids, "ffn_moe_hyb_cpu_ids", il);
        res.mask    = ggml_custom_4d(g.ctx0, GGML_TYPE_F32, 1, n_used, n_tokens, 1, args, 2, llama_moe_hybrid_mask_op, 1, nullptr);
        g.cb(res.mask, "ffn_moe_hyb_mask", il);
    }
    // one CPU split, before the device chain: the scheduler then queues the device chain and the CPU computes its experts
    // while the device uploads and computes the others
    for (ggml_tensor * t : { res.dev_ids, res.cpu_ids, res.mask }) {
        ggml_backend_sched_set_tensor_backend(g.sched, t, g.backend_cpu);
        ggml_build_forward_expand(g.gf, t);
    }
    return res;
}
