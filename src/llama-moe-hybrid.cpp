#include "llama-moe-hybrid.h"

#include "ggml.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <vector>

const llama_moe_hybrid_params & llama_moe_hybrid_get_params() {
    static const llama_moe_hybrid_params params = [] {
        llama_moe_hybrid_params p;
        const char * on = getenv("LLAMA_MOE_HYBRID");
        p.enabled = on != nullptr && atoi(on) != 0;
        if (const char * e = getenv("LLAMA_MOE_HYBRID_MIN")) {
            p.n_min = std::max<int64_t>(2, atoll(e));
        }
        if (const char * e = getenv("LLAMA_MOE_HYBRID_MAX")) {
            p.n_max = atoll(e);
        }
        if (const char * e = getenv("LLAMA_MOE_HYBRID_COSTS")) {
            float a = 0, c = 0, g = 0;
            if (sscanf(e, "%f,%f,%f", &a, &c, &g) == 3 && a > 0 && c >= 0 && g > 0) {
                p.us_assign = a; p.us_cpu_expert = c; p.us_gpu_expert = g;
            }
        }
        return p;
    }();
    return params;
}

bool llama_moe_hybrid_active(int64_t n_tokens) {
    const llama_moe_hybrid_params & p = llama_moe_hybrid_get_params();
    return p.enabled && n_tokens >= p.n_min && n_tokens <= p.n_max;
}

int32_t llama_moe_hybrid_choose_c(const int32_t * counts, int64_t n_expert, const llama_moe_hybrid_params & p) {
    // histogram of experts by token count
    int32_t max_count = 0;
    for (int64_t e = 0; e < n_expert; e++) {
        max_count = std::max(max_count, counts[e]);
    }
    std::vector<int64_t> n_at(max_count + 1, 0);
    for (int64_t e = 0; e < n_expert; e++) {
        n_at[counts[e]]++;
    }
    int64_t n_used = n_expert - n_at[0];
    // c = 0: every used expert on the GPU
    int32_t best_c = 0;
    double  best_t = p.us_gpu_expert * (double) n_used;
    int64_t cpu_experts = 0, cpu_assign = 0;
    for (int32_t c = 1; c <= max_count; c++) {
        cpu_experts += n_at[c];
        cpu_assign  += (int64_t) c * n_at[c];
        const double t_cpu = p.us_assign * (double) cpu_assign + p.us_cpu_expert * (double) cpu_experts;
        const double t_gpu = p.us_gpu_expert * (double) (n_used - cpu_experts);
        const double t = std::max(t_cpu, t_gpu);
        if (t < best_t) {
            best_t = t;
            best_c = c;
        }
    }
    return best_c;
}

// custom op bodies (single task)

static void hyb_counts(const ggml_tensor * ids, int64_t n_expert, std::vector<int32_t> & counts) {
    counts.assign(n_expert, 0);
    const int32_t * d = (const int32_t *) ids->data;
    const int64_t n = ggml_nelements(ids);
    for (int64_t i = 0; i < n; i++) {
        GGML_ASSERT(d[i] >= 0 && d[i] < n_expert);
        counts[d[i]]++;
    }
}

static void hyb_dev_ids_op(ggml_tensor * dst, int ith, int nth, void * userdata) {
    GGML_UNUSED(nth);
    if (ith != 0) {
        return;
    }
    const ggml_tensor * ids = dst->src[0];
    const int64_t n_expert = (int64_t) (intptr_t) userdata;
    const int64_t n_used   = ids->ne[0];
    const int64_t n_tokens = ids->ne[1];
    std::vector<int32_t> counts;
    hyb_counts(ids, n_expert, counts);
    const int32_t c = llama_moe_hybrid_choose_c(counts.data(), n_expert, llama_moe_hybrid_get_params());

    // Rows of CPU experts are recomputed by the device chain against some other expert and masked to zero. The device
    // mul_mat_id (MMQ) must never see the same expert twice in one token's row, so each token's CPU slots take distinct
    // stand-ins it is not routed to: GPU experts first (already uploaded), then any other expert (an extra upload).
    std::vector<int32_t> standins;
    for (int64_t e = 0; e < n_expert; e++) {
        if (counts[e] > c) {
            standins.push_back((int32_t) e);
        }
    }
    for (int64_t e = 0; e < n_expert; e++) {
        if (counts[e] <= c) {
            standins.push_back((int32_t) e);
        }
    }
    const int32_t * in  = (const int32_t *) ids->data;
    int32_t       * out = (int32_t *) dst->data;
    for (int64_t t = 0; t < n_tokens; t++) {
        const int32_t * r_in  = in  + t*n_used;
        int32_t       * r_out = out + t*n_used;
        size_t next = 0;
        for (int64_t k = 0; k < n_used; k++) {
            r_out[k] = counts[r_in[k]] > c ? r_in[k] : -1;
        }
        for (int64_t k = 0; k < n_used; k++) {
            if (r_out[k] >= 0) {
                continue;
            }
            for (; next < standins.size(); next++) {
                const int32_t s = standins[next];
                bool taken = false;
                for (int64_t j = 0; j < n_used; j++) {
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

static void hyb_mask_op(ggml_tensor * dst, int ith, int nth, void * userdata) {
    GGML_UNUSED(nth);
    GGML_UNUSED(userdata);
    if (ith != 0) {
        return;
    }
    const int32_t * ids = (const int32_t *) dst->src[0]->data;
    const int32_t * dev = (const int32_t *) dst->src[1]->data;
    float         * out = (float *) dst->data;
    const int64_t n = ggml_nelements(dst);
    for (int64_t i = 0; i < n; i++) {
        out[i] = ids[i] == dev[i] ? 1.0f : 0.0f;
    }
}

static void hyb_table_op(ggml_tensor * dst, int ith, int nth, void * userdata) {
    GGML_UNUSED(nth);
    GGML_UNUSED(userdata);
    if (ith != 0) {
        return;
    }
    const int32_t * ids = (const int32_t *) dst->src[0]->data;
    const int32_t * dev = (const int32_t *) dst->src[1]->data;
    int32_t       * tbl = (int32_t *) dst->data;
    const int64_t n_expert = dst->ne[0];
    for (int64_t e = 0; e < n_expert; e++) {
        tbl[e] = LLAMA_MOE_HYBRID_DUMMY;
    }
    const int64_t n = ggml_nelements(dst->src[0]);
    for (int64_t i = 0; i < n; i++) {
        if (ids[i] == dev[i]) {
            tbl[ids[i]] = 0; // the GPU computes this expert: the CPU op skips it
        }
    }
}

ggml_tensor * llama_moe_hybrid_dev_ids(ggml_context * ctx, ggml_tensor * ids, int64_t n_expert) {
    GGML_ASSERT(ids->type == GGML_TYPE_I32 && ggml_is_contiguous(ids));
    ggml_tensor * args[1] = { ids };
    return ggml_custom_4d(ctx, GGML_TYPE_I32, ids->ne[0], ids->ne[1], 1, 1, args, 1, hyb_dev_ids_op, 1,
            (void *) (intptr_t) n_expert);
}

ggml_tensor * llama_moe_hybrid_mask(ggml_context * ctx, ggml_tensor * ids, ggml_tensor * dev_ids) {
    ggml_tensor * args[2] = { ids, dev_ids };
    return ggml_custom_4d(ctx, GGML_TYPE_F32, 1, ids->ne[0], ids->ne[1], 1, args, 2, hyb_mask_op, 1, nullptr);
}

ggml_tensor * llama_moe_hybrid_host_table(ggml_context * ctx, ggml_tensor * ids, ggml_tensor * dev_ids, int64_t n_expert) {
    ggml_tensor * args[2] = { ids, dev_ids };
    return ggml_custom_4d(ctx, GGML_TYPE_I32, n_expert, 1, 1, 1, args, 2, hyb_table_op, 1, nullptr);
}
