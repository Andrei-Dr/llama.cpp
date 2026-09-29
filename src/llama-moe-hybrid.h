#pragma once

// Mid-batch MoE hybrid (research/moe-hybrid-design-2026-09-29.md in the local-ai repo): for a prompt ubatch of N tokens
// whose experts live in host memory, experts routed few tokens run on the CPU and the rest are streamed to the GPU, both
// at once. A CPU custom op reads the ubatch's routing and splits the experts by a cost model; its outputs drive the two
// chains of build_moe_ffn (the same structure as the expert cache: the CPU mul_mat_id skips ids whose table entry is not
// the dummy, a device chain computes them, the outputs are summed).
//
// Enabled with LLAMA_MOE_HYBRID=1 for N in [LLAMA_MOE_HYBRID_MIN, LLAMA_MOE_HYBRID_MAX] (default 32..512).
// Cost model per layer, in microseconds (measured on an i5-10400F + GTX 1650 SUPER): CPU = US_ASSIGN per routed (token, expert) + US_CPU_EXPERT per
// distinct CPU expert; GPU = US_GPU_EXPERT per streamed expert. Overrides: LLAMA_MOE_HYBRID_COSTS="assign,cpu_expert,gpu_expert".

#include <cstdint>

struct ggml_context;
struct ggml_tensor;

struct llama_moe_hybrid_params {
    bool    enabled       = false;
    int64_t n_min         = 32;
    int64_t n_max         = 512;
    float   us_assign     = 27.0f;
    float   us_cpu_expert = 40.0f;
    float   us_gpu_expert = 95.0f;
};

// process-wide parameters, read once from the environment
const llama_moe_hybrid_params & llama_moe_hybrid_get_params();

// true when a ubatch of n_tokens should run the hybrid
bool llama_moe_hybrid_active(int64_t n_tokens);

// ids: contiguous I32 [n_expert_used, n_tokens] (the router's top-k). Returns I32 [n_expert_used, n_tokens]: ids of experts
// sent to the GPU unchanged, the others remapped to one GPU expert (their rows are masked out).
ggml_tensor * llama_moe_hybrid_dev_ids(ggml_context * ctx, ggml_tensor * ids, int64_t n_expert);

// F32 [1, n_expert_used, n_tokens]: 1 where the device chain's row is real (dev_ids == ids), 0 where it is masked.
ggml_tensor * llama_moe_hybrid_mask(ggml_context * ctx, ggml_tensor * ids, ggml_tensor * dev_ids);

// I32 [n_expert] host table for the CPU mul_mat_id (src[3]): 0 for experts the GPU computes (skipped by the CPU op),
// LLAMA_MOE_HYBRID_DUMMY for the rest (computed by the CPU op). The op's op_params[0] must be LLAMA_MOE_HYBRID_DUMMY.
constexpr int32_t LLAMA_MOE_HYBRID_DUMMY = -1;
ggml_tensor * llama_moe_hybrid_host_table(ggml_context * ctx, ggml_tensor * ids, ggml_tensor * dev_ids, int64_t n_expert);

// the split point the policy picks for one layer (exposed for tests): experts with count <= returned c go to the CPU;
// c = 0 sends every used expert to the GPU
int32_t llama_moe_hybrid_choose_c(const int32_t * counts, int64_t n_expert, const llama_moe_hybrid_params & p);
