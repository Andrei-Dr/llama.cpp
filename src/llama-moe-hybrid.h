#pragma once

#include "ggml-backend.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

struct llama_model;
struct llama_moe_cache_layer;
struct llm_graph_context;

// Mid-batch MoE hybrid: a prompt ubatch that is too large for the MoE cache and too small to be offloaded computes the
// experts of its host-resident layers on the CPU, while the device waits. The hybrid splits the experts of each layer:
// the ones routed few tokens stay on the CPU, the others run on the device at the same time (the scheduler uploads the
// experts the device ids select, and copies the ones the MoE cache holds inside the device), and the outputs are summed.
//
// A CPU op reads the routing of the ubatch and splits the experts with a cost model, in microseconds per layer:
//   CPU    = us_assign per routed (token, expert) + us_cpu_expert per distinct expert on the CPU
//   device = us_gpu_expert per uploaded expert + us_gpu_cached per expert copied from the MoE cache
// The experts in the MoE cache go to the device. Of the others, the ones routed at most c tokens go to the CPU, with the
// c that minimizes the longer of the two sides.
//
// Environment: LLAMA_MOE_HYBRID=1 enables it, LLAMA_MOE_HYBRID_MIN / LLAMA_MOE_HYBRID_MAX set the ubatch sizes (default
// 33..127 tokens), LLAMA_MOE_HYBRID_COSTS="assign,cpu_expert,gpu_expert,gpu_cached" the cost model.

struct llama_moe_hybrid_params {
    int64_t n_min         = 33;
    int64_t n_max         = 127;
    float   us_assign     = 27.0f;
    float   us_cpu_expert = 40.0f;
    float   us_gpu_expert = 95.0f;
    float   us_gpu_cached = 10.0f;
};

// the split of the selected experts of a layer, see llama_moe_hybrid::build_lookup
struct llama_moe_hybrid_lookup {
    ggml_backend_t backend = nullptr; // the device that computes dev_ids, nullptr if the layer is not split
    ggml_tensor *  dev_ids = nullptr; // I32 [n_expert_used, n_tokens] the experts the device computes, a stand-in expert where the CPU computes
    ggml_tensor *  cpu_ids = nullptr; // I32 [n_expert_used, n_tokens] the experts the CPU computes, -1 where the device computes
    ggml_tensor *  mask    = nullptr; // F32 [1, n_expert_used, n_tokens] 1 where the device computes, 0 on the stand-in rows
};

// the experts of one layer as the split sees them
struct llama_moe_hybrid_split {
    int32_t c          = 0; // experts outside the MoE cache routed at most c tokens are computed on the CPU
    int64_t n_cpu      = 0; // experts on the CPU
    int64_t n_upload   = 0; // experts uploaded to the device
    int64_t n_cached   = 0; // experts the device reads from the MoE cache
    int64_t n_assign_cpu = 0; // routed (token, expert) pairs on the CPU
};

class llama_moe_hybrid {
public:
    // nullptr unless LLAMA_MOE_HYBRID is set to a non-zero value
    static std::unique_ptr<llama_moe_hybrid> init_from_env(const llama_model & model);

    llama_moe_hybrid(const llama_model & model, const llama_moe_hybrid_params & params);
    ~llama_moe_hybrid();

    const llama_moe_hybrid_params & get_params() const { return params; }

    // the split of the selected experts of a layer for this ubatch, built after the expert input and before the expert
    // matmuls; no backend if the layer is not split (ubatch size, experts not in plain host memory, no GPU, ...)
    // the three tensors are computed by one CPU split, added to the graph here
    llama_moe_hybrid_lookup build_lookup(const llm_graph_context & g, const llama_moe_cache_layer & layer, ggml_tensor * selected_experts) const;

    // the cost model: counts[e] = tokens routed to expert e, cached[e] != 0 if the MoE cache holds it (cached may be nullptr)
    static llama_moe_hybrid_split choose_split(const int32_t * counts, const uint8_t * cached, int64_t n_expert, const llama_moe_hybrid_params & params);

    // the device ids of one ubatch: ids [n_used, n_tokens] row by row, on_cpu[e] != 0 for the experts the CPU computes
    // an expert on the CPU is replaced by a stand-in the device computes anyway where possible; the ids of a token stay distinct
    static void fill_dev_ids(const int32_t * ids, int64_t n_used, int64_t n_tokens, const uint8_t * on_cpu, int64_t n_expert, int32_t * dev_ids);

    // the routing split of one layer, written by the split op (opaque)
    struct layer_state;

private:
    llama_moe_hybrid_params params;
    mutable std::vector<layer_state> layers;
};

using llama_moe_hybrid_ptr = std::unique_ptr<llama_moe_hybrid>;
