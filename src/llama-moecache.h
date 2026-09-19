#pragma once

// GPU-resident cache for MoE expert weights that -ot / --n-cpu-moe pinned to host memory.
// Based on ggml-org/llama.cpp#27861, extended for fused gate_up experts, per-expert scales,
// GELU experts and gated admission.
//
// Mechanism (no custom kernels):
//  - per cached layer, companion tensors of shape [ne0, ne1, n_slots+1] live in the device
//    buffer type of that layer; slot n_slots is permanently zero (the "dummy" slot).
//  - an I32 table maps expert id -> slot, or n_slots when uncached. One copy on the device
//    (read by get_rows to remap ids for the cache-side mul_mat_id chain) and one on the host
//    (read by the CPU mul_mat_id via src[3] to SKIP cached ids, zeroing their dst rows).
//  - the two down-projection outputs are summed; uncached ids contribute 0 through the cache
//    chain (zero slot) and cached ids contribute 0 through the CPU chain (skip), so the sum
//    is the full result.
//  - misses are always computed on the CPU; nothing ever stalls on an upload. Uploads run on
//    a worker thread and a slot is only published once its copy has completed.
//  - admission is gated: an expert is uploaded only after `admit` misses within the last
//    `window` tokens of its layer. Ungated admission churns (uploads ~= evictions) and
//    saturates PCIe; eviction is plain LRU.
//
// Enabled via llama_context_params.n_moe_cache_slots (CLI: --moe-expert-cache).

#include <cstdint>

struct llama_model;
struct ggml_tensor;

struct llama_moe_cache_params {
    int32_t n_slots     = 0;  // slots per host-resident expert layer (0 = disabled)
    int32_t max_inserts = 2;  // max expert uploads per layer per decode step
    int32_t window      = 16; // admission window, in tokens
    int32_t admit       = 3;  // misses within the window before an expert is uploaded (1 = ungated)
};

struct llama_moe_cache_layer {
    int il = -1;

    int32_t n_slots = 0;

    // host-resident source weights (the authoritative experts);
    // either gate_up_src, or up_src + gate_src
    ggml_tensor * up_src      = nullptr;
    ggml_tensor * gate_src    = nullptr;
    ggml_tensor * gate_up_src = nullptr;
    ggml_tensor * down_src    = nullptr;

    // device-resident cache slots, ne[2] == n_slots + 1 (last slot all zeros)
    ggml_tensor * up_c      = nullptr;
    ggml_tensor * gate_c    = nullptr;
    ggml_tensor * gate_up_c = nullptr;
    ggml_tensor * down_c    = nullptr;

    // per-expert scales readable on the device (a device copy when the model keeps them on the host), or nullptr;
    // up_s_c scales gate_up when the layer is fused
    ggml_tensor * up_s_c   = nullptr;
    ggml_tensor * gate_s_c = nullptr;
    ggml_tensor * down_s_c = nullptr;

    // expert id -> slot (or n_slots when uncached); I32 [1, n_expert]
    ggml_tensor * dev_table  = nullptr;
    ggml_tensor * host_table = nullptr;
};

// build the cache for every host-resident expert layer of the model.
// Safe to call more than once; only the first call does work.
void llama_moe_cache_init(const llama_model & model, const llama_moe_cache_params & params);

// true once a cache has been built
bool llama_moe_cache_active();

// key = the layer's gate_up_exps tensor when fused, else its up_exps tensor.
// nullptr when the cache is disabled or this tensor has no cached layer
const llama_moe_cache_layer * llama_moe_cache_lookup(const ggml_tensor * key);

// publish finished uploads and schedule new ones. Call between graph executions only,
// after the backend has been synchronized.
void llama_moe_cache_step();
