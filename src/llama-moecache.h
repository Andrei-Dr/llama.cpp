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
//  - the cache chain only serves (and the CPU op only observes) batches of 1-4 tokens, so a prompt
//    would teach the cache nothing. Batches of at least `warm` tokens therefore report their routing
//    through a CPU custom op, and the next step() re-ranks every slot by that frequency (experts the
//    batch never used keep their slots only as filler). The uploads are asynchronous like any other.
//
// Enabled via llama_context_params.n_moe_cache_slots (CLI: --moe-expert-cache).

#include "ggml-backend.h"

#include <cstddef>
#include <cstdint>

struct llama_model;
struct ggml_context;
struct ggml_tensor;

struct llama_moe_cache_params {
    int32_t n_slots     = 0;  // slots per host-resident expert layer (0 = disabled)
    int32_t max_inserts = 2;  // max expert uploads per layer per decode step
    int32_t window      = 16; // admission window, in tokens
    int32_t admit       = 3;  // misses within the window before an expert is uploaded (1 = ungated)
    int32_t warm        = 32; // batches of at least this many tokens re-rank the slots by their routing (0 = disabled)
    float   bias        = 0;  // cache-aware routing: a cached expert's selection score is its router prob * (1 + bias); 0 = exact routing
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

    // cache-aware routing (bias > 0 only): F32 [n_expert], 1 + bias for cached experts, 1 otherwise; nullptr when off
    ggml_tensor * sel_scale = nullptr;

    // expert id -> slot (or n_slots when uncached); I32 [1, n_expert]
    ggml_tensor * dev_table  = nullptr;
    ggml_tensor * host_table = nullptr;
};

// build the cache for every host-resident expert layer of the model.
// Safe to call more than once; only the first call does work.
void llama_moe_cache_init(const llama_model & model, const llama_moe_cache_params & params);

// true once a cache has been built
bool llama_moe_cache_active();

// true when the cache mirrors this model's experts (a draft / MTP model never owns it)
bool llama_moe_cache_owned_by(const llama_model & model);

// prefill mode: the cache chain only serves 1-4 token batches, so during a large batch the device slots are dead weight.
// suspend() drops queued uploads, waits for one in progress, frees the device slot buffers (the host tables stay) and makes
// lookups return nullptr, so graphs are built without the cache chain; returns the bytes freed. resume() re-allocates the
// slots empty (all-or-nothing; false = could not, the cache stays off and outputs are unchanged) and keeps the routing counts
// the prefill collected, so the next step() re-ranks the slots by the prompt. Call both between graph executions only,
// with the backends synchronized.
size_t llama_moe_cache_suspend();
bool   llama_moe_cache_resume();

// key = the layer's gate_up_exps tensor when fused, else its up_exps tensor.
// nullptr when the cache is disabled or this tensor has no cached layer
const llama_moe_cache_layer * llama_moe_cache_lookup(const ggml_tensor * key);

// graph hook for a batch that is too large for the cache chain: a CPU node that reports the batch's top-k
// expert ids [n_expert_used, n_tokens] to the cache. The caller expands it into the graph and pins it to the
// CPU backend. nullptr when this tensor has no cached layer, warm-up is disabled or the batch is too small
ggml_tensor * llama_moe_cache_build_warm_obs(ggml_context * ctx, const ggml_tensor * key, ggml_tensor * selected_experts);

// publish finished uploads and schedule new ones. Call between graph executions only,
// after the backend has been synchronized.
void llama_moe_cache_step();

// pre-gated prefetch (LLAMA_MOE_PREFETCH=N, N = max uploads per layer per decode step; 0/unset = off).
// During layer L of a decode step, layer L+1's router applied to layer L's MoE input predicts L+1's experts (top-k,
// LLAMA_MOE_PREFETCH_TOPK, default 16). A CPU node at the head of layer L's host-expert split uploads the predicted experts
// that are not cached into LRU slots of layer L+1 and publishes them IN THE SAME STEP: the slot copies and the device table
// are enqueued with set_tensor_async on the compute backend's stream (behind layer L's cache-hit kernels, ahead of every
// kernel of layer L+1), and the host table is written directly (layer L+1's host-expert op reads it later on this thread).
// Layer L+1 then computes those experts on the device and the host skips them. Same math as any cache hit.
// set_backend registers the compute backend that runs the cached layers (nullptr = unregister); only the owning model's
// context may register. build_prefetch returns the CPU node for the layer keyed by key_next, or nullptr.
bool          llama_moe_cache_set_backend(const llama_model & model, ggml_backend_t backend); // true = registered
// unregister `backend` if (and only if) it is the one registered (a context tearing down must not switch off another's)
void          llama_moe_cache_release_backend(const llama_model & model, ggml_backend_t backend);
int           llama_moe_cache_prefetch_budget(); // 0 = prefetch off (unset, no backend, suspended, or cache-aware routing)
int           llama_moe_cache_prefetch_topk();
bool          llama_moe_cache_prefetch_ok(const ggml_tensor * key_next); // budget > 0 and the layer keyed by key_next qualifies
ggml_tensor * llama_moe_cache_build_prefetch(ggml_context * ctx, const ggml_tensor * key_next, ggml_tensor * pred_ids);
// the matching tail node: expand it as the LAST node of the same host-expert split (after the host down-projection `after`)
ggml_tensor * llama_moe_cache_build_prefetch_join(ggml_context * ctx, ggml_tensor * after, int il, bool has_head);
int           llama_moe_cache_prefetch_ahead(); // layers ahead the prediction targets: 2 (window mode) or 1 (step mode)
