#pragma once

// Live statistics for GET /stats.
//
// The main loop records them where the work already happens (the decode, the slot selection, the slot release: the
// points print_timings() and release() log) and publishes an immutable snapshot at most every `period_us`, at a token
// boundary. The HTTP thread serves the latest snapshot: no queue task and no lock shared with the decode path (one
// atomic shared_ptr load). Rates are derived by the reader from cumulative counters, so any number of readers can poll
// without disturbing each other (unlike /metrics, whose gauges reset on every scrape).

#include "server-common.h"
#include "server-prompt-disk.h"

#include <cstdint>
#include <deque>
#include <functional>
#include <map>
#include <memory>
#include <string>
#include <vector>

// fixed-edge histogram: counts[i] = values in (edges[i-1], edges[i]]; counts.back() = values above the last edge.
// NaN (not measured) is ignored.
struct server_stats_hist {
    std::vector<double>   edges;
    std::vector<uint64_t> counts;
    uint64_t n   = 0;
    double   sum = 0.0;

    server_stats_hist() = default;
    explicit server_stats_hist(std::vector<double> edges);

    void add(double v);
    json to_json() const;
};

// a duration series in microseconds: count, total, last, max and an EMA (alpha 1/16, seeded by the first sample)
struct server_stats_timer {
    uint64_t n       = 0;
    uint64_t sum_us  = 0;
    uint64_t last_us = 0;
    uint64_t max_us  = 0;
    double   ema_us  = 0.0;

    void add(uint64_t us);
    json to_json() const;
};

// how a task's slot was chosen (get_available_slot)
struct server_stats_selection {
    int    id_task = -1;             // the task this record is for (a slot keeps it until that task is released)
    std::string how;                 // "id", "lcp", "lru"; empty = not recorded
    float  f_sim         = -1.0f;    // lcp: similarity of the best slot; -1 = not computed
    float  f_keep        = -1.0f;    // lcp: share of the slot's tokens the request keeps
    bool   cache_update  = false;    // the slot went through the prompt cache (save + load)
    bool   disk_restored = false;    // ... and the disk tier restored a stored prompt into it
    int32_t n_disk_tokens = 0;       // ... of this many tokens
    float  f_sim_disk    = -1.0f;    // ... reusing this share of the new prompt
    double t_cache_ms    = 0.0;      // time of the prompt cache update

    json to_json() const;
};

// one completed task: the fields print_timings(), release() and get_available_slot() log
struct server_stats_task {
    int     id       = -1;
    int     id_slot  = -1;
    int64_t t_end_us = 0;               // release time (ggml_time_us)

    double  t_queue_ms  = 0.0;          // posted -> prompt processing started (includes the prompt cache update); NaN = unknown
    double  t_prompt_ms = 0.0;          // slot launched -> last prompt batch computed (first token)
    double  t_gen_ms    = 0.0;          // first token -> last token

    int32_t n_prompt           = 0;     // the request's prompt tokens
    int32_t n_prompt_cached    = 0;     // reused from the slot or the prompt cache
    int32_t n_prompt_processed = 0;
    int32_t n_gen              = 0;
    int32_t n_gen_steps        = 0;     // decode steps of the generation (the first token is free)
    int32_t n_ctx              = 0;     // tokens in the slot at release
    bool    truncated          = false;

    int32_t n_draft             = 0;
    int32_t n_draft_accepted    = 0;
    int32_t n_draft_verif_steps = 0;
    std::vector<uint64_t> accepted_per_pos; // [j] = verification steps that accepted at least j+1 draft tokens

    int32_t     n_graphs_reused = 0;    // the target context's reused graphs at release (cumulative per context)
    std::string stop;                   // "eos", "word", "limit", "none" (error, cancel)

    server_stats_selection sel;

    double ttft_ms()          const; // NaN without a generated token
    double gen_ms_per_token() const; // NaN without a decode step
    double accept_rate()      const; // NaN without drafts

    json to_json() const;
};

// one slot as the requests table shows it
struct server_stats_slot {
    int         id      = -1;
    int         id_task = -1;
    const char * state = "idle";        // "idle", "waiting", "started", "prompt", "generating" (static strings)
    int32_t     n_prompt           = 0;
    int32_t     n_prompt_cached    = 0;
    int32_t     n_prompt_processed = 0;
    int32_t     n_gen              = 0;
    int32_t     n_ctx              = 0; // tokens in the slot (KV)
    int32_t     n_ctx_max          = 0;
    int32_t     n_draft            = 0;
    int32_t     n_draft_accepted   = 0;
    double      t_queue_ms         = 0.0;
    double      t_age_ms           = 0.0; // since the slot launched the task
    double      t_prompt_ms        = 0.0;
    double      t_gen_ms           = 0.0;

    json to_json() const;
};

// one GET /slots row: what server_slot::to_json() writes, captured as values so a snapshot copies no JSON.
// server_slot_row_json() is the one serializer for the task path and the snapshot path, so both answer byte for byte alike
struct server_slot_row {
    int      id            = -1;
    int32_t  n_ctx         = 0;
    bool     speculative   = false;
    bool     is_processing = false;
    bool     has_task      = false; // the slot ran a task (current or previous): the task fields below are set
    int      id_task       = -1;
    int32_t  n_prompt_tokens = 0;
    uint64_t n_prompt_tokens_processed = 0;
    uint64_t n_prompt_tokens_cache     = 0;
    std::shared_ptr<const json> params;  // the task's params.to_json(only_metrics); built once per task
    bool     has_next_token = false;
    bool     has_new_line   = false;
    int32_t  n_remain       = 0;
    uint64_t n_decoded      = 0;
};

json server_slot_row_json(const server_slot_row & r);

// what changes only when a task completes: rebuilt then and shared by every snapshot until the next completion
struct server_stats_done {
    server_stats_hist h_ttft_ms;
    server_stats_hist h_gen_ms_per_tok;
    server_stats_hist h_accept;

    uint64_t n_tasks     = 0;
    uint64_t n_truncated = 0;
    std::map<std::string, uint64_t> n_stop; // by stop reason
    std::map<std::string, uint64_t> n_sel;  // completed tasks by slot selection ("id", "lcp", "lru", "cache_update", "disk_restored")

    std::vector<server_stats_task> tasks;   // completed, oldest first
};

struct server_stats_snapshot {
    uint64_t seq        = 0;            // publishes so far
    int64_t  t_us       = 0;            // ggml_time_us at publish
    int64_t  t_start_us = 0;            // server start (metrics.t_start), ggml_time_us clock: identifies the server instance

    // upstream's cumulative counters (server_metrics); its *_bucket members are /metrics' scrape windows, unused here
    server_metrics metrics;
    int32_t        n_slots      = 0;
    int32_t        n_deferred   = 0; // tasks waiting for a slot
    int32_t        n_processing = 0; // slots processing a task

    // GET /slots served from the snapshot (rows serialized by the HTTP thread); false = not captured (/slots off, or
    // LLAMA_SERVER_SLOTS_DEBUG, whose rows carry the detokenized prompt)
    bool                         has_slot_rows = false;
    std::vector<server_slot_row> slot_rows;

    // CPU-split time of the generation steps (llama_get_cpu_timing): with the experts on the CPU, the expert phase
    struct {
        uint64_t             n_graphs = 0; // timed graphs (one per generation step)
        int64_t              cpu_us   = 0;
        std::vector<int64_t> layer_us;     // by layer index; empty when no split named a layer
    } phase;

    // decode wall time of one target batch with output (llama_decode + synchronize), by kind: gen = no prompt tokens.
    // prompt: only the synchronized (last) batch of a prompt is timed, and it can include earlier unsynchronized batches
    server_stats_timer decode_gen;
    server_stats_timer decode_prompt;
    // one main-loop iteration that only generated (drafting, decode, sampling, speculative accept): the decode step
    server_stats_timer step_gen;
    server_stats_hist  h_step_ms;

    std::vector<server_stats_slot> slots;
    std::shared_ptr<const server_stats_done> done; // never null in a published snapshot

    // the prompt cache disk tier (--cache-disk), when there is one
    bool                     has_disk     = false;
    server_prompt_disk_stats disk;
    uint64_t                 disk_entries = 0;
    uint64_t                 disk_bytes   = 0;
    uint64_t                 disk_limit   = 0; // 0 = no cap

    server_stats_timer publish; // cost of building and swapping in the snapshots before this one

    // max_tasks: the newest completed tasks to include (< 0 = the whole ring)
    json to_json(int max_tasks = -1) const;
    json disk_json() const;
    json phase_json() const;
};

// the MoE expert cache's latest snapshot (llama_moe_cache_get_stats), or null without a cache. Any thread: reads the
// snapshot the cache published at its last step, takes no lock the decode path uses
json server_stats_moe_cache_json();

class server_stats {
public:
    // n_ring: completed tasks kept; period_us: least time between two publishes (0 = never publish; recording continues)
    explicit server_stats(size_t n_ring = 128, int64_t period_us = 250000);

    // writer side: the server's main loop thread only

    // the working copy the hooks update; published by publish()
    server_stats_snapshot & cur() { return w; }

    // the per-completion part (histograms, counters, the ring): changing it marks it for the next publish
    server_stats_done & done() { done_dirty = true; return d; }

    // a completed task: the ring, the histograms and the counters
    void add_task(server_stats_task && t);

    // the period has elapsed since the last publish
    bool due(int64_t t_now_us) const;

    // fill (may be null) writes the volatile fields (metrics, slots) into the working copy, which is then copied into a
    // new immutable snapshot and swapped in; the per-completion part is copied only when it changed. The whole call is
    // timed and reported by the next snapshot.
    void publish(int64_t t_now_us, const std::function<void(server_stats_snapshot &)> & fill);

    int64_t period() const { return period_us; }

    // reader side: any thread. nullptr until the first publish
    std::shared_ptr<const server_stats_snapshot> latest() const;

private:
    const size_t  n_ring;
    const int64_t period_us;

    server_stats_snapshot         w;
    server_stats_done             d;  // tasks unused: the ring below is copied in at publish
    std::deque<server_stats_task> ring;
    bool                          done_dirty = true;
    int64_t                       t_last_publish_us = 0;
    bool                          published = false;

    std::shared_ptr<const server_stats_snapshot> snap; // std::atomic_load / std::atomic_store only
};
