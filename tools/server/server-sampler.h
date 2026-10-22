#pragma once

// The machine sampler of /stats and /stats/history: one thread of its own that wakes every period
// (LLAMA_SERVER_SAMPLE_MS, default 1000; 0 = no thread, no history, no machine data), reads the GPUs through NVML
// (libnvidia-ml loaded at run time, LLAMA_SERVER_NVML_LIB overrides the name, LLAMA_SERVER_NVML=0 skips it; without it the
// GPU fields are null), this process's CPU time and the host's CPU and memory (/proc on Linux), lets the server add the
// columns of its latest /stats snapshot, and appends the row to the history. It never posts a task, never takes a lock
// of the decode path and never queries a device from the main loop.

#include "server-stats.h"

#include <atomic>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

struct server_gpu_sample {
    std::string name;
    double util       = NAN; // %
    double mem_util   = NAN; // % of time device memory was read or written
    double vram_used  = NAN; // bytes
    double vram_total = NAN;
    double pcie_rx    = NAN; // bytes/s, host -> GPU
    double pcie_tx    = NAN; // bytes/s, GPU -> host
    double temp_c     = NAN;
    double power_w    = NAN;
    double clock_mhz  = NAN; // SM clock

    json to_json() const;
};

struct server_host_sample {
    double proc_cpu_us    = NAN; // this process, user + system, cumulative
    double host_busy_us   = NAN; // all cores, cumulative
    double host_total_us  = NAN;
    double host_iowait_us = NAN;
    double mem_avail      = NAN; // bytes
    double mem_total      = NAN;
    double swap_used      = NAN;

    json to_json() const;
};

struct server_machine_sample {
    int64_t t_us    = 0; // ggml_time_us
    int64_t unix_ms = 0;
    std::vector<server_gpu_sample> gpus;
    server_host_sample             host;

    // the sampler's own cost, all samples so far
    uint64_t n         = 0;
    uint64_t sum_us    = 0;
    uint64_t max_us    = 0;
    std::string nvml;    // "ok", "off", or why it is not available

    json to_json() const;
};

class server_nvml;

class server_sampler {
public:
    // fill: called on the sampler thread with a row whose machine columns are set, to add the server's columns
    using fill_t = std::function<void(server_stats_row &)>;

    // starts the thread when period_us > 0
    server_sampler(int64_t period_us, fill_t fill);
    ~server_sampler();

    server_sampler(const server_sampler &) = delete;
    server_sampler & operator=(const server_sampler &) = delete;

    int64_t period() const { return period_us; }

    // any thread; nullptr before the first sample (or with the sampler off)
    std::shared_ptr<const server_machine_sample> latest() const;
    std::shared_ptr<const server_stats_history::view> history() const;

    // the period from LLAMA_SERVER_SAMPLE_MS (default 1000 ms)
    static int64_t period_us_from_env();

private:
    void run();
    void sample_once(server_nvml & nvml, const std::string & nvml_state);

    const int64_t period_us;
    fill_t        fill;

    server_stats_history hist;
    server_machine_sample m; // the sampler thread's working copy (cost totals)

    std::shared_ptr<const server_machine_sample> snap; // std::atomic_load / std::atomic_store only

    std::mutex              mutex;
    std::condition_variable cv;
    bool                    stopping = false;
    std::thread             thread;
};
