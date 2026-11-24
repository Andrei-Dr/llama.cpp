#include "server-sampler.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#if !defined(_WIN32)
#include <dlfcn.h>
#include <sys/resource.h>
#include <unistd.h>
#endif

//
// NVML, loaded at run time: the build needs no CUDA toolkit and a machine without the driver just has no GPU fields.
// The declarations below are the NVML C API's (nvml.h), restated so no header is needed.
//

typedef int   nvml_return_t; // NVML_SUCCESS = 0
typedef void * nvml_device_t;
struct nvml_utilization_t { unsigned int gpu; unsigned int memory; };
struct nvml_memory_t      { unsigned long long total; unsigned long long free; unsigned long long used; };

static constexpr int k_nvml_pcie_tx_bytes  = 0; // NVML_PCIE_UTIL_TX_BYTES
static constexpr int k_nvml_pcie_rx_bytes  = 1; // NVML_PCIE_UTIL_RX_BYTES
static constexpr int k_nvml_temperature_gpu = 0; // NVML_TEMPERATURE_GPU
static constexpr int k_nvml_clock_sm       = 1; // NVML_CLOCK_SM

static constexpr int k_min_period_ms = 20;

class server_nvml {
public:
    // false with the reason in why
    bool open(std::string & why) {
#if defined(_WIN32)
        why = "not supported on this platform";
        return false;
#else
        const char * env = getenv("LLAMA_SERVER_NVML_LIB");
        const char * name = env && env[0] ? env : "libnvidia-ml.so.1";
        lib = dlopen(name, RTLD_NOW | RTLD_LOCAL);
        if (!lib) {
            const char * e = dlerror();
            why = std::string("cannot load ") + name + (e ? std::string(": ") + std::string(e).substr(0, 160) : std::string());
            return false;
        }
        bool all = true;
        auto sym = [&](const char * s) {
            void * p = dlsym(lib, s);
            all = all && p != nullptr;
            return p;
        };
        init      = (nvml_return_t (*)())                                       sym("nvmlInit_v2");
        shutdown  = (nvml_return_t (*)())                                       sym("nvmlShutdown");
        count     = (nvml_return_t (*)(unsigned int *))                         sym("nvmlDeviceGetCount_v2");
        handle    = (nvml_return_t (*)(unsigned int, nvml_device_t *))          sym("nvmlDeviceGetHandleByIndex_v2");
        get_name  = (nvml_return_t (*)(nvml_device_t, char *, unsigned int))    sym("nvmlDeviceGetName");
        util      = (nvml_return_t (*)(nvml_device_t, nvml_utilization_t *))    sym("nvmlDeviceGetUtilizationRates");
        mem       = (nvml_return_t (*)(nvml_device_t, nvml_memory_t *))         sym("nvmlDeviceGetMemoryInfo");
        pcie      = (nvml_return_t (*)(nvml_device_t, int, unsigned int *))     sym("nvmlDeviceGetPcieThroughput");
        temp      = (nvml_return_t (*)(nvml_device_t, int, unsigned int *))     sym("nvmlDeviceGetTemperature");
        power     = (nvml_return_t (*)(nvml_device_t, unsigned int *))          sym("nvmlDeviceGetPowerUsage");
        clock     = (nvml_return_t (*)(nvml_device_t, int, unsigned int *))     sym("nvmlDeviceGetClockInfo");
        if (!all) {
            why = std::string(name) + " lacks an NVML function";
            return false;
        }
        // optional: the PCIe link, for the page's bandwidth ceiling
        link_gen       = (nvml_return_t (*)(nvml_device_t, unsigned int *)) dlsym(lib, "nvmlDeviceGetCurrPcieLinkGeneration");
        link_width     = (nvml_return_t (*)(nvml_device_t, unsigned int *)) dlsym(lib, "nvmlDeviceGetCurrPcieLinkWidth");
        link_gen_max   = (nvml_return_t (*)(nvml_device_t, unsigned int *)) dlsym(lib, "nvmlDeviceGetMaxPcieLinkGeneration");
        link_width_max = (nvml_return_t (*)(nvml_device_t, unsigned int *)) dlsym(lib, "nvmlDeviceGetMaxPcieLinkWidth");
        if (init() != 0) {
            why = "nvmlInit failed";
            return false;
        }
        initialized = true;
        unsigned int n = 0;
        if (count(&n) != 0) {
            why = "nvmlDeviceGetCount failed";
            return false;
        }
        for (unsigned int i = 0; i < n; ++i) {
            nvml_device_t d = nullptr;
            if (handle(i, &d) != 0) {
                continue;
            }
            char buf[96] = {};
            devs.push_back(d);
            names.push_back(get_name(d, buf, sizeof(buf)) == 0 ? std::string(buf) : std::string());
        }
        if (devs.empty()) {
            why = "no NVML device";
            return false;
        }
        return true;
#endif
    }

    // one sample per device; a query that fails leaves its field null
    void sample(std::vector<server_gpu_sample> & out) const {
        out.resize(devs.size());
        for (size_t i = 0; i < devs.size(); ++i) {
            server_gpu_sample & g = out[i];
            g = server_gpu_sample();
            g.name = names[i];
            nvml_device_t d = devs[i];
            nvml_utilization_t u;
            if (util(d, &u) == 0) {
                g.util     = u.gpu;
                g.mem_util = u.memory;
            }
            nvml_memory_t mi;
            if (mem(d, &mi) == 0) {
                g.vram_used  = (double) mi.used;
                g.vram_total = (double) mi.total;
            }
            unsigned int v = 0;
            // NVML counts the bytes of one 20 ms window per call (KB/s, taken as 1024 bytes): a point sample, not an average
            if (pcie(d, k_nvml_pcie_rx_bytes, &v) == 0) {
                g.pcie_rx = 1024.0 * v;
            }
            if (pcie(d, k_nvml_pcie_tx_bytes, &v) == 0) {
                g.pcie_tx = 1024.0 * v;
            }
            if (temp(d, k_nvml_temperature_gpu, &v) == 0) {
                g.temp_c = v;
            }
            if (power(d, &v) == 0) {
                g.power_w = v / 1000.0; // mW
            }
            if (clock(d, k_nvml_clock_sm, &v) == 0) {
                g.clock_mhz = v;
            }
            auto link = [&](nvml_return_t (*f)(nvml_device_t, unsigned int *), double & out) {
                unsigned int x = 0;
                if (f && f(d, &x) == 0) {
                    out = x;
                }
            };
            link(link_gen,       g.pcie_gen);
            link(link_width,     g.pcie_width);
            link(link_gen_max,   g.pcie_gen_max);
            link(link_width_max, g.pcie_width_max);
        }
    }

    ~server_nvml() {
#if !defined(_WIN32)
        if (initialized && shutdown) {
            shutdown();
        }
        if (lib) {
            dlclose(lib);
        }
#endif
    }

private:
    void * lib         = nullptr;
    bool   initialized = false;

    nvml_return_t (*init)()                                    = nullptr;
    nvml_return_t (*shutdown)()                                = nullptr;
    nvml_return_t (*count)(unsigned int *)                     = nullptr;
    nvml_return_t (*handle)(unsigned int, nvml_device_t *)     = nullptr;
    nvml_return_t (*get_name)(nvml_device_t, char *, unsigned int) = nullptr;
    nvml_return_t (*util)(nvml_device_t, nvml_utilization_t *) = nullptr;
    nvml_return_t (*mem)(nvml_device_t, nvml_memory_t *)       = nullptr;
    nvml_return_t (*pcie)(nvml_device_t, int, unsigned int *)  = nullptr;
    nvml_return_t (*temp)(nvml_device_t, int, unsigned int *)  = nullptr;
    nvml_return_t (*power)(nvml_device_t, unsigned int *)      = nullptr;
    nvml_return_t (*clock)(nvml_device_t, int, unsigned int *) = nullptr;
    nvml_return_t (*link_gen)(nvml_device_t, unsigned int *)       = nullptr;
    nvml_return_t (*link_width)(nvml_device_t, unsigned int *)     = nullptr;
    nvml_return_t (*link_gen_max)(nvml_device_t, unsigned int *)   = nullptr;
    nvml_return_t (*link_width_max)(nvml_device_t, unsigned int *) = nullptr;

    std::vector<nvml_device_t> devs;
    std::vector<std::string>   names;
};

//
// host
//

static void sample_host(server_host_sample & h) {
    h = server_host_sample();
#if !defined(_WIN32)
    struct rusage ru;
    if (getrusage(RUSAGE_SELF, &ru) == 0) {
        h.proc_cpu_us = (double) ru.ru_utime.tv_sec*1e6 + ru.ru_utime.tv_usec + (double) ru.ru_stime.tv_sec*1e6 + ru.ru_stime.tv_usec;
    }
#endif
#if defined(__linux__)
    static const double us_per_tick = 1e6 / (double) std::max(1L, sysconf(_SC_CLK_TCK));
    if (FILE * f = fopen("/proc/stat", "r")) {
        unsigned long long user = 0, nice = 0, sys = 0, idle = 0, iowait = 0, irq = 0, softirq = 0, steal = 0;
        if (fscanf(f, "cpu %llu %llu %llu %llu %llu %llu %llu %llu", &user, &nice, &sys, &idle, &iowait, &irq, &softirq, &steal) >= 5) {
            const double busy = (double) (user + nice + sys + irq + softirq + steal);
            h.host_busy_us   = busy * us_per_tick;
            h.host_iowait_us = (double) iowait * us_per_tick;
            h.host_total_us  = (busy + idle + iowait) * us_per_tick;
        }
        fclose(f);
    }
    if (FILE * f = fopen("/proc/meminfo", "r")) {
        char key[64];
        unsigned long long kb = 0;
        double swap_total = NAN, swap_free = NAN;
        while (fscanf(f, "%63s %llu kB\n", key, &kb) == 2) {
            if (strcmp(key, "MemTotal:") == 0)     { h.mem_total = kb * 1024.0; }
            if (strcmp(key, "MemAvailable:") == 0) { h.mem_avail = kb * 1024.0; }
            if (strcmp(key, "SwapTotal:") == 0)    { swap_total = kb * 1024.0; }
            if (strcmp(key, "SwapFree:") == 0)     { swap_free  = kb * 1024.0; }
        }
        h.swap_used = swap_total - swap_free;
        fclose(f);
    }
#endif
}

//
// JSON (NaN = not measured = null)
//

static json num_or_null(double v) {
    return std::isnan(v) ? json(nullptr) : json(v);
}

json server_gpu_sample::to_json() const {
    return json {
        { "name",       name                   },
        { "util",       num_or_null(util)      },
        { "mem_util",   num_or_null(mem_util)  },
        { "vram_used",  num_or_null(vram_used) },
        { "vram_total", num_or_null(vram_total)},
        { "pcie_rx",    num_or_null(pcie_rx)   },
        { "pcie_tx",    num_or_null(pcie_tx)   },
        { "temp_c",     num_or_null(temp_c)    },
        { "power_w",    num_or_null(power_w)   },
        { "clock_mhz",  num_or_null(clock_mhz) },
        { "pcie_gen",       num_or_null(pcie_gen)       },
        { "pcie_width",     num_or_null(pcie_width)     },
        { "pcie_gen_max",   num_or_null(pcie_gen_max)   },
        { "pcie_width_max", num_or_null(pcie_width_max) },
    };
}

json server_host_sample::to_json() const {
    return json {
        { "proc_cpu_us",    num_or_null(proc_cpu_us)    },
        { "host_busy_us",   num_or_null(host_busy_us)   },
        { "host_total_us",  num_or_null(host_total_us)  },
        { "host_iowait_us", num_or_null(host_iowait_us) },
        { "mem_avail",      num_or_null(mem_avail)      },
        { "mem_total",      num_or_null(mem_total)      },
        { "swap_used",      num_or_null(swap_used)      },
    };
}

json server_machine_sample::to_json() const {
    json g = json::array();
    for (const auto & x : gpus) {
        g.push_back(x.to_json());
    }
    return json {
        { "t_us",    t_us            },
        { "unix_ms", unix_ms         },
        { "gpus",    std::move(g)    },
        { "host",    host.to_json()  },
        { "nvml",    nvml            },
        { "sampler", {
            { "n",      n      },
            { "sum_us", sum_us },
            { "max_us", max_us },
        }},
    };
}

//
// the sampler
//

int64_t server_sampler::period_us_from_env() {
    const char * e = getenv("LLAMA_SERVER_SAMPLE_MS");
    if (!e) {
        return 1000 * 1000LL;
    }
    const int ms = atoi(e);
    if (ms <= 0) {
        return 0;
    }
    // a sample takes tens of ms (NVML's PCIe counters wait 20 ms each): a shorter period would only sample back to back
    if (ms < k_min_period_ms) {
        fprintf(stderr, "%s: LLAMA_SERVER_SAMPLE_MS=%d raised to %d\n", __func__, ms, k_min_period_ms);
        return k_min_period_ms * 1000LL;
    }
    return ms * 1000LL;
}

server_sampler::server_sampler(int64_t period_us, fill_t fill) : period_us(period_us), fill(std::move(fill)), hist(period_us > 0 ? period_us : 1) {
    if (period_us > 0) {
        thread = std::thread([this] { run(); });
    }
}

server_sampler::~server_sampler() {
    {
        std::lock_guard<std::mutex> lock(mutex);
        stopping = true;
    }
    cv.notify_all();
    if (thread.joinable()) {
        thread.join();
    }
}

std::shared_ptr<const server_machine_sample> server_sampler::latest() const {
    return std::atomic_load(&snap);
}

std::shared_ptr<const server_stats_history::view> server_sampler::history() const {
    return period_us > 0 ? hist.latest() : nullptr;
}

void server_sampler::run() {
    server_nvml nvml;
    std::string state = "off";
    const char * e = getenv("LLAMA_SERVER_NVML");
    if (!(e && std::string(e) == "0")) {
        std::string why;
        state = nvml.open(why) ? "ok" : why;
    }
    const bool use_nvml = state == "ok";

    const auto period = std::chrono::microseconds(period_us);
    auto next = std::chrono::steady_clock::now();
    while (true) {
        m.nvml = state;
        if (!use_nvml) {
            m.gpus.clear();
        }
        const auto t0 = std::chrono::steady_clock::now();
        sample_once(nvml, use_nvml ? state : std::string());
        const auto now = std::chrono::steady_clock::now();

        next += period;
        if (now > next + period) {
            next = now; // fell behind (a suspended machine): no burst of catch-up samples
        }
        // never busier than half the time: a sample slower than the period waits as long as it took
        next = std::max(next, now + (now - t0));
        std::unique_lock<std::mutex> lock(mutex);
        if (cv.wait_until(lock, next, [this] { return stopping; })) {
            break;
        }
    }
}

void server_sampler::sample_once(server_nvml & nvml, const std::string & nvml_state) {
    const int64_t t0 = ggml_time_us();

    m.t_us    = t0;
    m.unix_ms = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
    if (!nvml_state.empty()) {
        nvml.sample(m.gpus);
    }
    sample_host(m.host);

    server_stats_row r = server_stats_row_empty();
    r[SSC_t_us]    = (double) m.t_us;
    r[SSC_unix_ms] = (double) m.unix_ms;

    // the GPUs, aggregated: shares and clocks averaged, bytes and watts summed, the hottest temperature
    auto agg = [&](int col, double server_gpu_sample::*f, bool mean, bool maxi) {
        double acc = maxi ? -INFINITY : 0.0;
        int n = 0;
        for (const auto & g : m.gpus) {
            const double v = g.*f;
            if (std::isnan(v)) {
                continue;
            }
            acc = maxi ? std::max(acc, v) : acc + v;
            n++;
        }
        r[col] = n == 0 ? NAN : (mean ? acc / n : acc);
    };
    agg(SSC_gpu_util,      &server_gpu_sample::util,       true,  false);
    agg(SSC_gpu_mem_util,  &server_gpu_sample::mem_util,   true,  false);
    agg(SSC_vram_used,     &server_gpu_sample::vram_used,  false, false);
    agg(SSC_vram_total,    &server_gpu_sample::vram_total, false, false);
    agg(SSC_pcie_rx,       &server_gpu_sample::pcie_rx,    false, false);
    agg(SSC_pcie_tx,       &server_gpu_sample::pcie_tx,    false, false);
    agg(SSC_gpu_temp_c,    &server_gpu_sample::temp_c,     false, true);
    agg(SSC_gpu_power_w,   &server_gpu_sample::power_w,    false, false);
    agg(SSC_gpu_clock_mhz, &server_gpu_sample::clock_mhz,  true,  false);

    r[SSC_proc_cpu_us]    = m.host.proc_cpu_us;
    r[SSC_host_busy_us]   = m.host.host_busy_us;
    r[SSC_host_total_us]  = m.host.host_total_us;
    r[SSC_host_iowait_us] = m.host.host_iowait_us;
    r[SSC_mem_avail]      = m.host.mem_avail;
    r[SSC_mem_total]      = m.host.mem_total;
    r[SSC_swap_used]      = m.host.swap_used;

    if (fill) {
        try {
            fill(r);
        } catch (const std::exception & e) {
            // a sample is not worth the thread: the row keeps its machine columns
            fprintf(stderr, "%s: server columns failed: %s\n", __func__, e.what());
        }
    }

    const uint64_t dt = (uint64_t) std::max<int64_t>(0, ggml_time_us() - t0);
    r[SSC_sample_us] = (double) dt;
    hist.append(r);

    m.n++;
    m.sum_us += dt;
    m.max_us  = std::max(m.max_us, dt);
    std::atomic_store(&snap, std::shared_ptr<const server_machine_sample>(std::make_shared<const server_machine_sample>(m)));
}
