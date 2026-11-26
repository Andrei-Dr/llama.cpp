// the split policy of the mid-batch MoE hybrid (src/llama-moe-hybrid.h)

#include "../src/llama-moe-hybrid.h"

#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

#define CHECK(cond) do { if (!(cond)) { fprintf(stderr, "%s:%d: check failed: %s\n", __FILE__, __LINE__, #cond); exit(1); } } while (0)

static void test_choose_split() {
    llama_moe_hybrid_params p; // 27, 40, 95, 10

    // nothing routed
    {
        std::vector<int32_t> counts(16, 0);
        const auto s = llama_moe_hybrid::choose_split(counts.data(), nullptr, 16, p);
        CHECK(s.c == 0 && s.n_cpu == 0 && s.n_upload == 0 && s.n_cached == 0);
    }
    // 10 experts with one token, 2 with twenty: the light ones cost the CPU 670, the device 950
    {
        std::vector<int32_t> counts(16, 0);
        for (int e = 0; e < 10; ++e) { counts[e] = 1; }
        counts[10] = counts[11] = 20;
        const auto s = llama_moe_hybrid::choose_split(counts.data(), nullptr, 16, p);
        CHECK(s.c == 1 && s.n_cpu == 10 && s.n_upload == 2 && s.n_cached == 0 && s.n_assign_cpu == 10);
    }
    // the experts in the cache go to the device whatever their count, and they are not counted as uploads
    {
        std::vector<int32_t> counts(16, 0);
        std::vector<uint8_t> cached(16, 0);
        for (int e = 0; e < 10; ++e) { counts[e] = 1; }
        counts[10] = counts[11] = 20;
        cached[0] = cached[1] = cached[10] = 1;
        cached[15] = 1; // not used
        const auto s = llama_moe_hybrid::choose_split(counts.data(), cached.data(), 16, p);
        CHECK(s.n_cached == 3 && s.n_cpu == 8 && s.n_upload == 1 && s.c == 1 && s.n_assign_cpu == 8);
    }
    // every split has the three parts add up to the used experts
    {
        std::mt19937 rng(7);
        for (int iter = 0; iter < 200; ++iter) {
            const int n_expert = 64;
            std::vector<int32_t> counts(n_expert);
            std::vector<uint8_t> cached(n_expert);
            int n_used = 0;
            for (int e = 0; e < n_expert; ++e) {
                counts[e] = rng() % 3 == 0 ? 0 : 1 + rng() % 9;
                cached[e] = rng() % 5 == 0;
                n_used += counts[e] > 0;
            }
            const auto s = llama_moe_hybrid::choose_split(counts.data(), cached.data(), n_expert, p);
            CHECK(s.n_cpu + s.n_upload + s.n_cached == n_used);
            int n_cpu = 0;
            for (int e = 0; e < n_expert; ++e) {
                n_cpu += counts[e] > 0 && !cached[e] && counts[e] <= s.c;
            }
            CHECK(n_cpu == s.n_cpu);
        }
    }
}

static void test_fill_dev_ids() {
    std::mt19937 rng(11);
    for (int iter = 0; iter < 300; ++iter) {
        const int n_expert = 32;
        const int n_used   = 8;
        const int n_tokens = 1 + rng() % 48;
        std::vector<int32_t> ids(n_used*n_tokens);
        for (int it = 0; it < n_tokens; ++it) {
            std::vector<int32_t> perm(n_expert);
            for (int e = 0; e < n_expert; ++e) { perm[e] = e; }
            std::shuffle(perm.begin(), perm.end(), rng);
            for (int k = 0; k < n_used; ++k) { ids[it*n_used + k] = perm[k]; }
        }
        // from "every expert on the device" to "every expert on the CPU"
        std::vector<uint8_t> on_cpu(n_expert);
        const int share = iter % 5;
        for (int e = 0; e < n_expert; ++e) {
            on_cpu[e] = share == 0 ? 0 : share == 4 ? 1 : (int) (rng() % 4) < share;
        }
        std::vector<int32_t> dev(ids.size(), -2);
        llama_moe_hybrid::fill_dev_ids(ids.data(), n_used, n_tokens, on_cpu.data(), n_expert, dev.data());
        for (int it = 0; it < n_tokens; ++it) {
            for (int k = 0; k < n_used; ++k) {
                const int32_t id = ids[it*n_used + k];
                const int32_t d  = dev[it*n_used + k];
                CHECK(d >= 0 && d < n_expert);
                // a device expert keeps its id, a CPU expert gets another one: the mask and the CPU ids are (id == dev id)
                CHECK(on_cpu[id] ? d != id : d == id);
                for (int j = 0; j < n_used; ++j) {
                    // distinct for the token, and a stand-in is not an expert the token is routed to
                    CHECK(j == k || dev[it*n_used + j] != d);
                    CHECK(!on_cpu[id] || ids[it*n_used + j] != d);
                }
            }
        }
    }
}

int main() {
    test_choose_split();
    test_fill_dev_ids();
    printf("OK\n");
    return 0;
}
