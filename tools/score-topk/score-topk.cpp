// llama-score-topk: teacher scoring for offline distillation.
//
// Reads rows of S token ids and writes, for every position s of every row, the model's top-K next-token
// log-probabilities (the distribution for token s+1), plus the log of the probability mass outside the top-K.
//
// input  --rows FILE     int32 [n_rows, S], raw little-endian, S = --ctx-size
// output --out-dir DIR   one file per --shard-rows rows: shard_NNNNNN.topk, written atomically (.tmp + rename);
//                        a shard whose .topk or .npz already exists is skipped, so a run resumes where it stopped
//
// shard file layout (little-endian):
//   char    magic[8]  "TOPKLP01"
//   int32   n_rows, S, K, n_vocab, row_start
//   then per row:  int32 ids[S][K]   (descending log-probability)
//                  f32   lp [S][K]   log p of those ids, from the full-vocabulary log-softmax
//                  f32   left[S]     log(sum of p outside the top-K)
//
// The rows of one decode are --batch-size / --ctx-size sequences, each in its own sequence id.
// --dump-logits FILE writes the raw f32 logits of the first row [S][n_vocab] (for testing).

#include "arg.h"
#include "common.h"
#include "log.h"
#include "llama.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <clocale>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <thread>
#include <vector>

#if defined(__AVX2__) && defined(__FMA__)
#include <immintrin.h>
#define SCORE_TOPK_AVX2 1
#endif

namespace fs = std::filesystem;

struct topk_args {
    std::string rows_path;
    std::string out_dir;
    std::string dump_logits;
    int32_t     k          = 32;
    int32_t     shard_rows = 32;
    int32_t     n_threads  = 0;   // top-K threads, 0 = hardware concurrency
    int64_t     row_limit  = -1;
};

// exp for x <= 0 (the argument is always logit - max), the Cephes expf scheme: n = round(x / ln 2),
// r = x - n ln 2 in two parts (Cody-Waite), e^r by a degree-7 polynomial, times 2^n; ~1 ulp, auto-vectorizes
static inline float exp_nonpos(float x) {
    x = std::max(x, -87.0f);
    const float n = std::floor(x * 1.44269504088896341f + 0.5f);
    const float r = x - n * 0.693359375f + n * 2.12194440e-4f;
    float p = 1.9875691500e-4f;
    p = p * r + 1.3981999507e-3f;
    p = p * r + 8.3334519073e-3f;
    p = p * r + 4.1665795894e-2f;
    p = p * r + 1.6666665459e-1f;
    p = p * r + 5.0000001201e-1f;
    p = p * r * r + r + 1.0f;
    int32_t bits;
    std::memcpy(&bits, &p, sizeof(bits));
    bits += (int32_t) n << 23;
    float e;
    std::memcpy(&e, &bits, sizeof(e));
    return e;
}

#ifdef SCORE_TOPK_AVX2
// exp_nonpos on 8 lanes, the same scheme and constants
static inline __m256 exp_nonpos8(__m256 x) {
    x = _mm256_max_ps(x, _mm256_set1_ps(-87.0f));
    const __m256 n = _mm256_floor_ps(_mm256_fmadd_ps(x, _mm256_set1_ps(1.44269504088896341f), _mm256_set1_ps(0.5f)));
    __m256 r = _mm256_fnmadd_ps(n, _mm256_set1_ps(0.693359375f), x);
    r = _mm256_fmadd_ps(n, _mm256_set1_ps(2.12194440e-4f), r);
    __m256 p = _mm256_set1_ps(1.9875691500e-4f);
    p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(1.3981999507e-3f));
    p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(8.3334519073e-3f));
    p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(4.1665795894e-2f));
    p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(1.6666665459e-1f));
    p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(5.0000001201e-1f));
    p = _mm256_fmadd_ps(p, _mm256_mul_ps(r, r), _mm256_add_ps(r, _mm256_set1_ps(1.0f)));
    const __m256i bits = _mm256_add_epi32(_mm256_castps_si256(p), _mm256_slli_epi32(_mm256_cvttps_epi32(n), 23));
    return _mm256_castsi256_ps(bits);
}
#endif

// top-K of one logits row; writes ids / lp (descending) and the leftover log-mass
static void topk_row(const float * x, int n_vocab, int K, int32_t * ids, float * lp, float * left) {
    constexpr int B = 64;

    // pass 1: max and the K largest (sorted descending); a block whose max cannot enter the top-K is skipped
    std::vector<float>   tv(K, -INFINITY);
    std::vector<int32_t> ti(K, -1);
    float mx = -INFINITY;
    for (int i0 = 0; i0 < n_vocab; i0 += B) {
        const int n = std::min(B, n_vocab - i0);
        float bm = -INFINITY;
        int   l0 = 0;
#ifdef SCORE_TOPK_AVX2
        if (n == B) {
            __m256 m8 = _mm256_loadu_ps(x + i0);
            for (int l = 8; l < B; l += 8) {
                m8 = _mm256_max_ps(m8, _mm256_loadu_ps(x + i0 + l));
            }
            __m128 m4 = _mm_max_ps(_mm256_castps256_ps128(m8), _mm256_extractf128_ps(m8, 1));
            m4 = _mm_max_ps(m4, _mm_movehl_ps(m4, m4));
            m4 = _mm_max_ss(m4, _mm_movehdup_ps(m4));
            bm = _mm_cvtss_f32(m4);
            l0 = B;
        }
#endif
        for (int l = l0; l < n; ++l) {
            bm = std::max(bm, x[i0 + l]);
        }
        mx = std::max(mx, bm);
        if (!(bm > tv[K - 1])) {
            continue;
        }
        for (int l = 0; l < n; ++l) {
            const float v = x[i0 + l];
            if (v > tv[K - 1]) {
                int j = K - 1;
                while (j > 0 && tv[j - 1] < v) {
                    tv[j] = tv[j - 1];
                    ti[j] = ti[j - 1];
                    --j;
                }
                tv[j] = v;
                ti[j] = i0 + l;
            }
        }
    }
    const float thr = tv[K - 1];

    // pass 2: sums of exp(x - max) below the threshold and at / above it, per block in independent lanes
    double lo = 0.0, hi = 0.0;
    int    n_ge = 0;
    float  e[B];
    for (int i0 = 0; i0 < n_vocab; i0 += B) {
        const int n = std::min(B, n_vocab - i0);
        float slo = 0.0f, shi = 0.0f;
        int   cge = 0;
        int   l0 = 0;
#ifdef SCORE_TOPK_AVX2
        if (n == B) {
            const __m256 vmx = _mm256_set1_ps(mx), vthr = _mm256_set1_ps(thr);
            __m256  alo = _mm256_setzero_ps(), ahi = _mm256_setzero_ps();
            __m256i acnt = _mm256_setzero_si256();
            for (int l = 0; l < B; l += 8) {
                const __m256 v  = _mm256_loadu_ps(x + i0 + l);
                const __m256 ev = exp_nonpos8(_mm256_sub_ps(v, vmx));
                const __m256 ge = _mm256_cmp_ps(v, vthr, _CMP_GE_OQ);
                ahi  = _mm256_add_ps(ahi, _mm256_and_ps(ge, ev));
                alo  = _mm256_add_ps(alo, _mm256_andnot_ps(ge, ev));
                acnt = _mm256_sub_epi32(acnt, _mm256_castps_si256(ge));  // ge lanes are all ones = -1
            }
            float fl[8], fh[8];
            int32_t ic[8];
            _mm256_storeu_ps(fl, alo);
            _mm256_storeu_ps(fh, ahi);
            _mm256_storeu_si256((__m256i *) ic, acnt);
            for (int j = 0; j < 8; ++j) {
                slo += fl[j];
                shi += fh[j];
                cge += ic[j];
            }
            l0 = B;
        }
#endif
        for (int l = l0; l < n; ++l) {
            e[l] = exp_nonpos(x[i0 + l] - mx);
        }
        for (int l = l0; l < n; ++l) {
            const bool ge = x[i0 + l] >= thr;
            slo += ge ? 0.0f : e[l];
            shi += ge ? e[l] : 0.0f;
            cge += ge;
        }
        lo += slo;
        hi += shi;
        n_ge += cge;
    }

    const double sum = lo + hi;
    const double lse = (double) mx + std::log(sum);
    for (int k = 0; k < K; ++k) {
        ids[k] = ti[k];
        lp[k]  = (float) ((double) tv[k] - lse);
    }
    // ties at the threshold that did not make the top-K belong to the rest
    const double rest = lo + (double) (n_ge - K) * std::exp((double) thr - (double) mx);
    *left = rest > 0.0 ? (float) (std::log(rest) - std::log(sum)) : -INFINITY;
}

static bool parse_own_args(int & argc, char ** argv, topk_args & a) {
    std::vector<char *> keep = { argv[0] };
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        auto next = [&]() -> const char * {
            if (i + 1 >= argc) {
                LOG_ERR("%s needs a value\n", arg.c_str());
                return nullptr;
            }
            return argv[++i];
        };
        const char * v = nullptr;
        if      (arg == "--rows")        { if (!(v = next())) return false; a.rows_path   = v; }
        else if (arg == "--out-dir")     { if (!(v = next())) return false; a.out_dir     = v; }
        else if (arg == "--dump-logits") { if (!(v = next())) return false; a.dump_logits = v; }
        else if (arg == "--top-k")       { if (!(v = next())) return false; a.k          = std::stoi(v); }
        else if (arg == "--shard-rows")  { if (!(v = next())) return false; a.shard_rows = std::stoi(v); }
        else if (arg == "--topk-threads"){ if (!(v = next())) return false; a.n_threads  = std::stoi(v); }
        else if (arg == "--row-limit")   { if (!(v = next())) return false; a.row_limit  = std::stoll(v); }
        else { keep.push_back(argv[i]); }
    }
    for (size_t i = 0; i < keep.size(); ++i) {
        argv[i] = keep[i];
    }
    argc = (int) keep.size();
    if (a.rows_path.empty() || a.out_dir.empty()) {
        LOG_ERR("usage: %s --rows ROWS.i32 --out-dir DIR [--top-k 32] [--shard-rows 32] [--topk-threads N] "
                "[--row-limit N] [--dump-logits FILE] -m MODEL -c S [-b n_seq*S] [llama options]\n", argv[0]);
        return false;
    }
    if (a.k <= 0 || a.shard_rows <= 0) {
        LOG_ERR("--top-k and --shard-rows must be > 0\n");
        return false;
    }
    return true;
}

int main(int argc, char ** argv) {
    std::setlocale(LC_NUMERIC, "C");

    topk_args ta;
    if (!parse_own_args(argc, argv, ta)) {
        return 1;
    }

    common_params params;
    params.n_ctx = 2048;
    common_init();
    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_PERPLEXITY)) {
        return 1;
    }

    const int32_t S = params.n_ctx;
    if (S <= 0) {
        LOG_ERR("--ctx-size (the row length S) must be > 0\n");
        return 1;
    }
    // whole rows per decode: n_seq sequences of S tokens
    const int32_t n_seq = std::max(1, params.n_batch / S);
    params.n_parallel = n_seq;
    params.n_ctx      = n_seq * S;
    params.n_batch    = n_seq * S;
    params.kv_unified = false;

    // rows
    std::ifstream fin(ta.rows_path, std::ios::binary | std::ios::ate);
    if (!fin) {
        LOG_ERR("cannot open %s\n", ta.rows_path.c_str());
        return 1;
    }
    const int64_t n_bytes = fin.tellg();
    if (n_bytes % (int64_t(S) * 4) != 0) {
        LOG_ERR("%s: %lld bytes is not a whole number of rows of %d int32\n", ta.rows_path.c_str(), (long long) n_bytes, S);
        return 1;
    }
    int64_t n_rows = n_bytes / (int64_t(S) * 4);
    if (ta.row_limit >= 0) {
        n_rows = std::min(n_rows, ta.row_limit);
    }
    std::vector<int32_t> rows(size_t(n_rows) * S);
    fin.seekg(0);
    fin.read((char *) rows.data(), std::streamsize(rows.size() * 4));

    fs::create_directories(ta.out_dir);
    const int64_t n_shards = (n_rows + ta.shard_rows - 1) / ta.shard_rows;
    auto shard_path = [&](int64_t sh, const char * ext) {
        char name[64];
        snprintf(name, sizeof(name), "shard_%06lld%s", (long long) sh, ext);
        return (fs::path(ta.out_dir) / name).string();
    };
    std::vector<int64_t> todo;
    for (int64_t sh = 0; sh < n_shards; ++sh) {
        if (!fs::exists(shard_path(sh, ".topk")) && !fs::exists(shard_path(sh, ".npz"))) {
            todo.push_back(sh);
        }
    }
    LOG_INF("rows %lld x %d, shards %lld (%lld to do), n_seq %d per decode, K %d\n",
            (long long) n_rows, S, (long long) n_shards, (long long) todo.size(), n_seq, ta.k);
    if (todo.empty()) {
        return 0;
    }

    llama_backend_init();
    llama_numa_init(params.numa);
    auto llama_init = common_init_from_params(params);
    auto * model = llama_init->model();
    auto * ctx   = llama_init->context();
    if (model == nullptr || ctx == nullptr) {
        LOG_ERR("unable to load the model or create the context\n");
        return 1;
    }
    const int n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));
    const int K = ta.k;
    if (K > n_vocab) {
        LOG_ERR("--top-k %d exceeds the vocabulary (%d)\n", K, n_vocab);
        return 1;
    }
    for (int64_t t = 0; t < (int64_t) rows.size(); ++t) {
        if (rows[t] < 0 || rows[t] >= n_vocab) {
            LOG_ERR("row %lld holds token %d, outside the vocabulary\n", (long long) (t / S), rows[t]);
            return 1;
        }
    }

    const int n_threads = ta.n_threads > 0 ? ta.n_threads : (int) std::max(1u, std::thread::hardware_concurrency());
    llama_batch batch = llama_batch_init(n_seq * S, 0, 1);

    std::vector<int32_t> ids;
    std::vector<float>   lps;
    std::vector<float>   lefts;
    double t_decode = 0.0, t_topk = 0.0;
    int64_t n_tok_done = 0;
    bool dumped = ta.dump_logits.empty();
    const auto t0 = std::chrono::steady_clock::now();

    for (int64_t sh : todo) {
        const int64_t r0 = sh * ta.shard_rows;
        const int64_t r1 = std::min(n_rows, r0 + ta.shard_rows);
        const int64_t nr = r1 - r0;
        ids.assign(size_t(nr) * S * K, 0);
        lps.assign(size_t(nr) * S * K, 0.0f);
        lefts.assign(size_t(nr) * S, 0.0f);

        for (int64_t rb = r0; rb < r1; rb += n_seq) {
            const int nb = (int) std::min<int64_t>(n_seq, r1 - rb);
            llama_memory_clear(llama_get_memory(ctx), true);
            batch.n_tokens = 0;
            for (int q = 0; q < nb; ++q) {
                for (int s = 0; s < S; ++s) {
                    common_batch_add(batch, rows[size_t(rb + q) * S + s], s, { q }, true);
                }
            }
            auto ta0 = std::chrono::steady_clock::now();
            if (llama_decode(ctx, batch) != 0) {
                LOG_ERR("llama_decode failed at row %lld\n", (long long) rb);
                return 1;
            }
            // the logits are read below; llama_get_logits_ith synchronizes
            const float * probe = llama_get_logits_ith(ctx, 0);
            GGML_UNUSED(probe);
            auto ta1 = std::chrono::steady_clock::now();

            if (!dumped) {
                std::ofstream fd(ta.dump_logits, std::ios::binary);
                for (int s = 0; s < S; ++s) {
                    fd.write((const char *) llama_get_logits_ith(ctx, s), std::streamsize(n_vocab) * 4);
                }
                dumped = true;
            }

            const int n_tok = nb * S;
            std::atomic<int> next{0};
            auto work = [&]() {
                for (int t; (t = next.fetch_add(1)) < n_tok; ) {
                    const size_t o = size_t(rb - r0) * S + t;  // row-major [row][s]
                    topk_row(llama_get_logits_ith(ctx, t), n_vocab, K, ids.data() + o * K, lps.data() + o * K, lefts.data() + o);
                }
            };
            std::vector<std::thread> pool;
            for (int w = 1; w < n_threads; ++w) {
                pool.emplace_back(work);
            }
            work();
            for (auto & th : pool) {
                th.join();
            }
            auto ta2 = std::chrono::steady_clock::now();
            t_decode += std::chrono::duration<double>(ta1 - ta0).count();
            t_topk   += std::chrono::duration<double>(ta2 - ta1).count();
            n_tok_done += n_tok;
        }

        const std::string tmp = shard_path(sh, ".topk.tmp");
        {
            std::ofstream fo(tmp, std::ios::binary);
            const int32_t hdr[5] = { (int32_t) nr, S, K, n_vocab, (int32_t) r0 };
            fo.write("TOPKLP01", 8);
            fo.write((const char *) hdr, sizeof(hdr));
            for (int64_t r = 0; r < nr; ++r) {
                fo.write((const char *) (ids.data()   + size_t(r) * S * K), std::streamsize(S) * K * 4);
                fo.write((const char *) (lps.data()   + size_t(r) * S * K), std::streamsize(S) * K * 4);
                fo.write((const char *) (lefts.data() + size_t(r) * S),     std::streamsize(S) * 4);
            }
            if (!fo) {
                LOG_ERR("write failed: %s\n", tmp.c_str());
                return 1;
            }
        }
        fs::rename(tmp, shard_path(sh, ".topk"));

        const double el = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        LOG_INF("shard %lld done: %lld tok, %.1f tok/s (decode %.1f s, top-K %.1f s)\n",
                (long long) sh, (long long) n_tok_done, n_tok_done / el, t_decode, t_topk);
    }

    llama_batch_free(batch);
    llama_backend_free();
    return 0;
}
