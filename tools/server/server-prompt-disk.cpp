#include "server-prompt-disk.h"

#include "server-task.h"

#include <cstdio>
#include <memory>
#include <system_error>

namespace fs = std::filesystem;

// .meta layout (written last: an entry exists only once its .meta does):
//   u32 magic, u32 version, u32 n_tokens, i32 tokens[n_tokens]
// .ckpt layout:
//   u32 magic, u32 version, u32 count, then per checkpoint:
//   i64 n_tokens, i32 id_task, i32 pos_min, i32 pos_max, (u64 len, bytes) x {tgt, dft, spec}
// .tgt / .dft are llama_state_seq_save_file() files of the target and draft contexts.
static constexpr uint32_t PDC_META_MAGIC = 0x4d434450;  // "PDCM"
static constexpr uint32_t PDC_CKPT_MAGIC = 0x4b434450;  // "PDCK"
static constexpr uint32_t PDC_VERSION    = 1;

std::string server_prompt_disk_hash(const void * data, size_t n) {
    uint64_t h = 1469598103934665603ull;
    const auto * p = static_cast<const uint8_t *>(data);
    for (size_t i = 0; i < n; ++i) {
        h ^= p[i];
        h *= 1099511628211ull;
    }
    char buf[17];
    snprintf(buf, sizeof(buf), "%016llx", (unsigned long long) h);
    return buf;
}

namespace {

struct file_closer {
    void operator()(FILE * f) const { if (f) { fclose(f); } }
};
using file_ptr = std::unique_ptr<FILE, file_closer>;

file_ptr open_file(const fs::path & p, const char * mode) {
    return file_ptr(fopen(p.string().c_str(), mode));
}

bool put(FILE * f, const void * src, size_t n) {
    return n == 0 || fwrite(src, 1, n, f) == n;
}

bool get(FILE * f, void * dst, size_t n) {
    return n == 0 || fread(dst, 1, n, f) == n;
}

template <typename T> bool put_v(FILE * f, T v) { return put(f, &v, sizeof(v)); }
template <typename T> bool get_v(FILE * f, T & v) { return get(f, &v, sizeof(v)); }

bool put_blob(FILE * f, const std::vector<uint8_t> & b) {
    return put_v<uint64_t>(f, b.size()) && put(f, b.data(), b.size());
}

bool get_blob(FILE * f, std::vector<uint8_t> & b, uint64_t max_len) {
    uint64_t n = 0;
    if (!get_v(f, n) || n > max_len) {
        return false;
    }
    b.resize(n);
    return get(f, b.data(), n);
}

uintmax_t file_size_or_0(const fs::path & p) {
    std::error_code ec;
    const auto n = fs::file_size(p, ec);
    return ec ? 0 : n;
}

bool write_meta(const fs::path & p, const llama_tokens & tokens) {
    auto f = open_file(p, "wb");
    return f && put_v(f.get(), PDC_META_MAGIC) && put_v(f.get(), PDC_VERSION)
             && put_v<uint32_t>(f.get(), tokens.size())
             && put(f.get(), tokens.data(), tokens.size() * sizeof(llama_token));
}

bool read_meta(const fs::path & p, llama_tokens & tokens) {
    auto f = open_file(p, "rb");
    uint32_t magic = 0, version = 0, n = 0;
    if (!f || !get_v(f.get(), magic) || !get_v(f.get(), version) || !get_v(f.get(), n)) {
        return false;
    }
    if (magic != PDC_META_MAGIC || version != PDC_VERSION || n == 0 || n > (1u << 26)) {
        return false;
    }
    tokens.resize(n);
    return get(f.get(), tokens.data(), n * sizeof(llama_token));
}

bool write_ckpts(const fs::path & p, const std::list<common_prompt_checkpoint> & ckpts) {
    auto f = open_file(p, "wb");
    if (!f || !put_v(f.get(), PDC_CKPT_MAGIC) || !put_v(f.get(), PDC_VERSION) || !put_v<uint32_t>(f.get(), ckpts.size())) {
        return false;
    }
    for (const auto & c : ckpts) {
        if (!put_v<int64_t>(f.get(), c.n_tokens) || !put_v<int32_t>(f.get(), c.id_task)
                || !put_v<int32_t>(f.get(), c.pos_min) || !put_v<int32_t>(f.get(), c.pos_max)
                || !put_blob(f.get(), c.data_tgt) || !put_blob(f.get(), c.data_dft) || !put_blob(f.get(), c.data_spec)) {
            return false;
        }
    }
    return true;
}

bool read_ckpts(const fs::path & p, std::list<common_prompt_checkpoint> & ckpts) {
    auto f = open_file(p, "rb");
    uint32_t magic = 0, version = 0, count = 0;
    if (!f || !get_v(f.get(), magic) || !get_v(f.get(), version) || !get_v(f.get(), count)) {
        return false;
    }
    if (magic != PDC_CKPT_MAGIC || version != PDC_VERSION || count > 4096) {
        return false;
    }
    const uint64_t max_len = file_size_or_0(p);
    for (uint32_t i = 0; i < count; ++i) {
        common_prompt_checkpoint c;
        int64_t n_tokens = 0;
        int32_t id_task = -1, pos_min = 0, pos_max = 0;
        if (!get_v(f.get(), n_tokens) || !get_v(f.get(), id_task) || !get_v(f.get(), pos_min) || !get_v(f.get(), pos_max)
                || !get_blob(f.get(), c.data_tgt, max_len) || !get_blob(f.get(), c.data_dft, max_len)
                || !get_blob(f.get(), c.data_spec, max_len)) {
            return false;
        }
        c.update_pos(n_tokens, pos_min, pos_max);
        c.id_task = id_task;
        ckpts.push_back(std::move(c));
    }
    return true;
}

// Write to <path>.tmp and rename, so a crash never leaves a half-written file under the real name.
template <typename F>
bool write_atomic(const fs::path & p, F && writer) {
    fs::path tmp = p;
    tmp += ".tmp";
    if (!writer(tmp)) {
        std::error_code ec;
        fs::remove(tmp, ec);
        return false;
    }
    std::error_code ec;
    fs::rename(tmp, p, ec);
    return !ec;
}

} // namespace

server_prompt_disk::server_prompt_disk(const std::string & root, const std::string & fingerprint,
                                       size_t limit_bytes, int64_t ttl_s, size_t reserve_bytes)
    : dir(fs::path(root) / fingerprint), limit_bytes(limit_bytes), ttl_s(ttl_s), reserve_bytes(reserve_bytes) {
    std::error_code ec;
    fs::create_directories(dir, ec);
    if (ec) {
        SRV_ERR("prompt disk cache: cannot create %s: %s\n", dir.string().c_str(), ec.message().c_str());
        return;
    }
    // the entries hold prompt contents: only the server's user may read them (never touch `root` itself, it may be
    // a shared mount)
    fs::permissions(dir, fs::perms::owner_all, fs::perm_options::replace, ec);
    ready = true;
    scan();
    evict();
    SRV_INF("prompt disk cache: %s, %zu entries, %.1f MiB (limit %.1f MiB, ttl %lld s, reserve %.1f MiB)\n",
            dir.string().c_str(), entries.size(), size() / 1048576.0, limit_bytes / 1048576.0, (long long) ttl_s,
            reserve_bytes / 1048576.0);
}

fs::path server_prompt_disk::path(const std::string & id, const char * ext) const {
    return dir / (id + ext);
}

void server_prompt_disk::scan() {
    std::error_code ec;
    for (const auto & de : fs::directory_iterator(dir, ec)) {
        const fs::path p = de.path();
        if (p.extension() == ".tmp") {
            fs::remove(p, ec);  // leftover of an interrupted write
            continue;
        }
        if (p.extension() != ".meta") {
            continue;
        }
        server_prompt_disk_entry e;
        e.id = p.stem().string();
        if (!read_meta(p, e.tokens) || !fs::exists(path(e.id, ".tgt"))) {
            SRV_WRN("prompt disk cache: dropping unreadable entry %s\n", e.id.c_str());
            for (const char * ext : { ".meta", ".tgt", ".dft", ".ckpt" }) {
                fs::remove(path(e.id, ext), ec);
            }
            continue;
        }
        for (const char * ext : { ".meta", ".tgt", ".dft", ".ckpt" }) {
            e.bytes += file_size_or_0(path(e.id, ext));
        }
        e.t_used = fs::last_write_time(p, ec);
        entries.push_back(std::move(e));
    }
    entries.sort([](const auto & a, const auto & b) { return a.t_used < b.t_used; });

    // files without a .meta belong to no entry (a crash between the state and the .meta write)
    for (const auto & de : fs::directory_iterator(dir, ec)) {
        const fs::path p = de.path();
        if (p.extension() != ".meta" && !fs::exists(path(p.stem().string(), ".meta"))) {
            fs::remove(p, ec);
        }
    }
}

size_t server_prompt_disk::size() const {
    size_t res = 0;
    for (const auto & e : entries) {
        res += e.bytes;
    }
    return res;
}

bool server_prompt_disk::expired(const server_prompt_disk_entry & e) const {
    if (ttl_s <= 0) {
        return false;
    }
    const auto age = fs::file_time_type::clock::now() - e.t_used;
    return std::chrono::duration_cast<std::chrono::seconds>(age).count() > ttl_s;
}

void server_prompt_disk::remove(std::list<server_prompt_disk_entry>::iterator it) {
    std::error_code ec;
    fs::remove(path(it->id, ".meta"), ec);  // uncommit first
    for (const char * ext : { ".tgt", ".dft", ".ckpt" }) {
        fs::remove(path(it->id, ext), ec);
    }
    entries.erase(it);
}

void server_prompt_disk::touch(std::list<server_prompt_disk_entry>::iterator it) {
    std::error_code ec;
    it->t_used = fs::file_time_type::clock::now();
    fs::last_write_time(path(it->id, ".meta"), it->t_used, ec);
    entries.splice(entries.end(), entries, it);
}

void server_prompt_disk::evict() {
    for (auto it = entries.begin(); it != entries.end();) {
        auto next = std::next(it);
        if (expired(*it)) {
            SRV_TRC("prompt disk cache: expired entry %s (%d tokens)\n", it->id.c_str(), (int) it->tokens.size());
            remove(it);
        }
        it = next;
    }
    auto free_bytes = [&]() {
        std::error_code ec;
        const auto sp = fs::space(dir, ec);
        return ec ? (uintmax_t) -1 : sp.available;
    };
    while (!entries.empty() && ((limit_bytes > 0 && size() > limit_bytes) || free_bytes() < reserve_bytes)) {
        SRV_TRC("prompt disk cache: evicting oldest entry %s (%.1f MiB)\n", entries.front().id.c_str(),
                entries.front().bytes / 1048576.0);
        remove(entries.begin());
    }
}

bool server_prompt_disk::save(const server_prompt & prompt, llama_context * ctx_tgt, llama_context * ctx_dft,
                              llama_seq_id seq) {
    if (!ready || prompt.tokens.size() == 0) {
        return false;
    }
    if (prompt.tokens.has_mtmd) {
        SRV_TRC("%s", "prompt disk cache: skipping a multimodal prompt\n");
        return false;
    }

    const llama_tokens tokens = prompt.tokens.get_text_tokens();

    for (auto it = entries.begin(); it != entries.end(); ++it) {
        const server_tokens et(it->tokens, false);
        if (et.get_common_prefix(prompt.tokens) == prompt.tokens.size()) {
            SRV_TRC("prompt disk cache: prompt already stored in %s\n", it->id.c_str());
            touch(it);
            return false;
        }
    }

    evict();

    const size_t need = llama_state_seq_get_size_ext(ctx_tgt, seq, LLAMA_STATE_SEQ_FLAGS_NONE)
                      + (ctx_dft ? llama_state_seq_get_size_ext(ctx_dft, seq, LLAMA_STATE_SEQ_FLAGS_NONE) : 0);
    std::error_code ec;
    const auto sp = fs::space(dir, ec);
    if (limit_bytes > 0 && need > limit_bytes) {
        SRV_WRN("prompt disk cache: state %.1f MiB exceeds the limit, not stored\n", need / 1048576.0);
        return false;
    }
    while (!entries.empty() && limit_bytes > 0 && size() + need > limit_bytes) {
        remove(entries.begin());
    }
    if (!ec && sp.available < need + reserve_bytes) {
        SRV_WRN("prompt disk cache: %.1f MiB free is below state + reserve, not stored\n", sp.available / 1048576.0);
        return false;
    }

    const int64_t t0 = ggml_time_us();
    server_prompt_disk_entry e;
    e.id = server_prompt_disk_hash(tokens.data(), tokens.size() * sizeof(llama_token));

    const bool ok =
        write_atomic(path(e.id, ".tgt"), [&](const fs::path & p) {
            return llama_state_seq_save_file(ctx_tgt, p.string().c_str(), seq, tokens.data(), tokens.size()) > 0;
        }) &&
        (!ctx_dft || write_atomic(path(e.id, ".dft"), [&](const fs::path & p) {
            return llama_state_seq_save_file(ctx_dft, p.string().c_str(), seq, tokens.data(), tokens.size()) > 0;
        })) &&
        (prompt.checkpoints.empty() || write_atomic(path(e.id, ".ckpt"), [&](const fs::path & p) {
            return write_ckpts(p, prompt.checkpoints);
        })) &&
        write_atomic(path(e.id, ".meta"), [&](const fs::path & p) { return write_meta(p, tokens); });

    if (!ok) {
        SRV_ERR("prompt disk cache: failed to write entry %s\n", e.id.c_str());
        for (const char * ext : { ".meta", ".tgt", ".dft", ".ckpt" }) {
            fs::remove(path(e.id, ext), ec);
        }
        return false;
    }

    // entries the new prompt fully contains are obsolete (the RAM tier's rule)
    const server_tokens nt(tokens, false);
    for (auto it = entries.begin(); it != entries.end();) {
        auto next = std::next(it);
        if (it->id == e.id || nt.get_common_prefix(server_tokens(it->tokens, false)) == it->tokens.size()) {
            remove(it);
        }
        it = next;
    }

    for (const char * ext : { ".meta", ".tgt", ".dft", ".ckpt" }) {
        e.bytes += file_size_or_0(path(e.id, ext));
    }
    e.t_used = fs::last_write_time(path(e.id, ".meta"), ec);
    e.tokens = tokens;
    entries.push_back(std::move(e));

    const double ms = (ggml_time_us() - t0) / 1000.0;
    SRV_INF("prompt disk cache: saved %d tokens, %.1f MiB, %zu checkpoints in %.0f ms (%zu entries, %.1f MiB)\n",
            (int) tokens.size(), entries.back().bytes / 1048576.0, prompt.checkpoints.size(), ms, entries.size(),
            size() / 1048576.0);
    return true;
}

const server_prompt_disk_entry * server_prompt_disk::find(const server_tokens & tokens_new, float f_keep_base,
                                                          float f_sim_base, float & f_keep_out, float & f_sim_out) const {
    const server_prompt_disk_entry * best = nullptr;
    f_keep_out = f_keep_base;
    f_sim_out  = f_sim_base;
    for (const auto & e : entries) {
        if (expired(e)) {
            continue;
        }
        const size_t lcp = server_tokens(e.tokens, false).get_common_prefix(tokens_new);
        const float f_keep = float(lcp) / e.tokens.size();
        const float f_sim  = float(lcp) / tokens_new.size();
        if (f_keep < 0.25f) {
            continue;  // the RAM tier's "don't trash large prompts" rule
        }
        if (f_keep_out < f_keep && f_sim_out < f_sim) {
            f_keep_out = f_keep;
            f_sim_out  = f_sim;
            best = &e;
        }
    }
    return best;
}

bool server_prompt_disk::load(const server_prompt_disk_entry & entry, server_prompt & prompt,
                              llama_context * ctx_tgt, llama_context * ctx_dft, llama_seq_id seq) {
    auto it = entries.begin();
    while (it != entries.end() && &*it != &entry) {
        ++it;
    }
    GGML_ASSERT(it != entries.end());

    const int64_t t0 = ggml_time_us();
    llama_tokens tokens(it->tokens.size());
    size_t n_read = 0;
    std::list<common_prompt_checkpoint> ckpts;

    bool ok = llama_state_seq_load_file(ctx_tgt, path(it->id, ".tgt").string().c_str(), seq,
                                        tokens.data(), tokens.size(), &n_read) > 0
              && n_read == it->tokens.size() && tokens == it->tokens;
    if (ok && ctx_dft) {
        size_t n_dft = 0;
        llama_tokens tokens_dft(it->tokens.size());
        ok = fs::exists(path(it->id, ".dft"))
             && llama_state_seq_load_file(ctx_dft, path(it->id, ".dft").string().c_str(), seq,
                                          tokens_dft.data(), tokens_dft.size(), &n_dft) > 0
             && n_dft == it->tokens.size();
    }
    if (ok && fs::exists(path(it->id, ".ckpt"))) {
        ok = read_ckpts(path(it->id, ".ckpt"), ckpts);
    }
    if (!ok) {
        SRV_ERR("prompt disk cache: failed to restore entry %s, deleting it\n", it->id.c_str());
        remove(it);
        return false;
    }

    prompt.tokens      = server_tokens(tokens, false);
    prompt.checkpoints = std::move(ckpts);
    touch(it);

    SRV_INF("prompt disk cache: restored %d tokens (%.1f MiB) in %.0f ms\n", (int) tokens.size(),
            entries.back().bytes / 1048576.0, (ggml_time_us() - t0) / 1000.0);
    return true;
}
