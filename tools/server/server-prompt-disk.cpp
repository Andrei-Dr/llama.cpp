#include "server-prompt-disk.h"

#include "server-task.h"

#include <openssl/crypto.h>
#include <openssl/evp.h>
#include <openssl/rand.h>

#include <cerrno>
#include <csignal>
#include <cstdio>
#include <cstring>
#include <memory>
#include <system_error>
#include <vector>

#ifndef _WIN32
#include <unistd.h>
#endif

namespace fs = std::filesystem;

// Every file of an entry (.tgt, .dft, .ckpt) is one encrypted stream:
//   header : u32 magic, u32 version, u8 encrypted, u8 nonce_prefix[8]
//   records: u32 len, u8 final, u8 data[len], u8 tag[16] (tag only when encrypted)
// With --no-cache-disk-encrypt the records carry plaintext and no tag (same framing, same lifecycle).
// Record i is AES-256-GCM with IV = nonce_prefix || u32 i and AAD = kind || u32 i || final, so records cannot be
// reordered, swapped between files or truncated (the last record carries final = 1 and nothing may follow it).
// .ckpt plaintext: u32 count, then per checkpoint i64 n_tokens, i32 id_task, i32 pos_min, i32 pos_max and
// (u64 len, bytes) for tgt, dft and spec.
static constexpr uint32_t PDC_MAGIC   = 0x45434450;  // "PDCE"
static constexpr uint32_t PDC_VERSION = 1;
static constexpr size_t   PDC_CHUNK   = 4u << 20;    // plaintext bytes per record
static constexpr size_t   PDC_TAG     = 16;

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

struct cipher_free {
    void operator()(EVP_CIPHER_CTX * c) const { EVP_CIPHER_CTX_free(c); }
};
using cipher_ptr = std::unique_ptr<EVP_CIPHER_CTX, cipher_free>;

void record_iv_aad(const uint8_t prefix[8], char kind, uint32_t index, uint8_t final_flag, uint8_t iv[12], uint8_t aad[6]) {
    memcpy(iv, prefix, 8);
    memcpy(iv + 8, &index, 4);
    aad[0] = (uint8_t) kind;
    memcpy(aad + 1, &index, 4);
    aad[5] = final_flag;
}

// Buffers plaintext and writes it as encrypted records.
class enc_writer {
public:
    enc_writer(const fs::path & p, const uint8_t * key, char kind) : key(key), kind(kind), f(fopen(p.string().c_str(), "wb")) {
        const uint8_t enc = key != nullptr;
        ok = f && (!enc || RAND_bytes(prefix, sizeof(prefix)) == 1);
        ok = ok && fwrite(&PDC_MAGIC, 4, 1, f.get()) == 1 && fwrite(&PDC_VERSION, 4, 1, f.get()) == 1
                && fwrite(&enc, 1, 1, f.get()) == 1 && fwrite(prefix, sizeof(prefix), 1, f.get()) == 1;
        buf.reserve(PDC_CHUNK);
    }

    ~enc_writer() { OPENSSL_cleanse(buf.data(), buf.capacity()); }

    bool write(const void * data, size_t n) {
        const auto * p = static_cast<const uint8_t *>(data);
        while (ok && n > 0) {
            const size_t take = std::min(n, PDC_CHUNK - buf.size());
            buf.insert(buf.end(), p, p + take);
            p += take;
            n -= take;
            if (buf.size() == PDC_CHUNK) {
                ok = flush(0);
            }
        }
        return ok;
    }

    bool finish() {
        ok = ok && flush(1) && fflush(f.get()) == 0;
        return ok;
    }

    static bool cb(const void * data, size_t n, void * self) {
        return static_cast<enc_writer *>(self)->write(data, n);
    }

private:
    const uint8_t * key;
    char kind;
    file_ptr f;
    bool ok = false;
    uint8_t prefix[8] = {};
    uint32_t index = 0;
    std::vector<uint8_t> buf;
    std::vector<uint8_t> out;

    bool flush(uint8_t final_flag) {
        if (key == nullptr) {
            const uint32_t n = (uint32_t) buf.size();
            const bool w = fwrite(&n, 4, 1, f.get()) == 1 && fwrite(&final_flag, 1, 1, f.get()) == 1
                        && (n == 0 || fwrite(buf.data(), n, 1, f.get()) == 1);
            buf.clear();
            ++index;
            return w;
        }
        uint8_t iv[12], aad[6], tag[PDC_TAG];
        record_iv_aad(prefix, kind, index, final_flag, iv, aad);
        cipher_ptr c(EVP_CIPHER_CTX_new());
        out.resize(buf.size());
        int len = 0, fin = 0;
        const uint32_t n = (uint32_t) buf.size();
        const bool enc = c
            && EVP_EncryptInit_ex(c.get(), EVP_aes_256_gcm(), nullptr, nullptr, nullptr) == 1
            && EVP_CIPHER_CTX_ctrl(c.get(), EVP_CTRL_GCM_SET_IVLEN, 12, nullptr) == 1
            && EVP_EncryptInit_ex(c.get(), nullptr, nullptr, key, iv) == 1
            && EVP_EncryptUpdate(c.get(), nullptr, &len, aad, sizeof(aad)) == 1
            && (n == 0 || EVP_EncryptUpdate(c.get(), out.data(), &len, buf.data(), (int) n) == 1)
            && EVP_EncryptFinal_ex(c.get(), out.data() + (n == 0 ? 0 : len), &fin) == 1
            && EVP_CIPHER_CTX_ctrl(c.get(), EVP_CTRL_GCM_GET_TAG, PDC_TAG, tag) == 1;
        OPENSSL_cleanse(buf.data(), buf.size());
        buf.clear();
        ++index;
        return enc && fwrite(&n, 4, 1, f.get()) == 1 && fwrite(&final_flag, 1, 1, f.get()) == 1
                   && (n == 0 || fwrite(out.data(), n, 1, f.get()) == 1) && fwrite(tag, PDC_TAG, 1, f.get()) == 1;
    }
};

// Reads and authenticates records, serving plaintext on demand.
class dec_reader {
public:
    dec_reader(const fs::path & p, const uint8_t * key, char kind) : key(key), kind(kind), f(fopen(p.string().c_str(), "rb")) {
        uint32_t magic = 0, version = 0;
        uint8_t enc = 0;
        ok = f && fread(&magic, 4, 1, f.get()) == 1 && fread(&version, 4, 1, f.get()) == 1
               && fread(&enc, 1, 1, f.get()) == 1 && fread(prefix, sizeof(prefix), 1, f.get()) == 1
               && magic == PDC_MAGIC && version == PDC_VERSION && enc == (key != nullptr);
    }

    ~dec_reader() { OPENSSL_cleanse(plain.data(), plain.capacity()); }

    bool read(void * data, size_t n) {
        auto * p = static_cast<uint8_t *>(data);
        while (ok && n > 0) {
            if (pos == plain.size()) {
                if (done || !next()) {
                    ok = false;
                    break;
                }
                continue;
            }
            const size_t take = std::min(n, plain.size() - pos);
            memcpy(p, plain.data() + pos, take);
            pos += take;
            p += take;
            n -= take;
        }
        return ok;
    }

    // true when every byte was consumed, the final record was seen and nothing follows it
    bool at_end() {
        while (ok && !done && pos == plain.size()) {
            if (!next()) {
                return false;
            }
        }
        return ok && done && pos == plain.size() && fgetc(f.get()) == EOF;
    }

    static bool cb(void * data, size_t n, void * self) {
        return static_cast<dec_reader *>(self)->read(data, n);
    }

private:
    const uint8_t * key;
    char kind;
    file_ptr f;
    bool ok = false;
    bool done = false;
    uint8_t prefix[8] = {};
    uint32_t index = 0;
    std::vector<uint8_t> plain;
    std::vector<uint8_t> in;
    size_t pos = 0;

    bool next() {
        uint32_t n = 0;
        uint8_t final_flag = 0, tag[PDC_TAG], iv[12], aad[6];
        if (fread(&n, 4, 1, f.get()) != 1 || fread(&final_flag, 1, 1, f.get()) != 1 || n > PDC_CHUNK || final_flag > 1) {
            return ok = false;
        }
        if (key == nullptr) {
            plain.resize(n);
            ok = n == 0 || fread(plain.data(), n, 1, f.get()) == 1;
            pos = 0;
            done = final_flag == 1;
            ++index;
            return ok;
        }
        in.resize(n);
        if ((n > 0 && fread(in.data(), n, 1, f.get()) != 1) || fread(tag, PDC_TAG, 1, f.get()) != 1) {
            return ok = false;
        }
        record_iv_aad(prefix, kind, index, final_flag, iv, aad);
        cipher_ptr c(EVP_CIPHER_CTX_new());
        OPENSSL_cleanse(plain.data(), plain.size());
        plain.resize(n);
        int len = 0, fin = 0;
        ok = c
            && EVP_DecryptInit_ex(c.get(), EVP_aes_256_gcm(), nullptr, nullptr, nullptr) == 1
            && EVP_CIPHER_CTX_ctrl(c.get(), EVP_CTRL_GCM_SET_IVLEN, 12, nullptr) == 1
            && EVP_DecryptInit_ex(c.get(), nullptr, nullptr, key, iv) == 1
            && EVP_DecryptUpdate(c.get(), nullptr, &len, aad, sizeof(aad)) == 1
            && (n == 0 || EVP_DecryptUpdate(c.get(), plain.data(), &len, in.data(), (int) n) == 1)
            && EVP_CIPHER_CTX_ctrl(c.get(), EVP_CTRL_GCM_SET_TAG, PDC_TAG, tag) == 1
            && EVP_DecryptFinal_ex(c.get(), plain.data() + (n == 0 ? 0 : len), &fin) == 1;  // authenticates
        pos = 0;
        done = final_flag == 1;
        ++index;
        return ok;
    }
};

template <typename T> bool put_v(enc_writer & w, T v) { return w.write(&v, sizeof(v)); }
template <typename T> bool get_v(dec_reader & r, T & v) { return r.read(&v, sizeof(v)); }

bool put_blob(enc_writer & w, const std::vector<uint8_t> & b) {
    return put_v<uint64_t>(w, b.size()) && w.write(b.data(), b.size());
}

bool get_blob(dec_reader & r, std::vector<uint8_t> & b) {
    uint64_t n = 0;
    if (!get_v(r, n) || n > (uint64_t(1) << 36)) {
        return false;
    }
    b.resize(n);
    return r.read(b.data(), n);
}

bool write_ckpts(const fs::path & p, const uint8_t * key, const std::list<common_prompt_checkpoint> & ckpts) {
    enc_writer w(p, key, 'c');
    if (!put_v<uint32_t>(w, ckpts.size())) {
        return false;
    }
    for (const auto & c : ckpts) {
        if (!put_v<int64_t>(w, c.n_tokens) || !put_v<int32_t>(w, c.id_task) || !put_v<int32_t>(w, c.pos_min)
                || !put_v<int32_t>(w, c.pos_max) || !put_blob(w, c.data_tgt) || !put_blob(w, c.data_dft)
                || !put_blob(w, c.data_spec)) {
            return false;
        }
    }
    return w.finish();
}

bool read_ckpts(const fs::path & p, const uint8_t * key, std::list<common_prompt_checkpoint> & ckpts) {
    dec_reader r(p, key, 'c');
    uint32_t count = 0;
    if (!get_v(r, count) || count > 4096) {
        return false;
    }
    for (uint32_t i = 0; i < count; ++i) {
        common_prompt_checkpoint c;
        int64_t n_tokens = 0;
        int32_t id_task = -1, pos_min = 0, pos_max = 0;
        if (!get_v(r, n_tokens) || !get_v(r, id_task) || !get_v(r, pos_min) || !get_v(r, pos_max)
                || !get_blob(r, c.data_tgt) || !get_blob(r, c.data_dft) || !get_blob(r, c.data_spec)) {
            return false;
        }
        c.update_pos(n_tokens, pos_min, pos_max);
        c.id_task = id_task;
        ckpts.push_back(std::move(c));
    }
    return r.at_end();
}

bool write_state(const fs::path & p, const uint8_t * key, char kind, llama_context * ctx, llama_seq_id seq) {
    enc_writer w(p, key, kind);
    return llama_state_seq_save_stream(ctx, seq, enc_writer::cb, &w) > 0 && w.finish();
}

bool read_state(const fs::path & p, const uint8_t * key, char kind, llama_context * ctx, llama_seq_id seq) {
    dec_reader r(p, key, kind);
    return llama_state_seq_load_stream(ctx, seq, dec_reader::cb, &r) > 0 && r.at_end();
}

uintmax_t file_size_or_0(const fs::path & p) {
    std::error_code ec;
    const auto n = fs::file_size(p, ec);
    return ec ? 0 : n;
}

bool pid_alive(long pid) {
#ifndef _WIN32
    return pid > 0 && (kill((pid_t) pid, 0) == 0 || errno == EPERM);
#else
    (void) pid;
    return false;
#endif
}

long current_pid() {
#ifndef _WIN32
    return (long) getpid();
#else
    return 0;
#endif
}

} // namespace

server_prompt_disk::server_prompt_disk(const std::string & root, size_t limit_bytes, int64_t ttl_s, size_t reserve_bytes,
                                       bool encrypt)
    : dir(fs::path(root) / ("run-" + std::to_string(current_pid()))),
      limit_bytes(limit_bytes), ttl_s(ttl_s), reserve_bytes(reserve_bytes), encrypt(encrypt) {
    std::error_code ec;
    fs::create_directories(root, ec);
    if (!ec) {
        remove_stale_runs(root);
        fs::remove_all(dir, ec);  // a dead process that had our pid
        fs::create_directory(dir, ec);
    }
    if (ec) {
        SRV_ERR("prompt disk cache: cannot create %s: %s\n", dir.string().c_str(), ec.message().c_str());
        return;
    }
    // the entries hold prompt contents: only the server's user may even list them (never touch `root` itself, it
    // may be a shared mount)
    fs::permissions(dir, fs::perms::owner_all, fs::perm_options::replace, ec);
    if (encrypt && RAND_bytes(key, sizeof(key)) != 1) {
        SRV_ERR("%s", "prompt disk cache: no randomness for the key, disk tier disabled\n");
        fs::remove_all(dir, ec);
        return;
    }
    ready = true;
    SRV_INF("prompt disk cache: %s, %s, limit %.1f MiB, ttl %lld s, reserve %.1f MiB\n", dir.string().c_str(),
            encrypt ? "encrypted (AES-256-GCM, per-run key)" : "NOT encrypted (--no-cache-disk-encrypt)",
            limit_bytes / 1048576.0, (long long) ttl_s, reserve_bytes / 1048576.0);
}

server_prompt_disk::~server_prompt_disk() {
    OPENSSL_cleanse(key, sizeof(key));
    std::error_code ec;
    fs::remove_all(dir, ec);
}

void server_prompt_disk::remove_stale_runs(const fs::path & root) const {
    std::error_code ec;
    for (const auto & de : fs::directory_iterator(root, ec)) {
        const std::string name = de.path().filename().string();
        if (!de.is_directory(ec) || name.rfind("run-", 0) != 0) {
            continue;
        }
        char * end = nullptr;
        const long pid = strtol(name.c_str() + 4, &end, 10);
        if (end == name.c_str() + 4 || *end != '\0' || pid == current_pid() || pid_alive(pid)) {
            continue;
        }
        SRV_INF("prompt disk cache: removing %s (its process is gone and so is its key)\n", de.path().string().c_str());
        fs::remove_all(de.path(), ec);
    }
}

fs::path server_prompt_disk::path(const std::string & id, const char * ext) const {
    return dir / (id + ext);
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
    const auto age = std::chrono::steady_clock::now() - e.t_used;
    return std::chrono::duration_cast<std::chrono::seconds>(age).count() > ttl_s;
}

void server_prompt_disk::remove(std::list<server_prompt_disk_entry>::iterator it) {
    std::error_code ec;
    for (const char * ext : { ".tgt", ".dft", ".ckpt" }) {
        fs::remove(path(it->id, ext), ec);
    }
    entries.erase(it);
}

void server_prompt_disk::touch(std::list<server_prompt_disk_entry>::iterator it) {
    it->t_used = std::chrono::steady_clock::now();
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

    for (auto it = entries.begin(); it != entries.end(); ++it) {
        if (server_tokens(it->tokens, false).get_common_prefix(prompt.tokens) == prompt.tokens.size()) {
            SRV_TRC("prompt disk cache: prompt already stored in %s\n", it->id.c_str());
            touch(it);
            return false;
        }
    }

    evict();

    size_t need = llama_state_seq_get_size_ext(ctx_tgt, seq, LLAMA_STATE_SEQ_FLAGS_NONE)
                + (ctx_dft ? llama_state_seq_get_size_ext(ctx_dft, seq, LLAMA_STATE_SEQ_FLAGS_NONE) : 0);
    for (const auto & c : prompt.checkpoints) {
        need += c.size();
    }
    if (limit_bytes > 0 && need > limit_bytes) {
        SRV_WRN("prompt disk cache: state %.1f MiB exceeds the limit, not stored\n", need / 1048576.0);
        return false;
    }
    while (!entries.empty() && limit_bytes > 0 && size() + need > limit_bytes) {
        remove(entries.begin());
    }
    std::error_code ec;
    const auto sp = fs::space(dir, ec);
    if (!ec && sp.available < need + reserve_bytes) {
        SRV_WRN("prompt disk cache: %.1f MiB free is below state + reserve, not stored\n", sp.available / 1048576.0);
        return false;
    }

    const int64_t t0 = ggml_time_us();
    const llama_tokens tokens = prompt.tokens.get_text_tokens();
    server_prompt_disk_entry e;
    e.id = server_prompt_disk_hash(tokens.data(), tokens.size() * sizeof(llama_token));
    for (auto it = entries.begin(); it != entries.end(); ++it) {
        if (it->id == e.id) {  // a hash collision with a different prompt: replace it
            remove(it);
            break;
        }
    }

    const bool ok = write_state(path(e.id, ".tgt"), key_ptr(), 't', ctx_tgt, seq)
        && (!ctx_dft || write_state(path(e.id, ".dft"), key_ptr(), 'd', ctx_dft, seq))
        && (prompt.checkpoints.empty() || write_ckpts(path(e.id, ".ckpt"), key_ptr(), prompt.checkpoints));
    if (!ok) {
        SRV_ERR("prompt disk cache: failed to write entry %s\n", e.id.c_str());
        for (const char * ext : { ".tgt", ".dft", ".ckpt" }) {
            fs::remove(path(e.id, ext), ec);
        }
        return false;
    }

    // entries the new prompt fully contains are obsolete (the RAM tier's rule)
    const server_tokens nt(tokens, false);
    for (auto it = entries.begin(); it != entries.end();) {
        auto next = std::next(it);
        if (nt.get_common_prefix(server_tokens(it->tokens, false)) == it->tokens.size()) {
            remove(it);
        }
        it = next;
    }

    for (const char * ext : { ".tgt", ".dft", ".ckpt" }) {
        e.bytes += file_size_or_0(path(e.id, ext));
    }
    e.tokens = tokens;
    e.t_used = std::chrono::steady_clock::now();
    entries.push_back(std::move(e));

    SRV_INF("prompt disk cache: saved %d tokens, %.1f MiB, %zu checkpoints in %.0f ms (%zu entries, %.1f MiB)\n",
            (int) tokens.size(), entries.back().bytes / 1048576.0, prompt.checkpoints.size(),
            (ggml_time_us() - t0) / 1000.0, entries.size(), size() / 1048576.0);
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
    std::list<common_prompt_checkpoint> ckpts;
    std::error_code ec;
    const bool ok = read_state(path(it->id, ".tgt"), key_ptr(), 't', ctx_tgt, seq)
        && (!ctx_dft || read_state(path(it->id, ".dft"), key_ptr(), 'd', ctx_dft, seq))
        && (!fs::exists(path(it->id, ".ckpt"), ec) || read_ckpts(path(it->id, ".ckpt"), key_ptr(), ckpts));
    if (!ok) {
        // the sequence may be half-restored: the caller clears the slot on false
        SRV_ERR("prompt disk cache: failed to restore entry %s, deleting it\n", it->id.c_str());
        remove(it);
        return false;
    }

    prompt.tokens      = server_tokens(it->tokens, false);
    prompt.checkpoints = std::move(ckpts);
    touch(it);

    SRV_INF("prompt disk cache: restored %d tokens (%.1f MiB) in %.0f ms\n", (int) entries.back().tokens.size(),
            entries.back().bytes / 1048576.0, (ggml_time_us() - t0) / 1000.0);
    return true;
}
