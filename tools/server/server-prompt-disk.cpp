#include "server-prompt-disk.h"

#include "server-task.h"

#ifdef LLAMA_PDC_OPENSSL
#include <openssl/crypto.h>
#include <openssl/evp.h>
#include <openssl/rand.h>
#endif

#include <algorithm>
#include <cerrno>
#include <cstdio>
#include <cstring>
#include <memory>
#include <new>
#include <system_error>
#include <vector>

#ifndef _WIN32
#include <fcntl.h>
#include <sys/file.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace fs = std::filesystem;

// Every file of an entry (.tgt, .dft, .ckpt) is one stream of records:
//   header : u32 magic, u32 version, u8 encrypted, u8 nonce_prefix[8]
//   record : u32 len, u8 final, u8 data[len], u8 tag[16]
// AAD of record i = kind || u32 i || final || entry id (16 hex chars). Encrypted: AES-256-GCM, IV = nonce_prefix ||
// u32 i, tag = the GCM tag. Plaintext (--no-cache-disk-encrypt): tag = a 64-bit keyless checksum of AAD and data,
// then the length; it catches corruption, not tampering. Records cannot be reordered, moved to another file or entry,
// or truncated (the last record has final = 1 and nothing may follow it).
// .ckpt plaintext: u32 count, then per checkpoint i64 n_tokens, i32 id_task, i32 pos_min, i32 pos_max and
// (u64 len, bytes) for tgt, dft and spec.
static constexpr uint32_t PDC_MAGIC     = 0x45434450;  // "PDCE"
static constexpr uint32_t PDC_VERSION   = 2;
static constexpr size_t   PDC_CHUNK     = 4u << 20;    // plaintext bytes per record
static constexpr size_t   PDC_TAG       = 16;
static constexpr size_t   PDC_ID        = 16;
static constexpr size_t   PDC_AAD       = 6 + PDC_ID;
static const char *       PDC_BASE      = "llama-pdc";
static const char *       PDC_MARKER    = "llama-pdc.run";
static const char *       PDC_LOCK      = "lock";

static uint64_t pdc_mix(uint64_t h, uint64_t w) {
    h ^= w;
    h *= 0xff51afd7ed558ccdull;
    return h ^ (h >> 32);
}

std::string server_prompt_disk_hash(const void * data, size_t n) {
    const auto * p = static_cast<const uint8_t *>(data);
    uint64_t h = 0x9e3779b97f4a7c15ull ^ n;
    size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        uint64_t w;
        memcpy(&w, p + i, 8);
        h = pdc_mix(h, w);
    }
    for (; i < n; ++i) {
        h = pdc_mix(h, p[i]);
    }
    h = pdc_mix(h, 0x2545f4914f6cdd1dull);
    char buf[17];
    snprintf(buf, sizeof(buf), "%016llx", (unsigned long long) h);
    return buf;
}

namespace {

void cleanse(void * p, size_t n) {
#ifdef LLAMA_PDC_OPENSSL
    OPENSSL_cleanse(p, n);
#else
    volatile uint8_t * v = static_cast<volatile uint8_t *>(p);
    while (n--) {
        *v++ = 0;
    }
#endif
}

void plain_tag(const uint8_t * aad, const uint8_t * data, uint32_t n, uint8_t tag[PDC_TAG]) {
    uint64_t h = 0x84222325cbf29ce4ull ^ n;
    for (size_t i = 0; i < PDC_AAD; ++i) {
        h = pdc_mix(h, aad[i]);
    }
    size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        uint64_t w;
        memcpy(&w, data + i, 8);
        h = pdc_mix(h, w);
    }
    for (; i < n; ++i) {
        h = pdc_mix(h, data[i]);
    }
    const uint64_t len = n;
    memcpy(tag, &h, 8);
    memcpy(tag + 8, &len, 8);
}

void record_iv_aad(const uint8_t prefix[8], char kind, uint32_t index, uint8_t final_flag, const std::string & id,
                   uint8_t iv[12], uint8_t aad[PDC_AAD]) {
    memcpy(iv, prefix, 8);
    memcpy(iv + 8, &index, 4);
    aad[0] = (uint8_t) kind;
    memcpy(aad + 1, &index, 4);
    aad[5] = final_flag;
    memset(aad + 6, 0, PDC_ID);
    memcpy(aad + 6, id.data(), std::min(id.size(), PDC_ID));
}

bool random_bytes(uint8_t * p, size_t n) {
#ifdef LLAMA_PDC_OPENSSL
    return RAND_bytes(p, (int) n) == 1;
#else
    FILE * f = fopen("/dev/urandom", "rb");
    const bool ok = f && fread(p, 1, n, f) == n;
    if (f) {
        fclose(f);
    }
    return ok;
#endif
}

struct file_closer {
    void operator()(FILE * f) const { if (f) { fclose(f); } }
};
using file_ptr = std::unique_ptr<FILE, file_closer>;

// Owner-only, never follows a symlink, never reuses an existing file.
file_ptr create_file(const fs::path & p) {
#ifndef _WIN32
    const int fd = open(p.c_str(), O_WRONLY | O_CREAT | O_EXCL | O_NOFOLLOW | O_CLOEXEC, 0600);
    if (fd < 0) {
        return nullptr;
    }
    FILE * f = fdopen(fd, "wb");
    if (!f) {
        close(fd);
    }
    return file_ptr(f);
#else
    return file_ptr(fopen(p.string().c_str(), "wb"));
#endif
}

file_ptr open_file(const fs::path & p) {
#ifndef _WIN32
    const int fd = open(p.c_str(), O_RDONLY | O_NOFOLLOW | O_CLOEXEC);
    if (fd < 0) {
        return nullptr;
    }
    FILE * f = fdopen(fd, "rb");
    if (!f) {
        close(fd);
    }
    return file_ptr(f);
#else
    return file_ptr(fopen(p.string().c_str(), "rb"));
#endif
}

// Written data goes to disk and leaves the page cache (the box has no RAM to spare for it).
bool sync_and_drop(FILE * f) {
    if (fflush(f) != 0) {
        return false;
    }
#if defined(__linux__)
    const int fd = fileno(f);
    if (fdatasync(fd) != 0) {
        return false;
    }
    posix_fadvise(fd, 0, 0, POSIX_FADV_DONTNEED);
#elif !defined(_WIN32)
    if (fsync(fileno(f)) != 0) {
        return false;
    }
#endif
    return true;
}

void drop_cache(FILE * f) {
#if defined(__linux__)
    posix_fadvise(fileno(f), 0, 0, POSIX_FADV_DONTNEED);
#else
    (void) f;
#endif
}

#ifdef LLAMA_PDC_OPENSSL
struct cipher_free {
    void operator()(EVP_CIPHER_CTX * c) const { EVP_CIPHER_CTX_free(c); }
};
using cipher_ptr = std::unique_ptr<EVP_CIPHER_CTX, cipher_free>;
#endif

// Buffers plaintext and writes it as records (encrypted when key != nullptr).
class rec_writer {
public:
    rec_writer(const fs::path & p, const uint8_t * key, char kind, const std::string & id)
        : key(key), kind(kind), id(id), f(create_file(p)) {
        const uint8_t enc = key != nullptr;
        ok = f && (!enc || random_bytes(prefix, sizeof(prefix)));
        ok = ok && fwrite(&PDC_MAGIC, 4, 1, f.get()) == 1 && fwrite(&PDC_VERSION, 4, 1, f.get()) == 1
                && fwrite(&enc, 1, 1, f.get()) == 1 && fwrite(prefix, sizeof(prefix), 1, f.get()) == 1;
        buf.reserve(PDC_CHUNK);
    }

    ~rec_writer() {
        cleanse(buf.data(), buf.size());
    }

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
        ok = ok && flush(1) && sync_and_drop(f.get());
        return ok;
    }

    static bool cb(const void * data, size_t n, void * self) {
        return static_cast<rec_writer *>(self)->write(data, n);
    }

private:
    const uint8_t * key;
    char            kind;
    std::string     id;
    file_ptr        f;
    bool            ok = false;
    uint8_t         prefix[8] = {};
    uint32_t        index = 0;
    std::vector<uint8_t> buf;
    std::vector<uint8_t> out;

    bool seal(uint8_t final_flag, uint8_t tag[PDC_TAG], const uint8_t *& payload) {
        uint8_t iv[12], aad[PDC_AAD];
        record_iv_aad(prefix, kind, index, final_flag, id, iv, aad);
        const uint32_t n = (uint32_t) buf.size();
        if (key == nullptr) {
            plain_tag(aad, buf.data(), n, tag);
            payload = buf.data();
            return true;
        }
#ifdef LLAMA_PDC_OPENSSL
        cipher_ptr c(EVP_CIPHER_CTX_new());
        out.resize(n);
        int len = 0, fin = 0;
        payload = out.data();
        return c
            && EVP_EncryptInit_ex(c.get(), EVP_aes_256_gcm(), nullptr, nullptr, nullptr) == 1
            && EVP_CIPHER_CTX_ctrl(c.get(), EVP_CTRL_GCM_SET_IVLEN, 12, nullptr) == 1
            && EVP_EncryptInit_ex(c.get(), nullptr, nullptr, key, iv) == 1
            && EVP_EncryptUpdate(c.get(), nullptr, &len, aad, sizeof(aad)) == 1
            && (n == 0 || EVP_EncryptUpdate(c.get(), out.data(), &len, buf.data(), (int) n) == 1)
            && EVP_EncryptFinal_ex(c.get(), out.data() + (n == 0 ? 0 : len), &fin) == 1
            && EVP_CIPHER_CTX_ctrl(c.get(), EVP_CTRL_GCM_GET_TAG, PDC_TAG, tag) == 1;
#else
        return false;
#endif
    }

    bool flush(uint8_t final_flag) {
        uint8_t tag[PDC_TAG];
        const uint8_t * payload = nullptr;
        const uint32_t n = (uint32_t) buf.size();
        const bool sealed = seal(final_flag, tag, payload);
        const bool w = sealed && fwrite(&n, 4, 1, f.get()) == 1 && fwrite(&final_flag, 1, 1, f.get()) == 1
                    && (n == 0 || fwrite(payload, n, 1, f.get()) == 1) && fwrite(tag, PDC_TAG, 1, f.get()) == 1;
        cleanse(buf.data(), buf.size());
        buf.clear();
        ++index;
        return w;
    }
};

// Reads and verifies records, serving plaintext on demand.
class rec_reader {
public:
    rec_reader(const fs::path & p, const uint8_t * key, char kind, const std::string & id)
        : key(key), kind(kind), id(id), f(open_file(p)) {
        uint32_t magic = 0, version = 0;
        uint8_t enc = 0;
        ok = f && fread(&magic, 4, 1, f.get()) == 1 && fread(&version, 4, 1, f.get()) == 1
               && fread(&enc, 1, 1, f.get()) == 1 && fread(prefix, sizeof(prefix), 1, f.get()) == 1
               && magic == PDC_MAGIC && version == PDC_VERSION && enc == (key != nullptr);
        std::error_code ec;
        file_bytes = fs::file_size(p, ec);
        ok = ok && !ec;
    }

    ~rec_reader() {
        cleanse(plain.data(), plain.size());
        if (f) {
            drop_cache(f.get());
        }
    }

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

    // every byte consumed, the final record seen and nothing after it
    bool at_end() {
        while (ok && !done && pos == plain.size()) {
            if (!next()) {
                return false;
            }
        }
        return ok && done && pos == plain.size() && fgetc(f.get()) == EOF;
    }

    size_t size() const { return file_bytes; }

    static bool cb(void * data, size_t n, void * self) {
        return static_cast<rec_reader *>(self)->read(data, n);
    }

private:
    const uint8_t * key;
    char            kind;
    std::string     id;
    file_ptr        f;
    bool            ok = false;
    bool            done = false;
    uint8_t         prefix[8] = {};
    uint32_t        index = 0;
    size_t          file_bytes = 0;
    std::vector<uint8_t> plain;
    std::vector<uint8_t> in;
    size_t          pos = 0;

    bool next() {
        uint32_t n = 0;
        uint8_t final_flag = 0, tag[PDC_TAG], iv[12], aad[PDC_AAD];
        if (fread(&n, 4, 1, f.get()) != 1 || fread(&final_flag, 1, 1, f.get()) != 1 || n > PDC_CHUNK
                || final_flag > 1) {
            return ok = false;
        }
        record_iv_aad(prefix, kind, index, final_flag, id, iv, aad);
        cleanse(plain.data(), plain.size());
        plain.resize(n);
        pos = 0;
        done = final_flag == 1;
        ++index;
        if (key == nullptr) {
            uint8_t want[PDC_TAG];
            ok = (n == 0 || fread(plain.data(), n, 1, f.get()) == 1) && fread(tag, PDC_TAG, 1, f.get()) == 1;
            plain_tag(aad, plain.data(), n, want);
            return ok = ok && memcmp(tag, want, PDC_TAG) == 0;
        }
#ifdef LLAMA_PDC_OPENSSL
        in.resize(n);
        if ((n > 0 && fread(in.data(), n, 1, f.get()) != 1) || fread(tag, PDC_TAG, 1, f.get()) != 1) {
            return ok = false;
        }
        cipher_ptr c(EVP_CIPHER_CTX_new());
        int len = 0, fin = 0;
        return ok = c
            && EVP_DecryptInit_ex(c.get(), EVP_aes_256_gcm(), nullptr, nullptr, nullptr) == 1
            && EVP_CIPHER_CTX_ctrl(c.get(), EVP_CTRL_GCM_SET_IVLEN, 12, nullptr) == 1
            && EVP_DecryptInit_ex(c.get(), nullptr, nullptr, key, iv) == 1
            && EVP_DecryptUpdate(c.get(), nullptr, &len, aad, sizeof(aad)) == 1
            && (n == 0 || EVP_DecryptUpdate(c.get(), plain.data(), &len, in.data(), (int) n) == 1)
            && EVP_CIPHER_CTX_ctrl(c.get(), EVP_CTRL_GCM_SET_TAG, PDC_TAG, tag) == 1
            && EVP_DecryptFinal_ex(c.get(), plain.data() + (n == 0 ? 0 : len), &fin) == 1;  // authenticates
#else
        return ok = false;
#endif
    }
};

template <typename T> bool put_v(rec_writer & w, T v) { return w.write(&v, sizeof(v)); }
template <typename T> bool get_v(rec_reader & r, T & v) { return r.read(&v, sizeof(v)); }

bool put_blob(rec_writer & w, const std::vector<uint8_t> & b) {
    return put_v<uint64_t>(w, b.size()) && w.write(b.data(), b.size());
}

bool get_blob(rec_reader & r, std::vector<uint8_t> & b) {
    uint64_t n = 0;
    if (!get_v(r, n) || n > r.size()) {  // a blob can never be larger than its file
        return false;
    }
    b.resize(n);
    return r.read(b.data(), n);
}

bool write_ckpts(const fs::path & p, const uint8_t * key, const std::string & id,
                 const std::list<common_prompt_checkpoint> & ckpts) {
    rec_writer w(p, key, 'c', id);
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

bool read_ckpts(const fs::path & p, const uint8_t * key, const std::string & id,
                std::list<common_prompt_checkpoint> & ckpts) {
    rec_reader r(p, key, 'c', id);
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

bool write_state(const fs::path & p, const uint8_t * key, char kind, const std::string & id, llama_context * ctx,
                 llama_seq_id seq) {
    rec_writer w(p, key, kind, id);
    return llama_state_seq_save_stream(ctx, seq, LLAMA_STATE_SEQ_FLAGS_NONE, rec_writer::cb, &w) > 0 && w.finish();
}

bool read_state(const fs::path & p, const uint8_t * key, char kind, const std::string & id, llama_context * ctx,
                llama_seq_id seq) {
    rec_reader r(p, key, kind, id);
    return llama_state_seq_load_stream(ctx, seq, LLAMA_STATE_SEQ_FLAGS_NONE, rec_reader::cb, &r) > 0 && r.at_end();
}

uintmax_t file_size_or_0(const fs::path & p) {
    std::error_code ec;
    const auto n = fs::file_size(p, ec);
    return ec ? 0 : n;
}

// The prompt's token ids, or false when it contains media (a LLAMA_TOKEN_NULL placeholder). Uses operator[] because
// get_tokens() asserts on slots of an mmproj server, which are all flagged has_mtmd.
bool text_tokens(const server_tokens & tokens, llama_tokens & out) {
    out.resize(tokens.size());
    for (size_t i = 0; i < tokens.size(); ++i) {
        out[i] = tokens[i];
        if (out[i] == LLAMA_TOKEN_NULL) {
            return false;
        }
    }
    return true;
}

} // namespace

server_prompt_disk::server_prompt_disk(const std::string & root, size_t limit_bytes, int64_t ttl_s,
                                       size_t reserve_bytes, bool encrypt, bool can_truncate)
    : limit_bytes(limit_bytes), ttl_s(ttl_s), reserve_bytes(reserve_bytes), encrypt(encrypt),
      can_truncate(can_truncate) {
#ifdef _WIN32
    (void) root;
    SRV_ERR("%s", "prompt disk cache: not supported on Windows\n");
#else
#ifndef LLAMA_PDC_OPENSSL
    if (encrypt) {
        SRV_ERR("%s", "prompt disk cache: built without OpenSSL, encryption unavailable (use --no-cache-disk-encrypt)\n");
        return;
    }
#endif
    std::error_code ec;
    const fs::path base = fs::path(root) / PDC_BASE;
    fs::create_directories(base, ec);
    if (ec) {
        SRV_ERR("prompt disk cache: cannot create %s: %s\n", base.string().c_str(), ec.message().c_str());
        return;
    }
    remove_stale_runs(base);

    std::string tmpl = (base / "run-XXXXXX").string();
    if (mkdtemp(tmpl.data()) == nullptr) {  // mode 0700
        SRV_ERR("prompt disk cache: cannot create a run directory in %s: %s\n", base.string().c_str(), strerror(errno));
        return;
    }
    dir = tmpl;

    // lock first, marker second: a concurrent startup never sees a marked directory that is not yet locked
    lock_fd = open((dir / PDC_LOCK).c_str(), O_RDWR | O_CREAT | O_EXCL | O_NOFOLLOW | O_CLOEXEC, 0600);
    if (lock_fd < 0 || flock(lock_fd, LOCK_EX | LOCK_NB) != 0 || !create_file(dir / PDC_MARKER)) {
        SRV_ERR("prompt disk cache: cannot lock %s: %s\n", dir.string().c_str(), strerror(errno));
        fs::remove_all(dir, ec);
        return;
    }

    if (encrypt) {
        key_page = (size_t) sysconf(_SC_PAGESIZE);
        void * page = mmap(nullptr, key_page, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        if (page == MAP_FAILED) {
            SRV_ERR("%s", "prompt disk cache: cannot map the key page\n");
            fs::remove_all(dir, ec);
            return;
        }
        key = static_cast<uint8_t *>(page);
        if (mlock(page, key_page) != 0) {
            SRV_WRN("%s", "prompt disk cache: cannot lock the key page in RAM (RLIMIT_MEMLOCK); it may reach swap\n");
        }
#ifdef MADV_DONTDUMP
        madvise(page, key_page, MADV_DONTDUMP);
#endif
        if (!random_bytes(key, 32)) {
            SRV_ERR("%s", "prompt disk cache: no randomness for the key\n");
            fs::remove_all(dir, ec);
            return;
        }
    }

    ready = true;
    SRV_INF("prompt disk cache: %s, %s, limit %.1f MiB, ttl %lld s, reserve %.1f MiB\n", dir.string().c_str(),
            encrypt ? "encrypted (AES-256-GCM, per-run key)" : "NOT encrypted (--no-cache-disk-encrypt)",
            limit_bytes / 1048576.0, (long long) ttl_s, reserve_bytes / 1048576.0);
#endif
}

server_prompt_disk::~server_prompt_disk() {
#ifndef _WIN32
    if (key != nullptr) {
        cleanse(key, 32);
        munlock(key, key_page);
        munmap(key, key_page);
    }
    std::error_code ec;
    if (!dir.empty()) {
        fs::remove_all(dir, ec);
    }
    if (lock_fd >= 0) {
        close(lock_fd);
    }
#endif
}

void server_prompt_disk::remove_stale_runs(const fs::path & base) const {
#ifndef _WIN32
    std::error_code ec;
    std::vector<fs::path> stale;
    for (auto it = fs::directory_iterator(base, ec); !ec && it != fs::directory_iterator(); it.increment(ec)) {
        const fs::path p = it->path();
        std::error_code ec2;
        if (p.filename().string().rfind("run-", 0) != 0 || !fs::is_directory(fs::symlink_status(p, ec2))
                || !fs::exists(p / PDC_MARKER, ec2)) {
            continue;  // not ours: never touched
        }
        const int fd = open((p / PDC_LOCK).c_str(), O_RDWR | O_NOFOLLOW | O_CLOEXEC);
        if (fd < 0) {
            continue;
        }
        if (flock(fd, LOCK_EX | LOCK_NB) == 0) {
            stale.push_back(p);  // no live server holds it: its key died with it
        }
        close(fd);
    }
    for (const auto & p : stale) {
        SRV_INF("prompt disk cache: removing %s (its server is gone and so is its key)\n", p.string().c_str());
        fs::remove_all(p, ec);
    }
#else
    (void) base;
#endif
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

uintmax_t server_prompt_disk::available() const {
    std::error_code ec;
    const auto sp = fs::space(dir, ec);
    return ec ? UINTMAX_MAX : sp.available;
}

bool server_prompt_disk::expired(const server_prompt_disk_entry & e) const {
    if (ttl_s <= 0) {
        return false;
    }
    const auto age = std::chrono::steady_clock::now() - e.t_used;
    return std::chrono::duration_cast<std::chrono::seconds>(age).count() > ttl_s;
}

size_t server_prompt_disk::remove(std::list<server_prompt_disk_entry>::iterator it) {
    std::error_code ec;
    for (const char * ext : { ".tgt", ".dft", ".ckpt" }) {
        fs::remove(path(it->id, ext), ec);
    }
    const size_t freed = it->bytes;
    entries.erase(it);
    return freed;
}

void server_prompt_disk::touch(std::list<server_prompt_disk_entry>::iterator it) {
    it->t_used = std::chrono::steady_clock::now();
    entries.splice(entries.end(), entries, it);
}

// usable prefix of entry e when lcp tokens match: the whole entry, any prefix when the memory can be rolled back,
// else the longest stored checkpoint within the match
size_t server_prompt_disk::reusable(const server_prompt_disk_entry & e, size_t lcp) const {
    if (lcp == e.tokens.size() || can_truncate) {
        return lcp;
    }
    size_t best = 0;
    for (const int64_t n : e.ckpt_tokens) {
        if (n > 0 && (size_t) n <= lcp) {
            best = std::max(best, (size_t) n);
        }
    }
    return best;
}

// entry e restores `tokens` exactly: tokens is e's prompt, or a prefix e can be rolled back to
bool server_prompt_disk::covers(const server_prompt_disk_entry & e, const server_tokens & tokens) const {
    const size_t n = tokens.size();
    if (n > e.tokens.size() || server_tokens(e.tokens, false).get_common_prefix(tokens) != n) {
        return false;
    }
    if (n == e.tokens.size() || can_truncate) {
        return true;
    }
    return std::find(e.ckpt_tokens.begin(), e.ckpt_tokens.end(), (int64_t) n) != e.ckpt_tokens.end();
}

bool server_prompt_disk::stored(const server_prompt & prompt) {
    for (auto it = entries.begin(); it != entries.end(); ++it) {
        if (!expired(*it) && covers(*it, prompt.tokens)) {
            touch(it);
            return true;
        }
    }
    return false;
}

void server_prompt_disk::evict() {
    size_t freed = 0;
    for (auto it = entries.begin(); it != entries.end();) {
        auto next = std::next(it);
        if (expired(*it)) {
            SRV_TRC("prompt disk cache: expired entry %s (%d tokens)\n", it->id.c_str(), (int) it->tokens.size());
            freed += remove(it);
        }
        it = next;
    }
    // btrfs and others report freed space only after their next commit: count what was freed ourselves
    const uintmax_t avail = available();
    while (!entries.empty() && ((limit_bytes > 0 && size() > limit_bytes) || avail + freed < reserve_bytes)) {
        SRV_TRC("prompt disk cache: evicting oldest entry %s (%.1f MiB)\n", entries.front().id.c_str(),
                entries.front().bytes / 1048576.0);
        freed += remove(entries.begin());
    }
}

bool server_prompt_disk::save(const server_prompt & prompt, llama_context * ctx_tgt, llama_context * ctx_dft,
                              llama_seq_id seq) {
    try {
        return save_impl(prompt, ctx_tgt, ctx_dft, seq);
    } catch (const std::exception & e) {
        SRV_ERR("prompt disk cache: save failed: %s\n", e.what());
        return false;
    }
}

bool server_prompt_disk::save_impl(const server_prompt & prompt, llama_context * ctx_tgt, llama_context * ctx_dft,
                                   llama_seq_id seq) {
    if (!ready || prompt.tokens.size() == 0) {
        return false;
    }
    llama_tokens tokens;
    if (!text_tokens(prompt.tokens, tokens)) {
        SRV_TRC("%s", "prompt disk cache: skipping a prompt that contains media\n");
        return false;
    }
    if (stored(prompt)) {
        SRV_TRC("%s", "prompt disk cache: prompt already stored\n");
        return false;
    }

    evict();

    size_t need = llama_state_seq_get_size_ext(ctx_tgt, seq, LLAMA_STATE_SEQ_FLAGS_NONE)
                + (ctx_dft ? llama_state_seq_get_size_ext(ctx_dft, seq, LLAMA_STATE_SEQ_FLAGS_NONE) : 0);
    for (const auto & c : prompt.checkpoints) {
        need += c.size();
    }
    need += need / 64 + (1u << 20);  // record framing
    if (limit_bytes > 0 && need > limit_bytes) {
        SRV_WRN("prompt disk cache: state %.1f MiB exceeds the limit, not stored\n", need / 1048576.0);
        return false;
    }
    // refuse before evicting anything if even an empty cache would not leave the reserve
    const uintmax_t avail = available();
    if (avail != UINTMAX_MAX && avail + size() < need + reserve_bytes) {
        SRV_WRN("prompt disk cache: %.1f MiB free is below state + reserve, not stored\n", avail / 1048576.0);
        return false;
    }
    size_t freed = 0;
    while (!entries.empty() && ((limit_bytes > 0 && size() + need > limit_bytes)
                                || (avail != UINTMAX_MAX && avail + freed < need + reserve_bytes))) {
        freed += remove(entries.begin());
    }

    const int64_t t0 = ggml_time_us();
    server_prompt_disk_entry e;
    e.id = server_prompt_disk_hash(tokens.data(), tokens.size() * sizeof(llama_token));
    for (auto it = entries.begin(); it != entries.end(); ++it) {
        if (it->id == e.id) {  // same hash, different prompt (a collision): replace it
            remove(it);
            break;
        }
    }

    const bool ok = write_state(path(e.id, ".tgt"), key_ptr(), 't', e.id, ctx_tgt, seq)
        && (!ctx_dft || write_state(path(e.id, ".dft"), key_ptr(), 'd', e.id, ctx_dft, seq))
        && (prompt.checkpoints.empty() || write_ckpts(path(e.id, ".ckpt"), key_ptr(), e.id, prompt.checkpoints));
    if (!ok) {
        SRV_ERR("prompt disk cache: failed to write entry %s\n", e.id.c_str());
        std::error_code ec;
        for (const char * ext : { ".tgt", ".dft", ".ckpt" }) {
            fs::remove(path(e.id, ext), ec);
        }
        return false;
    }

    e.tokens = tokens;
    for (const auto & c : prompt.checkpoints) {
        e.ckpt_tokens.push_back(c.n_tokens);
    }
    for (const char * ext : { ".tgt", ".dft", ".ckpt" }) {
        e.bytes += file_size_or_0(path(e.id, ext));
    }
    e.t_used = std::chrono::steady_clock::now();

    // entries the new one restores exactly are superseded
    for (auto it = entries.begin(); it != entries.end();) {
        auto next = std::next(it);
        if (covers(e, server_tokens(it->tokens, false))) {
            remove(it);
        }
        it = next;
    }
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
    if (!ready || tokens_new.size() == 0) {
        return nullptr;
    }
    for (const auto & e : entries) {
        if (expired(e)) {
            continue;
        }
        const size_t lcp   = server_tokens(e.tokens, false).get_common_prefix(tokens_new);
        const size_t reuse = reusable(e, lcp);
        const float f_keep = float(reuse) / e.tokens.size();
        const float f_sim  = float(reuse) / tokens_new.size();
        if (reuse == 0 || f_keep < 0.25f) {
            continue;  // the RAM tier's "don't trash large prompts" rule, on the part that is actually reusable
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
    if (it == entries.end()) {
        return false;
    }
    try {
        if (load_impl(it, prompt, ctx_tgt, ctx_dft, seq)) {
            return true;
        }
    } catch (const std::exception & e) {
        SRV_ERR("prompt disk cache: restore failed: %s\n", e.what());
    }
    llama_memory_seq_rm(llama_get_memory(ctx_tgt), seq, -1, -1);
    if (ctx_dft) {
        llama_memory_seq_rm(llama_get_memory(ctx_dft), seq, -1, -1);
    }
    SRV_ERR("prompt disk cache: failed to restore entry %s, deleting it\n", it->id.c_str());
    remove(it);
    return false;
}

bool server_prompt_disk::load_impl(std::list<server_prompt_disk_entry>::iterator it, server_prompt & prompt,
                                   llama_context * ctx_tgt, llama_context * ctx_dft, llama_seq_id seq) {
    const int64_t t0 = ggml_time_us();

    // the outgoing checkpoints are already stored or being replaced: free them before reading the new ones, so the
    // two sets are never in RAM together
    prompt.checkpoints.clear();

    std::list<common_prompt_checkpoint> ckpts;
    std::error_code ec;
    const bool ok = read_state(path(it->id, ".tgt"), key_ptr(), 't', it->id, ctx_tgt, seq)
        && (!ctx_dft || read_state(path(it->id, ".dft"), key_ptr(), 'd', it->id, ctx_dft, seq))
        && (it->ckpt_tokens.empty() || read_ckpts(path(it->id, ".ckpt"), key_ptr(), it->id, ckpts))
        && ckpts.size() == it->ckpt_tokens.size();
    if (!ok) {
        return false;
    }

    // keep the slot's multimodal flag: an mmproj server's slot must stay able to take media next
    prompt.tokens      = server_tokens(it->tokens, prompt.tokens.has_mtmd);
    prompt.checkpoints = std::move(ckpts);
    touch(it);

    SRV_INF("prompt disk cache: restored %d tokens (%.1f MiB) in %.0f ms\n", (int) entries.back().tokens.size(),
            entries.back().bytes / 1048576.0, (ggml_time_us() - t0) / 1000.0);
    return true;
}
