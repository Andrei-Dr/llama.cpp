#pragma once

// Disk tier of the server prompt cache: slot states are written to a directory and restored on a later request that
// shares their prefix, so a long prompt is read back instead of being prefilled again. States are written after the
// server has been idle for --cache-disk-idle ms (a new request cancels the write) and, if still unsaved, when a request
// evicts them.
//
// By default every file is encrypted with AES-256-GCM under a key drawn from the OS CSPRNG at startup and kept only in
// locked, non-dumpable memory: the files are unreadable to anyone else and become garbage when the process exits
// (crypto-shredding). --no-cache-disk-encrypt stores plaintext with a per-record checksum, for fully trusted disks.
//
// Layout: <root>/llama-pdc/run-XXXXXX (mkdtemp, 0700) per server run, holding a marker file and a lock file that the
// run keeps flock()ed. At startup, run directories that carry the marker and whose lock can be taken are removed;
// nothing without the marker is ever touched. The index (tokens, checkpoint positions, sizes, last use) lives in
// memory only. A new entry replaces older ones it reproduces >= 90% of (the same conversation, one turn later).
// Entries expire after a TTL; under the byte cap or the free-space reserve, old entries largely reproduced by newer
// ones go first, then the least recently used.

#include "server-common.h"

#include <chrono>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <list>
#include <string>
#include <vector>

struct server_prompt;

struct server_prompt_disk_entry {
    std::string          id;           // hex hash of the tokens: identical prompts share one entry
    llama_tokens         tokens;
    std::vector<int64_t> ckpt_tokens;  // n_tokens of each stored context checkpoint
    size_t               bytes = 0;
    std::chrono::steady_clock::time_point t_used;  // last save or restore
};

struct server_prompt_disk {
    // can_truncate: the model's memory can be rolled back to any position (no recurrent/SWA state), so a stored
    // prompt is reusable up to the common prefix; otherwise only up to the prompt's end or one of its checkpoints.
    server_prompt_disk(const std::string & root, size_t limit_bytes, int64_t ttl_s, size_t reserve_bytes, bool encrypt,
                       bool can_truncate);
    ~server_prompt_disk();

    server_prompt_disk(const server_prompt_disk &) = delete;
    server_prompt_disk & operator=(const server_prompt_disk &) = delete;

    bool ok() const { return ready; }

    // Reuse the slot itself offers for tokens_new under the same accounting as stored entries.
    size_t reusable_prompt(const server_prompt & prompt, size_t lcp) const;

    // Whether the prompt is already on disk in a form that restores it exactly.
    bool stored(const server_prompt & prompt);

    // Write the state of `seq` (target and draft contexts) and the prompt's checkpoints, unless already stored.
    // Removes entries the new one supersedes. Prompts containing media are not stored.
    // `abort` is polled between 4 MiB records; when it returns true the save stops and leaves nothing behind.
    bool save(const server_prompt & prompt, llama_context * ctx_tgt, llama_context * ctx_dft, llama_seq_id seq,
              const std::function<bool()> & abort = {});

    // The entry that reuses the most of tokens_new, more than f_sim_base (what the slot or RAM offers), or nullptr.
    // f_keep_base is accepted for symmetry with the RAM tier and not used.
    const server_prompt_disk_entry * find(const server_tokens & tokens_new, float f_keep_base, float f_sim_base,
                                          float & f_keep_out, float & f_sim_out) const;

    // Restore an entry into `seq` and replace `prompt` (tokens and checkpoints). On failure the entry is deleted and
    // `seq` is cleared; the caller must clear the slot.
    bool load(const server_prompt_disk_entry & entry, server_prompt & prompt,
              llama_context * ctx_tgt, llama_context * ctx_dft, llama_seq_id seq);

    // Drop expired entries, then (near-duplicates first, else least recently used) until the cap and reserve hold.
    void evict();

    size_t size() const;
    size_t count() const { return entries.size(); }

private:
    std::filesystem::path dir;
    size_t    limit_bytes;
    int64_t   ttl_s;
    size_t    reserve_bytes;
    bool      encrypt;
    bool      can_truncate;
    bool      ready   = false;
    int       lock_fd = -1;
    uint8_t * key     = nullptr;  // 32 bytes in a locked, non-dumpable page (encrypt only)
    size_t    key_page = 0;

    std::list<server_prompt_disk_entry> entries;  // least recently used first

    const uint8_t * key_ptr() const { return encrypt ? key : nullptr; }
    std::filesystem::path path(const std::string & id, const char * ext) const;
    bool   covers(const server_prompt_disk_entry & e, const server_tokens & tokens) const;
    size_t reusable(const server_prompt_disk_entry & e, size_t lcp) const;
    void   remove_stale_runs(const std::filesystem::path & base) const;
    size_t remove(std::list<server_prompt_disk_entry>::iterator it);
    void   touch(std::list<server_prompt_disk_entry>::iterator it);
    bool   expired(const server_prompt_disk_entry & e) const;
    uintmax_t available() const;
    bool   save_impl(const server_prompt & prompt, llama_context * ctx_tgt, llama_context * ctx_dft, llama_seq_id seq,
                     const std::function<bool()> & abort);
    float  redundancy(const server_prompt_disk_entry & o, const server_prompt_disk_entry & n) const;
    std::list<server_prompt_disk_entry>::iterator victim();
    bool   load_impl(std::list<server_prompt_disk_entry>::iterator it, server_prompt & prompt,
                     llama_context * ctx_tgt, llama_context * ctx_dft, llama_seq_id seq);
};

// Hex hash of a byte string; used for entry ids.
std::string server_prompt_disk_hash(const void * data, size_t n);
