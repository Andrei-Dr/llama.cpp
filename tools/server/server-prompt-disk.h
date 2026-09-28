#pragma once

// Disk tier of the server prompt cache: slot states that leave the slot are written to a directory and restored on a
// later request that shares their prefix, so a long prompt is read back instead of being prefilled again.
//
// By default everything on disk is encrypted with AES-256-GCM under a key drawn from the OS CSPRNG when the server starts and
// kept only in its memory: the files are unreadable to anyone else and become garbage when the process exits
// (crypto-shredding). --no-cache-disk-encrypt stores plaintext for fully trusted disks. Each run uses its own directory `<root>/run-<pid>`, removed on exit; directories of runs whose
// process is gone are removed at startup. The index (tokens, sizes, last use) lives in memory only.
//
// Entries expire after a TTL and are evicted oldest first to stay under a byte cap and above a free-space reserve.

#include "server-common.h"

#include <chrono>
#include <cstdint>
#include <filesystem>
#include <list>
#include <string>

struct server_prompt;

struct server_prompt_disk_entry {
    std::string  id;       // hex FNV-1a of the tokens: identical prompts share one entry
    llama_tokens tokens;
    size_t       bytes = 0;
    std::chrono::steady_clock::time_point t_used;  // last save or restore
};

struct server_prompt_disk {
    server_prompt_disk(const std::string & root, size_t limit_bytes, int64_t ttl_s, size_t reserve_bytes, bool encrypt);
    ~server_prompt_disk();

    server_prompt_disk(const server_prompt_disk &) = delete;
    server_prompt_disk & operator=(const server_prompt_disk &) = delete;

    bool ok() const { return ready; }

    // Write the state of `seq` (target and draft contexts) and the prompt's checkpoints. Skips a prompt that an entry
    // already contains and removes entries the new prompt contains. Multimodal prompts are not stored.
    bool save(const server_prompt & prompt, llama_context * ctx_tgt, llama_context * ctx_dft, llama_seq_id seq);

    // The entry that beats both f_keep_base and f_sim_base for tokens_new under the RAM tier's rule, or nullptr.
    const server_prompt_disk_entry * find(const server_tokens & tokens_new, float f_keep_base, float f_sim_base,
                                          float & f_keep_out, float & f_sim_out) const;

    // Restore an entry into `seq` and replace `prompt` (tokens and checkpoints). On failure the entry is deleted.
    bool load(const server_prompt_disk_entry & entry, server_prompt & prompt,
              llama_context * ctx_tgt, llama_context * ctx_dft, llama_seq_id seq);

    // Drop expired entries, then the oldest ones until the byte cap and the free-space reserve hold.
    void evict();

    size_t size() const;
    size_t count() const { return entries.size(); }

private:
    std::filesystem::path dir;
    size_t  limit_bytes;
    int64_t ttl_s;
    size_t  reserve_bytes;
    bool    ready = false;
    bool    encrypt;
    uint8_t key[32] = {};

    const uint8_t * key_ptr() const { return encrypt ? key : nullptr; }

    std::list<server_prompt_disk_entry> entries;  // least recently used first

    std::filesystem::path path(const std::string & id, const char * ext) const;
    void remove_stale_runs(const std::filesystem::path & root) const;
    void remove(std::list<server_prompt_disk_entry>::iterator it);
    void touch(std::list<server_prompt_disk_entry>::iterator it);
    bool expired(const server_prompt_disk_entry & e) const;
};

// Hex FNV-1a of a byte string; used for entry ids.
std::string server_prompt_disk_hash(const void * data, size_t n);
