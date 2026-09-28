#pragma once

// Disk tier of the server prompt cache: slot states that leave the slot are written to a directory and restored on a
// later request that shares their prefix, so a long prompt is read from disk instead of being prefilled again.
// Entries are bound to a fingerprint of the model and context setup (one subdirectory per fingerprint), expire after a
// TTL and are evicted oldest first to stay under a byte cap and above a free-space reserve.

#include "server-common.h"

#include <cstdint>
#include <filesystem>
#include <list>
#include <string>

struct server_prompt;

struct server_prompt_disk_entry {
    std::string  id;     // hex FNV-1a of the tokens: identical prompts share one entry
    llama_tokens tokens;
    size_t       bytes = 0;  // all files of the entry
    std::filesystem::file_time_type t_used;  // last save or restore (mtime of the .meta file)
};

struct server_prompt_disk {
    server_prompt_disk(const std::string & root, const std::string & fingerprint,
                       size_t limit_bytes, int64_t ttl_s, size_t reserve_bytes);

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

    std::list<server_prompt_disk_entry> entries;  // least recently used first

    std::filesystem::path path(const std::string & id, const char * ext) const;
    void scan();
    void remove(std::list<server_prompt_disk_entry>::iterator it);
    void touch(std::list<server_prompt_disk_entry>::iterator it);
    bool expired(const server_prompt_disk_entry & e) const;
};

// Hex FNV-1a of a byte string; used for entry ids and the fingerprint directory name.
std::string server_prompt_disk_hash(const void * data, size_t n);
