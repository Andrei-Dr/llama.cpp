import glob, os, time

import pytest
from utils import *

# the prompt cache disk tier (--cache-disk, tools/server/server-prompt-disk.h), seen from the responses and the directory

# the disk tier only restores a prompt that saves >= 256 tokens: a model with more context than stories15M's 256
MODEL_LONG_CTX_FILE_URL = "https://huggingface.co/bartowski/SmolLM2-135M-Instruct-GGUF/resolve/main/SmolLM2-135M-Instruct-Q4_K_M.gguf"

MODEL_DRAFT_FILE_URL = "https://huggingface.co/ggml-org-dev/tiny-llamas/resolve/main/stories15M-q4_0.gguf"

STORY = " ".join(f"Once upon a time there was a little cat number {i} who liked to play." for i in range(30))
OTHER = " ".join(f"The big dog {i} ran across the green field to find a bone." for i in range(30))

server = ServerPreset.stories15m_moe()


def long_ctx_server() -> ServerProcess:
    s = ServerPreset.stories15m_moe()
    s.model_hf_repo = None
    s.model_hf_file = None
    s.model_file = download_file(MODEL_LONG_CTX_FILE_URL)
    s.model_alias = "smollm2"
    return s


@pytest.fixture(autouse=True)
def create_server():
    global server
    server = long_ctx_server()


def disk_args(root, size_mib: int = 256) -> list:
    # no RAM tier and no idle save: a prompt goes to disk when another one takes its slot
    return ["--cache-ram", "0", "--cache-disk", str(root), "--cache-disk-size", str(size_mib),
            "--cache-disk-reserve", "64", "--cache-disk-idle", "600000"]


def disk_stats() -> dict:
    res = server.make_request("GET", "/stats")
    assert res.status_code == 200
    return res.body["disk_cache"]


def complete(prompt: str, n_predict: int) -> dict:
    res = server.make_request("POST", "/completion", data={
        "prompt": prompt,
        "n_predict": n_predict,
        "temperature": 0.0,
        "cache_prompt": True,
    })
    assert res.status_code == 200
    return res.body


def entry_files(root) -> list:
    return glob.glob(os.path.join(str(root), "llama-pdc", "run-*", "*.tgt"))


@pytest.mark.parametrize("encrypt", [True, False])
def test_prompt_disk_restore(tmp_path, encrypt):
    # one slot, no RAM tier: a second long prompt pushes the first out of the slot onto disk, the third brings it back
    server.extra_args = ["--cache-ram", "0", "--cache-disk", str(tmp_path), "--cache-disk-size", "256",
                         "--cache-disk-reserve", "64", "--cache-disk-idle", "600000"]
    if not encrypt:
        server.extra_args.append("--no-cache-disk-encrypt")
    server.start()

    first = complete(STORY + " The end.", 4)
    assert first["timings"]["cache_n"] == 0
    assert entry_files(tmp_path) == []  # nothing left the slot yet

    complete(OTHER + " The end.", 4)
    assert len(entry_files(tmp_path)) == 1  # the first prompt, evicted from the slot

    third = complete(STORY + " And then", 8)
    # the stored state came back: the shared prefix is reused instead of decoded again
    assert third["timings"]["cache_n"] > 256
    assert third["timings"]["prompt_n"] < 16
    n_reused = third["timings"]["cache_n"]

    # a plaintext entry is readable, an encrypted one is not: the tokens of the prompt are not in the file as stored
    files = entry_files(tmp_path)
    assert len(files) >= 1
    with open(files[0], "rb") as f:
        head = f.read(9)
    assert head[:4] == b"PDCE" and head[8] == (1 if encrypt else 0)

    # the same slot extended (a prefix hit, no restore needed)
    fourth = complete(STORY + " And then the cat", 4)
    assert fourth["timings"]["cache_n"] > 256

    # the restored state computes what a full prefill computes
    server.stop()
    server.extra_args = ["--cache-ram", "0"]
    server.start()
    fresh = complete(STORY + " And then", 8)
    assert fresh["timings"]["cache_n"] == 0
    assert fresh["timings"]["prompt_n"] > n_reused
    assert fresh["content"] == third["content"]

    # the run directory dies with its server
    server.stop()
    assert glob.glob(os.path.join(str(tmp_path), "llama-pdc", "run-*")) == []


def test_prompt_disk_encrypted_entry_holds_no_stored_state(tmp_path):
    # the same prompt stored three times: twice without encryption (the state as stored reproduces byte for byte, so
    # pieces of it are a pattern to look for) and once encrypted (none of those pieces is in the file)
    def stored_entry(root, encrypt):
        global server
        server = long_ctx_server()
        server.extra_args = disk_args(root)
        if not encrypt:
            server.extra_args.append("--no-cache-disk-encrypt")
        server.start()
        complete(STORY + " The end.", 4)
        complete(OTHER + " The end.", 4)  # pushes the first prompt out of the slot onto disk
        files = entry_files(root)
        assert len(files) == 1
        with open(files[0], "rb") as f:
            data = f.read()
        server.stop()
        return data

    plain = stored_entry(tmp_path / "a", False)
    again = stored_entry(tmp_path / "b", False)
    crypt = stored_entry(tmp_path / "c", True)
    assert plain[:4] == again[:4] == crypt[:4] == b"PDCE"
    assert plain[8] == 0 and again[8] == 0 and crypt[8] == 1
    assert len(plain) > 1024 * 1024

    # 32-byte pieces of the stored state, spread over the file (pieces of padding or zeros would prove nothing)
    step = len(plain) // 64
    pieces = [plain[o:o + 32] for o in range(4096, len(plain) - 32, step)]
    pieces = [w for w in pieces if len(set(w)) >= 12]
    assert len(pieces) >= 32
    assert all(w in again for w in pieces), "the plaintext state does not reproduce: the pieces are not a usable pattern"
    found = [w for w in pieces if w in crypt]
    assert found == [], f"{len(found)} of {len(pieces)} pieces of the plaintext state are in the encrypted file"


@pytest.mark.parametrize("encrypt", [True, False])
@pytest.mark.parametrize("damage", ["truncated", "flipped"])
def test_prompt_disk_damaged_entry_fails_the_restore_cleanly(tmp_path, encrypt, damage):
    # a stored state that was cut short or changed on disk is rejected (authentication tag / record checksum): the
    # entry is deleted, the slot starts empty and the request is answered by a full prefill, with the same reply
    server.server_metrics = True
    server.extra_args = disk_args(tmp_path)
    if not encrypt:
        server.extra_args.append("--no-cache-disk-encrypt")
    server.start()
    complete(STORY + " The end.", 4)
    complete(OTHER + " The end.", 4)  # pushes the first prompt out of the slot onto disk
    files = entry_files(tmp_path)
    assert len(files) == 1
    size = os.path.getsize(files[0])
    assert size > 1024 * 1024
    with open(files[0], "r+b") as f:
        if damage == "truncated":
            f.truncate(size // 2)
        else:
            f.seek(size // 2)
            byte = f.read(1)
            f.seek(size // 2)
            f.write(bytes([byte[0] ^ 0x01]))

    third = complete(STORY + " And then", 8)
    assert third["timings"]["cache_n"] == 0
    assert third["timings"]["prompt_n"] > 256
    d = disk_stats()
    assert d["find_hit"] == 1 and d["restore"] == 0 and d["restore_failed"] == 1
    assert not os.path.exists(files[0])  # the damaged entry is gone

    server.stop()
    server.extra_args = ["--cache-ram", "0"]
    server.start()
    fresh = complete(STORY + " And then", 8)
    assert fresh["timings"]["cache_n"] == 0
    assert fresh["content"] == third["content"]


def test_prompt_disk_size_cap_evicts_the_oldest_entry(tmp_path):
    # four long prompts through one slot under a cap that holds two stored states: the oldest entry goes first
    def story(animal):
        return " ".join(f"Once upon a time there was a little {animal} number {i} who liked to play." for i in range(30))

    server.server_metrics = True
    server.extra_args = disk_args(tmp_path, size_mib=32)
    server.start()
    complete(story("cat") + " The end.", 4)
    complete(story("dog") + " The end.", 4)   # cat goes to disk
    one = os.path.getsize(entry_files(tmp_path)[0])
    limit = 32 * 1024 * 1024
    assert 2 * one <= limit < 3 * one, f"an entry of {one} bytes: the cap no longer holds exactly two"
    complete(story("bird") + " The end.", 4)  # dog goes to disk
    d = disk_stats()
    assert d["entries"] == 2 and d["evict_space"] == 0
    complete(story("fish") + " The end.", 4)  # bird goes to disk: over the cap, the oldest entry (cat) is removed
    d = disk_stats()
    assert d["entries"] == 2 and d["evict_space"] == 1
    assert d["bytes"] <= d["limit_bytes"] == limit
    assert len(entry_files(tmp_path)) == 2

    # bird is still stored (saving fish's slot for it removes dog, the oldest by then); cat is not
    bird = complete(story("bird") + " And then", 4)
    assert bird["timings"]["cache_n"] > 256
    cat = complete(story("cat") + " And then", 4)
    assert cat["timings"]["cache_n"] < 64
    d = disk_stats()
    assert d["restore"] == 1 and d["restore_failed"] == 0
    assert d["bytes"] <= d["limit_bytes"]


def test_prompt_disk_restore_with_a_draft_context(tmp_path):
    # a server with a draft model stores both contexts of a slot (.tgt and .dft) and restores both: the reply and the
    # draft acceptance are those of a server whose slot never lost the prompt
    def run(other_prompt_first):
        global server
        # the preset of test_speculative.py; stories15M's 256-token context is below what the tier restores (>= 256
        # reused tokens), so the target runs with a longer one
        server = ServerPreset.stories15m_moe()
        server.override_kv = "llama.context_length=int:2048"
        server.server_metrics = True
        server.model_draft = download_file(MODEL_DRAFT_FILE_URL)
        server.spec_type = "draft-simple"
        server.spec_draft_n_min = 1
        server.spec_draft_n_max = 4
        server.fa = "off"
        server.extra_args = disk_args(tmp_path)
        server.start()
        complete(STORY + " The end.", 4)
        dft_bytes = 0
        if other_prompt_first:
            complete(OTHER + " The end.", 4)  # pushes the first prompt out of the slot onto disk
            dft = glob.glob(os.path.join(str(tmp_path), "llama-pdc", "run-*", "*.dft"))
            assert len(dft) == 1
            dft_bytes = os.path.getsize(dft[0])
        res = complete(STORY + " And then", 24)
        stats = server.make_request("GET", "/stats").body
        server.stop()
        return res, stats, dft_bytes

    direct, direct_stats, _ = run(False)
    restored, restored_stats, dft_bytes = run(True)

    assert dft_bytes > 256 * 1024, "the draft context's state was not stored"
    assert restored_stats["disk_cache"]["restore"] == 1 and restored_stats["disk_cache"]["restore_failed"] == 0
    assert restored["timings"]["cache_n"] > 256
    assert restored["timings"]["cache_n"] == direct["timings"]["cache_n"]
    assert restored["timings"]["prompt_n"] == direct["timings"]["prompt_n"]
    assert restored["content"] == direct["content"]
    # the draft works from the restored state as from the kept one
    a, b = direct_stats["tasks"][-1], restored_stats["tasks"][-1]
    assert a["n_draft"] > 0
    assert (b["n_draft"], b["n_draft_accepted"]) == (a["n_draft"], a["n_draft_accepted"])


def test_prompt_disk_idle_save(tmp_path):
    # an idle slot is written without another request (the queue's idle hook), and not written twice
    server.extra_args = ["--cache-ram", "0", "--cache-disk", str(tmp_path), "--cache-disk-reserve", "64",
                         "--cache-disk-idle", "100"]
    server.start()
    complete("Once upon a time there was a little cat", 4)
    deadline = time.time() + 10.0
    while not entry_files(tmp_path) and time.time() < deadline:
        time.sleep(0.1)
    files = entry_files(tmp_path)
    assert len(files) == 1
    mtime = os.path.getmtime(files[0])
    time.sleep(2.5)
    assert entry_files(tmp_path) == files and os.path.getmtime(files[0]) == mtime


def test_prompt_disk_needs_a_usable_directory(tmp_path):
    # an explicit --cache-disk that cannot start fails the startup instead of serving without it
    blocker = tmp_path / "file"
    blocker.write_text("x")
    server.extra_args = ["--cache-disk", str(blocker)]
    with pytest.raises(RuntimeError):
        server.start(timeout_seconds=15)


def test_prompt_disk_survives_a_model_reload(tmp_path):
    # the tier belongs to the server, not to the loaded model: entries written before a sleep restore after the wake
    server.sleep_idle_seconds = 1
    server.extra_args = ["--cache-ram", "0", "--cache-disk", str(tmp_path), "--cache-disk-size", "256",
                         "--cache-disk-reserve", "64", "--cache-disk-idle", "600000"]
    server.start()
    complete(STORY + " The end.", 4)
    complete(OTHER + " The end.", 4)  # pushes the first prompt out of the slot onto disk
    assert len(entry_files(tmp_path)) == 1

    deadline = time.time() + 10
    while not server.make_request("GET", "/props").body.get("is_sleeping") and time.time() < deadline:
        time.sleep(0.1)
    assert server.make_request("GET", "/props").body.get("is_sleeping")
    assert len(entry_files(tmp_path)) == 1  # the model is gone, the entry is not

    third = complete(STORY + " And then", 8)  # wakes the server: a fresh context, an empty slot
    assert third["timings"]["cache_n"] > 256
    assert third["timings"]["prompt_n"] < 16


LORA_FILE_URL = "https://huggingface.co/ggml-org-dev/stories15M_MOE/resolve/main/moe_shakespeare15M.gguf"


def test_prompt_disk_survives_a_model_reload_with_a_lora_adapter(tmp_path):
    # a reload frees the adapters and loads them again: an entry keeps its lora config by value (file and scale), so
    # the entry written before the sleep is found under the same config after the wake, and not under another scale
    global server
    server = ServerPreset.stories15m_moe()
    server.lora_files = [download_file(LORA_FILE_URL)]
    # the only test adapter is for stories15M, whose 256-token context is below what the tier restores (>= 256 reused
    # tokens): run it with a longer one
    server.override_kv = "llama.context_length=int:2048"
    server.sleep_idle_seconds = 1
    server.extra_args = ["--cache-ram", "0", "--cache-disk", str(tmp_path), "--cache-disk-size", "256",
                         "--cache-disk-reserve", "64", "--cache-disk-idle", "600000"]
    server.start()

    def ask(prompt, n_predict, scale=None):
        data = {"prompt": prompt, "n_predict": n_predict, "temperature": 0.0, "cache_prompt": True}
        if scale is not None:
            data["lora"] = [{"id": 0, "scale": scale}]
        res = server.make_request("POST", "/completion", data=data)
        assert res.status_code == 200
        return res.body

    def wait_sleeping():
        deadline = time.time() + 10
        while not server.make_request("GET", "/props").body.get("is_sleeping") and time.time() < deadline:
            time.sleep(0.1)
        assert server.make_request("GET", "/props").body.get("is_sleeping")

    ask(STORY + " The end.", 4)
    ask(OTHER + " The end.", 4)  # pushes the first prompt out of the slot onto disk
    assert len(entry_files(tmp_path)) == 1
    wait_sleeping()

    # the adapter at another scale: the stored state was not computed under this config
    other_scale = ask(STORY + " And then", 8, scale=0.5)
    assert other_scale["timings"]["cache_n"] == 0
    wait_sleeping()

    # the config of the entry (the adapter at its startup scale): restored
    same = ask(STORY + " And then", 8)
    assert same["timings"]["cache_n"] > 256
    assert same["timings"]["prompt_n"] < 16

    # and it computes what a server that never stored anything computes under that config
    server.stop()
    server.sleep_idle_seconds = None
    server.extra_args = ["--cache-ram", "0"]
    server.start()
    fresh = ask(STORY + " And then", 8)
    assert fresh["timings"]["cache_n"] == 0
    assert fresh["content"] == same["content"]


SWA_PROMPT = "The quick brown fox jumps over the lazy dog. " * 30


def test_prompt_disk_restore_with_context_checkpoints(tmp_path):
    # a sliding window smaller than the prompt: the memory cannot be rolled back to any position, so the slot keeps
    # context checkpoints and a stored entry is written with them (.ckpt) and read back with them
    def tokenize(text):
        res = server.make_request("POST", "/tokenize", data={"content": text})
        assert res.status_code == 200
        return res.body["tokens"]

    def ask(prompt, n_predict=8):
        res = server.make_request("POST", "/completion", data={
            "prompt": prompt, "n_predict": n_predict, "temperature": 0.0, "cache_prompt": True, "return_tokens": True})
        assert res.status_code == 200
        return res.body

    def run(other_prompt_first):
        global server
        server = ServerPreset.tinygemma3()
        server.override_kv = "gemma3.attention.sliding_window=int:64"
        server.n_slots = 1
        server.n_ubatch = 32
        server.temperature = 0.0
        server.extra_args = ["--cache-ram", "0", "--cache-disk", str(tmp_path), "--cache-disk-size", "256",
                             "--cache-disk-reserve", "64", "--cache-disk-idle", "600000"]
        server.start()
        first = ask(tokenize(SWA_PROMPT))
        # the first turn and its exact continuation, so the whole stored entry is a prefix of the next prompt
        nxt = tokenize(SWA_PROMPT) + first["tokens"] + tokenize(" The end.")
        n_ckpt_files = 0
        if other_prompt_first:
            ask("Once upon a time " * 60)  # pushes the first turn out of the slot onto disk
            n_ckpt_files = len(glob.glob(os.path.join(str(tmp_path), "llama-pdc", "run-*", "*.ckpt")))
        res = ask(nxt)
        server.stop()
        return res, n_ckpt_files, len(nxt)

    direct, _, n_next = run(False)
    restored, n_ckpt_files, _ = run(True)

    assert n_ckpt_files >= 1, "the evicted prompt was stored without its context checkpoints"
    assert n_next > 256 + 16
    # the entry came back with its checkpoints: the same prefix is reused as when the slot never lost it, and the same
    # tokens come out
    assert restored["timings"]["cache_n"] > 256
    assert restored["timings"]["cache_n"] == direct["timings"]["cache_n"]
    assert restored["timings"]["prompt_n"] == direct["timings"]["prompt_n"]
    assert restored["tokens"] == direct["tokens"]
