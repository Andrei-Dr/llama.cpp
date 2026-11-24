import glob, os, time

import pytest
from utils import *

# the prompt cache disk tier (--cache-disk, tools/server/server-prompt-disk.h), seen from the responses and the directory

# the disk tier only restores a prompt that saves >= 256 tokens: a model with more context than stories15M's 256
MODEL_LONG_CTX_FILE_URL = "https://huggingface.co/bartowski/SmolLM2-135M-Instruct-GGUF/resolve/main/SmolLM2-135M-Instruct-Q4_K_M.gguf"

STORY = " ".join(f"Once upon a time there was a little cat number {i} who liked to play." for i in range(30))
OTHER = " ".join(f"The big dog {i} ran across the green field to find a bone." for i in range(30))

server = ServerPreset.stories15m_moe()


@pytest.fixture(autouse=True)
def create_server():
    global server
    server = ServerPreset.stories15m_moe()
    server.model_hf_repo = None
    server.model_hf_file = None
    server.model_file = download_file(MODEL_LONG_CTX_FILE_URL)
    server.model_alias = "smollm2"


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
