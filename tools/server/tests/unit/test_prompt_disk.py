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
