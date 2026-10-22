import os
import shutil
import subprocess
import sys
import tempfile
import time

import pytest
from utils import *

# GET /stats/history and /stats "machine": the sampler thread's rows (tools/server/server-sampler.h)

MODEL_FILE_URL = "https://huggingface.co/ggml-org/stories15M_MOE/resolve/main/stories15M_MOE-F16.gguf"

server = ServerPreset.stories15m_moe()


@pytest.fixture(scope="module")
def fake_nvml() -> str:
    """libnvidia-ml stand-in with fixed readings (fake_nvml.c), built for this run"""
    cc = shutil.which("cc")
    if cc is None:
        pytest.skip("no C compiler for the fake NVML library")
    out = os.path.join(tempfile.mkdtemp(prefix="nvml-"), "libfake_nvml" + (".dylib" if sys.platform == "darwin" else ".so"))
    src = os.path.join(os.path.dirname(__file__), "fake_nvml.c")
    subprocess.run([cc, "-shared", "-fPIC", "-o", out, src], check=True)
    return out


@pytest.fixture(autouse=True)
def create_server(monkeypatch):
    global server
    server = ServerPreset.stories15m_moe()
    server.model_hf_repo = None
    server.model_hf_file = None
    server.model_file = download_file(MODEL_FILE_URL)
    server.server_metrics = True
    monkeypatch.setenv("LLAMA_SERVER_SAMPLE_MS", "100")
    monkeypatch.setenv("LLAMA_SERVER_NVML_LIB", "/nonexistent/libnvidia-ml.so.1")


def history(**params) -> dict:
    q = "&".join(f"{k}={v}" for k, v in params.items())
    res = server.make_request("GET", "/stats/history" + ("?" + q if q else ""))
    assert res.status_code == 200, res.body
    return res.body


def wait_history(cond, timeout_s: float = 5.0, **params) -> dict:
    deadline = time.time() + timeout_s
    while True:
        h = history(**params)
        if cond(h) or time.time() > deadline:
            return h
        time.sleep(0.05)


def test_history_needs_metrics():
    server.server_metrics = False
    server.start()
    res = server.make_request("GET", "/stats/history")
    assert res.status_code == 501


def test_history_rows_and_fake_gpu(fake_nvml, monkeypatch):
    monkeypatch.setenv("LLAMA_SERVER_NVML_LIB", fake_nvml)
    server.start()
    h = wait_history(lambda h: h["n"] >= 3)
    assert h["enabled"] is True
    assert h["tier"] == "s"
    assert h["period_ms"] == 100
    c = h["cols"]
    ts = c["t_us"]
    assert len(ts) == h["n"] >= 3
    assert all(b > a for a, b in zip(ts, ts[1:]))
    assert all(v == 42 for v in c["gpu_util"])
    assert all(v == 1 << 30 for v in c["vram_used"])
    assert all(v == 4 << 30 for v in c["vram_total"])
    assert all(v == 1000 * 1024 for v in c["pcie_rx"])
    assert all(v == 10 * 1024 for v in c["pcie_tx"])
    assert all(v == 61 for v in c["gpu_temp_c"])
    assert all(v == 70.5 for v in c["gpu_power_w"])
    assert all(v == 1905 for v in c["gpu_clock_mhz"])
    cpu = c["proc_cpu_us"]
    assert all(v is not None for v in cpu) and all(b >= a for a, b in zip(cpu, cpu[1:]))
    assert all(v is not None and v >= 0 for v in c["sample_us"])
    if sys.platform.startswith("linux"):
        assert all(v and v > 0 for v in c["mem_total"])
        assert all(v is not None for v in c["host_total_us"])

    s = server.make_request("GET", "/stats").body
    m = s["machine"]
    assert m["nvml"] == "ok"
    assert m["gpus"][0]["name"] == "Fake GPU"
    assert m["gpus"][0]["util"] == 42
    assert m["sampler"]["n"] >= 3


def test_history_follows_the_server():
    server.start()
    for i in range(2):
        res = server.make_request("POST", "/completion", data={"prompt": "Once upon a time", "n_predict": 8 + i})
        assert res.status_code == 200

    def done(h):
        return h["n"] > 0 and h["cols"]["tasks"][-1] == 2

    h = wait_history(done)
    assert done(h)
    s = server.make_request("GET", "/stats").body
    c = h["cols"]
    assert c["gen_tokens"][-1] == s["counters"]["gen_tokens"]
    assert c["prompt_tokens"][-1] == s["counters"]["prompt_tokens"]
    assert c["t_start_us"][-1] == s["t_start_us"]
    assert c["snap_t_us"][-1] <= c["t_us"][-1]
    assert c["kv_max"][-1] > 0
    # no GPU library: the GPU columns are null, the rest is there
    assert all(v is None for v in c["gpu_util"])
    assert s["machine"]["nvml"].startswith("cannot load")
    assert s["machine"]["gpus"] == []


def test_history_since_and_cols():
    server.start()
    h = wait_history(lambda h: h["n"] >= 4)
    ts = h["cols"]["t_us"]
    mid = ts[len(ts) // 2]
    later = history(since_us=mid)
    assert later["cols"]["t_us"] and all(t > mid for t in later["cols"]["t_us"])

    only = history(cols="gen_tokens,gpu_util")
    assert set(only["cols"]) == {"t_us", "gen_tokens", "gpu_util"}

    for bad in ("cols=nope", "tier=x", "since_us=abc"):
        res = server.make_request("GET", "/stats/history?" + bad)
        assert res.status_code == 400, bad


def test_history_minute_tier(fake_nvml, monkeypatch):
    monkeypatch.setenv("LLAMA_SERVER_SAMPLE_MS", "20") # a minute row every 60 samples = 1.2 s
    monkeypatch.setenv("LLAMA_SERVER_NVML_LIB", fake_nvml)
    server.start()
    h = wait_history(lambda h: h["n"] >= 1, timeout_s=10.0, tier="m")
    assert h["tier"] == "m"
    assert h["period_ms"] == 1200
    assert h["cols"]["gpu_util"][0] == 42


def test_sampler_off(monkeypatch):
    monkeypatch.setenv("LLAMA_SERVER_SAMPLE_MS", "0")
    server.start()
    h = history()
    assert h["enabled"] is False
    assert h["n"] == 0
    s = server.make_request("GET", "/stats").body
    assert s["machine"] is None


def test_nvml_off(fake_nvml, monkeypatch):
    monkeypatch.setenv("LLAMA_SERVER_NVML_LIB", fake_nvml)
    monkeypatch.setenv("LLAMA_SERVER_NVML", "0")
    server.start()
    h = wait_history(lambda h: h["n"] >= 1)
    assert all(v is None for v in h["cols"]["gpu_util"])
    assert server.make_request("GET", "/stats").body["machine"]["nvml"] == "off"
