import json, time

import pytest
from utils import *

# GET /stats: the snapshot the main loop publishes (tools/server/server-stats.h)

MODEL_LONG_CTX_FILE_URL = "https://huggingface.co/bartowski/SmolLM2-135M-Instruct-GGUF/resolve/main/SmolLM2-135M-Instruct-Q4_K_M.gguf"
MODEL_DRAFT_FILE_URL = "https://huggingface.co/ggml-org/tiny-llamas/resolve/main/stories15M-q4_0.gguf"

server = ServerPreset.stories15m_moe()


@pytest.fixture(autouse=True)
def create_server():
    global server
    server = ServerPreset.stories15m_moe()
    server.server_metrics = True


def get_stats() -> dict:
    res = server.make_request("GET", "/stats")
    assert res.status_code == 200
    return res.body


def wait_stats(cond, timeout_s: float = 3.0) -> dict:
    """the main loop publishes right after a task completes; the response can reach the client first"""
    deadline = time.time() + timeout_s
    while True:
        s = get_stats()
        if cond(s) or time.time() > deadline:
            return s
        time.sleep(0.02)


def complete(prompt: str, n_predict: int) -> dict:
    res = server.make_request("POST", "/completion", data={
        "prompt": prompt,
        "n_predict": n_predict,
        "temperature": 0.0,
    })
    assert res.status_code == 200
    return res.body


def test_stats_needs_metrics_flag():
    server.server_metrics = False
    server.start()
    res = server.make_request("GET", "/stats")
    assert res.status_code == 501


def test_stats_before_any_task():
    server.start()
    s = get_stats()
    assert s["schema"] == 1
    assert s["seq"] >= 0


def test_stats_records_completed_tasks():
    server.start()
    bodies = [complete("Once upon a time", 8 + i) for i in range(3)]

    s = wait_stats(lambda s: s.get("counters", {}).get("tasks", 0) >= 3)
    c = s["counters"]
    assert c["tasks"] == 3
    assert s["t_serve_us"] >= s["t_us"] > s["t_start_us"] > 0
    assert c["gen_tokens"] == sum(b["tokens_predicted"] for b in bodies)
    assert c["prompt_tokens"] + c["prompt_tokens_cached"] == sum(b["tokens_evaluated"] for b in bodies)
    assert c["stop"] == {"limit": 3}

    rows = s["tasks"]
    assert len(rows) == 3
    assert [r["n_gen"] for r in rows] == [b["tokens_predicted"] for b in bodies]
    for r, b in zip(rows, bodies):
        assert r["n_prompt"] == b["tokens_evaluated"]
        assert r["n_prompt_cached"] == b["timings"]["cache_n"]
        assert r["n_prompt_processed"] == b["timings"]["prompt_n"]
        assert r["n_ctx"] == b["tokens_cached"]  # the response's tokens_cached = tokens in the slot
        assert r["stop"] == "limit"
        assert r["ttft_ms"] >= r["t_prompt_ms"] > 0
        assert r["gen_ms_per_token"] > 0
        assert r["sel"]["how"] in ("lcp", "lru")
        # same as the timings the response reports (both read the slot's stats)
        assert abs(r["t_prompt_ms"] - b["timings"]["prompt_ms"]) < 1e-3
        assert abs(r["t_gen_ms"] - b["timings"]["predicted_ms"]) < 1e-3

    # the same prompt again: the slot keeps it, so later tasks reuse it
    assert rows[1]["n_prompt_cached"] > 0

    assert sum(s["selection"].get(k, 0) for k in ("id", "lcp", "lru")) == 3
    assert s["hist"]["ttft_ms"]["n"] == 3
    assert s["hist"]["gen_ms_per_token"]["n"] == 3
    assert s["hist"]["accept"]["n"] == 0  # no drafts
    assert s["decode"]["gen"]["n"] > 0
    assert s["decode"]["step_gen"]["n"] > 0
    assert s["hist"]["step_ms"]["n"] == s["decode"]["step_gen"]["n"]
    assert s["publish"]["n"] >= 1

    assert len(s["slots"]) == s["n_slots"] == 1
    assert s["n_deferred"] == 0
    assert s["slots"][0]["state"] == "idle"
    assert s["disk_cache"] is None


def test_stats_ring_is_bounded():
    server.start()
    for i in range(130):
        complete(f"Story {i}:", 1)
    s = wait_stats(lambda s: s.get("counters", {}).get("tasks", 0) >= 130)
    assert s["counters"]["tasks"] == 130
    assert len(s["tasks"]) == 128
    ids = [r["id"] for r in s["tasks"]]
    assert ids == sorted(ids)

    res = server.make_request("GET", "/stats?tasks=2")
    assert res.status_code == 200
    assert [r["id"] for r in res.body["tasks"]] == ids[-2:]
    assert server.make_request("GET", "/stats?tasks=0").body["tasks"] == []
    assert server.make_request("GET", "/stats?tasks=x").status_code == 400


def test_stats_publish_period_env():
    # 0 turns publishing off: the route still answers, with no snapshot
    os.environ["LLAMA_SERVER_STATS_MS"] = "0"
    try:
        server.start()
        complete("Once upon a time", 4)
        time.sleep(0.3)
        assert get_stats()["seq"] == 0
    finally:
        del os.environ["LLAMA_SERVER_STATS_MS"]


def test_stats_speculative_acceptance():
    server.model_draft = download_file(MODEL_DRAFT_FILE_URL)
    server.spec_type = "draft-simple"
    server.spec_draft_n_min = 1
    server.spec_draft_n_max = 4
    server.fa = "off"
    server.start()
    complete("I believe the meaning of life is", 32)

    s = wait_stats(lambda s: s.get("counters", {}).get("tasks", 0) >= 1)
    c = s["counters"]
    r = s["tasks"][0]
    assert c["draft_tokens"] > 0
    assert r["n_draft"] == c["draft_tokens"]
    assert r["n_draft_accepted"] == c["draft_accepted"]
    assert 0.0 <= r["accept_rate"] <= 1.0
    assert r["accepted_per_pos"] == c["accepted_per_pos"]
    # acceptance by position is cumulative: reaching position j+1 needs position j
    acc = r["accepted_per_pos"]
    assert all(acc[j] >= acc[j + 1] for j in range(len(acc) - 1))
    assert s["hist"]["accept"]["n"] == 1


def test_stats_moe_cache_absent():
    server.start()
    complete("Once upon a time", 4)
    assert wait_stats(lambda s: s.get("seq", 0) > 0)["moe_cache"] is None


def test_stats_moe_cache():
    # experts in host memory, a small device cache for them (llama_moe_cache_get_stats, read at every publish)
    server.n_gpu_layer = 99
    server.extra_args = ["-ot", "exps=CPU", "--moe-cache-mib", "8"]
    try:
        server.start()
    except RuntimeError:
        # the context refuses a MoE cache without a GPU backend (a CPU-only build)
        pytest.skip("no MoE cache: the cache needs a GPU backend to hold its slots")
    complete("Once upon a time", 48)
    complete("Once upon a time there was", 48)

    s = wait_stats(lambda s: s.get("counters", {}).get("tasks", 0) >= 2)
    m = s["moe_cache"]
    if m is None:
        pytest.skip("no MoE cache: the cache needs a GPU device to hold its slots")
    c = m["counters"]
    assert m["n_layer"] > 0 and m["n_slots"] > 0 and m["n_expert"] >= m["n_layer"]
    assert 0 < m["n_resident"] <= m["n_slots"]
    assert not m["suspended"] and m["device_bytes"] > 0
    assert c["hit"] + c["miss"] > 0
    assert c["hit"] > 0, "a cache over 96 decoded tokens serves some experts"
    assert c["miss_cpu"] <= c["miss"] and c["upload"] == c["miss"] - c["miss_cpu"]
    assert (c["upload_bytes"] > 0) == (c["upload"] > 0)
    for k in ("hit", "miss", "miss_cpu", "upload", "upload_bytes"):
        assert c[k] == m["ubatch_small"][k] + m["ubatch_large"][k], k

    # the history rows carry the same counters from the sampler's next row on
    want = "moe_hit,moe_miss,moe_miss_cpu,moe_upload,moe_upload_bytes,moe_resident"
    deadline = time.time() + 5.0
    while True:
        h = server.make_request("GET", "/stats/history?cols=" + want).body
        if (h["n"] and h["cols"]["moe_hit"][-1] == c["hit"]) or time.time() > deadline:
            break
        time.sleep(0.1)
    assert h["n"], "no history row within 5 s"
    assert h["cols"]["moe_hit"][-1] == c["hit"] and h["cols"]["moe_miss"][-1] == c["miss"]
    assert h["cols"]["moe_resident"][-1] == m["n_resident"]


def test_stats_disk_cache(tmp_path):
    # the disk tier only restores a prompt that saves >= 256 tokens: a model with more context than stories15M's 256
    server.model_hf_repo = None
    server.model_hf_file = None
    server.model_file = download_file(MODEL_LONG_CTX_FILE_URL)
    server.model_alias = "smollm2"
    # one slot: a second long prompt pushes the first out of the slot onto disk, the third brings it back
    server.extra_args = ["--cache-ram", "0", "--cache-disk", str(tmp_path), "--cache-disk-size", "256",
                         "--cache-disk-reserve", "64", "--cache-disk-idle", "600000"]
    server.start()
    story = " ".join(f"Once upon a time there was a little cat number {i} who liked to play." for i in range(30))
    other = " ".join(f"The big dog {i} ran across the green field to find a bone." for i in range(30))
    complete(story + " The end.", 4)
    complete(other + " The end.", 4)
    complete(story + " And then", 4)

    s = wait_stats(lambda s: s.get("counters", {}).get("tasks", 0) >= 3)
    d = s["disk_cache"]
    assert d is not None
    assert d["save"] >= 1 and d["save_bytes"] > 0 and d["save_tokens"] > 256
    assert d["entries"] >= 1 and d["bytes"] > 0
    assert d["limit_bytes"] == 256 * 1024 * 1024
    assert d["find"] >= d["find_hit"] >= 1
    assert d["restore"] == 1 and d["restore_failed"] == 0
    assert d["save_canceled"] == 0 and "save_cancelled" not in d
    assert 0.0 < d["f_sim_hit"] <= 1.0

    third = s["tasks"][2]["sel"]
    assert third["cache_update"] and third["disk_restored"]
    assert third["n_disk_tokens"] > 256
    assert s["tasks"][2]["n_prompt_cached"] > 256
    assert s["selection"]["disk_restored"] == 1

    # the restore replaces the slot's prompt, so the reused prefix of a restored task comes from the disk and is counted in
    # prompt_tokens_cached; prompt_tokens_cached_disk is that part (the restore may load more than the prompt reuses)
    c = s["counters"]
    assert c["prompt_tokens_cached_disk"] == s["tasks"][2]["n_prompt_cached"]
    assert c["prompt_tokens_cached"] >= c["prompt_tokens_cached_disk"]
    assert d["restore_tokens"] >= c["prompt_tokens_cached_disk"]

    # the history rows carry the same counters from the sampler's next row on
    want = "prompt_cached_disk,disk_restore_tokens,disk_restore_us,disk_save_us"
    deadline = time.time() + 5.0
    while True:
        h = server.make_request("GET", "/stats/history?cols=" + want).body
        if (h["n"] and h["cols"]["prompt_cached_disk"][-1] == c["prompt_tokens_cached_disk"]) or time.time() > deadline:
            break
        time.sleep(0.1)
    assert h["n"], "no history row within 5 s"
    cols = h["cols"]
    assert cols["prompt_cached_disk"][-1] == c["prompt_tokens_cached_disk"]
    assert cols["disk_restore_tokens"][-1] == d["restore_tokens"]
    assert cols["disk_restore_us"][-1] == d["restore_us"] > 0
    assert cols["disk_save_us"][-1] == d["save_us"] > 0

    # the same slot extended (a prefix hit, no restore): reused, but not from the disk
    complete(story + " And then the cat", 4)
    s4 = wait_stats(lambda s: s.get("counters", {}).get("tasks", 0) >= 4)
    assert not s4["tasks"][3]["sel"]["disk_restored"]
    assert s4["tasks"][3]["n_prompt_cached"] > 256
    assert s4["counters"]["prompt_tokens_cached"] == c["prompt_tokens_cached"] + s4["tasks"][3]["n_prompt_cached"]
    assert s4["counters"]["prompt_tokens_cached_disk"] == c["prompt_tokens_cached_disk"]


def test_stats_disk_idle_save_is_published(tmp_path):
    # an idle save runs outside the request path (the queue's idle hook): /stats must show it without another request
    server.extra_args = ["--cache-ram", "0", "--cache-disk", str(tmp_path), "--cache-disk-reserve", "64",
                         "--cache-disk-idle", "100"]
    server.start()
    complete("Once upon a time there was a little cat", 4)
    s = wait_stats(lambda s: (s.get("disk_cache") or {}).get("save", 0) >= 1, timeout_s=10.0)
    assert s["disk_cache"]["save"] == 1
    assert s["disk_cache"]["entries"] == 1


def _run_slots_sequence() -> tuple[list, str]:
    server.server_slots = True
    server.start()
    complete("Once upon a time", 12)
    complete("Once upon a time there was", 9)
    time.sleep(0.3)
    slots = server.make_request("GET", "/slots")
    metrics = server.make_request("GET", "/metrics")
    assert slots.status_code == 200 and metrics.status_code == 200
    server.stop()
    return slots.body, metrics.body


def _metric_values(text: str) -> dict:
    out = {}
    for line in text.splitlines():
        if line and not line.startswith("#"):
            k, v = line.rsplit(" ", 1)
            out[k] = float(v)
    return out


def test_slots_and_metrics_from_snapshot_match_the_task_path():
    # same deterministic sequence twice: published snapshot (default) vs the upstream task path (snapshots off)
    snap_slots, snap_metrics = _run_slots_sequence()
    os.environ["LLAMA_SERVER_STATS_MS"] = "0"
    try:
        task_slots, task_metrics = _run_slots_sequence()
    finally:
        del os.environ["LLAMA_SERVER_STATS_MS"]
    assert snap_slots == task_slots
    a, b = _metric_values(snap_metrics), _metric_values(task_metrics)
    assert a.keys() == b.keys()
    assert [l for l in snap_metrics.splitlines() if l.startswith("#")] == [l for l in task_metrics.splitlines() if l.startswith("#")]
    for k in ("llamacpp:prompt_tokens_total", "llamacpp:tokens_predicted_total", "llamacpp:prompt_tokens_cached_total",
              "llamacpp:n_decode_total", "llamacpp:requests_processing", "llamacpp:requests_deferred"):
        assert a[k] == b[k], k


def test_metrics_rate_window_resets_per_scrape():
    server.start()
    complete("Once upon a time", 16)
    time.sleep(0.3)
    first = _metric_values(server.make_request("GET", "/metrics").body)
    second = _metric_values(server.make_request("GET", "/metrics").body)
    assert first["llamacpp:predicted_tokens_seconds"] > 0
    assert second["llamacpp:predicted_tokens_seconds"] == 0, "nothing generated since the previous scrape"
    assert second["llamacpp:tokens_predicted_total"] == first["llamacpp:tokens_predicted_total"]
    complete("Once upon a time", 16)
    time.sleep(0.3)
    third = _metric_values(server.make_request("GET", "/metrics").body)
    assert third["llamacpp:predicted_tokens_seconds"] > 0


def test_slots_shows_a_running_task():
    server.server_slots = True
    server.n_predict = 200
    server.start()
    import threading
    t = threading.Thread(target=lambda: complete("Once upon a time", 200))
    t.start()
    seen = False
    deadline = time.time() + 10
    while time.time() < deadline and t.is_alive():
        rows = server.make_request("GET", "/slots").body
        if rows[0]["is_processing"]:
            seen = True
            break
        time.sleep(0.005)
    t.join()
    assert seen
    time.sleep(0.3)
    assert server.make_request("GET", "/slots").body[0]["is_processing"] is False
    assert server.make_request("GET", "/slots?fail_on_no_slot=1").status_code == 200


def test_slots_free_right_after_a_response_with_another_slot_busy():
    # two slots: one generates for a while, the other finishes; a /slots read right after the short response must show
    # the short one's slot free (the release publishes at once, not at the next period)
    server.server_slots = True
    server.n_slots = 2
    server.n_predict = 400
    server.start()
    import threading
    long_req = threading.Thread(target=lambda: complete("Once upon a time", 200))
    long_req.start()
    time.sleep(0.2)
    for _ in range(5):
        body = complete("The little dog", 4)
        rows = server.make_request("GET", "/slots").body  # no sleep: right after the response
        busy = [r for r in rows if r["is_processing"]]
        assert len(busy) <= 1, f"the finished request's slot still shows busy: {busy}"
    long_req.join()


def test_stats_phase_is_not_measured():
    # the CPU-split time of each layer has no source on this tree: the key stays, null
    server.start()
    complete("Once upon a time", 4)
    assert wait_stats(lambda s: s.get("seq", 0) > 0)["phase"] is None
