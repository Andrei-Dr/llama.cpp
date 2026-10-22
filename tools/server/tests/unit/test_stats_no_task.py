import http.client
import os
import tempfile
import time

import pytest
from utils import *

# the statistics reads (/stats, /stats/history, /slots, /metrics, /dash/) answer from published snapshots on the HTTP
# thread: none of them posts a task to the main loop. /stats "queue" counts the tasks posted, by type

MODEL_FILE_URL = "https://huggingface.co/ggml-org/stories15M_MOE/resolve/main/stories15M_MOE-F16.gguf"

server = ServerPreset.stories15m_moe()


@pytest.fixture(autouse=True)
def create_server():
    global server
    server = ServerPreset.stories15m_moe()
    server.model_hf_repo = None
    server.model_hf_file = None
    server.model_file = download_file(MODEL_FILE_URL)
    server.server_metrics = True
    server.server_slots = True
    root = tempfile.mkdtemp(prefix="dash-")
    with open(os.path.join(root, "index.html"), "wb") as f:
        f.write(b"<!doctype html>")
    server.extra_args = ["--dash-path", root]


def posted() -> dict:
    res = server.make_request("GET", "/stats")
    assert res.status_code == 200
    return res.body["queue"]["posted"]


def get_raw(path: str) -> int:
    c = http.client.HTTPConnection(server.server_host, server.server_port, timeout=10)
    try:
        c.request("GET", path)
        r = c.getresponse()
        r.read()
        return r.status
    finally:
        c.close()


def test_stats_reads_post_no_task():
    server.start()
    res = server.make_request("POST", "/completion", data={"prompt": "Once upon a time", "n_predict": 8})
    assert res.status_code == 200

    # the snapshot after the release is published before the next read sees it
    deadline = time.time() + 3
    while server.make_request("GET", "/stats").body["counters"]["tasks"] < 1 and time.time() < deadline:
        time.sleep(0.02)

    p0 = posted()
    assert p0["completion"] == 1
    for _ in range(25):
        for path in ("/stats", "/stats?tasks=4", "/stats/history", "/stats/history?tier=m", "/slots", "/metrics", "/props", "/dash/"):
            assert get_raw(path) == 200, path
    p1 = posted()
    assert p1 == p0

    # the counter itself works: an admission-time /slots read still asks the main loop
    assert get_raw("/slots?fail_on_no_slot=1") == 200
    p2 = posted()
    assert p2["slot_get"] == p0["slot_get"] + 1
