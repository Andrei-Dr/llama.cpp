import http.client
import os
import tempfile

import pytest
from utils import *

# GET /dash/*: the dashboard's static files from --dash-path (tools/server/server-dash.h), served by the HTTP thread

MODEL_FILE_URL = "https://huggingface.co/ggml-org/stories15M_MOE/resolve/main/stories15M_MOE-F16.gguf"

SECRET = b"SECRET-OUTSIDE-THE-ROOT"

server = ServerPreset.stories15m_moe()


def make_root() -> str:
    """a dash root with the usual layout, a secret beside it, a symlink out and a file of an unknown type"""
    base = tempfile.mkdtemp(prefix="dash-")
    root = os.path.join(base, "dash")
    os.makedirs(os.path.join(root, "css"))
    os.makedirs(os.path.join(root, "js"))
    with open(os.path.join(root, "index.html"), "wb") as f:
        f.write(b"<!doctype html><title>dash</title>")
    with open(os.path.join(root, "css", "base.css"), "wb") as f:
        f.write(b"body{}")
    with open(os.path.join(root, "js", "main.js"), "wb") as f:
        f.write(b"export const x = 1;")
    with open(os.path.join(root, "js", "tool.py"), "wb") as f:
        f.write(SECRET)
    with open(os.path.join(root, ".hidden.js"), "wb") as f:
        f.write(SECRET)
    with open(os.path.join(base, "secret.js"), "wb") as f:
        f.write(SECRET)
    os.symlink(os.path.join(base, "secret.js"), os.path.join(root, "js", "out.js"))
    return root


@pytest.fixture(autouse=True)
def create_server():
    global server
    server = ServerPreset.stories15m_moe()
    server.model_hf_repo = None
    server.model_hf_file = None
    server.model_file = download_file(MODEL_FILE_URL)


def raw(method: str, path: str, headers: dict | None = None):
    """a request with the path sent as is (no client-side normalization)"""
    c = http.client.HTTPConnection(server.server_host, server.server_port, timeout=10)
    try:
        c.request(method, path, headers=headers or {})
        r = c.getresponse()
        return r.status, {k.lower(): v for k, v in r.getheaders()}, r.read()
    finally:
        c.close()


def start_with_dash(root: str | None = None) -> str:
    root = root or make_root()
    server.extra_args = ["--dash-path", root]
    server.start()
    return root


def test_dash_off_without_the_option():
    server.start()
    status, _, body = raw("GET", "/dash/")
    assert status == 404
    status, _, _ = raw("GET", "/dash/js/main.js")
    assert status == 404


def test_dash_serves_index_and_types():
    start_with_dash()
    cases = [
        ("/dash/",             "text/html",       b"<!doctype html><title>dash</title>"),
        ("/dash/index.html",   "text/html",       b"<!doctype html><title>dash</title>"),
        ("/dash/css/base.css", "text/css",        b"body{}"),
        ("/dash/js/main.js",   "text/javascript", b"export const x = 1;"),
    ]
    for path, ctype, want in cases:
        status, h, body = raw("GET", path)
        assert status == 200, path
        assert h["content-type"].startswith(ctype), path
        assert body == want, path
        assert h["cache-control"] == "no-cache", path
        assert h["x-content-type-options"] == "nosniff", path
        csp = h["content-security-policy"]
        assert "script-src 'self'" in csp and "style-src 'self'" in csp and "unsafe-inline" not in csp, path


def test_dash_without_slash_redirects():
    start_with_dash()
    status, h, _ = raw("GET", "/dash")
    assert status in (301, 302, 307, 308)
    assert h["location"] == "/dash/"


def test_dash_etag_revalidation_and_change():
    root = start_with_dash()
    status, h, _ = raw("GET", "/dash/js/main.js")
    assert status == 200
    etag = h["etag"]
    assert etag.startswith('"') and etag.endswith('"')

    status, h2, body = raw("GET", "/dash/js/main.js", {"If-None-Match": etag})
    assert status == 304
    assert body == b""
    assert h2["etag"] == etag

    status, _, _ = raw("GET", "/dash/js/main.js", {"If-None-Match": '"nope", ' + etag})
    assert status == 304

    # a deploy is live at once: new bytes, new tag
    with open(os.path.join(root, "js", "main.js"), "wb") as f:
        f.write(b"export const x = 22;")
    status, h3, body = raw("GET", "/dash/js/main.js", {"If-None-Match": etag})
    assert status == 200
    assert body == b"export const x = 22;"
    assert h3["etag"] != etag


def test_dash_head():
    start_with_dash()
    status, h, body = raw("GET", "/dash/js/main.js")
    status_h, hh, body_h = raw("HEAD", "/dash/js/main.js")
    assert status_h == 200
    assert body_h == b""
    assert hh["etag"] == h["etag"]
    assert hh["content-type"] == h["content-type"]


def test_dash_refuses_traversal():
    start_with_dash()
    bad = [
        "/dash/../secret.js", "/dash/%2e%2e/secret.js", "/dash/..%2fsecret.js", "/dash/js/../../secret.js",
        "/dash/js/%2e%2e/%2e%2e/secret.js", "/dash//etc/passwd", "/dash/%2fetc%2fpasswd", "/dash/js/main.js%00.css",
        "/dash/..\\secret.js", "/dash/%5c..%5csecret.js", "/dash/%252e%252e/secret.js", "/dash/%252e%252e%252fsecret.js",
        "/dash/js/", "/dash/js", "/dash/js/nope.js", "/dash/js/tool.py", "/dash/js/out.js", "/dash/.hidden.js",
        "/dash/./index.html", "/dash/js/./main.js", "/dash/css//base.css",
    ]
    for path in bad:
        status, _, body = raw("GET", path)
        assert status in (400, 404), path
        assert SECRET not in body, path


def test_dash_is_public_with_an_api_key():
    server.api_key = "sekrit"
    start_with_dash()
    status, _, body = raw("GET", "/dash/")
    assert status == 200
    assert body.startswith(b"<!doctype html>")
    status, _, _ = raw("GET", "/dash/js/main.js")
    assert status == 200
    # the data the page reads keeps the key requirement, and so does any path that only starts like the mount
    for path in ("/slots", "/stats", "/stats/history", "/dashboard", "/dash-x", "/dashx/index.html"):
        status, _, _ = raw("GET", path)
        assert status == 401, path


def test_dash_path_must_be_a_directory():
    server.extra_args = ["--dash-path", os.path.join(tempfile.mkdtemp(), "missing")]
    with pytest.raises(RuntimeError):
        server.start(timeout_seconds=30)


def test_dash_under_an_api_prefix():
    # the harness waits on /health, which the prefix moves: launch by hand
    import subprocess, time
    bin_path = os.environ.get("LLAMA_SERVER_BIN_PATH", "../../../build/bin/llama-server")
    proc = subprocess.Popen([bin_path, "-m", server.model_file, "--port", str(server.server_port), "--api-prefix", "/llm",
                             "--dash-path", make_root(), "-c", "512"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        deadline = time.time() + 60
        while True:
            try:
                if raw("GET", "/llm/health")[0] == 200:
                    break
            except OSError:
                pass
            assert proc.poll() is None and time.time() < deadline, "server did not start"
            time.sleep(0.1)
        _check_prefixed()
    finally:
        proc.terminate()
        proc.wait(timeout=10)


def _check_prefixed():
    status, _, body = raw("GET", "/llm/dash/")
    assert status == 200 and body.startswith(b"<!doctype html>")
    status, h, _ = raw("GET", "/llm/dash")
    assert status in (301, 302, 307, 308) and h["location"] == "/llm/dash/"
    status, _, _ = raw("GET", "/dash/")
    assert status == 404


def test_dash_refuses_a_shadowing_static_path():
    pub = tempfile.mkdtemp(prefix="pub-")
    os.makedirs(os.path.join(pub, "dash"))
    server.extra_args = ["--dash-path", make_root(), "--path", pub]
    with pytest.raises(RuntimeError):
        server.start(timeout_seconds=30)
