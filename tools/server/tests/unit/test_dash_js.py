import glob
import os
import shutil
import subprocess

import pytest

# the dashboard under node: every module parses (one syntax error blanks the whole page), and the arithmetic
# (tools/server/dash/js/derive.js) passes tools/server/tests/dash/*.test.mjs


def test_dash_js():
    node = shutil.which("node")
    if node is None:
        pytest.skip("node is not installed")
    here = os.path.dirname(__file__)
    files = sorted(glob.glob(os.path.join(here, "..", "dash", "*.test.mjs")))
    assert files
    r = subprocess.run([node, "--test", *files], capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, r.stdout + r.stderr


def test_dash_js_modules_parse():
    node = shutil.which("node")
    if node is None:
        pytest.skip("node is not installed")
    root = os.path.join(os.path.dirname(__file__), "..", "..", "dash")
    files = sorted(glob.glob(os.path.join(root, "js", "**", "*.js"), recursive=True))
    assert len(files) >= 10
    for f in files:
        r = subprocess.run([node, "--check", f], capture_output=True, text=True, timeout=60)
        assert r.returncode == 0, f + ": " + r.stderr
