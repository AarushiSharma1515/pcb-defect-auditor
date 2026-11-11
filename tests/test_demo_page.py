"""Tests for the browser demo page served at GET /demo (app/static/demo.html)."""
import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import app.main as main_module


@pytest.fixture(scope="module")
def page():
    # No startup needed: /demo just returns a file
    response = TestClient(main_module.app).get("/demo")
    assert response.status_code == 200
    return response.text


def test_demo_returns_200_html_with_the_title():
    response = TestClient(main_module.app).get("/demo")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")
    assert "<title>PCB Defect Auditor Demo</title>" in response.text


def test_demo_is_not_listed_as_an_api_endpoint():
    assert "/demo" not in TestClient(main_module.app).get("/openapi.json").json()["paths"]


def test_page_loads_nothing_from_outside(page):
    # No build step, no frameworks, no CDN: everything is inline
    assert not re.search(r"<script[^>]*\bsrc=", page, re.I)
    assert not re.search(r"<link\b", page, re.I)
    assert "@import" not in page
    assert not re.search(r"<img[^>]*\bsrc=[\"']https?:", page, re.I)
    assert not re.search(r"url\(\s*[\"']?https?:", page, re.I)
    # The only absolute URLs are plain links a visitor can click
    for url in re.findall(r"https?://[^\s\"'<>)]+", page):
        assert url.startswith(("https://github.com/AarushiSharma1515/pcb-defect-auditor",
                               "https://github.com/tangsanli5201/DeepPCB")), url


def test_intro_links_and_footer(page):
    assert 'href="/docs"' in page
    assert 'href="https://github.com/AarushiSharma1515/pcb-defect-auditor"' in page
    assert ("Trained on DeepPCB (Kaggle repackaging, 1,200/150/150 split). "
            "Not validated on real factory images.") in page


def test_page_works_at_phone_width(page):
    assert '<meta name="viewport" content="width=device-width, initial-scale=1">' in page


# ---------------------------------------------------------------- running the page's JS in Node

DEMO_FILE = Path(main_module.DEMO_PAGE)
NODE = shutil.which("node")
needs_node = pytest.mark.skipif(NODE is None, reason="Node.js is not installed")

# Loads the page's <script> into a fresh JS context WITHOUT a page (no `document`), so only the
# helper functions are defined, then runs the test code and prints its result as JSON.
HARNESS = r"""
const vm = require("vm");
const fs = require("fs");
const [script, testCode] = [fs.readFileSync(process.argv[2], "utf8"), fs.readFileSync(process.argv[3], "utf8")];
const context = vm.createContext({ console, setTimeout, clearTimeout, AbortController });
vm.runInContext(script, context, { filename: "demo.html <script>" });
Promise.resolve(vm.runInContext("(async () => {" + testCode + "})()", context))
  .then((result) => console.log(JSON.stringify(result === undefined ? null : result)))
  .catch((error) => { console.error(error); process.exit(1); });
"""


def page_script() -> str:
    scripts = re.findall(r"<script>(.*?)</script>", DEMO_FILE.read_text(encoding="utf-8"), re.S)
    assert len(scripts) == 1, "expected exactly one inline <script>"
    return scripts[0]


def run_js(tmp_path, test_code: str):
    """Run `test_code` (the body of an async function) after the page's script; return its result."""
    (tmp_path / "harness.js").write_text(HARNESS, encoding="utf-8")
    (tmp_path / "page.js").write_text(page_script(), encoding="utf-8")
    (tmp_path / "test.js").write_text(test_code, encoding="utf-8")
    result = subprocess.run([NODE, str(tmp_path / "harness.js"), str(tmp_path / "page.js"),
                             str(tmp_path / "test.js")], capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout.strip().splitlines()[-1])


@needs_node
def test_page_script_has_no_syntax_errors(tmp_path):
    assert run_js(tmp_path, "return typeof startPage;") == "function"


# ---------------------------------------------------------------- step: waking the server

def test_waking_message_is_shown_while_waiting(page):
    message = ("Waking up the server (free hosting sleeps when idle). "
               "This can take up to 60 seconds.")
    assert page.count(message) == 2   # in the initial HTML, and in the script for "Try again"


@needs_node
def test_health_timeout_is_60_seconds(tmp_path):
    assert run_js(tmp_path, "return HEALTH_TIMEOUT_MS;") == 60000


@needs_node
@pytest.mark.parametrize("health, expected_detail", [
    ({"model_loaded": True, "database": "ok", "version": "1.1.0"}, "Server version 1.1.0."),
    ({"model_loaded": False, "database": "ok", "version": "1.1.0"}, "model is not loaded"),
    ({"model_loaded": True, "database": "unreachable", "version": "1.1.0"}, "database is unreachable"),
])
def test_ready_message_from_health(tmp_path, health, expected_detail):
    text = run_js(tmp_path, f"return describeHealth({json.dumps(health)});")
    assert text["headline"] == "Ready"
    assert expected_detail in text["detail"]


@needs_node
def test_health_is_retried_while_the_server_wakes(tmp_path):
    # Connection refused, then 502 from the host, then the real answer
    result = run_js(tmp_path, """
        let calls = 0;
        const fakeFetch = async () => {
          calls += 1;
          if (calls === 1) throw new TypeError("Failed to fetch");
          if (calls === 2) return { ok: false, status: 502 };
          return { ok: true, json: async () => ({ model_loaded: true, database: "ok" }) };
        };
        const health = await waitForHealth(fakeFetch, 5000, 10);
        return { health, calls };
    """)
    assert result == {"health": {"model_loaded": True, "database": "ok"}, "calls": 3}


@needs_node
def test_health_gives_up_when_the_time_runs_out(tmp_path):
    # A server that never answers: each request hangs until it is aborted
    result = run_js(tmp_path, """
        const hangingFetch = (url, options) => new Promise((_, reject) => {
          options.signal.addEventListener("abort", () => reject(new Error("aborted")));
        });
        const started = Date.now();
        const health = await waitForHealth(hangingFetch, 300, 10);
        return { health, ms: Date.now() - started };
    """)
    assert result["health"] is None
    assert result["ms"] < 2000            # gave up close to the 300 ms budget, didn't hang
