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
const context = vm.createContext({ console, setTimeout, clearTimeout, AbortController, FormData, Blob });
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


# ---------------------------------------------------------------- step: upload, samples, errors

JPEG_START = [0xFF, 0xD8, 0xFF, 0xE0, 0, 0, 0, 0]
PNG_START = [0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A]
PDF_START = list(b"%PDF-1.4")


@needs_node
def test_upload_limit_matches_the_server(tmp_path):
    assert run_js(tmp_path, "return MAX_UPLOAD_BYTES;") == main_module.MAX_UPLOAD_BYTES


@needs_node
@pytest.mark.parametrize("size, first_bytes, expected", [
    (50_000, JPEG_START, None),
    (50_000, PNG_START, None),
    (10 * 1024 * 1024, JPEG_START, None),                     # exactly 10 MB is allowed
    (10 * 1024 * 1024 + 1, JPEG_START, "the limit is 10 MB"),
    (12_900_000, PNG_START, "That file is 12.3 MB"),
    (0, [], "The file is empty"),
    (50_000, PDF_START, "isn't a JPEG or PNG"),                # checked by content, not by name
])
def test_file_is_checked_before_upload(tmp_path, size, first_bytes, expected):
    message = run_js(tmp_path, f"return checkFile({size}, {json.dumps(first_bytes)});")
    if expected is None:
        assert message is None
    else:
        assert expected in message


@needs_node
@pytest.mark.parametrize("status, body, expected", [
    (400, {"detail": "The file looks like a JPEG but could not be decoded."}, "could not be decoded"),
    (413, {"detail": "The image is 12000x12000 pixels; the limit is 25 megapixels."}, "12000x12000"),
    (422, {"detail": [{"loc": ["body", "file"], "msg": "Field required"}]}, "request was incomplete"),
    (500, None, "unexpected error"),
    (503, {"detail": "The database is unavailable right now."}, "temporarily unavailable"),
    (502, None, "temporarily unavailable"),
    ("network", None, "Couldn't reach the server"),
    ("timeout", None, "took too long"),
    (418, None, "HTTP 418"),
])
def test_every_error_gets_a_plain_english_message(tmp_path, status, body, expected):
    message = run_js(tmp_path, f"return errorMessage({json.dumps(status)}, {json.dumps(body)});")
    assert expected in message
    assert "loc" not in message and "[object" not in message   # never dumps raw error data


def inspect_with(tmp_path, fake_response_js: str):
    """Call inspectImage() with a fake fetch; return its outcome and what was sent."""
    return run_js(tmp_path, f"""
        let sent = null;
        const fakeFetch = async (url, options) => {{
          sent = {{ url, method: options.method, board_id: options.body.get("board_id"),
                    filename: options.body.get("file").name }};
          {fake_response_js}
        }};
        const outcome = await inspectImage(fakeFetch, new Blob(["x"]), "board.png", 500);
        return {{ outcome, sent }};
    """)


@needs_node
def test_inspect_sends_the_image_with_board_id_demo(tmp_path):
    r = inspect_with(tmp_path, 'return { ok: true, status: 200, json: async () => ({ status: "success", telemetry: {} }) };')
    assert r["sent"] == {"url": "/inspect", "method": "POST", "board_id": "demo", "filename": "board.png"}
    assert r["outcome"]["ok"] is True


@needs_node
@pytest.mark.parametrize("fake_response_js, expected", [
    ('return { ok: false, status: 413, json: async () => ({ detail: "The file is larger than the 10 MB limit." }) };',
     "larger than the 10 MB limit"),
    ('return { ok: false, status: 502, json: async () => { throw new SyntaxError("HTML page"); } };',
     "temporarily unavailable"),                                   # host's HTML error page
    ('throw new TypeError("Failed to fetch");', "Couldn't reach the server"),
    ('return new Promise((_, reject) => options.signal.addEventListener("abort", () => reject(new Error("aborted"))));',
     "took too long"),                                             # never answers -> 500 ms timeout
    ('return { ok: true, status: 200, json: async () => { throw new SyntaxError("not JSON"); } };',
     "couldn't read"),
])
def test_inspect_failures_become_messages_not_crashes(tmp_path, fake_response_js, expected):
    r = inspect_with(tmp_path, fake_response_js)
    assert r["outcome"]["ok"] is False
    assert expected in r["outcome"]["message"]


def test_no_samples_note_and_link(page):
    note = re.search(r'<p id="no-samples-note" hidden>(.*?)</p>', page, re.S).group(1)
    assert re.sub(r"<[^>]+>|\s+", " ", note).split() == (
        "No sample board handy? Download test images from the DeepPCB dataset (research use only)".split())
    assert '<a href="https://github.com/tangsanli5201/DeepPCB">' in note
    assert '<section class="card" id="samples-section" hidden>' in page   # hidden until samples arrive


def test_partial_success_notice(page):
    assert "Result is valid; saving it to the database failed." in page


def test_upload_box(page):
    assert "Upload your own PCB image" in page
    assert 'accept=".jpg,.jpeg,.png,image/jpeg,image/png"' in page


def test_server_error_details_used_by_the_page_are_plain_english(tmp_path):
    # The page shows the server's own "detail" for 400 and 413, so it must be a readable sentence
    client = TestClient(main_module.app)
    response = client.post("/inspect", data={"board_id": "demo"},
                           files={"file": ("doc.pdf", b"%PDF-1.4 not an image", "application/pdf")})
    assert response.status_code == 400
    assert isinstance(response.json()["detail"], str)
    assert response.json()["detail"].endswith(".")


# ---------------------------------------------------------------- step: canvas and table

@needs_node
def test_boxes_are_scaled_with_image_width_and_height(tmp_path):
    # A 1280x320 image drawn on a 640x160 canvas: everything halves
    box = run_js(tmp_path, "return scaleBox([200, 100, 400, 150], 1280, 320, 640, 160);")
    assert box == {"x": 100, "y": 50, "w": 100, "h": 25}


@needs_node
def test_big_images_are_drawn_at_most_2048_px(tmp_path):
    assert run_js(tmp_path, "return canvasSize(5000, 4000);") == {"width": 2048, "height": 1638}
    assert run_js(tmp_path, "return canvasSize(640, 640);") == {"width": 640, "height": 640}


@needs_node
def test_every_defect_type_has_its_own_colour(tmp_path):
    from app.ml.inference import DEFECT_CLASSES
    colours = run_js(tmp_path, f"return {json.dumps(DEFECT_CLASSES)}.map(colourFor);")
    assert len(set(colours)) == len(DEFECT_CLASSES) == 6
    assert "#a9a9a9" not in colours                         # none falls back to "unknown" grey
    assert run_js(tmp_path, 'return colourFor("something new");') == "#a9a9a9"


@needs_node
def test_labels_are_readable_on_their_colour(tmp_path):
    # WCAG AA asks for a contrast of at least 4.5 : 1 for normal text
    ratios = run_js(tmp_path, """
        return Object.values(DEFECT_COLOURS).concat([UNKNOWN_COLOUR])
          .map((c) => contrastRatio(c, textColourOn(c)));
    """)
    assert min(ratios) >= 4.5, ratios


@needs_node
def test_labels_stay_inside_the_canvas(tmp_path):
    placed = run_js(tmp_path, """
        return {
          normal:   placeLabel({x: 100, y: 100, w: 50, h: 50}, 80, 20, 640, 640),
          topEdge:  placeLabel({x: 100, y: 5,   w: 50, h: 50}, 80, 20, 640, 640),
          rightEdge: placeLabel({x: 600, y: 100, w: 40, h: 40}, 80, 20, 640, 640),
        };
    """)
    assert placed["normal"] == {"x": 100, "y": 80}       # just above the box
    assert placed["topEdge"] == {"x": 100, "y": 5}       # no room above: inside the box's top
    assert placed["rightEdge"] == {"x": 560, "y": 80}    # pulled back so it isn't cut off


@needs_node
def test_label_and_table_formats(tmp_path):
    out = run_js(tmp_path, """
        const det = { defect_type: "Short", confidence: 0.8707, box: [136, 484, 192, 520] };
        return [labelText(det), formatPercent(det.confidence, 1), formatBox(det.box)];
    """)
    assert out == ["Short 87%", "87.1%", "[136, 484, 192, 520]"]


def test_result_area_has_canvas_table_and_no_defects_text(page):
    assert '<canvas id="result-canvas"' in page
    assert "<strong>No defects detected</strong>" in page
    for heading in ("Defect", "Confidence", "Box [x1, y1, x2, y2] (px)"):
        assert f"<th>{heading}</th>" in page
    for fact in ("Processing time:", "Needs human review:"):
        assert fact in page


def test_results_are_never_inserted_as_html(page):
    # Server text (defect names, error details) must go in as text, so it can't inject markup
    assert "innerHTML" not in page.split("<script>")[1].replace("never innerHTML", "")


@needs_node
def test_labels_of_clustered_defects_do_not_cover_each_other(tmp_path):
    # The real cluster from DeepPCB 01_PCB__10: Short, Mousebite and Spur within ~50 px
    result = run_js(tmp_path, """
        const boxes = [[136, 484, 192, 520], [187, 445, 220, 472], [137, 454, 176, 485]];
        const items = boxes.map(([x1, y1, x2, y2]) => ({ box: {x: x1, y: y1, w: x2 - x1, h: y2 - y1}, w: 90, h: 20 }));
        const spots = layoutLabels(items, 640, 640);
        const rects = spots.map((s) => ({ x: s.x, y: s.y, w: 90, h: 20 }));
        let overlapping = 0;
        for (let i = 0; i < rects.length; i++)
          for (let j = i + 1; j < rects.length; j++)
            if (overlaps(rects[i], rects[j])) overlapping++;
        const inside = rects.every((r) => r.x >= 0 && r.y >= 0 && r.x + r.w <= 640 && r.y + r.h <= 640);
        return { overlapping, inside, first: spots[0] };
    """)
    assert result["overlapping"] == 0
    assert result["inside"]
    assert result["first"] == {"x": 136, "y": 464}    # the best detection keeps the spot right above its box
