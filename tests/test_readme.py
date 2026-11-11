"""Keep the README's API section in step with the app.

If an endpoint is added, removed or renamed, or a response gains or loses a field,
these tests fail until README.md is updated too.
"""
import json
import re
from pathlib import Path

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

import app.main as main_module
from app.ml.inference import PCBDefectModel
from tests.test_inference import fake_output

README = (Path(__file__).resolve().parents[1] / "README.md").read_text(encoding="utf-8")
API_SECTION = README.split("\n## API\n", 1)[1].split("\n## ", 1)[0]

# Pages FastAPI adds by itself; not ours to document one by one
BUILT_IN = {"/openapi.json", "/docs", "/docs/oauth2-redirect", "/redoc"}


def documented_endpoints():
    """{("GET", "/health"), ...} from the **`METHOD /path`** headings in the API section."""
    return {(method, path) for method, path in
            re.findall(r"\*\*`(GET|POST|PUT|PATCH|DELETE) (/[^`?\s]*)[^`]*`\*\*", API_SECTION)}


def app_endpoints():
    return {(method, route.path)
            for route in main_module.app.routes if route.path not in BUILT_IN
            for method in route.methods if method != "HEAD"}


def example_json(method_and_path: str) -> dict:
    """The first ```json block after the **`METHOD /path...`** heading."""
    # The path must end right there ("`" or "?query"), so "GET /" doesn't match "GET /health"
    heading = re.search(re.escape(f"**`{method_and_path}") + r"[`?]", API_SECTION)
    block = re.search(r"```json\n(.*?)\n```", API_SECTION[heading.start():], re.S).group(1)
    return json.loads(block)


def test_every_endpoint_is_documented_and_every_documented_one_exists():
    assert documented_endpoints() == app_endpoints()


class OneDefectSession:
    """The real model code, with an ONNX session that always 'finds' one Short."""
    def run(self, _names, _inputs):
        return fake_output([(300, 300, 20, 20, 1, 0.9)])


@pytest.fixture
def client():
    with TestClient(main_module.app) as test_client:   # startup loads the real model...
        model = PCBDefectModel()
        model.session = OneDefectSession()             # ...and this one returns a known detection
        main_module.ml_engine["model"] = model
        yield test_client


def test_inspect_example_has_the_same_fields_as_a_real_response(client):
    image = cv2.imencode(".png", np.zeros((480, 640, 3), np.uint8))[1].tobytes()
    real = client.post("/inspect", data={"board_id": "README-check"},
                       files={"file": ("board.png", image, "image/png")}).json()
    example = example_json("POST /inspect")

    assert real["status"] == "success"
    assert set(example) == set(real)
    assert set(example["telemetry"]) == set(real["telemetry"])
    assert set(example["detections"][0]) == set(real["detections"][0])


@pytest.mark.parametrize("path", ["/health", "/analytics/summary", "/"])
def test_get_examples_have_the_same_fields_as_real_responses(client, path):
    assert set(example_json(f"GET {path}")) == set(client.get(path).json())
