"""Tests for the browser demo page served at GET /demo (app/static/demo.html)."""
import re

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
