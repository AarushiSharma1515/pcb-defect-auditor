"""Tests for GET /health."""
import threading
import time

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine

import app.main as main_module
from app import __version__


@pytest.fixture
def client():
    with TestClient(main_module.app) as test_client:   # runs startup, so the real model loads
        yield test_client


def test_health_when_everything_is_up(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {
        "model_loaded": True,
        "model_file": "pcb_defect_v1.onnx",
        "mode": "onnx",
        "database": "ok",            # the temporary SQLite database from conftest.py
        "version": __version__,
    }


def test_health_is_200_when_database_refuses_connections(client, monkeypatch, tmp_path):
    # A SQLite file inside a folder that doesn't exist: every connection attempt fails
    down = create_engine("sqlite:///" + (tmp_path / "missing_folder" / "x.db").as_posix())
    monkeypatch.setattr(main_module, "engine", down)

    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["database"] == "unreachable"
    assert response.json()["model_loaded"] is True     # the rest of the report still works


class HangingEngine:
    """A database that never answers (like an unreachable host that drops packets)."""
    def __init__(self):
        self.release = threading.Event()

    def connect(self):
        self.release.wait(timeout=10)
        raise RuntimeError("gave up")


def test_health_does_not_wait_for_a_hanging_database(client, monkeypatch):
    hanging = HangingEngine()
    monkeypatch.setattr(main_module, "engine", hanging)
    monkeypatch.setattr(main_module, "HEALTH_DB_TIMEOUT_SECONDS", 0.3)
    try:
        start = time.perf_counter()
        response = client.get("/health")
        seconds = time.perf_counter() - start
    finally:
        hanging.release.set()        # let the stuck background check finish

    assert response.status_code == 200
    assert response.json()["database"] == "unreachable"
    assert seconds < 2.0


def test_health_reports_when_the_model_is_not_loaded(monkeypatch):
    monkeypatch.setattr(main_module, "ml_engine", {})
    # No "with": startup doesn't run, so no model is loaded
    response = TestClient(main_module.app).get("/health")
    assert response.status_code == 200
    assert response.json()["model_loaded"] is False
    assert response.json()["model_file"] == "pcb_defect_v1.onnx"   # the file it expects
    assert response.json()["mode"] == "onnx"
