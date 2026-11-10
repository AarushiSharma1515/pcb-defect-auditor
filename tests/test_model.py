"""Tests for loading the ONNX model (app/ml/inference.py)."""
import pytest
from fastapi.testclient import TestClient

import app.main as main_module
from app.ml.inference import DEFAULT_MODEL_PATH, PCBDefectModel


def test_missing_model_fails_loudly(tmp_path):
    missing = tmp_path / "no_such_model.onnx"
    with pytest.raises(FileNotFoundError) as error:
        PCBDefectModel(model_path=missing)
    assert str(missing) in str(error.value)          # the message says exactly where it looked


def test_app_refuses_to_start_without_the_model(monkeypatch, tmp_path):
    missing = tmp_path / "no_such_model.onnx"
    monkeypatch.setattr(main_module, "PCBDefectModel", lambda: PCBDefectModel(model_path=missing))
    with pytest.raises(FileNotFoundError):
        with TestClient(main_module.app):            # entering runs the startup code
            pass


def test_model_is_found_from_any_working_directory(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)                      # start "the app" from an unrelated folder
    model = PCBDefectModel()
    assert DEFAULT_MODEL_PATH.is_absolute()
    assert model.model_path == DEFAULT_MODEL_PATH
