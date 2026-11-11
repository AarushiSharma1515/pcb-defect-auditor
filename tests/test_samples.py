"""Tests for GET /samples and GET /samples/{name}."""
import os

import pytest
from fastapi.testclient import TestClient

import app.main as main_module

SECRET = b"this must never be served"


@pytest.fixture
def client():
    # No startup needed: these endpoints don't use the model or the database
    return TestClient(main_module.app)


@pytest.fixture
def samples(tmp_path, monkeypatch):
    """An empty samples/ folder, with secret files NEXT TO it for traversal attempts to aim at."""
    folder = tmp_path / "samples"
    folder.mkdir()
    (tmp_path / "secret.jpg").write_bytes(SECRET)
    (tmp_path / ".env").write_bytes(SECRET)
    monkeypatch.setattr(main_module, "SAMPLES_DIR", folder)
    return folder


def test_missing_folder_gives_empty_list(client, tmp_path, monkeypatch):
    monkeypatch.setattr(main_module, "SAMPLES_DIR", tmp_path / "does_not_exist")
    response = client.get("/samples")
    assert response.status_code == 200
    assert response.json() == []


def test_empty_folder_gives_empty_list(client, samples):
    assert client.get("/samples").json() == []


def test_lists_only_jpg_and_png_images(client, samples):
    for name in ["b_short.png", "a_open.jpg", "C_SPUR.PNG", "d.jpeg", "README.md", "notes.txt", "e.gif"]:
        (samples / name).write_bytes(b"x")
    (samples / "folder.jpg").mkdir()          # a directory with an image-like name is not a file

    assert client.get("/samples").json() == ["C_SPUR.PNG", "a_open.jpg", "b_short.png", "d.jpeg"]


def test_serves_a_sample_with_the_right_content_type(client, samples):
    (samples / "open.jpg").write_bytes(b"\xff\xd8\xff fake jpeg")
    (samples / "short.png").write_bytes(b"\x89PNG fake png")

    jpg = client.get("/samples/open.jpg")
    assert jpg.status_code == 200
    assert jpg.content == b"\xff\xd8\xff fake jpeg"
    assert jpg.headers["content-type"] == "image/jpeg"

    assert client.get("/samples/short.png").headers["content-type"] == "image/png"


@pytest.mark.parametrize("name", ["missing.jpg", "README.md", "notes.txt"])
def test_unknown_or_non_image_names_are_404(client, samples, name):
    (samples / "README.md").write_text("about the samples")
    (samples / "notes.txt").write_text("notes")
    response = client.get(f"/samples/{name}")
    assert response.status_code == 404


@pytest.mark.parametrize("path", [
    "/samples/..%5Csecret.jpg",        # ..\secret.jpg  (a path on Windows)
    "/samples/..%2Fsecret.jpg",        # ../secret.jpg  with the slash encoded
    "/samples/%2E%2E%2F.env",          # ../.env        fully encoded
    "/samples/..%5C.env",              # ..\.env
    "/samples/%2E%2E",                 # ..
    "/samples/../secret.jpg",          # plain ../ (clients usually collapse this themselves)
])
def test_path_traversal_is_rejected(client, samples, path):
    (samples / "open.jpg").write_bytes(b"x")
    response = client.get(path)
    assert response.status_code in (400, 404)
    assert SECRET not in response.content


def test_backslash_name_gets_a_clear_400(client, samples):
    response = client.get("/samples/..%5Csecret.jpg")
    assert response.status_code == 400
    assert "not a path" in response.json()["detail"]


def test_link_pointing_outside_the_folder_is_not_served(client, samples, tmp_path):
    try:
        os.symlink(tmp_path / "secret.jpg", samples / "linked.jpg")
    except (OSError, NotImplementedError):
        pytest.skip("this system doesn't allow creating symlinks (Windows without Developer Mode)")

    assert "linked.jpg" not in client.get("/samples").json()
    assert client.get("/samples/linked.jpg").status_code == 404
