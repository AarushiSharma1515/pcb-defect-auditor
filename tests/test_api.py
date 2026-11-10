import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.pool import StaticPool
from sqlalchemy.orm import sessionmaker
from sqlalchemy.exc import IntegrityError, OperationalError
from unittest.mock import MagicMock

import cv2
import numpy as np

from app.main import app
from app.db.database import Base, get_db
from app.db.models import Inspection

# 1. Setup an isolated, temporary SQLite database for testing
SQLALCHEMY_DATABASE_URL = "sqlite:///:memory:"
engine = create_engine(SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False}, poolclass=StaticPool)
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base.metadata.create_all(bind=engine)

def override_get_db():
    try:
        db = TestingSessionLocal()
        yield db
    finally:
        db.close()

# Force FastAPI to use the temporary database instead of Neon
app.dependency_overrides[get_db] = override_get_db


def make_image(extension: str = ".jpg") -> bytes:
    """A tiny real image (64x64 gray square), encoded as ".jpg" or ".png"."""
    return cv2.imencode(extension, np.full((64, 64, 3), 128, np.uint8))[1].tobytes()


@pytest.fixture
def client():
    with TestClient(app) as test_client:
        yield test_client


def test_root_endpoint(client):
    response = client.get("/")
    assert response.status_code == 200


def test_inspect_invalid_file_type(client):
    files = {"file": ("document.pdf", b"fake pdf bytes", "application/pdf")}
    data = {"board_id": "test_board_001"}
    response = client.post("/inspect", data=data, files=files)
    assert response.status_code == 400


def test_inspect_missing_board_id(client):
    files = {"file": ("test.jpg", b"fake image bytes", "image/jpeg")}
    response = client.post("/inspect", files=files)
    assert response.status_code == 422


def test_inspect_corrupted_image(client):
    # Send random bytes instead of a real image
    response = client.post(
        "/inspect/",
        data={"board_id": "BRD-999"},
        files={"file": ("corrupt.jpg", b"not a real image", "image/jpeg")}
    )
    assert response.status_code == 400


def test_inspect_valid_image(client):
    # 2. Create a fake model to bypass the heavy ONNX processing
    fake_model = MagicMock()
    fake_model.predict.return_value = {
        "defect_type": "short_circuit",
        "confidence": 0.985,
        "processing_ms": 115
    }
    
    # Inject the fake model into your app's memory
    from app.main import ml_engine
    ml_engine["model"] = fake_model
    
    files = {"file": ("valid.jpg", make_image(".jpg"), "image/jpeg")}
    data = {"board_id": "test_board_003"}
    
    response = client.post("/inspect", data=data, files=files)
    
    # 3. Verify the 200 response and correct telemetry parsing
    assert response.status_code == 200
    assert response.json()["status"] == "success"
    assert response.json()["board_id"] == "test_board_003"
    assert response.json()["telemetry"]["defect_type"] == "short_circuit"


# ---------------------------------------------------------------- DB retry tests

# What Neon's dropped idle connection looks like to SQLAlchemy
CONNECTION_DROPPED = OperationalError(
    "INSERT INTO inspections ...", {}, Exception("SSL connection has been closed unexpectedly")
)


def use_fake_model():
    """Replace the ONNX model with a fake one, so these tests only exercise the database logic."""
    from app.main import ml_engine
    fake_model = MagicMock()
    fake_model.predict.return_value = {"defect_type": "Short", "confidence": 0.985, "processing_ms": 115}
    ml_engine["model"] = fake_model


def post_with_failing_commits(client, board_id, errors):
    """POST /inspect while db.commit() raises each error in `errors` once, then works normally.

    Returns the response and how many times commit() was called.
    """
    session = TestingSessionLocal()
    real_commit = session.commit
    calls = []

    def flaky_commit():
        calls.append(1)
        if len(calls) <= len(errors):
            raise errors[len(calls) - 1]
        real_commit()

    session.commit = flaky_commit
    app.dependency_overrides[get_db] = lambda: session
    try:
        response = client.post(
            "/inspect",
            data={"board_id": board_id},
            files={"file": ("board.jpg", make_image(".jpg"), "image/jpeg")},
        )
    finally:
        app.dependency_overrides[get_db] = override_get_db
        session.close()
    return response, len(calls)


def count_rows(board_id):
    with TestingSessionLocal() as db:
        return db.query(Inspection).filter_by(board_id=board_id).count()


def test_insert_is_retried_once_after_dropped_connection(client):
    use_fake_model()
    response, commits = post_with_failing_commits(client, "retry_ok", [CONNECTION_DROPPED])

    assert response.status_code == 200
    assert response.json()["status"] == "success"
    assert commits == 2                    # failed once, succeeded on the retry
    assert count_rows("retry_ok") == 1     # saved exactly once, no duplicate row


def test_partial_success_when_retry_also_fails(client):
    use_fake_model()
    response, commits = post_with_failing_commits(
        client, "retry_fail", [CONNECTION_DROPPED, CONNECTION_DROPPED]
    )

    assert response.status_code == 200
    assert response.json()["status"] == "partial_success"
    assert response.json()["telemetry"]["defect_type"] == "Short"  # prediction still returned
    assert commits == 2                    # one try + one retry, never more
    assert count_rows("retry_fail") == 0


def test_non_connection_db_error_is_not_retried(client):
    use_fake_model()
    data_error = IntegrityError("INSERT INTO inspections ...", {}, Exception("constraint violated"))
    response, commits = post_with_failing_commits(client, "no_retry", [data_error])

    assert response.json()["status"] == "partial_success"
    assert commits == 1                    # retrying would fail the same way, so we don't


# ---------------------------------------------------------------- bad uploads

def post_without_content_type(client, content: bytes):
    """POST /inspect where the file part has no Content-Type header at all.
    (Test clients always add one, so the multipart body is built by hand.)"""
    boundary = "TESTBOUNDARY"
    body = (
        f'--{boundary}\r\nContent-Disposition: form-data; name="board_id"\r\n\r\nno_ctype\r\n'
        f'--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="board.jpg"\r\n\r\n'
    ).encode() + content + f"\r\n--{boundary}--\r\n".encode()
    return client.post("/inspect", content=body,
                       headers={"Content-Type": f"multipart/form-data; boundary={boundary}"})


def test_empty_file_is_a_clean_400(client):
    response = client.post("/inspect", data={"board_id": "empty"},
                           files={"file": ("empty.jpg", b"", "image/jpeg")})
    assert response.status_code == 400
    assert "empty" in response.json()["detail"]


def test_missing_content_type_is_a_clean_400(client):
    response = post_without_content_type(client, b"not an image")
    assert response.status_code == 400
    assert "detail" in response.json()


def test_real_png_sent_as_octet_stream_is_accepted(client):
    # The real model runs here: the bytes are decoded and inferred, whatever the label says
    response = client.post("/inspect", data={"board_id": "octet"},
                           files={"file": ("board.png", make_image(".png"), "application/octet-stream")})
    assert response.status_code == 200
    assert response.json()["status"] == "success"


def test_real_jpeg_without_content_type_is_accepted(client):
    response = post_without_content_type(client, make_image(".jpg"))
    assert response.status_code == 200


def test_non_image_message_says_the_content_is_wrong(client):
    response = client.post("/inspect", data={"board_id": "pdf"},
                           files={"file": ("fake.png", b"%PDF-1.4 not an image", "image/png")})
    assert response.status_code == 400
    assert "not a JPEG or PNG" in response.json()["detail"]


def test_truncated_jpeg_message_says_it_is_corrupted(client):
    truncated = make_image(".jpg")[:40]  # starts like a JPEG, but most of the file is missing
    response = client.post("/inspect", data={"board_id": "truncated"},
                           files={"file": ("cut.jpg", truncated, "image/jpeg")})
    assert response.status_code == 400
    assert "truncated or corrupted" in response.json()["detail"]


# ---------------------------------------------------------------- size limits

def png_header_only(width: int, height: int) -> bytes:
    """The first bytes of a PNG that CLAIMS to be width x height, with no pixel data at all.
    Enough for the header check; it would fail if anyone tried to decode it."""
    ihdr = width.to_bytes(4, "big") + height.to_bytes(4, "big") + bytes([8, 2, 0, 0, 0])
    return b"\x89PNG\r\n\x1a\n" + (13).to_bytes(4, "big") + b"IHDR" + ihdr + b"\x00" * 4


def test_file_over_10_mb_is_rejected_with_413(client):
    too_big = make_image(".jpg") + b"\x00" * (10 * 1024 * 1024)
    response = client.post("/inspect", data={"board_id": "big_file"},
                           files={"file": ("big.jpg", too_big, "image/jpeg")})
    assert response.status_code == 413
    assert "10 MB" in response.json()["detail"]


def test_too_many_pixels_is_rejected_before_decoding(client):
    use_fake_model()
    from app.main import ml_engine
    response = client.post("/inspect", data={"board_id": "pixel_bomb"},
                           files={"file": ("huge.png", png_header_only(12000, 12000), "image/png")})
    assert response.status_code == 413
    assert "12000x12000" in response.json()["detail"]
    ml_engine["model"].predict.assert_not_called()   # rejected from the header alone


def test_exactly_25_megapixels_passes_the_pixel_check(client):
    # 5000 x 5000 = 25,000,000 is allowed; this fake file then fails to decode (400, not 413)
    response = client.post("/inspect", data={"board_id": "limit"},
                           files={"file": ("limit.png", png_header_only(5000, 5000), "image/png")})
    assert response.status_code == 400
