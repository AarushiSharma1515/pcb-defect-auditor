import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.pool import StaticPool
from sqlalchemy.orm import sessionmaker
from sqlalchemy.exc import IntegrityError, OperationalError
from unittest.mock import MagicMock

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
    
    files = {"file": ("valid.jpg", b"fake valid image bytes", "image/jpeg")}
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
            files={"file": ("board.jpg", b"fake image bytes", "image/jpeg")},
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