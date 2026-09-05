import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.pool import StaticPool
from sqlalchemy.orm import sessionmaker
from unittest.mock import MagicMock

from app.main import app
from app.db.database import Base, get_db

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
    files = {"file": ("corrupted.jpg", b"not real image data", "image/jpeg")}
    data = {"board_id": "test_board_002"}
    response = client.post("/inspect", data=data, files=files)
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