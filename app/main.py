from dotenv import load_dotenv
load_dotenv()

import logging
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from pathlib import Path
from fastapi import FastAPI, Depends, File, UploadFile, Form, HTTPException, Query, Request
from fastapi.responses import FileResponse, JSONResponse
from sqlalchemy.exc import OperationalError, SQLAlchemyError
from sqlalchemy import func, text
from sqlalchemy.orm import Session
from app import __version__
from app.db.database import get_db, engine, Base
from app.db.models import Inspection
from app.ml.inference import DEFAULT_MODEL_PATH, PCBDefectModel
from app.ml.image_checks import detect_image_format, read_image_size

MAX_UPLOAD_BYTES = 10 * 1024 * 1024   # 10 MB
MAX_PIXELS = 25_000_000               # 25 megapixels (e.g. 5000 x 5000)
HEALTH_DB_TIMEOUT_SECONDS = 2.0       # /health reports "unreachable" if SELECT 1 takes longer

# Demo images served by /samples, found relative to this file (app/main.py -> repo root)
SAMPLES_DIR = Path(__file__).resolve().parents[1] / "samples"
SAMPLE_EXTENSIONS = {".jpg", ".jpeg", ".png"}

logger = logging.getLogger(__name__)

ml_engine = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Create the table if it's missing. If the database is down right now, don't crash:
    # the model can still answer /inspect (with partial_success, since nothing is saved).
    try:
        Base.metadata.create_all(bind=engine)
    except SQLAlchemyError as e:
        logger.warning(
            "Database unavailable at startup, so tables were not checked or created. "
            "The API is starting anyway; /inspect will return partial_success until the "
            "database is reachable again. Error: %s", e
        )
    ml_engine["model"] = PCBDefectModel()
    yield
    ml_engine.clear()

app = FastAPI(title="PCB Defect Auditor", version=__version__, lifespan=lifespan)


def save_inspection(db: Session, record: dict) -> bool:
    """Save one inspection row. Returns True if it was saved, False if the database failed.

    If the connection itself fails (OperationalError, e.g. Neon closed an idle
    connection: "SSL connection has been closed unexpectedly"), try one more time.
    rollback() hands the broken connection back; SQLAlchemy throws away connections
    that failed like this, and pool_pre_ping tests the next one, so the second
    attempt runs on a fresh connection.
    Other database errors (e.g. a value too long for a column) would just fail the
    same way again, so they are not retried.
    """
    for attempt in (1, 2):
        try:
            db.add(Inspection(**record))
            db.commit()
            return True
        except OperationalError as e:
            db.rollback()
            print(f"Database connection error (attempt {attempt} of 2): {e}")
        except SQLAlchemyError as e:
            db.rollback()
            print(f"Database error (not retried): {e}")
            return False
    return False


@app.get("/")
def root():
    return {"status": "PCB Defect Auditor is running"}


# Two small background threads for the /health database check. Running SELECT 1 there lets
# /health stop waiting after HEALTH_DB_TIMEOUT_SECONDS, even if the database hangs instead
# of refusing the connection. (A stuck check still ends on its own when Postgres's 5 s
# connect_timeout expires, and only two can run at once, so they can't pile up threads.)
_health_db_checker = ThreadPoolExecutor(max_workers=2, thread_name_prefix="health-db")


def _select_one():
    with engine.connect() as connection:
        connection.execute(text("SELECT 1"))


def check_database() -> str:
    """'ok' if the database answers SELECT 1 within the timeout, otherwise 'unreachable'.
    Never raises: a health check must not crash because the thing it checks is down."""
    try:
        _health_db_checker.submit(_select_one).result(timeout=HEALTH_DB_TIMEOUT_SECONDS)
        return "ok"
    except Exception as e:   # connection refused, bad credentials, timeout, anything else
        logger.warning("Health check: database unreachable: %s", e or type(e).__name__)
        return "unreachable"


@app.get("/health")
def health():
    """Always 200, so a monitor can read the details; check the fields to see what's wrong.
    Plain "def": the database check blocks, so FastAPI runs this in the thread pool."""
    model = ml_engine.get("model")
    model_loaded = isinstance(model, PCBDefectModel)
    return {
        "model_loaded": model_loaded,
        "model_file": (model.model_path if model_loaded else DEFAULT_MODEL_PATH).name,
        "mode": "onnx",
        "database": check_database(),
        "version": __version__,
    }


def list_sample_files() -> dict:
    """{file name: path} for the images directly inside SAMPLES_DIR; {} if the folder is missing.
    Only real files whose resolved location is inside the folder count, so a link that
    points somewhere else on the disk is never served."""
    folder = SAMPLES_DIR.resolve()
    if not folder.is_dir():
        return {}
    return {
        path.name: path
        for path in folder.iterdir()
        if path.suffix.lower() in SAMPLE_EXTENSIONS
        and path.is_file()
        and path.resolve().parent == folder
    }


@app.get("/samples")
def list_samples():
    """Names of the demo images in samples/ (an empty list if there are none)."""
    return sorted(list_sample_files())


@app.get("/samples/{name}", responses={
    400: {"description": "The name contains a path (/, \\ or ..)"},
    404: {"description": "No such sample image"},
})
def get_sample(name: str):
    # Only a bare file name is accepted, never a path like ../.env or ..\secret.jpg
    if "/" in name or "\\" in name or ".." in name:
        raise HTTPException(status_code=400, detail="Give a sample file name, not a path.")
    # And it must be one of the files /samples lists (an allowlist, not a path lookup)
    path = list_sample_files().get(name)
    if path is None:
        raise HTTPException(status_code=404, detail=f"No sample image named '{name}'.")
    return FileResponse(path)

@app.exception_handler(OperationalError)
async def database_unavailable(request: Request, exc: OperationalError):
    """Any endpoint that can't reach the database answers 503 (try again later) with a
    JSON message, instead of a plain-text 500. /inspect handles its own errors and
    returns partial_success, so it never gets here."""
    logger.warning("Database unavailable during %s %s: %s", request.method, request.url.path, exc)
    return JSONResponse(
        status_code=503,
        content={"detail": "The database is unavailable right now. Please try again shortly."},
    )


DATABASE_DOWN = {503: {"description": "The database is unavailable"}}


@app.get("/inspections", responses=DATABASE_DOWN)
def list_inspections(
    limit: int = Query(50, ge=1, le=200, description="How many inspections to return (1-200)"),
    offset: int = Query(0, ge=0, description="How many inspections to skip (for the next page)"),
    db: Session = Depends(get_db),
):
    # Newest first, so the first page shows the most recent inspections
    return db.query(Inspection).order_by(Inspection.id.desc()).offset(offset).limit(limit).all()

@app.get("/analytics/summary", responses=DATABASE_DOWN)
def get_analytics_summary(db: Session = Depends(get_db)):
    # 1. Get total inspections
    total = db.query(Inspection).count()
    
    # 2. Get average confidence across all runs
    avg_conf = db.query(func.avg(Inspection.confidence)).scalar() or 0.0
    
    # 3. Get count breakdown by defect type
    breakdown = db.query(
        Inspection.defect_type, 
        func.count(Inspection.id)
    ).group_by(Inspection.defect_type).all()
    
    return {
        "total_inspections": total,
        "average_confidence": round(avg_conf, 4),
        "defect_breakdown": {row[0]: row[1] for row in breakdown}
    }

@app.post("/inspect", responses={
    400: {"description": "Empty file, not a JPEG/PNG, or a corrupted image"},
    413: {"description": "File over 10 MB or image over 25 megapixels"},
})
def inspect_pcb(
    # Plain "def" (not "async def") on purpose: everything in here blocks (the ONNX model,
    # the database commit). FastAPI runs plain-def endpoints in a thread pool, so a slow
    # request can't freeze the event loop that answers every other request.
    # Same limit as the database column (String(100)). Checking it here gives the same
    # 422 error on SQLite and Postgres, before any work is done.
    board_id: str = Form(..., max_length=100),
    file: UploadFile = File(...),
    db: Session = Depends(get_db)
):
    # 1. Read raw bytes, but never more than the limit (+1 byte, so we can tell it was too big).
    #    An empty file would crash OpenCV, so reject it here.
    file_bytes = file.file.read(MAX_UPLOAD_BYTES + 1)   # file.file = the uploaded data (no await needed)
    if len(file_bytes) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="The file is larger than the 10 MB limit.")
    if not file_bytes:
        raise HTTPException(status_code=400, detail="The uploaded file is empty (0 bytes).")

    # 2. Decide if it's an image by looking at the bytes, not at the Content-Type label
    #    (clients often send real images as application/octet-stream, or with no label).
    image_format = detect_image_format(file_bytes)
    if image_format is None:
        raise HTTPException(
            status_code=400,
            detail=(f"The file's content is not a JPEG or PNG image "
                    f"(it was sent as '{file.content_type or 'no content type'}'). Please upload a JPEG or PNG."),
        )

    # 2b. Check width x height from the file header BEFORE decoding: a small file can
    #     unpack into a huge image (a 0.5 MB PNG can need 432 MB once decoded).
    size = read_image_size(file_bytes, image_format)
    if size is None or 0 in size:
        raise HTTPException(
            status_code=400,
            detail=f"The file looks like a {image_format} but its header could not be read; it is probably truncated or corrupted.",
        )
    width, height = size
    if width * height > MAX_PIXELS:
        raise HTTPException(
            status_code=413,
            detail=f"The image is {width}x{height} pixels; the limit is {MAX_PIXELS // 1_000_000} megapixels.",
        )

    # 3. Run ONNX Inference and extract dictionary values safely
    try:
        prediction_result = ml_engine["model"].predict(file_bytes)
        defect_type = prediction_result["defect_type"]
        confidence_float = float(prediction_result["confidence"])
        processing_ms = int(prediction_result["processing_ms"])
        detections = prediction_result["detections"]
        image_width = int(prediction_result["image_width"])
        image_height = int(prediction_result["image_height"])
    except ValueError:
        # predict() raises ValueError when OpenCV can't decode the image
        raise HTTPException(
            status_code=400,
            detail=f"The file looks like a {image_format} but could not be decoded; it is probably truncated or corrupted.",
        )
    except TypeError as e:
        # Fallback just in case predict() returns a tuple instead of a dictionary
        raise HTTPException(status_code=500, detail="Model returned an unexpected data format.")

    # 4. Flag low-confidence predictions
    requires_human_review = bool(confidence_float < 0.50)
    # 5. Database logging (retried once if the connection was dropped)
    saved = save_inspection(db, {
        "board_id": board_id,
        "image_path": file.filename,
        "defect_type": defect_type,
        "confidence": confidence_float,
        "processing_ms": processing_ms,
    })
    # 6. Response. "telemetry" is the top defect (what was saved); "detections" lists every
    #    defect found, with boxes [x1, y1, x2, y2] in the ORIGINAL image's pixels.
    response = {
        "status": "success",
        "board_id": board_id,
        "requires_human_review": requires_human_review,
        "telemetry": {
            "defect_type": defect_type,
            "confidence": round(confidence_float, 4),
            "processing_ms": processing_ms
        },
        "detections": detections,
        "image_width": image_width,
        "image_height": image_height,
    }
    if not saved:
        response["status"] = "partial_success"
        response["message"] = "Inference completed but database logging failed."
    return response