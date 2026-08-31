from dotenv import load_dotenv
load_dotenv()

from contextlib import asynccontextmanager
from fastapi import FastAPI, Depends, File, UploadFile, Form, HTTPException
from sqlalchemy.orm import Session
from app.db.database import get_db, engine, Base
from app.db.models import Inspection
from app.ml.inference import PCBDefectModel

# Global dictionary to hold the warmed-up model
ml_engine = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Executes exactly once on server startup
    ml_engine["model"] = PCBDefectModel()
    yield
    # Cleans up resources on server shutdown
    ml_engine.clear()


app = FastAPI(title="PCB Defect Auditor", lifespan=lifespan)

Base.metadata.create_all(bind=engine)


@app.get("/")
def root():
    return {"status": "PCB Defect Auditor is running"}


@app.get("/inspections")
def list_inspections(db: Session = Depends(get_db)):
    return db.query(Inspection).all()


@app.post("/inspect")
async def inspect_pcb(
    board_id: str = Form(...),
    file: UploadFile = File(...),
    db: Session = Depends(get_db)
):
    # 0. Basic input validation
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be an image")

    # 1. Ingestion
    image_bytes = await file.read()

    if len(image_bytes) == 0:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")

    # 2. Inference
    try:
        prediction = ml_engine["model"].predict(image_bytes)
    except Exception:
        raise HTTPException(
            status_code=422,
            detail="Inference failed — the file may be corrupted or not a valid image"
        )

    # 3. Persistence
    new_inspection = Inspection(
        board_id=board_id,
        image_path=file.filename,  # filename only — image itself is not persisted to storage
        defect_type=prediction["defect_type"],
        confidence=prediction["confidence"],
        processing_ms=prediction["processing_ms"]
    )

    try:
        db.add(new_inspection)
        db.commit()
        db.refresh(new_inspection)
    except Exception:
        db.rollback()
        raise HTTPException(status_code=500, detail="Failed to log inspection result to database")

    # 4. Response
    return {
        "status": "success",
        "board_id": board_id,
        "telemetry": prediction
    }