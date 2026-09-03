from dotenv import load_dotenv
load_dotenv()

from contextlib import asynccontextmanager
from fastapi import FastAPI, Depends, File, UploadFile, Form, HTTPException
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session
from app.db.database import get_db, engine, Base
from app.db.models import Inspection
from app.ml.inference import PCBDefectModel

ml_engine = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    ml_engine["model"] = PCBDefectModel()
    yield
    ml_engine.clear()

app = FastAPI(title="PCB Defect Auditor", lifespan=lifespan)

Base.metadata.create_all(bind=engine)

@app.get("/")
def root():
    return {"status": "PCB Defect Auditor is running"}

@app.get("/inspections")
def list_inspections(db: Session = Depends(get_db)):
    return db.query(Inspection).all()

@app.post("/inspect", responses={
    400: {"description": "Invalid file type or corrupted image"}
})
async def inspect_pcb(
    board_id: str = Form(...),
    file: UploadFile = File(...),
    db: Session = Depends(get_db)
):
    # 1. Validate file type
    if not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400, 
            detail=f"Invalid file type: {file.content_type}. Please upload a JPEG or PNG."
        )

    # 2. Read raw bytes
    file_bytes = await file.read()

    # 3. Run ONNX Inference and extract dictionary values safely
    try:
        prediction_result = ml_engine["model"].predict(file_bytes)
        defect_type = prediction_result["defect_type"]
        confidence_float = float(prediction_result["confidence"])
        processing_ms = int(prediction_result["processing_ms"])
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except TypeError as e:
        # Fallback just in case predict() returns a tuple instead of a dictionary
        raise HTTPException(status_code=500, detail="Model returned an unexpected data format.")

    # 4. Flag low-confidence predictions
    requires_human_review = bool(confidence_float < 0.50)

    # 5. Database logging
    try:
        new_inspection = Inspection(
            board_id=board_id,
            image_path=file.filename,
            defect_type=defect_type,
            confidence=confidence_float,
            processing_ms=processing_ms
        )
        db.add(new_inspection)
        db.commit()
    except SQLAlchemyError as e:
        print(f"Database error: {e}")
        db.rollback()
        return {
            "status": "partial_success",
            "warning": "Inference succeeded but database logging failed.",
            "board_id": board_id,
            "telemetry": {"defect_type": defect_type, "confidence": round(confidence_float, 4)}
        }

    # 6. Final Response
    return {
        "status": "success",
        "board_id": board_id,
        "requires_human_review": requires_human_review,
        "telemetry": {
            "defect_type": defect_type,
            "confidence": round(confidence_float, 4),
            "processing_ms": processing_ms
        }
    }