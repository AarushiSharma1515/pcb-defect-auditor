from dotenv import load_dotenv
load_dotenv()
from fastapi import FastAPI, Depends
from sqlalchemy.orm import Session
from app.db.database import get_db, engine, Base
from app.db.models import Inspection

app = FastAPI(title="PCB Defect Auditor")

Base.metadata.create_all(bind=engine)

@app.get("/")
def root():
    return {"status": "PCB Defect Auditor is running"}

@app.get("/inspections")
def list_inspections(db: Session = Depends(get_db)):
    return db.query(Inspection).all()