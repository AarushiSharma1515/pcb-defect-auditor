import os
from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base

load_dotenv()

# Fallback to local Docker credentials if no .env is loaded
DATABASE_URL = os.getenv(
    "DATABASE_URL", 
    "postgresql://pcbaudit:localdevpassword@localhost:5432/pcb_defect_auditor"
)
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()