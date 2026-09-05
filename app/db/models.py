from sqlalchemy import Column, Integer, String, Text, Numeric, DateTime, func
from app.db.database import Base

class Inspection(Base):
    __tablename__ = "inspections"

    id = Column(Integer, primary_key=True, index=True)
    board_id = Column(String(100), nullable=False)
    image_path = Column(Text, nullable=False)
    defect_type = Column(String(50), nullable=False, index=True)
    confidence = Column(Numeric(5, 4), nullable=False)
    inspected_at = Column(DateTime(timezone=True), server_default=func.now(), index=True)
    processing_ms = Column(Integer)