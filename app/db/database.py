import os
from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.engine import make_url
from sqlalchemy.orm import sessionmaker, declarative_base

load_dotenv()

# Fallback to local Docker credentials if no .env is loaded
DATABASE_URL = os.getenv(
    "DATABASE_URL", 
    "postgresql://pcbaudit:localdevpassword@localhost:5432/pcb_defect_auditor"
)

# Neon's free tier closes connections that sit idle, so a pooled connection can be
# dead by the time we use it ("SSL connection has been closed unexpectedly").
#   pool_pre_ping=True -> before reusing a pooled connection, send a tiny test query;
#                         if it fails, throw that connection away and open a new one
#   pool_recycle=300   -> never reuse a connection that is older than 5 minutes
#   connect_timeout=5  -> give up opening a new connection after 5 seconds
#                         (a Postgres-only option, so SQLite, used in tests, skips it)
connect_args = {}
if make_url(DATABASE_URL).get_backend_name() == "postgresql":
    connect_args["connect_timeout"] = 5

engine = create_engine(
    DATABASE_URL,
    pool_pre_ping=True,
    pool_recycle=300,
    connect_args=connect_args,
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()