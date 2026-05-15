"""
Configuring SQLAlchemy engine, session factory, and declarative base.
"""

import os
from pathlib import Path
from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker

_PROJECT_ROOT = Path(__file__).parent.parent
DB_PATH       = os.getenv("SQLITE_DB_PATH", str(_PROJECT_ROOT / "storage" / "agentbench.db"))
DATABASE_URL = f"sqlite:///{DB_PATH}"

Path(DB_PATH).parent.mkdir(parents=True, exist_ok=True)

engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})

SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)

Base = declarative_base()