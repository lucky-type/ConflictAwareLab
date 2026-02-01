"""Database configuration for ConflictAwareLab FastAPI application."""
from __future__ import annotations

from contextlib import contextmanager
from typing import Generator

from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker

# SQLite database - for production consider PostgreSQL
SQLALCHEMY_DATABASE_URL = "sqlite:///./conflictawarelab.db"

engine = create_engine(
    SQLALCHEMY_DATABASE_URL, 
    connect_args={"check_same_thread": False, "timeout": 30},
    pool_size=20,
    max_overflow=40,
    echo=False  # Set to True for SQL debugging
)

# Enable Write-Ahead Logging (WAL) for better concurrency performance
# This allows readers and writers to coexist
# try:
#     from sqlalchemy import text
#     with engine.connect() as connection:
#         connection.execute(text("PRAGMA journal_mode=WAL;"))
#         connection.execute(text("PRAGMA synchronous=NORMAL;"))
# except Exception as e:
#     print(f"Warning: Could not enable WAL mode: {e}")
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()


@contextmanager
def session_scope() -> Generator:
    """Provide a transactional scope around a series of operations."""
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def get_db() -> Generator:
    """FastAPI dependency that yields a SQLAlchemy session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
