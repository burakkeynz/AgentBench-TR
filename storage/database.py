"""
Providing database initialization and session context manager.
"""

from contextlib import contextmanager
from storage.base import Base, engine, SessionLocal


def init_db():
    """Initializing database — creating all tables."""
    import storage.models  # noqa: F401 — registering models with Base
    Base.metadata.create_all(bind=engine)
    print("Initializing database — creating tables: traces, agent_logs, eval_results.")


@contextmanager
def get_db():
    """Providing a transactional database session."""
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()