from sqlalchemy import create_engine, text
from core.config import settings

engine = create_engine(settings.database_url)


def init_db() -> None:
    """Enable the pgvector extension. Called once at app startup."""
    with engine.connect() as conn:
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        conn.commit()
