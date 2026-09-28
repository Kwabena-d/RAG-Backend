from sqlalchemy import create_engine, text
from core.config import settings

engine = create_engine(settings.database_url)


def init_db() -> None:
    """Enable the pgvector extension. Called once at app startup."""
    with engine.connect() as conn:
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        conn.commit()


def init_conversation_tables() -> None:
    """Create the conversation-related tables if they do not already exist."""
    with engine.connect() as conn:
        conn.execute(
            text(
                """
                CREATE TABLE IF NOT EXISTS sv_documents (
                    id          TEXT PRIMARY KEY,
                    tenant_id   TEXT NOT NULL,
                    doc_id      TEXT NOT NULL,
                    filename    TEXT NOT NULL,
                    source_type TEXT NOT NULL,
                    chunks_stored INT DEFAULT 0,
                    created_at  TIMESTAMPTZ DEFAULT now(),
                    UNIQUE (tenant_id, doc_id)
                )
                """
            )
        )

        conn.execute(
            text(
                """
                CREATE TABLE IF NOT EXISTS sv_conversations (
                    id         TEXT PRIMARY KEY,
                    tenant_id  TEXT NOT NULL,
                    title      TEXT NOT NULL,
                    created_at TIMESTAMPTZ DEFAULT now(),
                    updated_at TIMESTAMPTZ DEFAULT now()
                )
                """
            )
        )

        conn.execute(
            text(
                """
                CREATE INDEX IF NOT EXISTS idx_sv_conversations_tenant
                ON sv_conversations (tenant_id, updated_at DESC)
                """
            )
        )

        conn.execute(
            text(
                """
                CREATE TABLE IF NOT EXISTS sv_messages (
                    id              TEXT PRIMARY KEY,
                    conversation_id TEXT NOT NULL REFERENCES sv_conversations(id) ON DELETE CASCADE,
                    role            TEXT NOT NULL,
                    content         TEXT NOT NULL,
                    sources         JSONB DEFAULT '[]'::jsonb,
                    created_at      TIMESTAMPTZ DEFAULT now()
                )
                """
            )
        )

        conn.execute(
            text(
                """
                CREATE INDEX IF NOT EXISTS idx_sv_messages_conv
                ON sv_messages (conversation_id, created_at)
                """
            )
        )

        conn.execute(
            text(
                """
                CREATE TABLE IF NOT EXISTS sv_conversation_documents (
                    conversation_id TEXT NOT NULL REFERENCES sv_conversations(id) ON DELETE CASCADE,
                    doc_id          TEXT NOT NULL,
                    filename        TEXT NOT NULL,
                    source_type     TEXT NOT NULL,
                    created_at      TIMESTAMPTZ DEFAULT now(),
                    PRIMARY KEY (conversation_id, doc_id)
                )
                """
            )
        )

        conn.commit()
