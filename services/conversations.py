"""
services/conversations.py

SQLAlchemy database service for conversation persistence.
Uses the shared engine from core.database.
"""

import json
import uuid

from sqlalchemy import text

from core.database import engine


def create_conversation(tenant_id: str, title: str) -> str:
    """Insert a new conversation into sv_conversations. Returns the new id as a string."""
    conv_id = str(uuid.uuid4())
    with engine.connect() as conn:
        conn.execute(
            text(
                """
                INSERT INTO sv_conversations (id, tenant_id, title)
                VALUES (:id, :tenant_id, :title)
                """
            ),
            {"id": conv_id, "tenant_id": tenant_id, "title": title},
        )
        conn.commit()
    return conv_id


def list_conversations(tenant_id: str) -> list[dict]:
    """Return up to 100 conversations for a tenant, newest first."""
    with engine.connect() as conn:
        rows = conn.execute(
            text(
                """
                SELECT id, title, created_at, updated_at
                FROM sv_conversations
                WHERE tenant_id = :tenant_id
                ORDER BY updated_at DESC
                LIMIT 100
                """
            ),
            {"tenant_id": tenant_id},
        ).fetchall()

    result = []
    for row in rows:
        result.append(
            {
                "id": row[0],
                "title": row[1],
                "created_at": row[2].isoformat() if row[2] else "",
                "updated_at": row[3].isoformat() if row[3] else "",
            }
        )
    return result


def get_conversation(conv_id: str, tenant_id: str) -> dict | None:
    """
    Return full conversation data including messages and documents.
    Returns None if the conversation doesn't exist or belongs to a different tenant.
    """
    with engine.connect() as conn:
        conv_row = conn.execute(
            text(
                """
                SELECT id, tenant_id, title, created_at, updated_at
                FROM sv_conversations
                WHERE id = :id AND tenant_id = :tenant_id
                """
            ),
            {"id": conv_id, "tenant_id": tenant_id},
        ).fetchone()

        if conv_row is None:
            return None

        msg_rows = conn.execute(
            text(
                """
                SELECT role, content, sources, created_at
                FROM sv_messages
                WHERE conversation_id = :conv_id
                ORDER BY created_at ASC
                """
            ),
            {"conv_id": conv_id},
        ).fetchall()

        doc_rows = conn.execute(
            text(
                """
                SELECT doc_id, filename, source_type, created_at
                FROM sv_conversation_documents
                WHERE conversation_id = :conv_id
                ORDER BY created_at ASC
                """
            ),
            {"conv_id": conv_id},
        ).fetchall()

    messages = []
    for row in msg_rows:
        raw_sources = row[2]
        if raw_sources is None:
            sources = []
        elif isinstance(raw_sources, list):
            sources = raw_sources
        elif isinstance(raw_sources, str):
            try:
                sources = json.loads(raw_sources)
            except Exception:
                sources = []
        else:
            sources = list(raw_sources) if raw_sources else []

        messages.append(
            {
                "role": row[0],
                "content": row[1],
                "sources": sources,
            }
        )

    documents = []
    for row in doc_rows:
        documents.append(
            {
                "doc_id": row[0],
                "filename": row[1],
                "source_type": row[2],
                "created_at": row[3].isoformat() if row[3] else "",
            }
        )

    return {
        "id": conv_row[0],
        "tenant_id": conv_row[1],
        "title": conv_row[2],
        "created_at": conv_row[3].isoformat() if conv_row[3] else "",
        "updated_at": conv_row[4].isoformat() if conv_row[4] else "",
        "messages": messages,
        "documents": documents,
    }


def add_message(
    conv_id: str,
    role: str,
    content: str,
    sources: list | None = None,
) -> None:
    """Insert a message into sv_messages and bump the conversation's updated_at."""
    msg_id = str(uuid.uuid4())
    sources_json = json.dumps(sources if sources is not None else [])

    with engine.connect() as conn:
        conn.execute(
            text(
                """
                INSERT INTO sv_messages (id, conversation_id, role, content, sources)
                VALUES (:id, :conversation_id, :role, :content, :sources::jsonb)
                """
            ),
            {
                "id": msg_id,
                "conversation_id": conv_id,
                "role": role,
                "content": content,
                "sources": sources_json,
            },
        )
        conn.execute(
            text(
                """
                UPDATE sv_conversations
                SET updated_at = now()
                WHERE id = :conv_id
                """
            ),
            {"conv_id": conv_id},
        )
        conn.commit()


def document_exists(tenant_id: str, doc_id: str) -> bool:
    """Return True if a document with this doc_id already exists for the tenant."""
    with engine.connect() as conn:
        row = conn.execute(
            text(
                """
                SELECT 1 FROM sv_documents
                WHERE tenant_id = :tenant_id AND doc_id = :doc_id
                LIMIT 1
                """
            ),
            {"tenant_id": tenant_id, "doc_id": doc_id},
        ).fetchone()
    return row is not None


def register_document(
    tenant_id: str,
    doc_id: str,
    filename: str,
    source_type: str,
    chunks_stored: int,
) -> None:
    """
    Upsert a document record into sv_documents.
    Uses ON CONFLICT DO NOTHING so it is safe to call even if the document already exists.
    """
    record_id = str(uuid.uuid4())
    with engine.connect() as conn:
        conn.execute(
            text(
                """
                INSERT INTO sv_documents (id, tenant_id, doc_id, filename, source_type, chunks_stored)
                VALUES (:id, :tenant_id, :doc_id, :filename, :source_type, :chunks_stored)
                ON CONFLICT (tenant_id, doc_id) DO NOTHING
                """
            ),
            {
                "id": record_id,
                "tenant_id": tenant_id,
                "doc_id": doc_id,
                "filename": filename,
                "source_type": source_type,
                "chunks_stored": chunks_stored,
            },
        )
        conn.commit()


def link_document_to_conversation(
    conv_id: str,
    doc_id: str,
    filename: str,
    source_type: str,
) -> None:
    """
    Link a document to a conversation in sv_conversation_documents.
    Uses ON CONFLICT DO NOTHING. Also bumps the conversation's updated_at.
    """
    with engine.connect() as conn:
        conn.execute(
            text(
                """
                INSERT INTO sv_conversation_documents (conversation_id, doc_id, filename, source_type)
                VALUES (:conversation_id, :doc_id, :filename, :source_type)
                ON CONFLICT (conversation_id, doc_id) DO NOTHING
                """
            ),
            {
                "conversation_id": conv_id,
                "doc_id": doc_id,
                "filename": filename,
                "source_type": source_type,
            },
        )
        conn.execute(
            text(
                """
                UPDATE sv_conversations
                SET updated_at = now()
                WHERE id = :conv_id
                """
            ),
            {"conv_id": conv_id},
        )
        conn.commit()


def get_conversation_doc_ids(conv_id: str) -> list[str]:
    """Return a list of doc_id strings linked to a conversation."""
    with engine.connect() as conn:
        rows = conn.execute(
            text(
                """
                SELECT doc_id FROM sv_conversation_documents
                WHERE conversation_id = :conv_id
                """
            ),
            {"conv_id": conv_id},
        ).fetchall()
    return [row[0] for row in rows]


def update_title(conv_id: str, title: str) -> None:
    """Update the title of a conversation."""
    with engine.connect() as conn:
        conn.execute(
            text(
                """
                UPDATE sv_conversations
                SET title = :title, updated_at = now()
                WHERE id = :conv_id
                """
            ),
            {"title": title, "conv_id": conv_id},
        )
        conn.commit()
