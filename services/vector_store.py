"""
services/vector_store.py

Replaces the in-memory FAISS index with pgvector (Postgres).
Each tenant gets its own collection so documents are fully isolated.

Public API:
    store_chunks(chunks, tenant_id)        — embed and persist chunks
    similarity_search(query, tenant_id, k) — retrieve top-k relevant chunks
    get_store(tenant_id)                   — return the raw PGVector store
                                             (used by the RAG chain)
"""

from typing import List

from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_postgres import PGVector

from core.config import settings

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

_embeddings: HuggingFaceEmbeddings | None = None


def get_embeddings() -> HuggingFaceEmbeddings:
    global _embeddings
    if _embeddings is None:
        _embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    return _embeddings


def get_store(tenant_id: str) -> PGVector:
    """Return a PGVector store scoped to this tenant."""
    return PGVector(
        embeddings=get_embeddings(),
        collection_name=f"tenant_{tenant_id}",
        connection=settings.database_url,
        use_jsonb=True,
    )


def store_chunks(chunks: List[Document], tenant_id: str) -> int:
    """Embed chunks and persist them to the tenant's pgvector collection.
    Returns the number of chunks stored."""
    if not chunks:
        raise ValueError("No chunks to store.")
    store = get_store(tenant_id)
    store.add_documents(chunks)
    return len(chunks)


def similarity_search(query: str, tenant_id: str, k: int = 4) -> List[Document]:
    """Return the top-k most relevant chunks for this tenant."""
    store = get_store(tenant_id)
    return store.similarity_search(query, k=k)
