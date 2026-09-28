"""
routers/ingest.py

Endpoints for ingesting files into a tenant's vector store.

POST /ingest/file     — upload a PDF, DOCX, CSV, Excel, text, JSON, or image file
POST /ingest/database — connect to a SQL database and index its rows
"""

import hashlib
import os
import tempfile
import uuid
from typing import Optional

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from models.schemas import (
    IngestDatabaseRequest,
    IngestDatabaseResponse,
    IngestFileResponse,
    TENANT_ID_RE,
)
from services.ingestion import load_source, split_documents
from services.vector_store import store_chunks
from services.conversations import (
    document_exists,
    get_conversation,
    link_document_to_conversation,
    register_document,
)

router = APIRouter(prefix="/ingest", tags=["Ingestion"])

EXTENSION_TO_SOURCE_TYPE = {
    ".pdf":  "pdf",
    ".docx": "docx",
    ".csv":  "csv",
    ".xlsx": "excel",
    ".xls":  "excel",
    ".txt":  "text",
    ".md":   "text",
    ".json": "json",
    ".jpg":  "image",
    ".jpeg": "image",
    ".png":  "image",
    ".gif":  "image",
    ".webp": "image",
    ".bmp":  "image",
    ".tiff": "image",
    ".tif":  "image",
}


def _validate_tenant(tenant_id: str) -> None:
    if not tenant_id or not tenant_id.strip():
        raise HTTPException(status_code=422, detail="tenant_id must not be empty")
    if not TENANT_ID_RE.match(tenant_id):
        raise HTTPException(
            status_code=422,
            detail="tenant_id must be 1–64 characters: letters, digits, hyphens, underscores only",
        )


@router.post("/file", response_model=IngestFileResponse)
async def ingest_file(
    tenant_id: str = Form(..., description="Unique identifier for the user or organisation"),
    file: UploadFile = File(...),
    conversation_id: Optional[str] = Form(None),
):
    _validate_tenant(tenant_id)

    ext = os.path.splitext(file.filename or "")[1].lower()
    source_type = EXTENSION_TO_SOURCE_TYPE.get(ext)
    if not source_type:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{ext}'. Allowed: pdf, docx, csv, xlsx, xls, txt, md, json, jpg, jpeg, png, gif, webp, bmp, tiff",
        )

    content = await file.read()

    if len(content) == 0:
        raise HTTPException(status_code=422, detail="Uploaded file is empty")

    # Compute content hash for deduplication
    doc_id = hashlib.md5(content).hexdigest()

    # Verify conversation ownership if provided
    if conversation_id:
        conv = get_conversation(conversation_id, tenant_id)
        if conv is None:
            raise HTTPException(status_code=404, detail="Conversation not found")

    already_existed = document_exists(tenant_id, doc_id)

    if not already_existed:
        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
            tmp.write(content)
            tmp_path = tmp.name

        try:
            docs = load_source(source_type, tmp_path)
            # Stamp doc_id onto every document's metadata before splitting
            for doc in docs:
                doc.metadata["doc_id"] = doc_id
            chunks = split_documents(docs)
            chunks_stored = store_chunks(chunks, tenant_id)
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
        finally:
            os.unlink(tmp_path)
    else:
        chunks_stored = 0

    # Register (safe with ON CONFLICT DO NOTHING)
    register_document(tenant_id, doc_id, file.filename or "", source_type, chunks_stored)

    # Link to conversation if provided
    if conversation_id:
        link_document_to_conversation(conversation_id, doc_id, file.filename or "", source_type)

    return IngestFileResponse(
        tenant_id=tenant_id,
        filename=file.filename or "",
        chunks_stored=chunks_stored,
        source_type=source_type,
        doc_id=doc_id,
        already_existed=already_existed,
    )


@router.post("/database", response_model=IngestDatabaseResponse)
def ingest_database(body: IngestDatabaseRequest):
    # Generate a unique doc_id for this database ingestion
    doc_id = str(uuid.uuid4())

    # Verify conversation ownership if provided
    if body.conversation_id:
        conv = get_conversation(body.conversation_id, body.tenant_id)
        if conv is None:
            raise HTTPException(status_code=404, detail="Conversation not found")

    try:
        from services.ingestion import load_database
        docs = load_database(body.connection_string, body.max_rows_per_table)
        # Stamp doc_id onto every document's metadata before splitting
        for doc in docs:
            doc.metadata["doc_id"] = doc_id
        chunks = split_documents(docs)
        count = store_chunks(chunks, body.tenant_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    if body.conversation_id:
        register_document(body.tenant_id, doc_id, "database", "database", count)
        link_document_to_conversation(body.conversation_id, doc_id, "database", "database")

    return IngestDatabaseResponse(
        tenant_id=body.tenant_id,
        chunks_stored=count,
        doc_id=doc_id,
    )
