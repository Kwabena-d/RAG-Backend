"""
routers/ingest.py

Endpoints for ingesting documents into a tenant's vector store.

POST /ingest/file     — upload a PDF, DOCX, CSV, or Excel file
POST /ingest/database — connect to a SQL database and index its rows
"""

import os
import tempfile

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from models.schemas import (
    IngestDatabaseRequest,
    IngestDatabaseResponse,
    IngestFileResponse,
    TENANT_ID_RE,
)
from services.ingestion import load_source, split_documents
from services.vector_store import store_chunks

router = APIRouter(prefix="/ingest", tags=["Ingestion"])

EXTENSION_TO_SOURCE_TYPE = {
    ".pdf":  "pdf",
    ".docx": "docx",
    ".csv":  "csv",
    ".xlsx": "excel",
    ".xls":  "excel",
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
):
    _validate_tenant(tenant_id)

    ext = os.path.splitext(file.filename or "")[1].lower()
    source_type = EXTENSION_TO_SOURCE_TYPE.get(ext)
    if not source_type:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{ext}'. Allowed: pdf, docx, csv, xlsx, xls",
        )

    content = await file.read()

    if len(content) == 0:
        raise HTTPException(status_code=422, detail="Uploaded file is empty")

    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
        tmp.write(content)
        tmp_path = tmp.name

    try:
        docs = load_source(source_type, tmp_path)
        chunks = split_documents(docs)
        count = store_chunks(chunks, tenant_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        os.unlink(tmp_path)

    return IngestFileResponse(
        tenant_id=tenant_id,
        filename=file.filename or "",
        chunks_stored=count,
        source_type=source_type,
    )


@router.post("/database", response_model=IngestDatabaseResponse)
def ingest_database(body: IngestDatabaseRequest):
    try:
        from services.ingestion import load_database
        docs = load_database(body.connection_string, body.max_rows_per_table)
        chunks = split_documents(docs)
        count = store_chunks(chunks, body.tenant_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return IngestDatabaseResponse(tenant_id=body.tenant_id, chunks_stored=count)
