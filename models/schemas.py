from pydantic import BaseModel
from typing import Optional


class IngestFileResponse(BaseModel):
    tenant_id: str
    filename: str
    chunks_stored: int
    source_type: str


class IngestDatabaseRequest(BaseModel):
    tenant_id: str
    connection_string: str
    max_rows_per_table: Optional[int] = 500


class IngestDatabaseResponse(BaseModel):
    tenant_id: str
    chunks_stored: int


class ChatRequest(BaseModel):
    tenant_id: str
    question: str
    k: Optional[int] = 4


class SourceChunk(BaseModel):
    content: str
    metadata: dict


class ChatResponse(BaseModel):
    tenant_id: str
    question: str
    answer: str
    sources: list[SourceChunk]
