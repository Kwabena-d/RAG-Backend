import re
from typing import Optional

from pydantic import BaseModel, field_validator

TENANT_ID_RE = re.compile(r"^[a-zA-Z0-9_-]{1,64}$")


def _check_tenant_id(v: str) -> str:
    if not v or not v.strip():
        raise ValueError("tenant_id must not be empty")
    if not TENANT_ID_RE.match(v):
        raise ValueError(
            "tenant_id must be 1–64 characters: letters, digits, hyphens, underscores only"
        )
    return v


class IngestFileResponse(BaseModel):
    tenant_id: str
    filename: str
    chunks_stored: int
    source_type: str


class IngestDatabaseRequest(BaseModel):
    tenant_id: str
    connection_string: str
    max_rows_per_table: Optional[int] = 500

    @field_validator("tenant_id")
    @classmethod
    def validate_tenant_id(cls, v: str) -> str:
        return _check_tenant_id(v)

    @field_validator("connection_string")
    @classmethod
    def validate_connection_string(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("connection_string must not be empty")
        return v


class IngestDatabaseResponse(BaseModel):
    tenant_id: str
    chunks_stored: int


class ChatRequest(BaseModel):
    tenant_id: str
    question: str
    k: Optional[int] = 4

    @field_validator("tenant_id")
    @classmethod
    def validate_tenant_id(cls, v: str) -> str:
        return _check_tenant_id(v)

    @field_validator("question")
    @classmethod
    def validate_question(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("question must not be empty")
        if len(v) > 2000:
            raise ValueError("question must be 2000 characters or fewer")
        return v


class SourceChunk(BaseModel):
    content: str
    metadata: dict


class ChatResponse(BaseModel):
    tenant_id: str
    question: str
    answer: str
    sources: list[SourceChunk]
