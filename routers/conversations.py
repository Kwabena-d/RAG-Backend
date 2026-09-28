"""
routers/conversations.py

Endpoints for managing conversations.

POST /conversations           — create a new conversation
GET  /conversations           — list conversations for a tenant
GET  /conversations/{conv_id} — get a single conversation with messages and docs
"""

from fastapi import APIRouter, HTTPException, Query

from models.schemas import (
    ConversationCreate,
    ConversationListItem,
    ConversationResponse,
    MessageModel,
    TENANT_ID_RE,
)
from services.conversations import (
    create_conversation,
    get_conversation,
    list_conversations,
)

router = APIRouter(prefix="/conversations", tags=["Conversations"])


@router.post("", response_model=ConversationResponse)
def create_conversation_endpoint(body: ConversationCreate) -> ConversationResponse:
    """Create a new conversation and return its full representation."""
    conv_id = create_conversation(body.tenant_id, body.title)
    conv = get_conversation(conv_id, body.tenant_id)
    if conv is None:
        raise HTTPException(status_code=500, detail="Failed to retrieve conversation after creation")

    return ConversationResponse(
        id=conv["id"],
        tenant_id=conv["tenant_id"],
        title=conv["title"],
        created_at=conv["created_at"],
        updated_at=conv["updated_at"],
        messages=[MessageModel(**m) for m in conv["messages"]],
        documents=conv["documents"],
    )


@router.get("", response_model=list[ConversationListItem])
def list_conversations_endpoint(
    tenant_id: str = Query(..., description="Tenant identifier"),
) -> list[ConversationListItem]:
    """Return the most recent conversations for a tenant."""
    if not tenant_id or not tenant_id.strip():
        raise HTTPException(status_code=422, detail="tenant_id must not be empty")
    if not TENANT_ID_RE.match(tenant_id):
        raise HTTPException(
            status_code=422,
            detail="tenant_id must be 1–64 characters: letters, digits, hyphens, underscores only",
        )

    convs = list_conversations(tenant_id)
    return [
        ConversationListItem(
            id=c["id"],
            title=c["title"],
            created_at=c["created_at"],
            updated_at=c["updated_at"],
        )
        for c in convs
    ]


@router.get("/{conv_id}", response_model=ConversationResponse)
def get_conversation_endpoint(
    conv_id: str,
    tenant_id: str = Query(..., description="Tenant identifier"),
) -> ConversationResponse:
    """Return a single conversation (with messages and documents) or 404."""
    if not tenant_id or not tenant_id.strip():
        raise HTTPException(status_code=422, detail="tenant_id must not be empty")
    if not TENANT_ID_RE.match(tenant_id):
        raise HTTPException(
            status_code=422,
            detail="tenant_id must be 1–64 characters: letters, digits, hyphens, underscores only",
        )

    conv = get_conversation(conv_id, tenant_id)
    if conv is None:
        raise HTTPException(status_code=404, detail="Conversation not found")

    return ConversationResponse(
        id=conv["id"],
        tenant_id=conv["tenant_id"],
        title=conv["title"],
        created_at=conv["created_at"],
        updated_at=conv["updated_at"],
        messages=[MessageModel(**m) for m in conv["messages"]],
        documents=conv["documents"],
    )
