"""
routers/chat.py

POST /chat — ask a question against a tenant's indexed documents.
"""

from fastapi import APIRouter, HTTPException

from models.schemas import ChatRequest, ChatResponse, SourceChunk
from services.rag_chain import ask, build_rag_chain, build_rag_chain_scoped
from services.conversations import get_conversation, get_conversation_doc_ids, add_message

router = APIRouter(prefix="/chat", tags=["Chat"])


@router.post("", response_model=ChatResponse)
def chat(body: ChatRequest):
    doc_ids: list[str] = []

    if body.conversation_id:
        conv = get_conversation(body.conversation_id, body.tenant_id)
        if conv is None:
            raise HTTPException(status_code=404, detail="Conversation not found")
        doc_ids = get_conversation_doc_ids(body.conversation_id)

    try:
        if body.conversation_id:
            chain = build_rag_chain_scoped(body.tenant_id, body.question, doc_ids)
        else:
            chain = build_rag_chain(body.tenant_id, body.question)
        answer, source_docs = ask(chain, body.question)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    sources = [
        SourceChunk(content=doc.page_content[:500], metadata=doc.metadata)
        for doc in source_docs
    ]

    # Persist to DB (don't fail the response if persistence errors)
    if body.conversation_id:
        try:
            src_list = [s.model_dump() for s in sources]
            add_message(body.conversation_id, "user", body.question)
            add_message(body.conversation_id, "assistant", answer, src_list)
        except Exception:
            pass

    return ChatResponse(
        tenant_id=body.tenant_id,
        question=body.question,
        answer=answer,
        sources=sources,
    )
