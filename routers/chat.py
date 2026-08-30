"""
routers/chat.py

POST /chat — ask a question against a tenant's indexed documents.
"""

from fastapi import APIRouter, HTTPException

from models.schemas import ChatRequest, ChatResponse, SourceChunk
from services.rag_chain import ask, build_rag_chain

router = APIRouter(prefix="/chat", tags=["Chat"])


@router.post("", response_model=ChatResponse)
def chat(body: ChatRequest):
    try:
        chain = build_rag_chain(body.tenant_id, k=body.k)
        answer, source_docs = ask(chain, body.question)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    sources = [
        SourceChunk(content=doc.page_content[:500], metadata=doc.metadata)
        for doc in source_docs
    ]

    return ChatResponse(
        tenant_id=body.tenant_id,
        question=body.question,
        answer=answer,
        sources=sources,
    )
