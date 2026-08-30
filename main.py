"""
main.py

FastAPI application entry point.

Run with:
    uvicorn main:app --reload

Interactive API docs available at:
    http://localhost:8000/docs
"""

from fastapi import FastAPI
from dotenv import load_dotenv

load_dotenv()

from core.database import init_db
from routers import ingest, chat

app = FastAPI(
    title="RAG Backend API",
    description="Retrieval-Augmented Generation API — ingest documents and chat with Claude.",
    version="1.0.0",
)


@app.on_event("startup")
def on_startup():
    init_db()


app.include_router(ingest.router)
app.include_router(chat.router)


@app.get("/health")
def health():
    return {"status": "ok"}
