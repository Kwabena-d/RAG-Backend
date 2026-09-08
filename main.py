"""
main.py

FastAPI application entry point.

Run with:
    uvicorn main:app --reload

Interactive API docs available at:
    http://localhost:8000/docs
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

load_dotenv()

from core.database import init_db
from routers import ingest, chat


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    yield


app = FastAPI(
    title="RAG Backend API",
    description="Retrieval-Augmented Generation API — ingest documents and chat with Claude.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(ingest.router)
app.include_router(chat.router)


@app.get("/health")
def health():
    return {"status": "ok"}
