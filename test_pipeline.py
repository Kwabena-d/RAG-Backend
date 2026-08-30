"""
test_pipeline.py

End-to-end smoke test with tenant_id="test".
No auth, no HTTP — calls the service functions directly.

Steps:
  1. Enable pgvector extension
  2. Create test documents from a hardcoded string
  3. Split into chunks
  4. Store in pgvector under tenant "test"
  5. Run a similarity search
  6. Ask Claude a question via the full RAG chain

Run with:
    python test_pipeline.py
"""

from dotenv import load_dotenv
load_dotenv()

from langchain_core.documents import Document

from core.database import init_db
from services.ingestion import split_documents
from services.vector_store import store_chunks, similarity_search
from services.rag_chain import build_rag_chain, ask

TENANT = "test"

SAMPLE_TEXT = """
Artificial Intelligence (AI) refers to the simulation of human intelligence in machines.
Machine learning is a subset of AI that allows systems to learn from data automatically.
Deep learning uses neural networks with many layers to analyse complex patterns.
Large Language Models (LLMs) such as Claude are trained on vast text datasets and can
generate human-like text, answer questions, summarise documents, and write code.
Retrieval-Augmented Generation (RAG) combines a retrieval system with an LLM so the
model can answer questions grounded in specific documents rather than relying solely on
its training data. RAG reduces hallucinations and keeps answers up to date.
"""


def main():
    print("=== Step 1: Initialising database (pgvector extension) ===")
    init_db()
    print("Done.\n")

    print("=== Step 2: Creating test documents ===")
    docs = [Document(page_content=SAMPLE_TEXT, metadata={"source_type": "test"})]
    print(f"Created {len(docs)} document(s).\n")

    print("=== Step 3: Splitting into chunks ===")
    chunks = split_documents(docs)
    print(f"Split into {len(chunks)} chunk(s).\n")

    print("=== Step 4: Storing chunks in pgvector (tenant='{TENANT}') ===")
    count = store_chunks(chunks, TENANT)
    print(f"Stored {count} chunk(s).\n")

    print("=== Step 5: Similarity search ===")
    query = "What is RAG?"
    results = similarity_search(query, TENANT, k=2)
    print(f"Query: '{query}'")
    for i, doc in enumerate(results, 1):
        print(f"  Result {i}: {doc.page_content[:120]}...")
    print()

    print("=== Step 6: Full RAG chain — asking Claude ===")
    chain = build_rag_chain(TENANT, k=3)
    answer, sources = ask(chain, "Explain what RAG is and why it reduces hallucinations.")
    print(f"Answer:\n{answer}\n")
    print(f"Sources used: {len(sources)}")
    print("\n=== All steps passed ===")


if __name__ == "__main__":
    main()
