"""
services/rag_chain.py

Builds the RAG pipeline on top of the pgvector store:
  pgvector retriever → prompt + retrieved context → Claude → answer

Public API:
    build_rag_chain(tenant_id, k)   — create the chain for a tenant
    ask(chain, question)            — run a question, return (answer, sources)
"""

from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains import create_retrieval_chain
from services.vector_store import get_store

SYSTEM_PROMPT = (
    "You are a helpful assistant answering questions using ONLY the provided "
    "context. If the answer is not in the context, say you don't know rather "
    "than guessing.\n\nContext:\n{context}"
)

CLAUDE_MODEL = "claude-sonnet-4-6"


def build_rag_chain(tenant_id: str, k: int = 4, model_name: str = CLAUDE_MODEL):
    """Wire together a pgvector retriever and Claude into a retrieval chain."""
    store = get_store(tenant_id)
    retriever = store.as_retriever(search_kwargs={"k": k})

    llm = ChatAnthropic(model=model_name, temperature=0)

    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", "{input}"),
    ])

    combine_docs_chain = create_stuff_documents_chain(llm, prompt)
    return create_retrieval_chain(retriever, combine_docs_chain)


def ask(chain, question: str) -> tuple[str, list]:
    """Run a question through the chain. Returns (answer, source_documents)."""
    result = chain.invoke({"input": question})
    return result["answer"], result.get("context", [])
