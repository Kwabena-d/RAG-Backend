"""
services/ingestion.py

Loads documents from PDF, DOCX, CSV, Excel, or a SQL database and splits
them into chunks ready for embedding. Ported directly from document_loaders.py
and vector_store.py in the original Streamlit app.
"""

from typing import List
import pandas as pd
from langchain_core.documents import Document
from langchain_community.document_loaders import Docx2txtLoader, PyPDFLoader
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sqlalchemy import create_engine, inspect, text


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def load_pdf(file_path: str) -> List[Document]:
    loader = PyPDFLoader(file_path)
    docs = loader.load()
    for d in docs:
        d.metadata["source_type"] = "pdf"
    return docs


def load_docx(file_path: str) -> List[Document]:
    loader = Docx2txtLoader(file_path)
    docs = loader.load()
    for d in docs:
        d.metadata["source_type"] = "docx"
    return docs


def load_csv(file_path: str) -> List[Document]:
    loader = CSVLoader(file_path)
    docs = loader.load()
    for d in docs:
        d.metadata["source_type"] = "csv"
    return docs


def load_excel(file_path: str) -> List[Document]:
    docs: List[Document] = []
    with pd.ExcelFile(file_path) as xl:
        for sheet_name in xl.sheet_names:
            df = xl.parse(sheet_name)
            for _, row in df.iterrows():
                content = "\n".join(
                    f"{col}: {val}"
                    for col, val in row.items()
                    if pd.notna(val)
                )
                if content.strip():
                    docs.append(
                        Document(
                            page_content=content,
                            metadata={"source_type": "excel", "sheet": sheet_name},
                        )
                    )
    return docs


def load_database(connection_string: str, max_rows_per_table: int = 500) -> List[Document]:
    engine = create_engine(connection_string)
    inspector = inspect(engine)
    docs: List[Document] = []

    with engine.connect() as conn:
        for table_name in inspector.get_table_names():
            columns = [col["name"] for col in inspector.get_columns(table_name)]
            result = conn.execute(
                text(f'SELECT * FROM "{table_name}" LIMIT :limit'),
                {"limit": max_rows_per_table},
            )
            for row in result:
                row_dict = dict(zip(columns, row))
                content = "\n".join(f"{col}: {val}" for col, val in row_dict.items())
                docs.append(
                    Document(
                        page_content=content,
                        metadata={"source_type": "database", "table": table_name},
                    )
                )
    return docs


def load_source(source_type: str, source: str) -> List[Document]:
    dispatch = {
        "pdf":      load_pdf,
        "docx":     load_docx,
        "csv":      load_csv,
        "excel":    load_excel,
        "database": load_database,
    }
    if source_type not in dispatch:
        raise ValueError(f"Unsupported source_type: {source_type!r}")
    return dispatch[source_type](source)


# ---------------------------------------------------------------------------
# Splitter
# ---------------------------------------------------------------------------

def split_documents(
    docs: List[Document],
    chunk_size: int = 1000,
    chunk_overlap: int = 150,
) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    return splitter.split_documents(docs)
