"""
services/ingestion.py

Loads files from PDF, DOCX, CSV, Excel, plain text, JSON, images, or a SQL
database and splits them into chunks ready for embedding.
"""

import base64
import io
import json as json_lib
from pathlib import Path
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


def load_text(file_path: str) -> List[Document]:
    with open(file_path, "r", encoding="utf-8", errors="replace") as f:
        content = f.read()
    return [Document(page_content=content, metadata={"source_type": "text"})]


def load_json(file_path: str) -> List[Document]:
    with open(file_path, "r", encoding="utf-8") as f:
        data = json_lib.load(f)
    content = json_lib.dumps(data, indent=2, ensure_ascii=False)
    return [Document(page_content=content, metadata={"source_type": "json"})]


def load_image(file_path: str) -> List[Document]:
    from anthropic import Anthropic
    from PIL import Image

    client = Anthropic()
    ext = Path(file_path).suffix.lower()

    # Anthropic Vision supports jpeg/png/gif/webp — convert others to PNG first
    if ext in (".bmp", ".tiff", ".tif"):
        img = Image.open(file_path)
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        image_data = base64.standard_b64encode(buf.getvalue()).decode("utf-8")
        media_type = "image/png"
    else:
        with open(file_path, "rb") as f:
            image_data = base64.standard_b64encode(f.read()).decode("utf-8")
        media_type = {
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".png": "image/png",
            ".gif": "image/gif",
            ".webp": "image/webp",
        }.get(ext, "image/png")

    message = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=4096,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": media_type,
                            "data": image_data,
                        },
                    },
                    {
                        "type": "text",
                        "text": (
                            "Extract and describe all content from this image in detail. "
                            "Transcribe any visible text verbatim. Describe charts, tables, "
                            "diagrams, and other visual elements with their data."
                        ),
                    },
                ],
            }
        ],
    )

    content = message.content[0].text
    return [Document(page_content=content, metadata={"source_type": "image"})]


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
        "text":     load_text,
        "json":     load_json,
        "image":    load_image,
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
