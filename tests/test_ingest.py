import io
from unittest.mock import patch

import pytest
from langchain_core.documents import Document

_FAKE_DOCS = [Document(page_content="Sample text", metadata={})]
_FAKE_CHUNKS = [Document(page_content="Sample chunk", metadata={})]

_PATCH_LOAD = patch("routers.ingest.load_source", return_value=_FAKE_DOCS)
_PATCH_SPLIT = patch("routers.ingest.split_documents", return_value=_FAKE_CHUNKS)


def _upload(client, tenant_id, filename, content=b"fake content"):
    return client.post(
        "/ingest/file",
        files={"file": (filename, io.BytesIO(content), "application/octet-stream")},
        data={"tenant_id": tenant_id},
    )


# ---------------------------------------------------------------------------
# Happy-path uploads
# ---------------------------------------------------------------------------

class TestIngestFileSuccess:
    def test_pdf(self, client):
        with _PATCH_LOAD, _PATCH_SPLIT, patch("routers.ingest.store_chunks", return_value=3):
            r = _upload(client, "acme", "report.pdf", b"%PDF-1.4 fake content")
        assert r.status_code == 200
        body = r.json()
        assert body["tenant_id"] == "acme"
        assert body["filename"] == "report.pdf"
        assert body["chunks_stored"] == 3
        assert body["source_type"] == "pdf"

    def test_docx(self, client):
        with _PATCH_LOAD, _PATCH_SPLIT, patch("routers.ingest.store_chunks", return_value=1):
            r = _upload(client, "tenant-1", "notes.docx")
        assert r.status_code == 200
        assert r.json()["source_type"] == "docx"

    def test_csv(self, client):
        with _PATCH_LOAD, _PATCH_SPLIT, patch("routers.ingest.store_chunks", return_value=5):
            r = _upload(client, "org_x", "data.csv", b"a,b\n1,2")
        assert r.status_code == 200
        assert r.json()["source_type"] == "csv"

    def test_xlsx(self, client):
        with _PATCH_LOAD, _PATCH_SPLIT, patch("routers.ingest.store_chunks", return_value=2):
            r = _upload(client, "org_x", "sheet.xlsx")
        assert r.status_code == 200
        assert r.json()["source_type"] == "excel"

    def test_xls(self, client):
        with _PATCH_LOAD, _PATCH_SPLIT, patch("routers.ingest.store_chunks", return_value=2):
            r = _upload(client, "org_x", "sheet.xls")
        assert r.status_code == 200
        assert r.json()["source_type"] == "excel"


# ---------------------------------------------------------------------------
# File-level validation
# ---------------------------------------------------------------------------

class TestIngestFileValidation:
    def test_unsupported_extension(self, client):
        r = _upload(client, "acme", "notes.txt")
        assert r.status_code == 400
        assert "Unsupported file type" in r.json()["detail"]

    def test_no_extension(self, client):
        r = _upload(client, "acme", "README")
        assert r.status_code == 400

    def test_empty_file(self, client):
        r = _upload(client, "acme", "empty.pdf", b"")
        assert r.status_code == 422
        assert "empty" in r.json()["detail"].lower()

    def test_missing_file_field(self, client):
        r = client.post("/ingest/file", data={"tenant_id": "acme"})
        assert r.status_code == 422

    def test_missing_tenant_id_field(self, client):
        r = client.post(
            "/ingest/file",
            files={"file": ("doc.pdf", io.BytesIO(b"content"), "application/pdf")},
        )
        assert r.status_code == 422


# ---------------------------------------------------------------------------
# tenant_id validation (Form field)
# ---------------------------------------------------------------------------

class TestIngestFileTenantValidation:
    def test_empty_string(self, client):
        r = _upload(client, "", "doc.pdf")
        assert r.status_code == 422

    def test_whitespace_only(self, client):
        r = _upload(client, "   ", "doc.pdf")
        assert r.status_code == 422

    def test_invalid_chars(self, client):
        r = _upload(client, "tenant/id", "doc.pdf")
        assert r.status_code == 422

    def test_too_long(self, client):
        r = _upload(client, "a" * 65, "doc.pdf")
        assert r.status_code == 422

    def test_valid_with_hyphens_and_underscores(self, client):
        with _PATCH_LOAD, _PATCH_SPLIT, patch("routers.ingest.store_chunks", return_value=1):
            r = _upload(client, "my-org_123", "doc.pdf", b"content")
        assert r.status_code == 200


# ---------------------------------------------------------------------------
# Service error propagation
# ---------------------------------------------------------------------------

class TestIngestFileServiceErrors:
    def test_load_failure_returns_500(self, client):
        with _PATCH_SPLIT, \
             patch("routers.ingest.load_source", side_effect=RuntimeError("parse error")), \
             patch("routers.ingest.store_chunks", return_value=0):
            r = _upload(client, "acme", "broken.pdf", b"content")
        assert r.status_code == 500
        assert "parse error" in r.json()["detail"]

    def test_store_failure_returns_500(self, client):
        with _PATCH_LOAD, _PATCH_SPLIT, \
             patch("routers.ingest.store_chunks", side_effect=RuntimeError("DB down")):
            r = _upload(client, "acme", "doc.pdf", b"content")
        assert r.status_code == 500


# ---------------------------------------------------------------------------
# /ingest/database endpoint
# ---------------------------------------------------------------------------

class TestIngestDatabase:
    def test_happy_path(self, client):
        with patch("services.ingestion.load_database", return_value=_FAKE_DOCS), \
             patch("routers.ingest.split_documents", return_value=_FAKE_CHUNKS), \
             patch("routers.ingest.store_chunks", return_value=10):
            r = client.post("/ingest/database", json={
                "tenant_id": "acme",
                "connection_string": "postgresql://user:pass@localhost/db",
            })
        assert r.status_code == 200
        body = r.json()
        assert body["tenant_id"] == "acme"
        assert body["chunks_stored"] == 10

    def test_custom_max_rows(self, client):
        with patch("services.ingestion.load_database", return_value=_FAKE_DOCS) as mock_load, \
             patch("routers.ingest.split_documents", return_value=_FAKE_CHUNKS), \
             patch("routers.ingest.store_chunks", return_value=1):
            client.post("/ingest/database", json={
                "tenant_id": "acme",
                "connection_string": "postgresql://user:pass@localhost/db",
                "max_rows_per_table": 100,
            })
        mock_load.assert_called_once_with("postgresql://user:pass@localhost/db", 100)

    def test_empty_tenant_id(self, client):
        r = client.post("/ingest/database", json={
            "tenant_id": "",
            "connection_string": "postgresql://user:pass@localhost/db",
        })
        assert r.status_code == 422

    def test_invalid_tenant_id_chars(self, client):
        r = client.post("/ingest/database", json={
            "tenant_id": "bad tenant",
            "connection_string": "postgresql://user:pass@localhost/db",
        })
        assert r.status_code == 422

    def test_empty_connection_string(self, client):
        r = client.post("/ingest/database", json={
            "tenant_id": "acme",
            "connection_string": "",
        })
        assert r.status_code == 422

    def test_whitespace_connection_string(self, client):
        r = client.post("/ingest/database", json={
            "tenant_id": "acme",
            "connection_string": "   ",
        })
        assert r.status_code == 422

    def test_missing_tenant_id(self, client):
        r = client.post("/ingest/database", json={
            "connection_string": "postgresql://user:pass@localhost/db",
        })
        assert r.status_code == 422

    def test_service_error_returns_500(self, client):
        with patch("services.ingestion.load_database", side_effect=RuntimeError("conn refused")):
            r = client.post("/ingest/database", json={
                "tenant_id": "acme",
                "connection_string": "postgresql://user:pass@localhost/db",
            })
        assert r.status_code == 500
        assert "conn refused" in r.json()["detail"]
