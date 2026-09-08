from unittest.mock import MagicMock, patch

from langchain_core.documents import Document

_FAKE_SOURCE = Document(
    page_content="RAG combines retrieval with generation.",
    metadata={"source": "guide.pdf", "page": 1},
)


def _ask(client, payload):
    return client.post("/chat", json=payload)


# ---------------------------------------------------------------------------
# Happy-path
# ---------------------------------------------------------------------------

class TestChatSuccess:
    def test_basic_question(self, client):
        mock_chain = MagicMock()
        with patch("routers.chat.build_rag_chain", return_value=mock_chain), \
             patch("routers.chat.ask", return_value=("RAG is great.", [_FAKE_SOURCE])):
            r = _ask(client, {"tenant_id": "acme", "question": "What is RAG?"})
        assert r.status_code == 200
        body = r.json()
        assert body["answer"] == "RAG is great."
        assert body["tenant_id"] == "acme"
        assert body["question"] == "What is RAG?"
        assert len(body["sources"]) == 1
        assert "RAG combines" in body["sources"][0]["content"]

    def test_no_sources(self, client):
        mock_chain = MagicMock()
        with patch("routers.chat.build_rag_chain", return_value=mock_chain), \
             patch("routers.chat.ask", return_value=("I don't know.", [])):
            r = _ask(client, {"tenant_id": "acme", "question": "Unrelated question?"})
        assert r.status_code == 200
        assert r.json()["sources"] == []

    def test_custom_k_is_forwarded(self, client):
        mock_chain = MagicMock()
        with patch("routers.chat.build_rag_chain", return_value=mock_chain) as mock_build, \
             patch("routers.chat.ask", return_value=("answer", [])):
            _ask(client, {"tenant_id": "acme", "question": "Q?", "k": 8})
        mock_build.assert_called_once_with("acme", k=8)

    def test_default_k_is_four(self, client):
        mock_chain = MagicMock()
        with patch("routers.chat.build_rag_chain", return_value=mock_chain) as mock_build, \
             patch("routers.chat.ask", return_value=("answer", [])):
            _ask(client, {"tenant_id": "acme", "question": "Q?"})
        mock_build.assert_called_once_with("acme", k=4)


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------

class TestChatValidation:
    def test_empty_question(self, client):
        r = _ask(client, {"tenant_id": "acme", "question": ""})
        assert r.status_code == 422

    def test_whitespace_only_question(self, client):
        r = _ask(client, {"tenant_id": "acme", "question": "   "})
        assert r.status_code == 422

    def test_question_too_long(self, client):
        r = _ask(client, {"tenant_id": "acme", "question": "a" * 2001})
        assert r.status_code == 422

    def test_question_at_max_length_is_accepted(self, client):
        mock_chain = MagicMock()
        with patch("routers.chat.build_rag_chain", return_value=mock_chain), \
             patch("routers.chat.ask", return_value=("ok", [])):
            r = _ask(client, {"tenant_id": "acme", "question": "a" * 2000})
        assert r.status_code == 200

    def test_empty_tenant_id(self, client):
        r = _ask(client, {"tenant_id": "", "question": "What is RAG?"})
        assert r.status_code == 422

    def test_invalid_tenant_id_chars(self, client):
        r = _ask(client, {"tenant_id": "bad tenant!", "question": "Q?"})
        assert r.status_code == 422

    def test_tenant_id_too_long(self, client):
        r = _ask(client, {"tenant_id": "a" * 65, "question": "Q?"})
        assert r.status_code == 422

    def test_missing_question(self, client):
        r = _ask(client, {"tenant_id": "acme"})
        assert r.status_code == 422

    def test_missing_tenant_id(self, client):
        r = _ask(client, {"question": "What is RAG?"})
        assert r.status_code == 422


# ---------------------------------------------------------------------------
# Service error propagation
# ---------------------------------------------------------------------------

class TestChatServiceErrors:
    def test_chain_build_failure_returns_500(self, client):
        with patch("routers.chat.build_rag_chain", side_effect=RuntimeError("no vectors")):
            r = _ask(client, {"tenant_id": "acme", "question": "Q?"})
        assert r.status_code == 500
        assert "no vectors" in r.json()["detail"]

    def test_ask_failure_returns_500(self, client):
        mock_chain = MagicMock()
        with patch("routers.chat.build_rag_chain", return_value=mock_chain), \
             patch("routers.chat.ask", side_effect=RuntimeError("LLM timeout")):
            r = _ask(client, {"tenant_id": "acme", "question": "Q?"})
        assert r.status_code == 500
        assert "LLM timeout" in r.json()["detail"]
