"""
frontend/app.py

Streamlit frontend for the RAG Backend API.

Run with:
    streamlit run frontend/app.py
"""

import requests
import os
import streamlit as st
# Locally falls back to localhost; on Streamlit Cloud set this secret.
API_BASE = os.getenv("API_BASE")

if not API_BASE:
    try:
        API_BASE = st.secrets["API_BASE"]
    except Exception:
        API_BASE = "http://localhost:8000"


# ── Helpers ────────────────────────────────────────────────────────────────────

def _api(method: str, path: str, **kwargs):
    """Call the API. Returns (response, None) or (None, error_message)."""
    try:
        r = requests.request(method, f"{API_BASE}{path}", timeout=120, **kwargs)
        return r, None
    except requests.exceptions.ConnectionError:
        return None, "Cannot reach the API — is uvicorn running on port 8000?"


def _file_size_label(n_bytes: int) -> str:
    if n_bytes >= 1024 * 1024:
        return f"{n_bytes / (1024 * 1024):.1f} MB"
    return f"{n_bytes / 1024:.1f} KB"


def _render_sources(sources: list) -> None:
    if not sources:
        return
    with st.expander(f"Sources ({len(sources)})"):
        for i, src in enumerate(sources, 1):
            meta = src.get("metadata", {})
            st.markdown(f"**Source {i}**")
            tags = []
            if meta.get("source_type"):
                tags.append(meta["source_type"].upper())
            if meta.get("page") is not None:
                tags.append(f"page {meta['page']}")
            if meta.get("table"):
                tags.append(f"table: {meta['table']}")
            if meta.get("sheet"):
                tags.append(f"sheet: {meta['sheet']}")
            if tags:
                st.caption("  ·  ".join(tags))
            st.text(src["content"])
            if i < len(sources):
                st.divider()


# ── Page config ────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="SkadVault AI",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Sidebar ────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("SkadVault AI")
    st.caption("Chat with your own documents.")

    st.divider()

    tenant_id = st.text_input(
        "Workspace ID",
        value=st.session_state.get("tenant_id", ""),
        placeholder="e.g. acme-corp",
        help="All uploads and conversations are scoped to this ID.",
    )
    st.session_state["tenant_id"] = tenant_id.strip()

    k = st.slider(
        "Sources to retrieve (k)",
        min_value=1,
        max_value=20,
        value=4,
        help="Number of document chunks retrieved per question.",
    )

    st.divider()

    r_health, _ = _api("GET", "/health")
    if r_health and r_health.status_code == 200:
        st.success("API online")
    else:
        st.error("API offline")

    st.divider()

    if st.button("Clear conversation", use_container_width=True):
        st.session_state["messages"] = []
        st.rerun()

# ── Guard: workspace required ──────────────────────────────────────────────────

if not st.session_state.get("tenant_id"):
    st.info("👈 Enter a **Workspace ID** in the sidebar to get started.")
    st.stop()

tenant_id = st.session_state["tenant_id"]

# ── Main tabs ──────────────────────────────────────────────────────────────────

ingest_tab, chat_tab = st.tabs(["📂 Ingest Documents", "💬 Chat"])

# ══════════════════════════════════════════════════════════════════════════════
# INGEST TAB
# ══════════════════════════════════════════════════════════════════════════════

with ingest_tab:
    st.subheader(f"Workspace · {tenant_id}")

    file_sub, db_sub = st.tabs(["Upload File", "Connect Database"])

    # ── File upload ──────────────────────────────────────────────────────────
    with file_sub:
        st.write("Upload a file to chunk, embed, and store it in your workspace.")

        uploaded = st.file_uploader(
            "Supported formats: PDF, DOCX, CSV, XLSX, XLS",
            type=["pdf", "docx", "csv", "xlsx", "xls"],
            label_visibility="visible",
        )

        if uploaded:
            col_info, col_btn = st.columns([4, 1])
            with col_info:
                st.caption(
                    f"**{uploaded.name}** · {_file_size_label(uploaded.size)}"
                )
            with col_btn:
                do_ingest = st.button("Ingest", type="primary", use_container_width=True)

            if do_ingest:
                with st.spinner(f"Processing {uploaded.name}…"):
                    r, err = _api(
                        "POST",
                        "/ingest/file",
                        data={"tenant_id": tenant_id},
                        files={
                            "file": (
                                uploaded.name,
                                uploaded.getvalue(),
                                uploaded.type or "application/octet-stream",
                            )
                        },
                    )

                if err:
                    st.error(err)
                elif r.status_code == 200:
                    body = r.json()
                    st.success(
                        f"**{body['filename']}** ingested — "
                        f"{body['chunks_stored']} chunks stored."
                    )
                else:
                    st.error(f"Error {r.status_code}: {r.json().get('detail', r.text)}")

    # ── Database ─────────────────────────────────────────────────────────────
    with db_sub:
        st.write(
            "Connect to a SQL database. Rows from every table will be "
            "embedded and stored in your workspace."
        )

        conn_str = st.text_input(
            "Connection string",
            placeholder="postgresql://user:password@host:5432/dbname",
            type="password",
        )
        max_rows = st.number_input(
            "Max rows per table",
            min_value=1,
            max_value=100_000,
            value=500,
            step=100,
            help="Cap on how many rows are read from each table.",
        )

        if st.button("Connect & Ingest", type="primary"):
            if not conn_str.strip():
                st.error("Enter a connection string.")
            else:
                with st.spinner("Connecting and indexing…"):
                    r, err = _api(
                        "POST",
                        "/ingest/database",
                        json={
                            "tenant_id": tenant_id,
                            "connection_string": conn_str,
                            "max_rows_per_table": int(max_rows),
                        },
                    )

                if err:
                    st.error(err)
                elif r.status_code == 200:
                    body = r.json()
                    st.success(f"Done — {body['chunks_stored']} chunks stored.")
                else:
                    st.error(f"Error {r.status_code}: {r.json().get('detail', r.text)}")

# ══════════════════════════════════════════════════════════════════════════════
# CHAT TAB
# ══════════════════════════════════════════════════════════════════════════════

with chat_tab:
    if "messages" not in st.session_state:
        st.session_state["messages"] = []

    # Render history
    for msg in st.session_state["messages"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] == "assistant":
                _render_sources(msg.get("sources", []))

    # Input
    question = st.chat_input("Ask a question about your documents…")

    if question:
        st.session_state["messages"].append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner("Thinking…"):
                r, err = _api(
                    "POST",
                    "/chat",
                    json={"tenant_id": tenant_id, "question": question, "k": k},
                )

            if err:
                st.error(err)
            elif r.status_code == 200:
                body = r.json()
                answer = body["answer"]
                sources = body.get("sources", [])
                st.markdown(answer)
                _render_sources(sources)
                st.session_state["messages"].append(
                    {"role": "assistant", "content": answer, "sources": sources}
                )
            else:
                detail = r.json().get("detail", r.text)
                st.error(f"Error {r.status_code}: {detail}")
