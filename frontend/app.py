"""
frontend/app.py

SkadVault AI — unified conversational workspace
"""

import hashlib
import os

import requests
import streamlit as st

# ─────────────────────────────────────────────────────────────────────────────
# API configuration
# ─────────────────────────────────────────────────────────────────────────────

API_BASE = os.getenv("API_BASE")
if not API_BASE:
    try:
        API_BASE = st.secrets["API_BASE"]
    except Exception:
        API_BASE = "http://localhost:8000"

# ─────────────────────────────────────────────────────────────────────────────
# File type constants
# ─────────────────────────────────────────────────────────────────────────────

SUPPORTED_TYPES = [
    "pdf", "docx",
    "csv", "xlsx", "xls", "txt", "md", "json",
    "jpg", "jpeg", "png", "gif", "webp", "bmp", "tiff", "tif",
]

IMAGE_EXTS = frozenset(["jpg", "jpeg", "png", "gif", "webp", "bmp", "tiff", "tif"])

_ICONS: dict[str, str] = {
    "pdf": "📄", "docx": "📝",
    "csv": "📊", "xlsx": "📊", "xls": "📊",
    "txt": "📃", "md": "📃", "json": "📋",
}
for _e in IMAGE_EXTS:
    _ICONS[_e] = "🖼️"

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _icon(filename: str) -> str:
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    return _ICONS.get(ext, "📁")


def _size_label(n: int) -> str:
    return f"{n / 1_048_576:.1f} MB" if n >= 1_048_576 else f"{n / 1024:.1f} KB"


def _file_key(tenant_id: str, name: str, size: int) -> str:
    """Stable key for a (workspace, file) pair — prevents re-ingestion on reruns."""
    return hashlib.md5(f"{tenant_id}||{name}||{size}".encode()).hexdigest()


def _api(method: str, path: str, **kwargs):
    try:
        r = requests.request(method, f"{API_BASE}{path}", timeout=300, **kwargs)
        return r, None
    except requests.exceptions.ConnectionError:
        return None, "Cannot reach the server. Please check that it is running."
    except requests.exceptions.Timeout:
        return None, "Processing is taking longer than expected. Please try again."
    except Exception as exc:
        return None, str(exc)


def _parse_json(r) -> tuple:
    """Returns (body_dict, None) on success or (None, error_str) on failure."""
    try:
        body = r.json()
    except Exception:
        body = None
    if r.status_code == 200:
        return (body or {}), None
    detail = (body.get("detail") if isinstance(body, dict) else None) or r.text[:300]
    return None, f"Error {r.status_code}: {detail}"


def _render_sources(sources: list) -> None:
    if not sources:
        return
    n = len(sources)
    with st.expander(f"Sources · {n} chunk{'s' if n != 1 else ''}"):
        for i, src in enumerate(sources, 1):
            meta = src.get("metadata", {})
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
            st.text(src.get("content", ""))
            if i < n:
                st.divider()


def _init_state() -> None:
    defaults: dict = {
        "messages": [],
        "ingested_files": {},   # fk -> {tenant_id, name, size, chunks}
        "file_errors": {},      # fk -> {tenant_id, name, size, error}
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

# ─────────────────────────────────────────────────────────────────────────────
# Page setup
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="SkadVault AI",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)
_init_state()

# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("SkadVault AI")
    st.caption("Your knowledge. Your answers.")
    st.divider()

    raw = st.text_input(
        "Workspace Name",
        value=st.session_state.get("tenant_id", ""),
        placeholder="e.g. my-project",
        help="Letters, numbers, hyphens and underscores only. Files you add are stored in this workspace.",
    )
    tenant_id = raw.strip()
    st.session_state["tenant_id"] = tenant_id

    st.divider()

    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("New", use_container_width=True, help="Start a new conversation"):
            st.session_state["messages"] = []
            st.rerun()
    with col_b:
        if st.button("Clear", use_container_width=True, help="Clear conversation history"):
            st.session_state["messages"] = []
            st.rerun()

    st.divider()

    with st.expander("🗄️ Connect Database"):
        conn_str = st.text_input(
            "Connection string",
            placeholder="postgresql://user:pass@host:5432/db",
            type="password",
            key="db_conn_str",
        )
        max_rows = st.number_input(
            "Max rows per table",
            min_value=1, max_value=100_000, value=500, step=100,
            key="db_max_rows",
        )
        if st.button("Connect & Index", type="primary", use_container_width=True):
            if not tenant_id:
                st.error("Enter a Workspace Name first.")
            elif not conn_str.strip():
                st.error("Enter a connection string.")
            else:
                with st.spinner("Connecting…"):
                    r, err = _api(
                        "POST", "/ingest/database",
                        json={
                            "tenant_id": tenant_id,
                            "connection_string": conn_str,
                            "max_rows_per_table": int(max_rows),
                        },
                    )
                if err:
                    st.error(err)
                else:
                    body, api_err = _parse_json(r)
                    if api_err:
                        st.error(api_err)
                    else:
                        st.success(f"Done — {body.get('chunks_stored', '?')} chunks indexed.")

# ─────────────────────────────────────────────────────────────────────────────
# Workspace guard
# ─────────────────────────────────────────────────────────────────────────────

if not tenant_id:
    st.markdown(
        "<div style='text-align:center;padding:5rem 1rem'>"
        "<h1>SkadVault AI</h1>"
        "<p style='color:#888;font-size:1.1rem'>Your knowledge. Your answers.</p>"
        "<br><p>Enter a <strong>Workspace Name</strong> in the sidebar to get started.</p>"
        "</div>",
        unsafe_allow_html=True,
    )
    st.stop()

# ─────────────────────────────────────────────────────────────────────────────
# Welcome state
# ─────────────────────────────────────────────────────────────────────────────

if not st.session_state["messages"]:
    st.markdown(
        "<div style='text-align:center;padding:3rem 1rem 2rem'>"
        "<h1>SkadVault AI</h1>"
        "<p style='color:#888;font-size:1.1rem'>Your knowledge. Your answers.</p>"
        "<p style='color:#aaa;margin-top:1.5rem'>"
        "Attach a file below, then ask questions about it."
        "</p>"
        "</div>",
        unsafe_allow_html=True,
    )

# ─────────────────────────────────────────────────────────────────────────────
# Chat history
# ─────────────────────────────────────────────────────────────────────────────

for msg in st.session_state["messages"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant":
            _render_sources(msg.get("sources", []))

# ─────────────────────────────────────────────────────────────────────────────
# Workspace file strip
# ─────────────────────────────────────────────────────────────────────────────

done = [v for v in st.session_state["ingested_files"].values() if v["tenant_id"] == tenant_id]
failed = [(k, v) for k, v in st.session_state["file_errors"].items() if v["tenant_id"] == tenant_id]

if done or failed:
    st.caption(f"**{tenant_id}** — files in workspace")
    if done:
        n_cols = min(len(done), 5)
        cols = st.columns(n_cols)
        for i, info in enumerate(done):
            with cols[i % n_cols]:
                st.success(f"{_icon(info['name'])} {info['name']}", icon=None)
    for fk, info in failed:
        c1, c2 = st.columns([5, 1])
        with c1:
            st.error(f"{_icon(info['name'])} {info['name']} — {info['error'][:120]}")
        with c2:
            if st.button("✕", key=f"dismiss_{fk}", help="Dismiss error"):
                del st.session_state["file_errors"][fk]
                st.rerun()

st.divider()

# ─────────────────────────────────────────────────────────────────────────────
# File attachment
# ─────────────────────────────────────────────────────────────────────────────

st.markdown("**📎 Attach a file**")
uploaded = st.file_uploader(
    "PDF, DOCX, CSV, XLSX, XLS, TXT, MD, JSON · Images: JPG, PNG, GIF, WEBP, BMP, TIFF",
    type=SUPPORTED_TYPES,
    label_visibility="visible",
    key="file_uploader",
)

if uploaded:
    fk = _file_key(tenant_id, uploaded.name, uploaded.size)
    ext = uploaded.name.rsplit(".", 1)[-1].lower() if "." in uploaded.name else ""
    icon = _icon(uploaded.name)
    size_lbl = _size_label(uploaded.size)

    if fk in st.session_state["ingested_files"]:
        # Already successfully processed — show status only
        info = st.session_state["ingested_files"][fk]
        st.success(
            f"{icon} **{uploaded.name}** · {size_lbl} · "
            f"{info['chunks']} chunks · ✓ Ready"
        )
        if ext in IMAGE_EXTS:
            try:
                st.image(uploaded.getvalue(), width=200)
            except Exception:
                pass

    elif fk in st.session_state["file_errors"]:
        # Previously failed — show error with retry option
        err_info = st.session_state["file_errors"][fk]
        c1, c2 = st.columns([5, 1])
        with c1:
            st.error(f"{icon} **{uploaded.name}** — {err_info['error']}")
        with c2:
            if st.button("Retry", key=f"retry_{fk}"):
                del st.session_state["file_errors"][fk]
                st.rerun()

    else:
        # New file — preview then auto-process
        st.info(f"{icon} **{uploaded.name}** · {size_lbl}")
        if ext in IMAGE_EXTS:
            try:
                st.image(uploaded.getvalue(), width=200)
            except Exception:
                pass

        with st.spinner(f"Processing {uploaded.name}…"):
            r, err = _api(
                "POST", "/ingest/file",
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
            st.session_state["file_errors"][fk] = {
                "tenant_id": tenant_id, "name": uploaded.name,
                "size": uploaded.size, "error": err,
            }
            st.error(f"SkadVault AI couldn't process this file. {err}")
        else:
            body, api_err = _parse_json(r)
            if api_err:
                st.session_state["file_errors"][fk] = {
                    "tenant_id": tenant_id, "name": uploaded.name,
                    "size": uploaded.size, "error": api_err,
                }
                st.error(f"Upload failed. {api_err}")
            else:
                st.session_state["ingested_files"][fk] = {
                    "tenant_id": tenant_id, "name": uploaded.name,
                    "size": uploaded.size, "chunks": body.get("chunks_stored", 0),
                }
                st.rerun()

# ─────────────────────────────────────────────────────────────────────────────
# Chat input
# ─────────────────────────────────────────────────────────────────────────────

question = st.chat_input("Ask about your files…")

if question:
    st.session_state["messages"].append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Thinking…"):
            r, err = _api(
                "POST", "/chat",
                json={"tenant_id": tenant_id, "question": question},
            )

        if err:
            msg = f"Service temporarily unavailable. {err}"
            st.error(msg)
            st.session_state["messages"].append(
                {"role": "assistant", "content": msg, "sources": []}
            )
        else:
            body, api_err = _parse_json(r)
            if api_err:
                st.error(api_err)
                st.session_state["messages"].append(
                    {"role": "assistant", "content": api_err, "sources": []}
                )
            else:
                answer = body.get("answer", "")
                sources = body.get("sources", [])
                st.markdown(answer)
                _render_sources(sources)
                st.session_state["messages"].append(
                    {"role": "assistant", "content": answer, "sources": sources}
                )
