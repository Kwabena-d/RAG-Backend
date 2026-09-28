"""
frontend/app.py

SkadVault AI — unified conversational workspace
"""

import hashlib
import os
import re
from datetime import datetime, timedelta, timezone

import requests
import streamlit as st
from export import build_csv, build_docx, build_pdf

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


def _parse_dt(s: str) -> datetime:
    dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def _init_state() -> None:
    defaults: dict = {
        "tenant_id": "",
        "current_conv_id": None,
        "conv_messages": [],
        "conv_docs": [],
        "conv_list": [],
        "conv_list_tenant": "",
        "refresh_conv_list": False,
        "processed_doc_ids": set(),
        "export_bytes": None,
        "export_filename": "",
        "export_mime": "",
        "export_msg_count": 0,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def _load_conversation(conv_id: str, tenant_id: str) -> bool:
    """Load a conversation from the API and update session state. Returns True on success."""
    r, err = _api("GET", f"/conversations/{conv_id}", params={"tenant_id": tenant_id})
    if err or r is None:
        return False
    body, api_err = _parse_json(r)
    if api_err or body is None:
        return False

    st.session_state["current_conv_id"] = conv_id
    st.session_state["conv_messages"] = body.get("messages", [])
    st.session_state["conv_docs"] = body.get("documents", [])
    st.session_state["export_bytes"] = None
    st.session_state["export_msg_count"] = 0
    return True


def _create_conversation(tenant_id: str, title: str) -> str | None:
    """Create a new conversation via the API. Returns the conv_id or None on failure."""
    r, err = _api("POST", "/conversations", json={"tenant_id": tenant_id, "title": title})
    if err or r is None:
        return None
    body, api_err = _parse_json(r)
    if api_err or body is None:
        return None
    conv_id = body.get("id")
    if conv_id:
        st.session_state["refresh_conv_list"] = True
    return conv_id


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

    # New Chat button
    if st.button("＋  New Chat", use_container_width=True, type="primary"):
        st.session_state["current_conv_id"] = None
        st.session_state["conv_messages"] = []
        st.session_state["conv_docs"] = []
        st.session_state["export_bytes"] = None
        st.session_state["export_msg_count"] = 0
        st.rerun()

    st.divider()
    st.markdown("**Workspace**")

    raw = st.text_input(
        "Workspace Name",
        value=st.session_state.get("tenant_id", ""),
        placeholder="e.g. my-project",
        help="Letters, numbers, hyphens and underscores only. Files you add are stored in this workspace.",
        label_visibility="collapsed",
    )
    tenant_id = raw.strip()

    # Detect workspace change — reset conversation list cache
    if tenant_id != st.session_state.get("tenant_id", ""):
        st.session_state["conv_list_tenant"] = ""
        st.session_state["current_conv_id"] = None
        st.session_state["conv_messages"] = []
        st.session_state["conv_docs"] = []

    st.session_state["tenant_id"] = tenant_id

    st.divider()

    # Conversation list
    if tenant_id:
        needs_reload = (
            st.session_state.get("conv_list_tenant") != tenant_id
            or st.session_state.get("refresh_conv_list", False)
        )
        if needs_reload:
            r, err = _api("GET", "/conversations", params={"tenant_id": tenant_id})
            if not err and r is not None:
                body, api_err = _parse_json(r)
                if not api_err and isinstance(body, list):
                    st.session_state["conv_list"] = body
                    st.session_state["conv_list_tenant"] = tenant_id
            st.session_state["refresh_conv_list"] = False

        conv_list = st.session_state.get("conv_list", [])
        current_conv_id = st.session_state.get("current_conv_id")

        if conv_list:
            now_utc = datetime.now(timezone.utc)
            today_start = now_utc.replace(hour=0, minute=0, second=0, microsecond=0)
            yesterday_start = today_start - timedelta(days=1)
            week_start = today_start - timedelta(days=7)

            groups: dict[str, list] = {
                "Today": [],
                "Yesterday": [],
                "Previous 7 days": [],
                "Older": [],
            }

            for conv in conv_list:
                try:
                    updated = _parse_dt(conv["updated_at"])
                except Exception:
                    updated = now_utc

                if updated >= today_start:
                    groups["Today"].append(conv)
                elif updated >= yesterday_start:
                    groups["Yesterday"].append(conv)
                elif updated >= week_start:
                    groups["Previous 7 days"].append(conv)
                else:
                    groups["Older"].append(conv)

            for group_name, items in groups.items():
                if not items:
                    continue
                st.caption(group_name)
                for conv in items:
                    label = conv["title"]
                    if len(label) > 35:
                        label = label[:35] + "…"
                    is_active = conv["id"] == current_conv_id
                    btn_type = "primary" if is_active else "secondary"
                    if st.button(
                        label,
                        key=f"conv_{conv['id']}",
                        use_container_width=True,
                        type=btn_type,
                    ):
                        _load_conversation(conv["id"], tenant_id)
                        st.rerun()

        st.divider()

    # Connect Database expander (sidebar)
    with st.expander("🗄️ Connect Database"):
        conn_str = st.text_input(
            "Connection string",
            placeholder="postgresql://user:pass@host:5432/db",
            type="password",
            key="db_conn_str_sidebar",
        )
        max_rows = st.number_input(
            "Max rows per table",
            min_value=1, max_value=100_000, value=500, step=100,
            key="db_max_rows_sidebar",
        )
        if st.button("Connect & Index", type="primary", use_container_width=True, key="db_connect_sidebar"):
            if not tenant_id:
                st.error("Enter a Workspace Name first.")
            elif not conn_str.strip():
                st.error("Enter a connection string.")
            else:
                current_conv_id_sidebar = st.session_state.get("current_conv_id")
                if not current_conv_id_sidebar:
                    new_id = _create_conversation(tenant_id, "Database Connection")
                    if new_id:
                        st.session_state["current_conv_id"] = new_id
                        current_conv_id_sidebar = new_id
                    else:
                        st.error("Could not create a conversation. Please try again.")
                        current_conv_id_sidebar = None

                if current_conv_id_sidebar:
                    with st.spinner("Connecting…"):
                        r, err = _api(
                            "POST", "/ingest/database",
                            json={
                                "tenant_id": tenant_id,
                                "connection_string": conn_str,
                                "max_rows_per_table": int(max_rows),
                                "conversation_id": current_conv_id_sidebar,
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
                            _load_conversation(current_conv_id_sidebar, tenant_id)
                            st.rerun()

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

current_conv_id = st.session_state.get("current_conv_id")
conv_messages = st.session_state.get("conv_messages", [])

if current_conv_id is None and len(conv_messages) == 0:
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

for msg in st.session_state["conv_messages"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant":
            _render_sources(msg.get("sources", []))

# ─────────────────────────────────────────────────────────────────────────────
# Export (only shown when there is at least one assistant message)
# ─────────────────────────────────────────────────────────────────────────────

if any(m["role"] == "assistant" for m in st.session_state["conv_messages"]):
    with st.expander("📥 Export"):
        col_fmt, col_scope = st.columns(2)
        with col_fmt:
            export_fmt = st.radio(
                "Format",
                ["PDF", "Word (.docx)", "CSV"],
                key="export_fmt",
            )
        with col_scope:
            export_scope = st.radio(
                "Scope",
                ["Last answer", "Full conversation"],
                key="export_scope",
            )

        current_msg_count = len(st.session_state["conv_messages"])
        if (
            st.session_state["export_bytes"] is not None
            and st.session_state["export_msg_count"] != current_msg_count
        ):
            st.caption("Conversation has changed — click Generate to refresh.")

        if st.button("Generate export", key="export_generate"):
            scope_key = "last" if export_scope == "Last answer" else "full"
            workspace_safe = re.sub(r"[^a-zA-Z0-9_-]", "_", tenant_id)
            ts_safe = datetime.now().strftime("%Y%m%d_%H%M%S")
            try:
                if export_fmt == "PDF":
                    data = build_pdf(st.session_state["conv_messages"], scope_key, tenant_id)
                    fname = f"skadvault_{workspace_safe}_{ts_safe}.pdf"
                    mime = "application/pdf"
                elif export_fmt == "Word (.docx)":
                    data = build_docx(st.session_state["conv_messages"], scope_key, tenant_id)
                    fname = f"skadvault_{workspace_safe}_{ts_safe}.docx"
                    mime = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
                else:
                    data = build_csv(st.session_state["conv_messages"], scope_key, tenant_id)
                    fname = f"skadvault_{workspace_safe}_{ts_safe}.csv"
                    mime = "text/csv"

                st.session_state["export_bytes"] = data
                st.session_state["export_filename"] = fname
                st.session_state["export_mime"] = mime
                st.session_state["export_msg_count"] = current_msg_count

            except Exception:
                st.error("Unable to generate the export. Please try again.")

        if st.session_state["export_bytes"] is not None:
            st.download_button(
                "⬇ Download",
                data=st.session_state["export_bytes"],
                file_name=st.session_state["export_filename"],
                mime=st.session_state["export_mime"],
                key="export_download",
            )

# ─────────────────────────────────────────────────────────────────────────────
# Attachment chips — compact status row above the composer
# ─────────────────────────────────────────────────────────────────────────────

conv_docs = st.session_state.get("conv_docs", [])
if conv_docs:
    chips_html = " &nbsp; ".join(
        f'<span style="background:#f0f2f6;border-radius:999px;padding:3px 12px;'
        f'font-size:0.82em;white-space:nowrap">'
        f'{_icon(d["filename"])} {d["filename"]} ✓</span>'
        for d in conv_docs
    )
    st.markdown(f'<div style="margin-bottom:4px">{chips_html}</div>', unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Unified composer — native file attachment built into the chat input
# The + button is provided by Streamlit 1.44+ natively via accept_file.
# ─────────────────────────────────────────────────────────────────────────────

submission = st.chat_input(
    "Ask about your files…",
    accept_file="multiple",
    file_type=SUPPORTED_TYPES,
)

if submission is not None:
    files = submission.files   # list[UploadedFile], empty when only text sent
    text = submission.text.strip()

    # ── Ingest any attached files ────────────────────────────────────────────
    any_new_file = False
    for uploaded in files:
        file_bytes = uploaded.getvalue()
        doc_id = hashlib.md5(file_bytes).hexdigest()
        icon = _icon(uploaded.name)

        conv_doc_ids = {d["doc_id"] for d in st.session_state["conv_docs"]}
        if doc_id in conv_doc_ids or doc_id in st.session_state["processed_doc_ids"]:
            continue  # already ingested — no duplicate

        # Lazy conversation creation on first attachment
        if not st.session_state.get("current_conv_id"):
            new_id = _create_conversation(tenant_id, uploaded.name[:50])
            if not new_id:
                st.error("Could not start a conversation. Please try again.")
                continue
            st.session_state["current_conv_id"] = new_id

        with st.spinner(f"Processing {icon} {uploaded.name}…"):
            r, err = _api(
                "POST", "/ingest/file",
                data={
                    "tenant_id": tenant_id,
                    "conversation_id": st.session_state["current_conv_id"],
                },
                files={
                    "file": (uploaded.name, file_bytes, uploaded.type or "application/octet-stream")
                },
            )

        if err:
            st.error(f"Could not process {uploaded.name}. {err}")
        else:
            body, api_err = _parse_json(r)
            if api_err:
                st.error(f"Upload failed for {uploaded.name}. {api_err}")
            else:
                returned_doc_id = body.get("doc_id", doc_id)
                st.session_state["processed_doc_ids"].add(returned_doc_id)
                _load_conversation(st.session_state["current_conv_id"], tenant_id)
                any_new_file = True

    # Files-only submission: rerun to refresh chips and sidebar
    if files and not text:
        if any_new_file:
            st.session_state["refresh_conv_list"] = True
        st.rerun()

    # ── RAG query ────────────────────────────────────────────────────────────
    if text:
        # Lazy conversation creation on first message
        if not st.session_state.get("current_conv_id"):
            new_id = _create_conversation(tenant_id, text[:50])
            if not new_id:
                st.error("Could not start conversation. Please try again.")
                st.stop()
            st.session_state["current_conv_id"] = new_id

        current_conv_id = st.session_state["current_conv_id"]

        st.session_state["conv_messages"].append({"role": "user", "content": text, "sources": []})

        with st.chat_message("user"):
            st.markdown(text)

        with st.chat_message("assistant"):
            with st.spinner("Thinking…"):
                r, err = _api("POST", "/chat", json={
                    "tenant_id": tenant_id,
                    "question": text,
                    "conversation_id": current_conv_id,
                })

            if err:
                msg = f"Service temporarily unavailable. {err}"
                st.error(msg)
                st.session_state["conv_messages"].append({"role": "assistant", "content": msg, "sources": []})
            else:
                body, api_err = _parse_json(r)
                if api_err:
                    st.error(api_err)
                    st.session_state["conv_messages"].append({"role": "assistant", "content": api_err, "sources": []})
                else:
                    answer = body.get("answer", "")
                    sources = body.get("sources", [])
                    st.markdown(answer)
                    _render_sources(sources)
                    st.session_state["conv_messages"].append({"role": "assistant", "content": answer, "sources": sources})

        st.session_state["refresh_conv_list"] = True
