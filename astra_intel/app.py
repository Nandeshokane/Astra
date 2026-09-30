"""
ASTRA INTEL — Main Application (Day 2)
=========================================
Streamlit UI with multi-document workspace, Cloud/Air-Gapped mode toggle,
grounding verification badges, comparative analysis, and Intel Dossier export.
"""

import streamlit as st
import traceback
from datetime import datetime, timezone

from core.document_loader import (
    load_pdf,
    get_doc_statistics,
    DocumentLoadError,
    ScannedOrEmptyPDFError,
    LoadedDocument,
)
from core.vector_store import AstraVectorStore
from core.llm_manager import LLMClient, InferenceMode
from core.rag_pipeline import (
    generate_executive_summary,
    answer_question,
    compare_documents,
    export_intel_dossier,
    AnswerResult,
)
from core.verification import GroundingLevel


# ── Page Config ─────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="ASTRA INTEL | Defence Document Intelligence",
    page_icon="\U0001f6e1\ufe0f",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ── Custom CSS — Tactical Dark Theme ───────────────────────────────────────

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;500;600;700&family=Inter:wght@300;400;500;600;700&display=swap');

:root {
    --bg-primary: #0a0f1a;
    --bg-secondary: #111827;
    --bg-card: #1a2332;
    --bg-card-hover: #1f2b3d;
    --border-color: #2d3a4d;
    --border-glow: #10b981;
    --text-primary: #e2e8f0;
    --text-secondary: #94a3b8;
    --text-muted: #64748b;
    --accent-emerald: #10b981;
    --accent-cyan: #06b6d4;
    --accent-amber: #f59e0b;
    --accent-red: #ef4444;
    --gradient-primary: linear-gradient(135deg, #10b981 0%, #06b6d4 100%);
}

.stApp {
    background-color: var(--bg-primary) !important;
    color: var(--text-primary) !important;
    font-family: 'Inter', sans-serif !important;
}

section[data-testid="stSidebar"] {
    background-color: var(--bg-secondary) !important;
    border-right: 1px solid var(--border-color) !important;
}

section[data-testid="stSidebar"] .stMarkdown {
    color: var(--text-primary) !important;
}

h1, h2, h3, h4, h5, h6 {
    font-family: 'JetBrains Mono', monospace !important;
    color: var(--text-primary) !important;
}

.astra-header {
    background: linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #0f172a 100%);
    border: 1px solid var(--border-color);
    border-left: 4px solid var(--accent-emerald);
    border-radius: 8px;
    padding: 1.5rem 2rem;
    margin-bottom: 1.5rem;
}

.astra-header h1 {
    background: var(--gradient-primary);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-size: 1.8rem !important;
    margin: 0 0 0.25rem 0 !important;
    letter-spacing: 0.05em;
}

.astra-header .subtitle {
    color: var(--text-secondary);
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.8rem;
    letter-spacing: 0.15em;
    text-transform: uppercase;
}

.status-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 4px 12px;
    border-radius: 20px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.75rem;
    font-weight: 500;
    letter-spacing: 0.05em;
}

.status-ready {
    background: rgba(16, 185, 129, 0.15);
    color: #10b981;
    border: 1px solid rgba(16, 185, 129, 0.3);
}

.status-error {
    background: rgba(239, 68, 68, 0.15);
    color: #ef4444;
    border: 1px solid rgba(239, 68, 68, 0.3);
}

.status-warning {
    background: rgba(245, 158, 11, 0.15);
    color: #f59e0b;
    border: 1px solid rgba(245, 158, 11, 0.3);
}

.meta-card {
    background: var(--bg-card);
    border: 1px solid var(--border-color);
    border-radius: 8px;
    padding: 1rem 1.2rem;
    margin: 0.5rem 0;
}

.meta-card .label {
    color: var(--text-muted);
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.7rem;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin-bottom: 4px;
}

.meta-card .value {
    color: var(--text-primary);
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.9rem;
    font-weight: 600;
}

.citation-badge {
    display: inline-flex;
    align-items: center;
    gap: 4px;
    background: rgba(6, 182, 212, 0.12);
    color: #06b6d4;
    border: 1px solid rgba(6, 182, 212, 0.25);
    border-radius: 6px;
    padding: 3px 10px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.75rem;
    font-weight: 500;
    margin: 2px 4px 2px 0;
}

.grounding-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 6px 14px;
    border-radius: 8px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.78rem;
    font-weight: 600;
    letter-spacing: 0.04em;
    margin: 8px 0;
}

.grounding-high {
    background: rgba(16, 185, 129, 0.12);
    color: #10b981;
    border: 1px solid rgba(16, 185, 129, 0.3);
}

.grounding-medium {
    background: rgba(245, 158, 11, 0.12);
    color: #f59e0b;
    border: 1px solid rgba(245, 158, 11, 0.3);
}

.grounding-low {
    background: rgba(249, 115, 22, 0.12);
    color: #f97316;
    border: 1px solid rgba(249, 115, 22, 0.3);
}

.grounding-ungrounded {
    background: rgba(239, 68, 68, 0.12);
    color: #ef4444;
    border: 1px solid rgba(239, 68, 68, 0.3);
}

.empty-state {
    text-align: center;
    padding: 4rem 2rem;
    color: var(--text-muted);
}

.empty-state .icon { font-size: 3rem; margin-bottom: 1rem; opacity: 0.5; }
.empty-state .title {
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.1rem;
    color: var(--text-secondary);
    margin-bottom: 0.5rem;
}
.empty-state .desc { font-size: 0.85rem; line-height: 1.6; }

.chat-msg-assistant {
    background: var(--bg-card);
    border: 1px solid var(--border-color);
    border-left: 3px solid var(--accent-emerald);
    border-radius: 8px;
    padding: 1rem 1.25rem;
    margin: 0.75rem 0;
}

.chat-msg-user {
    background: rgba(6, 182, 212, 0.08);
    border: 1px solid rgba(6, 182, 212, 0.2);
    border-left: 3px solid var(--accent-cyan);
    border-radius: 8px;
    padding: 1rem 1.25rem;
    margin: 0.75rem 0;
}

.sidebar-logo {
    text-align: center;
    padding: 0.75rem 0 1rem 0;
    border-bottom: 1px solid var(--border-color);
    margin-bottom: 1rem;
}

.sidebar-logo .logo-text {
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.4rem;
    font-weight: 700;
    background: var(--gradient-primary);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    letter-spacing: 0.1em;
}

.sidebar-logo .logo-sub {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.6rem;
    color: var(--text-muted);
    letter-spacing: 0.2em;
    text-transform: uppercase;
}

.doc-table {
    width: 100%;
    border-collapse: collapse;
    margin: 0.5rem 0;
    font-size: 0.78rem;
}

.doc-table th {
    background: var(--bg-card);
    color: var(--text-muted);
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.68rem;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    padding: 8px 10px;
    text-align: left;
    border-bottom: 1px solid var(--border-color);
}

.doc-table td {
    padding: 8px 10px;
    border-bottom: 1px solid rgba(45, 58, 77, 0.5);
    color: var(--text-primary);
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.78rem;
}

.mode-info {
    background: rgba(6, 182, 212, 0.06);
    border: 1px solid rgba(6, 182, 212, 0.15);
    border-radius: 6px;
    padding: 8px 12px;
    font-size: 0.72rem;
    color: var(--text-secondary);
    margin-top: 6px;
    line-height: 1.4;
}
</style>
""", unsafe_allow_html=True)


# ── Session State Init ──────────────────────────────────────────────────────

def _init_state():
    defaults = {
        "loaded_docs": {},          # {doc_name: LoadedDocument}
        "doc_stats_list": [],       # [{stats dict}, ...]
        "doc_summaries": {},        # {doc_name: summary_str}
        "vector_store": None,
        "llm_client": None,
        "chat_history": [],
        "processing": False,
        "inference_mode": "cloud",
        "compare_mode": False,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init_state()


# ── LLM Client Management ──────────────────────────────────────────────────

def _get_or_create_llm() -> LLMClient:
    """Get or create the LLM client matching the current inference mode."""
    mode = InferenceMode.CLOUD if st.session_state.inference_mode == "cloud" else InferenceMode.LOCAL
    if st.session_state.llm_client is None or st.session_state.llm_client.mode != mode:
        st.session_state.llm_client = LLMClient(mode=mode)
    return st.session_state.llm_client


# ── Sidebar ─────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("""
    <div class="sidebar-logo">
        <div class="logo-text">\U0001f6e1\ufe0f ASTRA INTEL</div>
        <div class="logo-sub">Defence Document Intelligence</div>
    </div>
    """, unsafe_allow_html=True)

    # ── Multi-File Upload ───────────────────────────────────────────────────
    st.markdown("#### \U0001f4c4 Document Upload")
    uploaded_files = st.file_uploader(
        "Upload classified PDF documents",
        type=["pdf"],
        accept_multiple_files=True,
        help="Upload one or more text-layer PDFs. Scanned image-only PDFs are not supported.",
        label_visibility="collapsed",
    )

    st.markdown("---")

    # ── Inference Mode Toggle ───────────────────────────────────────────────
    st.markdown("#### \u2699\ufe0f Inference Mode")
    mode_choice = st.radio(
        "Select inference mode",
        options=["cloud", "local"],
        format_func=lambda x: "\u2601\ufe0f Cloud API" if x == "cloud" else "\U0001f512 Local Air-Gapped",
        index=0 if st.session_state.inference_mode == "cloud" else 1,
        label_visibility="collapsed",
        horizontal=True,
    )
    if mode_choice != st.session_state.inference_mode:
        st.session_state.inference_mode = mode_choice
        st.session_state.llm_client = None  # force re-creation

    llm = _get_or_create_llm()
    status = llm.get_status()

    status_class = "status-ready" if status.status == "Ready" else "status-error"
    st.markdown(f"""
    <div class="meta-card">
        <div class="label">Mode</div>
        <div class="value" style="font-size:0.78rem;">{status.mode}</div>
    </div>
    <div class="meta-card">
        <div class="label">Provider / Model</div>
        <div class="value" style="font-size:0.75rem;">{status.provider} / {status.model}</div>
    </div>
    <div class="meta-card">
        <div class="label">Status</div>
        <div><span class="status-badge {status_class}">\u25cf {status.status}</span></div>
    </div>
    """, unsafe_allow_html=True)

    if mode_choice == "local":
        st.markdown("""
        <div class="mode-info">
            \U0001f512 <strong>Air-Gapped Mode</strong> — All inference runs locally via Ollama.
            No data leaves this machine. Required for classified documents.
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # ── Compare Mode Toggle ─────────────────────────────────────────────────
    if len(st.session_state.loaded_docs) > 1:
        st.markdown("#### \U0001f50d Analysis Mode")
        compare = st.toggle(
            "Cross-Document Comparison",
            value=st.session_state.compare_mode,
            help="Enable to ask comparative questions across all loaded documents.",
        )
        st.session_state.compare_mode = compare
        st.markdown("---")

    # ── Document Inventory ──────────────────────────────────────────────────
    st.markdown("#### \U0001f4ca Document Inventory")
    if st.session_state.doc_stats_list:
        table_rows = ""
        for s in st.session_state.doc_stats_list:
            name = s["doc_name"]
            if len(name) > 25:
                name = name[:22] + "..."
            table_rows += f"<tr><td>{name}</td><td>{s['total_pages']}</td><td>{s['total_chunks']}</td><td>{s['total_words']:,}</td></tr>"

        st.markdown(f"""
        <table class="doc-table">
            <thead><tr><th>Document</th><th>Pages</th><th>Chunks</th><th>Words</th></tr></thead>
            <tbody>{table_rows}</tbody>
        </table>
        """, unsafe_allow_html=True)

        if st.button("\U0001f5d1\ufe0f Clear All Documents", use_container_width=True):
            st.session_state.loaded_docs = {}
            st.session_state.doc_stats_list = []
            st.session_state.doc_summaries = {}
            st.session_state.chat_history = []
            if st.session_state.vector_store:
                st.session_state.vector_store.reset()
            st.rerun()
    else:
        st.markdown("""
        <div class="meta-card">
            <div class="label">Status</div>
            <div class="value" style="color: var(--text-muted);">No documents loaded</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # ── Export Intel Dossier ─────────────────────────────────────────────────
    st.markdown("#### \U0001f4e5 Export")
    if st.session_state.doc_stats_list:
        if st.button("\U0001f4cb Export Intel Dossier", use_container_width=True):
            dossier = export_intel_dossier(
                doc_summaries=st.session_state.doc_summaries,
                chat_history=st.session_state.chat_history,
                doc_stats=st.session_state.doc_stats_list,
            )
            st.download_button(
                label="\u2b07\ufe0f Download Dossier (.md)",
                data=dossier,
                file_name=f"ASTRA_INTEL_Dossier_{datetime.now().strftime('%Y%m%d_%H%M')}.md",
                mime="text/markdown",
                use_container_width=True,
            )


# ── Document Processing ────────────────────────────────────────────────────

def process_documents(files):
    """Ingest, chunk, embed, index, and summarise uploaded PDFs."""
    llm = _get_or_create_llm()

    # Create vector store if needed
    if st.session_state.vector_store is None:
        st.session_state.vector_store = AstraVectorStore()

    vs = st.session_state.vector_store

    for uploaded_file in files:
        doc_name = uploaded_file.name

        # Skip already-loaded documents
        if doc_name in st.session_state.loaded_docs:
            continue

        progress = st.status(f"\U0001f512 Processing: **{doc_name}**...", expanded=True)

        try:
            # Extract
            progress.write("\U0001f4e5 Extracting text from PDF...")
            raw_bytes = uploaded_file.getvalue()
            loaded_doc = load_pdf(file_input=raw_bytes, doc_name=doc_name)
            stats = get_doc_statistics(loaded_doc)

            st.session_state.loaded_docs[doc_name] = loaded_doc
            st.session_state.doc_stats_list.append(stats)
            progress.write(
                f"\u2705 Extracted **{stats['total_pages']}** pages, "
                f"**{stats['total_words']:,}** words, "
                f"**{stats['total_chunks']}** chunks"
            )

            # Embed & Index
            progress.write("\U0001f9e0 Generating embeddings and indexing...")
            count = vs.add_chunks(loaded_doc.chunks)
            progress.write(f"\u2705 Indexed **{count}** chunks")

            # Executive Summary
            progress.write("\U0001f4cb Generating Executive Summary...")
            all_text = "\n\n".join(
                f"[Page {p.page_number}]\n{p.text}"
                for p in loaded_doc.pages if p.text.strip()
            )
            summary = generate_executive_summary(all_text, llm, doc_name=doc_name)
            st.session_state.doc_summaries[doc_name] = summary
            progress.write("\u2705 Summary generated")

            progress.update(label=f"\u2705 **{doc_name}** processed", state="complete")

        except ScannedOrEmptyPDFError as exc:
            progress.update(label=f"\u26a0\ufe0f **{doc_name}** — Error", state="error")
            st.error(str(exc))

        except DocumentLoadError as exc:
            progress.update(label=f"\u274c **{doc_name}** — Load Error", state="error")
            st.error(str(exc))

        except RuntimeError as exc:
            progress.update(label=f"\u26a0\ufe0f **{doc_name}** — AI Error", state="error")
            st.error(f"**AI Model Error:** {exc}\n\nCheck your API key or switch inference mode.")

        except Exception as exc:
            progress.update(label=f"\u274c **{doc_name}** — Error", state="error")
            st.error(f"Unexpected error: {exc}")
            st.code(traceback.format_exc())


# Trigger processing on new uploads
if uploaded_files:
    new_files = [f for f in uploaded_files if f.name not in st.session_state.loaded_docs]
    if new_files:
        process_documents(new_files)


# ── Main Canvas ─────────────────────────────────────────────────────────────

st.markdown("""
<div class="astra-header">
    <h1>ASTRA INTEL</h1>
    <div class="subtitle">AI-Powered Defence Document Intelligence System</div>
</div>
""", unsafe_allow_html=True)

# ── No Document State ──
if not st.session_state.loaded_docs:
    st.markdown("""
    <div class="empty-state">
        <div class="icon">\U0001f6e1\ufe0f</div>
        <div class="title">No Documents Loaded</div>
        <div class="desc">
            Upload one or more classified PDF documents using the sidebar.<br>
            ASTRA INTEL will extract, index, and generate executive intelligence<br>
            summaries automatically. Multi-document comparison is supported.
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown("""
        <div class="meta-card" style="text-align:center; padding:1.5rem;">
            <div style="font-size:1.5rem; margin-bottom:0.5rem;">\U0001f4c4</div>
            <div class="label">Multi-Doc Ingestion</div>
            <div style="color:var(--text-secondary); font-size:0.78rem; margin-top:0.5rem;">
                Upload multiple PDFs simultaneously
            </div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="meta-card" style="text-align:center; padding:1.5rem;">
            <div style="font-size:1.5rem; margin-bottom:0.5rem;">\U0001f9e0</div>
            <div class="label">Conversational QA</div>
            <div style="color:var(--text-secondary); font-size:0.78rem; margin-top:0.5rem;">
                Multi-turn dialogue with context memory
            </div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div class="meta-card" style="text-align:center; padding:1.5rem;">
            <div style="font-size:1.5rem; margin-bottom:0.5rem;">\U0001f512</div>
            <div class="label">Air-Gapped Mode</div>
            <div style="color:var(--text-secondary); font-size:0.78rem; margin-top:0.5rem;">
                Fully offline inference via Ollama
            </div>
        </div>
        """, unsafe_allow_html=True)
    with col4:
        st.markdown("""
        <div class="meta-card" style="text-align:center; padding:1.5rem;">
            <div style="font-size:1.5rem; margin-bottom:0.5rem;">\u2705</div>
            <div class="label">Grounding Verification</div>
            <div style="color:var(--text-secondary); font-size:0.78rem; margin-top:0.5rem;">
                Anti-hallucination guardrails
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.stop()


# ── Documents Loaded — Intelligence Dashboard ──

# Executive Summaries
if st.session_state.doc_summaries:
    with st.expander(
        f"\U0001f4cb **Executive Defence Intelligence Summaries** ({len(st.session_state.doc_summaries)} doc{'s' if len(st.session_state.doc_summaries) > 1 else ''})",
        expanded=True,
    ):
        if len(st.session_state.doc_summaries) == 1:
            doc_name, summary = next(iter(st.session_state.doc_summaries.items()))
            st.markdown(summary)
        else:
            tabs = st.tabs(list(st.session_state.doc_summaries.keys()))
            for tab, (doc_name, summary) in zip(tabs, st.session_state.doc_summaries.items()):
                with tab:
                    st.markdown(summary)

st.markdown("---")

# ── Chat Section ────────────────────────────────────────────────────────────

mode_label = "\U0001f50d Cross-Document Comparison" if st.session_state.compare_mode else "\U0001f4ac Document Intelligence Q&A"
st.markdown(f"### {mode_label}")

if st.session_state.compare_mode:
    st.caption("Ask comparative questions across all loaded documents. Example: 'Compare the specifications between both documents.'")
else:
    st.caption("Ask questions about the uploaded document(s). Multi-turn conversation with context memory is active.")


def _render_grounding_badge(grounding_data):
    """Render the grounding verification badge."""
    if not grounding_data:
        return

    level = grounding_data.get("level", "medium")
    badge_emoji = grounding_data.get("badge_emoji", "\u26aa")
    badge_label = grounding_data.get("badge_label", "Unknown")
    confidence = grounding_data.get("confidence", 0)

    css_class = {
        "high": "grounding-high",
        "medium": "grounding-medium",
        "low": "grounding-low",
        "ungrounded": "grounding-ungrounded",
    }.get(level, "grounding-medium")

    st.markdown(
        f'<div class="grounding-badge {css_class}">'
        f'{badge_emoji} {badge_label} &mdash; Confidence: {confidence:.0%}'
        f'</div>',
        unsafe_allow_html=True,
    )

    # Warning banner for low/ungrounded
    if level in ("low", "ungrounded"):
        st.warning(
            "\u26a0\ufe0f **Low grounding score.** Some claims could not be verified "
            "against the source text. Exercise caution.",
            icon="\u26a0\ufe0f",
        )

    if grounding_data.get("flagged_claims"):
        with st.expander("\U0001f6a9 Flagged Claims"):
            for claim in grounding_data["flagged_claims"]:
                st.markdown(f"- {claim}")


def _render_citations(citations):
    """Render citation badges and snippets."""
    if not citations:
        return

    with st.expander("\U0001f517 **Verified Intelligence Sources & Citations**"):
        for cit in citations:
            src_label = cit.get("source", "")
            pg = cit.get("page", "?")
            relevance = cit.get("relevance", 0)
            snippet = cit.get("snippet", "")

            rel_pct = f"{relevance:.0%}" if relevance else ""
            badge_text = f"\U0001f4cc Page {pg} | {src_label}"
            if rel_pct:
                badge_text += f" | {rel_pct}"

            st.markdown(
                f'<span class="citation-badge">{badge_text}</span>',
                unsafe_allow_html=True,
            )
            st.markdown(f"> {snippet}")
            st.markdown("---")


# Display chat history
for entry in st.session_state.chat_history:
    st.markdown(
        f'<div class="chat-msg-user"><strong>\U0001f50d Query:</strong> {entry["question"]}</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        f'<div class="chat-msg-assistant">{entry["answer"]}</div>',
        unsafe_allow_html=True,
    )
    _render_grounding_badge(entry.get("grounding"))
    _render_citations(entry.get("citations"))


# Chat input
question = st.chat_input(
    placeholder="Ask a question about the document(s)..." if not st.session_state.compare_mode
    else "Ask a comparative question across documents...",
    key="qa_input",
)

if question:
    if st.session_state.vector_store is None or st.session_state.vector_store.count == 0:
        st.error("No documents indexed. Please upload a document first.")
    else:
        llm = _get_or_create_llm()

        st.markdown(
            f'<div class="chat-msg-user"><strong>\U0001f50d Query:</strong> {question}</div>',
            unsafe_allow_html=True,
        )

        with st.spinner("\U0001f9e0 Analysing document context..."):
            try:
                if st.session_state.compare_mode and len(st.session_state.loaded_docs) > 1:
                    result = compare_documents(
                        question=question,
                        vector_store=st.session_state.vector_store,
                        llm=llm,
                        k_per_doc=3,
                    )
                else:
                    result = answer_question(
                        question=question,
                        vector_store=st.session_state.vector_store,
                        llm=llm,
                        k=5,
                        chat_history=st.session_state.chat_history,
                    )

                # Render answer
                st.markdown(
                    f'<div class="chat-msg-assistant">{result.answer}</div>',
                    unsafe_allow_html=True,
                )

                # Grounding badge
                grounding_data = None
                if result.grounding:
                    grounding_data = {
                        "level": result.grounding.level.value if hasattr(result.grounding.level, 'value') else str(result.grounding.level),
                        "badge_emoji": result.grounding.badge_emoji,
                        "badge_label": result.grounding.badge_label,
                        "confidence": result.grounding.confidence,
                        "flagged_claims": result.grounding.flagged_claims,
                    }
                    _render_grounding_badge(grounding_data)

                # Citations
                citation_data = [
                    {
                        "page": c.page,
                        "source": c.source,
                        "snippet": c.snippet,
                        "relevance": c.relevance,
                    }
                    for c in result.citations
                ]
                _render_citations(citation_data)

                # Save to history
                st.session_state.chat_history.append({
                    "question": question,
                    "answer": result.answer,
                    "citations": citation_data,
                    "grounding": grounding_data,
                })

            except RuntimeError as exc:
                st.error(
                    f"**AI Model Error:** {exc}\n\n"
                    "This may be a rate limit or connectivity issue. "
                    "Try switching inference mode or waiting a moment."
                )
            except Exception as exc:
                st.error(f"Unexpected error: {exc}")
                st.code(traceback.format_exc())
