"""
ASTRA INTEL — Main Application
================================
Streamlit-based Defence Document Intelligence UI.
Dark tactical theme with emerald/cyan accents.
"""

import streamlit as st
import time
import traceback

from core.document_loader import (
    load_pdf,
    get_doc_statistics,
    DocumentLoadError,
    ScannedOrEmptyPDFError,
    LoadedDocument,
)
from core.vector_store import AstraVectorStore
from core.rag_pipeline import (
    generate_executive_summary,
    answer_question,
    get_active_provider_info,
    AnswerResult,
)


# ── Page Config ─────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="ASTRA INTEL | Defence Document Intelligence",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ── Custom CSS — Tactical Dark Theme ───────────────────────────────────────

st.markdown("""
<style>
/* ── Import monospace + modern fonts ── */
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;500;600;700&family=Inter:wght@300;400;500;600;700&display=swap');

/* ── Root variables ── */
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

/* ── Global overrides ── */
.stApp {
    background-color: var(--bg-primary) !important;
    color: var(--text-primary) !important;
    font-family: 'Inter', sans-serif !important;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background-color: var(--bg-secondary) !important;
    border-right: 1px solid var(--border-color) !important;
}

section[data-testid="stSidebar"] .stMarkdown {
    color: var(--text-primary) !important;
}

/* Headers */
h1, h2, h3, h4, h5, h6 {
    font-family: 'JetBrains Mono', monospace !important;
    color: var(--text-primary) !important;
}

/* ── Custom Components ── */
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

.status-processing {
    background: rgba(245, 158, 11, 0.15);
    color: #f59e0b;
    border: 1px solid rgba(245, 158, 11, 0.3);
}

.status-error {
    background: rgba(239, 68, 68, 0.15);
    color: #ef4444;
    border: 1px solid rgba(239, 68, 68, 0.3);
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

.empty-state {
    text-align: center;
    padding: 4rem 2rem;
    color: var(--text-muted);
}

.empty-state .icon {
    font-size: 3rem;
    margin-bottom: 1rem;
    opacity: 0.5;
}

.empty-state .title {
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.1rem;
    color: var(--text-secondary);
    margin-bottom: 0.5rem;
}

.empty-state .desc {
    font-size: 0.85rem;
    line-height: 1.6;
}

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

/* Streamlit component overrides */
.stTextInput > div > div {
    background-color: var(--bg-card) !important;
    border-color: var(--border-color) !important;
    color: var(--text-primary) !important;
}

.stTextInput input {
    color: var(--text-primary) !important;
    font-family: 'Inter', sans-serif !important;
}

.stFileUploader {
    border-color: var(--border-color) !important;
}

/* Expander styling */
.streamlit-expanderHeader {
    background-color: var(--bg-card) !important;
    color: var(--text-primary) !important;
    font-family: 'JetBrains Mono', monospace !important;
    border-color: var(--border-color) !important;
}

.streamlit-expanderContent {
    background-color: var(--bg-secondary) !important;
    border-color: var(--border-color) !important;
}

/* Chat input */
.stChatInput {
    border-color: var(--border-color) !important;
}

.stChatInput textarea {
    color: var(--text-primary) !important;
    font-family: 'Inter', sans-serif !important;
}
</style>
""", unsafe_allow_html=True)


# ── Session State Init ──────────────────────────────────────────────────────

def _init_state():
    defaults = {
        "loaded_doc": None,
        "doc_stats": None,
        "vector_store": None,
        "executive_summary": None,
        "chat_history": [],
        "processing": False,
        "last_uploaded_name": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init_state()


# ── Sidebar ─────────────────────────────────────────────────────────────────

with st.sidebar:
    # Logo
    st.markdown("""
    <div class="sidebar-logo">
        <div class="logo-text">🛡️ ASTRA INTEL</div>
        <div class="logo-sub">Defence Document Intelligence</div>
    </div>
    """, unsafe_allow_html=True)

    # Upload
    st.markdown("#### 📄 Document Upload")
    uploaded_file = st.file_uploader(
        "Upload a classified PDF document",
        type=["pdf"],
        help="Accepts text-layer PDFs. Scanned image-only PDFs are not supported.",
        label_visibility="collapsed",
    )

    st.markdown("---")

    # Provider status
    st.markdown("#### ⚙️ AI Model Status")
    provider_info = get_active_provider_info()
    status_class = "status-ready" if provider_info["status"] == "Ready" else "status-error"
    st.markdown(f"""
    <div class="meta-card">
        <div class="label">Provider</div>
        <div class="value">{provider_info['provider']}</div>
    </div>
    <div class="meta-card">
        <div class="label">Model</div>
        <div class="value" style="font-size:0.78rem;">{provider_info['model']}</div>
    </div>
    <div class="meta-card">
        <div class="label">Status</div>
        <div><span class="status-badge {status_class}">● {provider_info['status']}</span></div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # Doc metadata
    st.markdown("#### 📊 Document Metadata")
    if st.session_state.doc_stats:
        stats = st.session_state.doc_stats
        st.markdown(f"""
        <div class="meta-card">
            <div class="label">File Name</div>
            <div class="value" style="font-size:0.78rem; word-break:break-all;">{stats['doc_name']}</div>
        </div>
        <div class="meta-card">
            <div class="label">Total Pages</div>
            <div class="value">{stats['total_pages']}</div>
        </div>
        <div class="meta-card">
            <div class="label">Text Chunks</div>
            <div class="value">{stats['total_chunks']}</div>
        </div>
        <div class="meta-card">
            <div class="label">Word Count</div>
            <div class="value">{stats['total_words']:,}</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="meta-card">
            <div class="label">Status</div>
            <div class="value" style="color: var(--text-muted);">No document loaded</div>
        </div>
        """, unsafe_allow_html=True)


# ── Document Processing ────────────────────────────────────────────────────

def process_document(uploaded_file):
    """Ingest, chunk, embed, index, and summarise the uploaded PDF."""
    doc_name = uploaded_file.name

    # Reset state for new doc
    st.session_state.executive_summary = None
    st.session_state.chat_history = []

    progress = st.status("🔒 **ASTRA INTEL** — Processing classified document...", expanded=True)

    try:
        # Step 1: Extract
        progress.write("📥 Extracting text from PDF...")
        raw_bytes = uploaded_file.getvalue()
        loaded_doc: LoadedDocument = load_pdf(
            file_input=raw_bytes,
            doc_name=doc_name,
        )
        stats = get_doc_statistics(loaded_doc)
        st.session_state.loaded_doc = loaded_doc
        st.session_state.doc_stats = stats
        progress.write(
            f"✅ Extracted **{stats['total_pages']}** pages, "
            f"**{stats['total_words']:,}** words, "
            f"**{stats['total_chunks']}** chunks"
        )

        # Step 2: Embed & Index
        progress.write("🧠 Generating embeddings and indexing...")
        vs = AstraVectorStore()
        vs.reset()
        count = vs.add_chunks(loaded_doc.chunks)
        st.session_state.vector_store = vs
        progress.write(f"✅ Indexed **{count}** chunks into vector store")

        # Step 3: Executive Summary
        progress.write("📋 Generating Executive Defence Intelligence Summary...")
        all_text = "\n\n".join(
            f"[Page {p.page_number}]\n{p.text}"
            for p in loaded_doc.pages if p.text.strip()
        )
        summary = generate_executive_summary(all_text)
        st.session_state.executive_summary = summary
        progress.write("✅ Executive summary generated")

        st.session_state.last_uploaded_name = doc_name
        progress.update(label="✅ **Document processed successfully**", state="complete")

    except ScannedOrEmptyPDFError as exc:
        progress.update(label="⚠️ **Document Error**", state="error")
        st.error(str(exc))
        st.session_state.loaded_doc = None
        st.session_state.doc_stats = None

    except DocumentLoadError as exc:
        progress.update(label="❌ **Load Error**", state="error")
        st.error(str(exc))
        st.session_state.loaded_doc = None
        st.session_state.doc_stats = None

    except RuntimeError as exc:
        progress.update(label="⚠️ **AI Model Error**", state="error")
        st.error(
            f"**AI Model Error:** {exc}\n\n"
            "Please check your API key configuration in `.env` and try again."
        )

    except Exception as exc:
        progress.update(label="❌ **Unexpected Error**", state="error")
        st.error(f"An unexpected error occurred: {exc}")
        st.write("```\n" + traceback.format_exc() + "\n```")


# Trigger processing on new upload
if uploaded_file is not None:
    if st.session_state.last_uploaded_name != uploaded_file.name:
        process_document(uploaded_file)


# ── Main Canvas ─────────────────────────────────────────────────────────────

# Header
st.markdown("""
<div class="astra-header">
    <h1>ASTRA INTEL</h1>
    <div class="subtitle">AI-Powered Defence Document Intelligence System</div>
</div>
""", unsafe_allow_html=True)

# ── No Document State ──
if st.session_state.loaded_doc is None:
    st.markdown("""
    <div class="empty-state">
        <div class="icon">🛡️</div>
        <div class="title">No Document Loaded</div>
        <div class="desc">
            Upload a classified PDF document using the sidebar panel to begin analysis.<br>
            ASTRA INTEL will extract, index, and generate an executive intelligence<br>
            summary automatically upon ingestion.
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Capability cards
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        <div class="meta-card" style="text-align:center; padding:1.5rem;">
            <div style="font-size:1.5rem; margin-bottom:0.5rem;">📄</div>
            <div class="label">Document Ingestion</div>
            <div style="color:var(--text-secondary); font-size:0.8rem; margin-top:0.5rem;">
                PyMuPDF extraction with page-level tracking
            </div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="meta-card" style="text-align:center; padding:1.5rem;">
            <div style="font-size:1.5rem; margin-bottom:0.5rem;">🧠</div>
            <div class="label">Semantic Intelligence</div>
            <div style="color:var(--text-secondary); font-size:0.8rem; margin-top:0.5rem;">
                Vector embeddings with ChromaDB retrieval
            </div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div class="meta-card" style="text-align:center; padding:1.5rem;">
            <div style="font-size:1.5rem; margin-bottom:0.5rem;">🔒</div>
            <div class="label">Zero-Hallucination QA</div>
            <div style="color:var(--text-secondary); font-size:0.8rem; margin-top:0.5rem;">
                Grounded answers with page-level citations
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.stop()


# ── Document Loaded — Show Intelligence Dashboard ──

# Executive Summary
if st.session_state.executive_summary:
    with st.expander("📋 **Executive Defence Intelligence Summary**", expanded=True):
        st.markdown(st.session_state.executive_summary)

st.markdown("---")

# ── Chat / Q&A Section ─────────────────────────────────────────────────────

st.markdown("### 💬 Document Intelligence Q&A")
st.caption("Ask questions about the uploaded document. All answers are grounded with page-level citations.")

# Display chat history
for entry in st.session_state.chat_history:
    # User message
    st.markdown(
        f'<div class="chat-msg-user"><strong>🔍 Query:</strong> {entry["question"]}</div>',
        unsafe_allow_html=True,
    )

    # Assistant answer
    st.markdown(
        f'<div class="chat-msg-assistant">{entry["answer"]}</div>',
        unsafe_allow_html=True,
    )

    # Citations expander
    if entry.get("citations"):
        with st.expander("🔗 **Verified Sources & Citations**"):
            for cit in entry["citations"]:
                st.markdown(
                    f'<span class="citation-badge">📌 Page {cit["page"]}</span>',
                    unsafe_allow_html=True,
                )
                st.markdown(
                    f"> {cit['snippet']}",
                )
                st.markdown("---")

# Chat input
question = st.chat_input(
    placeholder="Ask a question about the document...",
    key="qa_input",
)

if question:
    if st.session_state.vector_store is None:
        st.error("Vector store not initialised. Please re-upload the document.")
    else:
        # Show the question immediately
        st.markdown(
            f'<div class="chat-msg-user"><strong>🔍 Query:</strong> {question}</div>',
            unsafe_allow_html=True,
        )

        with st.spinner("🧠 Analysing document context..."):
            try:
                result: AnswerResult = answer_question(
                    question=question,
                    vector_store=st.session_state.vector_store,
                    k=4,
                )

                # Display the answer
                st.markdown(
                    f'<div class="chat-msg-assistant">{result.answer}</div>',
                    unsafe_allow_html=True,
                )

                # Display citations
                if result.citations:
                    with st.expander("🔗 **Verified Sources & Citations**", expanded=True):
                        for cit in result.citations:
                            st.markdown(
                                f'<span class="citation-badge">📌 Page {cit.page}</span>',
                                unsafe_allow_html=True,
                            )
                            st.markdown(f"> {cit.snippet}")
                            st.markdown("---")

                # Save to history
                st.session_state.chat_history.append({
                    "question": question,
                    "answer": result.answer,
                    "citations": [
                        {"page": c.page, "snippet": c.snippet}
                        for c in result.citations
                    ],
                })

            except RuntimeError as exc:
                st.error(
                    f"**AI Model Error:** {exc}\n\n"
                    "This may be a rate limit or API connectivity issue. "
                    "Please wait a moment and try again."
                )
            except Exception as exc:
                st.error(f"An unexpected error occurred: {exc}")
