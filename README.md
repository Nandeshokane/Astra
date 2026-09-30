# 🛡️ ASTRA INTEL — Defence Document Intelligence System

> **Defence-Grade AI Document Intelligence Platform** with Zero-Hallucination RAG, Page-Level Citations, Multi-Document Comparative Analysis, Grounding Verification Guardrail, and Dual Cloud / Air-Gapped Local Inference.

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.64.0-FF4B4B.svg)](https://streamlit.io/)
[![ChromaDB](https://img.shields.io/badge/ChromaDB-VectorStore-green.svg)](https://www.trychroma.com/)
[![License](https://img.shields.io/badge/License-Proprietary-darkred.svg)]()

---

## ⚡ Key Highlights

- **Anti-Hallucination Grounding Guardrail**: Evaluates every generated claim against retrieved source chunks with a real-time confidence badge (High / Moderate / Low).
- **Exact Page-Level Citations**: Explicit `[Page X, DocName]` citations linked directly to verified document source excerpts.
- **Multi-Document Workspace**: Upload multiple defence PDFs, cross-reference intel, and perform comparative QA across documents simultaneously.
- **Dual Inference (Cloud / Air-Gapped)**:
  - 🌐 **Cloud Accelerated**: Groq (Llama-3.3-70B), Google Gemini (Gemini 2.0 Flash), OpenAI (GPT-4o-mini).
  - 🔒 **Secure Local (Air-Gapped)**: 100% offline local inference via Ollama (`llama3.2`, `mistral`, etc.) with local embeddings (`all-MiniLM-L6-v2`) — zero data exfiltration.
- **Conversational Memory & Query Reformulation**: Contextual rewrite of follow-up questions to retain multi-turn mission dialogue.
- **Classified Intel Dossier Export**: Generate structured operational debriefing reports in Markdown format.

---

## 🚀 Quickstart

```bash
# 1. Clone repository
git clone https://github.com/Nandeshokane/Astra.git
cd Astra/astra_intel

# 2. Set up virtual environment
python3 -m venv venv
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Set up environment variables
cp .env.example .env
# Edit .env and supply your preferred API key (GROQ_API_KEY, GOOGLE_API_KEY, or OPENAI_API_KEY)

# 5. Launch Defence Console
streamlit run app.py
```

Access the UI at `http://localhost:8501`.

---

## 📁 Repository Structure

```
Astra/
├── README.md                       # Root overview
├── .gitignore                      # Git ignore patterns
└── astra_intel/                    # Core application package
    ├── app.py                      # Tactical Streamlit UI
    ├── core/
    │   ├── __init__.py             # Module exports
    │   ├── document_loader.py      # PyMuPDF extraction & chunking
    │   ├── vector_store.py         # ChromaDB multi-doc indexing
    │   ├── rag_pipeline.py         # Summary, conversational QA & Dossier export
    │   ├── llm_manager.py          # Unified multi-provider LLM client
    │   └── verification.py         # Grounding & anti-hallucination verification
    ├── requirements.txt            # Python dependencies
    ├── .env.example                # Sample environment file
    └── README.md                   # Detailed application guide
```

---

For detailed documentation, refer to [astra_intel/README.md](astra_intel/README.md).
