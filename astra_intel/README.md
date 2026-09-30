# 🛡️ ASTRA INTEL — Defence Document Intelligence System

> AI-powered document analysis with zero-hallucination Q&A and page-level citations.

---

## 🚀 Quickstart

### 1. Clone & Navigate

```bash
cd astra_intel
```

### 2. Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate   # macOS/Linux
# venv\Scripts\activate    # Windows
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure API Keys

```bash
cp .env.example .env
```

Edit `.env` and add at least **one** API key. Provider priority:

| Priority | Provider | Key Variable | Notes |
|----------|----------|-------------|-------|
| 1st | **Groq** | `GROQ_API_KEY` | Fastest inference (recommended) |
| 2nd | **Google Gemini** | `GOOGLE_API_KEY` | Gemini 2.0 Flash |
| 3rd | **OpenAI** | `OPENAI_API_KEY` | GPT-4o-mini |
| 4th | **Ollama** | *(none needed)* | Local fallback — requires Ollama running |

### 5. Run the Application

```bash
streamlit run app.py
```

The app will open at `http://localhost:8501`.

---

## 📁 Project Structure

```
astra_intel/
├── app.py                  # Streamlit UI & interaction workflow
├── core/
│   ├── __init__.py
│   ├── document_loader.py  # PyMuPDF extraction & chunking
│   ├── vector_store.py     # ChromaDB indexing & retrieval
│   └── rag_pipeline.py     # Summary & Q&A engines
├── requirements.txt
├── .env.example
└── README.md
```

## 🔐 Features (Day 1 MVP)

- **PDF Ingestion** — Page-by-page text extraction with PyMuPDF
- **Scanned PDF Detection** — Flags image-only PDFs automatically
- **Executive Summary** — Structured defence brief generated on upload
- **Grounded Q&A** — Zero-hallucination answers with `[Page X]` citations
- **Citation Verification** — Collapsible source snippets for every answer
- **Multi-Provider LLM** — Groq / Gemini / OpenAI / Ollama support
- **Tactical Dark UI** — Defence-themed dark interface with emerald/cyan accents

---

## ⚠️ Notes

- Only **text-layer PDFs** are supported. Scanned image-PDFs will be rejected with a clear message.
- The first run downloads the `all-MiniLM-L6-v2` embedding model (~80MB).
- For Ollama fallback, ensure `ollama serve` is running and the model is pulled.
