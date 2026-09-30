# 🛡️ ASTRA INTEL — Defence Document Intelligence System

> **Defence-Grade AI Document Intelligence Platform** with Zero-Hallucination RAG, Page-Level Citations, Multi-Document Comparative Analysis, Grounding Verification Guardrail, and Dual Cloud / Air-Gapped Local Inference.

---

## 🚀 Quickstart

### 1. Clone & Navigate

```bash
git clone https://github.com/Nandeshokane/Astra.git
cd Astra/astra_intel
```

### 2. Create Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate   # macOS / Linux
# venv\Scripts\activate    # Windows
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure Environment

```bash
cp .env.example .env
```

Edit `.env` to configure your preferred LLM provider or local endpoint:

| Provider | Key / Setting | Default / Recommended Model | Description |
|---|---|---|---|
| **Groq (Cloud)** | `GROQ_API_KEY` | `llama-3.3-70b-versatile` | Ultra-fast inference (<1s TTFT) |
| **Google Gemini (Cloud)** | `GOOGLE_API_KEY` | `gemini-2.0-flash` | Multimodal & large context |
| **OpenAI (Cloud)** | `OPENAI_API_KEY` | `gpt-4o-mini` | High-accuracy generalist |
| **Ollama (Air-Gapped)** | `OLLAMA_BASE_URL` | `llama3.2:latest` | Local zero-data-leakage inference |

> **Air-Gapped Operation**: In `SECURE LOCAL (AIR-GAPPED)` mode, zero tokens leave the host machine. Embeddings (`all-MiniLM-L6-v2`) and LLM inference run 100% locally.

### 5. Run the Application

```bash
streamlit run app.py
```

Open your browser at `http://localhost:8501`.

---

## 📁 Architecture & Project Structure

```
Astra/
├── astra_intel/
│   ├── app.py                      # Streamlit UI & Defence Operations Console
│   ├── core/
│   │   ├── __init__.py             # Module exports
│   │   ├── document_loader.py      # PyMuPDF extraction, OCR check & chunking
│   │   ├── vector_store.py         # ChromaDB multi-doc indexing & metadata filtering
│   │   ├── rag_pipeline.py         # Summary, conversational QA & Dossier export
│   │   ├── llm_manager.py          # Unified multi-provider LLM client with health check
│   │   └── verification.py         # Anti-hallucination grounding verification engine
│   ├── requirements.txt            # Locked dependencies
│   ├── .env.example                # Sample environment configuration
│   └── README.md
├── .gitignore
└── README.md                       # Repository overview
```

---

## ⚡ Core Capabilities

### 1. Document Ingestion & Chunking
- **PyMuPDF Engine**: Fast extraction preserving exact page coordinates and page numbers.
- **Scanned PDF Guardrail**: Rejects image-only/empty PDFs before embedding to prevent junk vector indexing.
- **Context-Preserving Chunking**: 800-character chunks with 150-character overlap for optimal boundary recall.

### 2. Multi-Document Intelligence Workspace
- Upload single or multiple defence documents simultaneously (SOPs, procurement specifications, mission briefings).
- Document switcher to query a specific document or toggle **Cross-Document Comparative Intel** mode to synthesize answers across multiple files.
- Dynamic document lifecycle: add, isolate, or remove documents on the fly with live vector store cleanup.

### 3. Conversational Memory & Query Reformulation
- Retains multi-turn operational history.
- Contextual query rewriting: automatically resolves anaphoric references (e.g., "What was its maximum payload?" → "What was the MQ-9 Reaper maximum payload?") before vector retrieval.

### 4. Grounding Verification Engine (Anti-Hallucination Guardrail)
- Real-time fact-checking of generated answers against retrieved source context.
- Dual-metric confidence scoring (Lexical entity overlap + semantic entailment check).
- Tactical Badges:
  - 🟢 **HIGH CONFIDENCE (GROUNDED)** (≥ 75%)
  - 🟡 **MODERATE CONFIDENCE** (50% – 74%)
  - 🔴 **LOW CONFIDENCE / POTENTIAL HALLUCINATION** (< 50%)

### 5. Dual Inference Modes
- 🌐 **CLOUD ACCELERATED**: Low latency cloud endpoints (Groq Llama-3.3-70B, Gemini 2.0 Flash, OpenAI GPT-4o-mini).
- 🔒 **SECURE LOCAL (AIR-GAPPED)**: 100% offline local inference via Ollama (`llama3.2`, `mistral`, `deepseek-r1`) ensuring defence data never leaves the enclave.

### 6. Classified Intel Dossier Export
- Export complete intelligence sessions into formatted Markdown or text dossiers with operational timestamps, document manifests, grounding scores, full query-answer pairs, and exact page citations.

---

## 🛡️ Zero-Hallucination Directive

ASTRA INTEL strictly enforces an anti-hallucination defence prompt contract:
1. Answers must derive **exclusively** from the retrieved context chunks.
2. Every single claim is tagged with `[Page X, DocumentName]`.
3. If the context does not contain sufficient factual evidence, the model explicitly responds: *"INSUFFICIENT RETRIEVED INTELLIGENCE TO VERIFY THIS QUERY."*

---

## 📜 License & Compliance

Built for defence-tech operational evaluation and AI-assisted document intelligence.
