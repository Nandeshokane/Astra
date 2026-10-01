# ASTRA INTEL — Pre-Submission Verification Checklist

> **Section 8.7 Compliance Check**  
> Run through every item below before submitting. All items must be ✅.

---

## 1. Repository Hygiene & Security (Section 8.1)

- [ ] **No API keys committed** — Run `git log --all -p | grep -iE "(sk-|gsk_|AIza|Bearer|api.key|secret)" ` and verify zero matches
- [ ] **`.env` is gitignored** — Confirm `.env` appears in `.gitignore` and is NOT tracked: `git ls-files .env` returns nothing
- [ ] **`.env.example` IS committed** — Confirm it exists with placeholder values (no real keys)
- [ ] **No `__pycache__/` directories committed** — `git ls-files | grep __pycache__` returns nothing
- [ ] **No `venv/` committed** — `git ls-files | grep venv/` returns nothing
- [ ] **No ChromaDB data files committed** — `git ls-files | grep -E "(\.sqlite|chroma)"` returns nothing
- [ ] **No model cache files committed** — `git ls-files | grep -E "(\.bin|\.onnx|\.safetensors)"` returns nothing

## 2. Core Functionality Verification

- [ ] **Application starts** — `streamlit run app.py` opens without errors
- [ ] **PDF upload works** — Upload a test PDF; text extraction, chunking, and indexing complete
- [ ] **Executive summary generates** — Summary appears in the expander with structured sections
- [ ] **Q&A with citations works** — Ask a factual question; answer includes `[Page X]` citations
- [ ] **Anti-hallucination works** — Ask an out-of-domain question; system returns refusal message
- [ ] **Grounding badges display** — 🟢 / 🟡 / 🟠 / 🔴 badges render correctly next to answers
- [ ] **Citation drill-down works** — Expanding the citations section shows source snippets with page numbers
- [ ] **Error handling works** — Upload a non-PDF or corrupted file; error message appears (no crash)

## 3. Advanced Features Verification

- [ ] **Multi-document upload** — Upload two PDFs; both appear in the Document Inventory table
- [ ] **Cross-document comparison** — Enable "Cross-Document Comparison" toggle; ask a comparative question
- [ ] **Multi-turn conversation** — Ask a follow-up question (e.g., "What about its range?"); context is preserved
- [ ] **Cloud/Local mode switch** — Toggle between Cloud and Local in the sidebar; status updates correctly
- [ ] **Intel Dossier export** — Click "Export Intel Dossier"; download the `.md` file and verify it contains summaries + Q&A

## 4. Test Suite (Section 8.7)

- [ ] **Tests pass** — `python -m pytest tests/test_pipeline.py -v --tb=short` → all tests GREEN
- [ ] **No import errors** — Tests run without `ModuleNotFoundError` or `ImportError`
- [ ] **Synthetic PDFs work** — Tests create in-memory PDFs via PyMuPDF (no external test data required)

## 5. Documentation (Sections 8.5 & 8.6)

- [ ] **README.md exists** in `astra_intel/` with complete content
- [ ] **README has Project Overview** — Problem statement + defence relevance
- [ ] **README has Features table** — Must Have / Should Have / Bonus breakdown
- [ ] **README has Tech Stack** — Complete table of all technologies
- [ ] **README has Architecture diagram** — Mermaid.js flowchart renders correctly
- [ ] **README has Setup & Quickstart** — Prerequisites, install, env config, run command
- [ ] **README has AI/ML Rationale** — Chunking, embedding, vector store, prompt guardrail justifications
- [ ] **README has Testing section** — Test case summary table + known failure modes
- [ ] **README has Limitations** — Honest evaluation of system boundaries
- [ ] **README has AI Usage Disclosure** — Mandatory block per Section 8.6 with tool list + breakdown table

## 6. Video Submission (Section 8.3)

- [ ] **Video script reviewed** — `VIDEO_SCRIPT.md` covers all 7 mandatory parts
- [ ] **Video duration** — Target 5–8 minutes (script estimates ~7:30 at natural pace)
- [ ] **Live demo recorded** — Part 2 shows real-time PDF upload, Q&A, citations, and anti-hallucination
- [ ] **Architecture explained** — Part 3 walks through the Mermaid diagram with narration
- [ ] **AI usage disclosed** — Part 5 honestly covers tools used vs. manual implementation
- [ ] **Video uploaded** — To the submission platform in the required format

## 7. Final Git Commit

- [ ] **All files staged** — `git status` shows no untracked critical files
- [ ] **Commit message is clean** — e.g., "Day 3: Testing, documentation, deployment prep"
- [ ] **Push to remote** — `git push origin main` succeeds
- [ ] **Verify on GitHub** — Repository is accessible; README renders correctly; `.env` is NOT visible

---

## Quick Verification Commands

```bash
# Run the full check sequence:

# 1. Test suite
python -m pytest tests/test_pipeline.py -v --tb=short

# 2. Secret scan (should return nothing)
git log --all -p 2>/dev/null | grep -iE "(sk-[a-zA-Z0-9]{20}|gsk_[a-zA-Z0-9]{20}|AIza[a-zA-Z0-9]{30})" | head -5

# 3. Check tracked files for sensitive patterns
git ls-files | xargs grep -l "API_KEY=" 2>/dev/null | grep -v ".example"

# 4. Verify .env is NOT tracked
git ls-files .env

# 5. Application smoke test
timeout 10 streamlit run app.py --server.headless true 2>&1 | head -5

# 6. File structure check
find . -name "*.py" -not -path "./venv/*" | head -20
```

---

> **Submission is ready when every checkbox above is ✅.**
