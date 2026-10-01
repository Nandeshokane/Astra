# ASTRA INTEL — Video Submission Script

> **Target Duration:** 5–8 minutes  
> **Format:** Screen recording with voiceover  
> **Structure:** 7 mandatory parts per Section 8.3  

---

## Pre-Recording Checklist

- [ ] Application running: `streamlit run app.py` → `http://localhost:8501`
- [ ] `.env` configured with working Groq API key (fastest inference for live demo)
- [ ] Two test PDFs ready: one defence whitepaper (e.g., MQ-9 Reaper datasheet) and one shorter technical document
- [ ] Screen recording software open (OBS / QuickTime / Loom)
- [ ] Browser tab at the ASTRA INTEL landing page (empty state visible)
- [ ] Terminal visible for test execution (Part 7 or as supplementary footage)

---

## Part 1: Problem Understanding (~45 seconds)

> *Start with the ASTRA INTEL landing page visible, empty state showing the four feature cards.*

**[SPEAK]:**

"Hi, I'm [YOUR NAME], and this is ASTRA INTEL — an AI-powered defence document intelligence system built for Challenge 1 of the ASTRA evaluation.

The problem is straightforward but operationally critical: defence analysts deal with hundreds of dense technical PDFs — threat assessments, equipment specifications, doctrine manuals. Manually reading and cross-referencing these documents under time pressure is slow, error-prone, and doesn't scale.

ASTRA INTEL solves this by ingesting PDFs, embedding them into a searchable vector index, and providing grounded, citation-backed answers — so an analyst can query a 200-page whitepaper in seconds instead of hours.

The core objective was to build a Retrieval-Augmented Generation pipeline that doesn't just answer questions, but *proves* its answers by citing the exact page and document — and actively refuses to answer when the source material doesn't support it."

---

## Part 2: Live Product Demonstration (~90 seconds)

> *This is the most critical section. Perform each action live on screen.*

### Step 2a — Upload and Process (~30s)

**[ACTION]:** Drag the defence whitepaper PDF into the sidebar file uploader.

**[SPEAK]:**

"Let me upload a real defence technical document — this is a publicly available MQ-9 Reaper unmanned aerial vehicle specification sheet.

Watch the processing pipeline — it extracts text page by page, generates embeddings, indexes the chunks in ChromaDB, and then produces an executive intelligence summary. You can see in the sidebar: the document inventory shows the page count, chunk count, and word count."

> *Wait for processing to complete. Point to the Document Inventory table and the Executive Summary expander.*

### Step 2b — Executive Summary (~15s)

**[ACTION]:** Expand the Executive Summary section if not already expanded.

**[SPEAK]:**

"The executive summary is structured into four sections — document title, strategic objectives, key technical specifications, and operational constraints. This is generated from the document content, not from the model's training data."

### Step 2c — Grounded Q&A with Citations (~25s)

**[ACTION]:** Type a question like: *"What is the maximum altitude and endurance of the MQ-9?"*

**[SPEAK]:**

"Now I'll ask a technical question. Notice the response includes inline citations — Page 1, Page 2 — pointing to exactly where each fact was found. Below the answer, you can see the grounding verification badge — the green badge means the system has verified that every claim in the answer is supported by the source text."

> *Point to the 🟢 Grounded badge and expand the Citations section to show the source snippets.*

### Step 2d — Anti-Hallucination Demo (~20s)

**[ACTION]:** Type an out-of-domain question like: *"What is the best recipe for sourdough bread?"*

**[SPEAK]:**

"Now, critically — let me deliberately ask something the document can't answer. I'm asking about baking recipes in a drone specification document.

The system correctly responds: 'The uploaded document does not contain sufficient information to answer this question.' It refuses to hallucinate. The grounding badge stays green because a proper refusal *is* a grounded response — the model correctly identified that the context doesn't support the query."

---

## Part 3: Complete System Architecture Walkthrough (~90 seconds)

> *Switch to the README or show the Mermaid architecture diagram on screen. Ideally render it in a browser or markdown viewer.*

**[SPEAK]:**

"Let me walk through the full system architecture.

**Input Layer:** The analyst uploads one or more PDFs through the Streamlit sidebar. Multi-file upload is supported.

**Processing Layer:** PyMuPDF extracts text page by page. Before any further processing, a word-count guard checks whether the document has sufficient extractable text — if it's a scanned image-only PDF with no text layer, the system rejects it immediately with a clear error message rather than producing garbage results.

The text then passes through a recursive text splitter. This breaks each page into roughly 700-character chunks with 100-character overlap, trying paragraph boundaries first, then sentence boundaries, then word boundaries. The critical detail is that chunking happens *per page* — so every chunk retains a deterministic page number. This is what makes citations accurate.

**AI/ML Layer:** Each chunk is embedded using the `all-MiniLM-L6-v2` sentence-transformer model — this runs entirely locally, no API call needed. The 384-dimensional vectors are indexed in ChromaDB using HNSW indexing with cosine similarity.

When the analyst asks a question, a query condenser first rewrites follow-up questions into standalone queries using the conversation history. Then the standalone query is embedded and matched against the vector index to retrieve the top-K most relevant chunks.

These chunks, along with their page and source metadata, are injected into a structured system prompt. The LLM — either a cloud provider like Groq or a local Ollama model — generates a response constrained to *only* use the provided context.

**Verification Layer:** After generation, the answer passes through a hybrid grounding verifier. It first does a fast lexical check — keyword overlap, bigram overlap, and exact 4-gram phrase matching. If the score is borderline, it escalates to a secondary LLM-based semantic check. The combined score produces the grounding badge you saw in the demo.

**Output Layer:** The analyst gets the cited answer, the grounding badge, expandable source snippets, and can export a full Intel Dossier as a Markdown report."

---

## Part 4: Technical Implementation & Engineering Decisions (~90 seconds)

**[SPEAK]:**

"Let me explain four key engineering decisions.

**First — Chunk sizing at 700 characters with 100-character overlap.** This wasn't arbitrary. Defence documents have dense paragraphs averaging 400–800 characters. A 700-character target captures most paragraphs whole, which preserves semantic completeness for the embedding. The 100-character overlap prevents information loss when a key fact sits at a paragraph boundary. Smaller chunks — say 300 characters — would have better retrieval precision but lose context. Larger chunks — say 1500 — would dilute the embedding signal. 700 is the engineering trade-off.

**Second — Per-page chunking for deterministic citations.** Most RAG systems chunk the entire document as one blob, then try to reverse-engineer page numbers from character offsets. That's fragile and breaks when pages have varying lengths. We chunk each page independently, so the page number is *inherent* to the chunk — not calculated after the fact. This is why our `[Page X]` citations are accurate.

**Third — ChromaDB with cosine similarity over Euclidean.** For normalised sentence-transformer embeddings, cosine and Euclidean are mathematically equivalent — but cosine similarity gives us a natural 0-to-1 relevance score by computing `1 - distance`. This score flows directly into the citation display and grounding verification without any post-processing.

**Fourth — The hybrid grounding verification.** A pure lexical check is fast but misses semantic paraphrasing. A pure LLM check is accurate but slow and costly. We run the lexical check first. If the confidence is above 0.65, we accept it. Only borderline cases (below 0.65) escalate to the LLM verifier. This gives us the reliability of semantic verification without paying the latency cost on every query."

---

## Part 5: AI Tool Usage Transparency (~45 seconds)

**[SPEAK]:**

"Let me be transparent about AI tool usage.

I used **GitHub Copilot** for code autocompletion — boilerplate like HTTP request templates, dataclass definitions, and import statements. I used **Claude and ChatGPT** for architecture discussions, prompt engineering iteration, and debugging assistance.

What was *not* AI-generated: the recursive text splitter logic, the per-page chunking design, the hybrid grounding verification scoring weights, the citation parser regex, and the multi-provider fallback chain in the LLM manager. These were designed, implemented, and debugged manually through iterative development.

The key principle: AI tools were accelerators for boilerplate and ideation. Every critical design decision — chunking boundaries, grounding thresholds, prompt guardrails — was manually evaluated and tested."

---

## Part 6: Challenges & Debugging (~45 seconds)

**[SPEAK]:**

"The hardest engineering problem was preserving page boundaries across chunk splits.

When I first implemented chunking, I chunked the entire document as one string and then tried to map each chunk back to a page by counting character offsets. This broke immediately — different pages have wildly different text lengths, and the character offset calculation drifted after the first few pages. Citations were pointing to the wrong pages.

The fix was architectural, not algorithmic: I switched to *per-page chunking*. Each page's text is chunked independently, so the page number is attached at creation time — not reverse-engineered. This introduced a new problem: chunks at page boundaries lose cross-page context. I solved that with the 100-character overlap, which carries forward the tail of the previous chunk to preserve continuity.

A second real challenge was context drift in multi-turn dialogue. Follow-up questions like 'What about its range?' lose meaning without the prior context. The query condenser — which rewrites follow-ups into standalone queries using the conversation history — was essential. Without it, the vector retrieval would return irrelevant chunks because 'its range' has no semantic anchor."

---

## Part 7: Limitations & Future Roadmap (~45 seconds)

**[SPEAK]:**

"I want to be honest about the system's current limitations.

**Tables and schematics** — PyMuPDF extracts table text as interleaved strings without structural understanding. A production system would integrate Camelot or Tabula for structure-aware table parsing.

**Scanned PDFs** — We detect and reject them, but we don't process them. A full deployment would add an offline OCR pipeline using Tesseract or EasyOCR.

**Session persistence** — Everything lives in Streamlit session state. Refreshing the page loses all data. A production version would use ChromaDB's persistent mode with SQLite backing.

On the roadmap: **multi-document knowledge graphs** — building entity-relationship graphs across documents so analysts can ask 'Which documents mention this radar system?' without uploading everything into one session. And **classification-level gating** — tagging documents with classification markings and enforcing access control at the query level.

Thank you for watching. This is ASTRA INTEL — defence-grade document intelligence."

> *End recording.*

---

## Timing Summary

| Part | Topic | Target Duration |
|------|-------|----------------|
| 1 | Problem Understanding | ~45s |
| 2 | Live Product Demo | ~90s |
| 3 | Architecture Walkthrough | ~90s |
| 4 | Technical Decisions | ~90s |
| 5 | AI Usage Transparency | ~45s |
| 6 | Challenges & Debugging | ~45s |
| 7 | Limitations & Roadmap | ~45s |
| | **Total** | **~7 min 30s** |

---

## Recording Tips

1. **Practice the demo flow once** before recording to ensure the API responds quickly and there are no Streamlit errors.
2. **Use Groq** as the cloud provider — it has the fastest inference latency (~1–2s responses), which makes the live demo feel smooth.
3. **Zoom into the grounding badge and citations** when demonstrating them — evaluators need to see these clearly.
4. **Speak at a measured pace.** The script is designed to fill ~7.5 minutes at natural speaking speed. Don't rush.
5. **If the LLM takes long to respond** during the demo, fill the silence by explaining what's happening ("The system is now embedding the query and searching the vector index...").
