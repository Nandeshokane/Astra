"""
ASTRA INTEL — RAG Pipeline (Day 2)
=====================================
Executive Summary engine, Multi-turn query condenser,
Grounded Q&A with multi-doc [Page X, DocName] citations,
Comparative cross-document analysis, and Intel Dossier export.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from core.llm_manager import LLMClient
from core.vector_store import AstraVectorStore, RetrievedChunk
from core.verification import GroundingResult, verify_grounding


# ── Data Structures ─────────────────────────────────────────────────────────

@dataclass
class Citation:
    """A single page-level citation attached to an answer."""
    page: int
    source: str      # document filename
    snippet: str
    relevance: float = 0.0


@dataclass
class AnswerResult:
    """Full result returned by the Q&A engine."""
    answer: str
    citations: List[Citation] = field(default_factory=list)
    raw_chunks: List[RetrievedChunk] = field(default_factory=list)
    grounding: Optional[GroundingResult] = None


# ── Query Condenser (Multi-Turn Context Reformulation) ──────────────────────

CONDENSER_SYSTEM = """You are a search query optimizer for a defence document intelligence system.

Given a CHAT HISTORY and a FOLLOW-UP QUESTION, rewrite the follow-up into a STANDALONE search query
that captures all necessary context from the conversation.

Rules:
- Output ONLY the rewritten standalone query, nothing else.
- Preserve all technical terms, names, designations, and specifics.
- If the follow-up is already standalone, return it unchanged.
- Do NOT answer the question. Just rewrite it.
"""


def condense_question(
    chat_history: List[Dict[str, str]],
    latest_question: str,
    llm: LLMClient,
) -> str:
    """
    Rewrite a follow-up question as a standalone query using chat history.

    Example: "What about its range?" -> "What is the operational range of
    the MQ-9 Reaper drone discussed earlier?"
    """
    if not chat_history:
        return latest_question

    # Build history block (last 6 exchanges max)
    recent = chat_history[-6:]
    history_lines = []
    for entry in recent:
        history_lines.append(f"User: {entry.get('question', '')}")
        # Truncate long answers
        ans = entry.get("answer", "")
        if len(ans) > 300:
            ans = ans[:300] + "..."
        history_lines.append(f"Assistant: {ans}")

    user_prompt = (
        f"CHAT HISTORY:\n" + "\n".join(history_lines) +
        f"\n\nFOLLOW-UP QUESTION: {latest_question}"
    )

    try:
        rewritten = llm.call(CONDENSER_SYSTEM, user_prompt).strip()
        # Safety: if the LLM returned something too long or obviously wrong, fall back
        if len(rewritten) > 500 or len(rewritten) < 3:
            return latest_question
        return rewritten
    except Exception:
        return latest_question


# ── Executive Summary Engine ───────────────────────────────────────────────

SUMMARY_SYSTEM_PROMPT = """You are ASTRA INTEL, a defence-grade document intelligence analyst.

Given the full text of a document, produce a structured Executive Defence Intelligence Summary.
Use ONLY facts present in the document. Do NOT fabricate or infer beyond what is stated.

Format your output EXACTLY as follows (use Markdown):

## Document Title / Primary Subject
<title or main subject as stated in the document>

## Strategic / Operational Objectives
<bullet points summarising the main strategic or operational goals>

## Key Technical Specifications / Core Capabilities
<bullet points listing core technical details, systems, or capabilities mentioned>

## Operational Constraints or Risks
<bullet points on limitations, risks, dependencies, or constraints noted>

If a section has no relevant content, write "Not explicitly stated in the document."
Keep the summary concise (under 500 words total).
"""


def generate_executive_summary(
    all_text: str,
    llm: LLMClient,
    doc_name: str = "",
    max_chars: int = 12000,
) -> str:
    """Generate an executive summary from document text."""
    truncated = all_text[:max_chars]
    if len(all_text) > max_chars:
        truncated += "\n\n[... document text truncated for summary generation ...]"

    header = f"Document: {doc_name}\n\n" if doc_name else ""
    user_prompt = (
        f"{header}Analyse the following document text and produce the Executive Defence "
        "Intelligence Summary as instructed.\n\n"
        "--- DOCUMENT TEXT START ---\n"
        f"{truncated}\n"
        "--- DOCUMENT TEXT END ---"
    )

    return llm.call(SUMMARY_SYSTEM_PROMPT, user_prompt)


# ── Grounded Q&A Engine ────────────────────────────────────────────────────

QA_SYSTEM_PROMPT = """You are ASTRA INTEL, a defence-grade document intelligence analyst.

STRICT RULES — you must follow ALL of them:
1. Answer the user's question ONLY using the CONTEXT SNIPPETS provided below.
2. You must NEVER use outside knowledge, training data, or assumptions.
3. For every factual claim, include an inline citation in the format [Page X, DocName] where DocName is the source document filename.
4. If all context is from a single document, you may use the shorter format [Page X].
5. If the provided context does not contain enough information to answer the question, reply EXACTLY:
   "The uploaded document does not contain sufficient information to answer this question."
6. Be concise, precise, and professional in a defence/intelligence briefing style.
7. If multiple pages support the same fact, cite all relevant pages: [Page 3, Page 7].
"""


def answer_question(
    question: str,
    vector_store: AstraVectorStore,
    llm: LLMClient,
    k: int = 5,
    chat_history: Optional[List[Dict]] = None,
) -> AnswerResult:
    """
    Full Q&A pipeline:
    1. Condense question (if chat history exists).
    2. Retrieve relevant chunks.
    3. Generate grounded answer.
    4. Parse citations.
    5. Verify grounding.
    """
    # 1. Condense
    search_query = question
    if chat_history:
        search_query = condense_question(chat_history, question, llm)

    # 2. Retrieve
    chunks = vector_store.query(query_text=search_query, k=k)

    if not chunks:
        return AnswerResult(
            answer="No relevant context was found in the document to answer this question.",
            citations=[],
            raw_chunks=[],
            grounding=GroundingResult(
                is_grounded=True, confidence=1.0,
                level="high", explanation="Refusal response — no context available.",
            ),
        )

    # 3. Build context block with multi-doc labels
    multi_doc = len(set(ch.source for ch in chunks)) > 1
    context_parts: List[str] = []
    for i, ch in enumerate(chunks, 1):
        label = f"[Snippet {i} | {ch.source} | Page {ch.page} | Relevance: {ch.relevance_score:.0%}]"
        context_parts.append(f"{label}\n{ch.text}\n")
    context_block = "\n---\n".join(context_parts)

    user_prompt = (
        f"CONTEXT SNIPPETS:\n{context_block}\n\n"
        f"USER QUESTION:\n{question}"
    )

    # 4. Call LLM
    raw_answer = llm.call(QA_SYSTEM_PROMPT, user_prompt)

    # 5. Parse citations
    citations = _extract_citations(raw_answer, chunks)

    # 6. Verify grounding
    grounding = verify_grounding(raw_answer, chunks, llm_client=llm)

    return AnswerResult(
        answer=raw_answer,
        citations=citations,
        raw_chunks=chunks,
        grounding=grounding,
    )


# ── Comparative Cross-Document Analysis ────────────────────────────────────

COMPARE_SYSTEM = """You are ASTRA INTEL, a defence-grade comparative intelligence analyst.

You are given context snippets from MULTIPLE documents. Your task is to:
1. Compare and contrast the information across documents regarding the user's question.
2. Highlight similarities, differences, and gaps between the documents.
3. Cite every fact with [Page X, DocName] format.
4. If a document does not address a specific aspect, note it explicitly.
5. Use ONLY the provided context. Do NOT use outside knowledge.
6. Structure your response with clear headers for each document and a synthesis section.
"""


def compare_documents(
    question: str,
    vector_store: AstraVectorStore,
    llm: LLMClient,
    k_per_doc: int = 3,
) -> AnswerResult:
    """
    Cross-document comparative analysis.

    Retrieves relevant chunks from each indexed document separately,
    then asks the LLM to produce a comparative briefing.
    """
    cross_results = vector_store.query_cross_document(question, k_per_doc=k_per_doc)

    if not cross_results:
        return AnswerResult(
            answer="No relevant context was found across any documents.",
            citations=[],
            raw_chunks=[],
        )

    # Flatten all chunks and build a labelled context
    all_chunks: List[RetrievedChunk] = []
    context_parts: List[str] = []
    idx = 1

    for doc_name, chunks in cross_results.items():
        context_parts.append(f"\n=== DOCUMENT: {doc_name} ===")
        for ch in chunks:
            all_chunks.append(ch)
            context_parts.append(
                f"[Snippet {idx} | {doc_name} | Page {ch.page} | Relevance: {ch.relevance_score:.0%}]\n"
                f"{ch.text}\n"
            )
            idx += 1

    context_block = "\n---\n".join(context_parts)

    user_prompt = (
        f"CONTEXT SNIPPETS FROM MULTIPLE DOCUMENTS:\n{context_block}\n\n"
        f"USER QUESTION:\n{question}"
    )

    raw_answer = llm.call(COMPARE_SYSTEM, user_prompt)
    citations = _extract_citations(raw_answer, all_chunks)
    grounding = verify_grounding(raw_answer, all_chunks, llm_client=llm)

    return AnswerResult(
        answer=raw_answer,
        citations=citations,
        raw_chunks=all_chunks,
        grounding=grounding,
    )


# ── Citation Parser ─────────────────────────────────────────────────────────

def _extract_citations(
    answer_text: str,
    chunks: List[RetrievedChunk],
) -> List[Citation]:
    """
    Parse [Page X] and [Page X, DocName] markers from *answer_text*
    and map them to retrieved chunks for source display.
    """
    # Find all page numbers cited
    cited_pages: Dict[int, Optional[str]] = {}

    # Pattern: [Page 3, filename.pdf] or [Page 3]
    for match in re.finditer(
        r"\[Page\s+(\d+)(?:\s*,\s*([^\]]+))?\]", answer_text, re.IGNORECASE
    ):
        page = int(match.group(1))
        doc = match.group(2).strip() if match.group(2) else None
        cited_pages[page] = doc

    # Pattern: [Pages 2, 5] (comma-separated pages)
    for match in re.finditer(r"\[Pages?\s+([\d,\s]+)\]", answer_text, re.IGNORECASE):
        for num_str in re.findall(r"\d+", match.group(1)):
            cited_pages.setdefault(int(num_str), None)

    # Build citation objects
    citations: List[Citation] = []
    seen_keys: set = set()

    for page_num, doc_hint in sorted(cited_pages.items()):
        matching = [
            c for c in chunks
            if c.page == page_num and (doc_hint is None or doc_hint in c.source)
        ]
        if matching:
            best = matching[0]
            key = (best.source, page_num)
            if key not in seen_keys:
                snippet = best.text[:300].strip()
                if len(best.text) > 300:
                    snippet += "..."
                citations.append(Citation(
                    page=page_num,
                    source=best.source,
                    snippet=snippet,
                    relevance=best.relevance_score,
                ))
                seen_keys.add(key)

    # Add any retrieved chunks not explicitly cited (transparency)
    for ch in chunks:
        key = (ch.source, ch.page)
        if key not in seen_keys:
            snippet = ch.text[:300].strip()
            if len(ch.text) > 300:
                snippet += "..."
            citations.append(Citation(
                page=ch.page,
                source=ch.source,
                snippet=snippet,
                relevance=ch.relevance_score,
            ))
            seen_keys.add(key)

    return citations


# ── Intel Dossier Export ────────────────────────────────────────────────────

def export_intel_dossier(
    doc_summaries: Dict[str, str],
    chat_history: List[Dict],
    doc_stats: List[Dict],
) -> str:
    """
    Generate a structured Markdown Intel Dossier containing:
    - Document metadata
    - Executive summaries
    - Full conversation transcript with citations
    """
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    lines = [
        "# ASTRA INTEL — Intelligence Dossier",
        f"**Generated:** {now}",
        f"**Classification:** UNCLASSIFIED // FOR OFFICIAL USE ONLY",
        "",
        "---",
        "",
        "## 1. Document Inventory",
        "",
    ]

    for stats in doc_stats:
        lines.extend([
            f"### {stats.get('doc_name', 'Unknown')}",
            f"- **Pages:** {stats.get('total_pages', '?')}",
            f"- **Words:** {stats.get('total_words', '?'):,}",
            f"- **Chunks indexed:** {stats.get('total_chunks', '?')}",
            "",
        ])

    lines.extend(["---", "", "## 2. Executive Intelligence Summaries", ""])

    for doc_name, summary in doc_summaries.items():
        lines.extend([
            f"### {doc_name}",
            "",
            summary,
            "",
            "---",
            "",
        ])

    lines.extend(["## 3. Intelligence Q&A Transcript", ""])

    if not chat_history:
        lines.append("*No queries recorded in this session.*")
    else:
        for i, entry in enumerate(chat_history, 1):
            lines.extend([
                f"### Q{i}: {entry.get('question', '')}",
                "",
                f"**Answer:**",
                entry.get("answer", ""),
                "",
            ])
            # Grounding badge
            grounding = entry.get("grounding", {})
            if grounding:
                badge = grounding.get("badge_label", "")
                conf = grounding.get("confidence", 0)
                lines.append(f"**Grounding:** {badge} ({conf:.0%})")
                lines.append("")

            cits = entry.get("citations", [])
            if cits:
                lines.append("**Citations:**")
                for c in cits:
                    src = c.get("source", "")
                    pg = c.get("page", "?")
                    snip = c.get("snippet", "")[:200]
                    lines.append(f"- **[Page {pg}, {src}]** — {snip}")
                lines.append("")

            lines.extend(["---", ""])

    lines.extend([
        "## 4. Audit Trail",
        "",
        f"- **Session timestamp:** {now}",
        f"- **Total queries:** {len(chat_history)}",
        f"- **Documents analysed:** {len(doc_stats)}",
        "",
        "---",
        "*Generated by ASTRA INTEL — Defence Document Intelligence System*",
    ])

    return "\n".join(lines)
