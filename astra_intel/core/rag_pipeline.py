"""
ASTRA INTEL — RAG Pipeline
============================
Executive Summary engine and Grounded Q&A engine with citation parsing.
Supports Groq, Google Gemini, OpenAI, and a local Ollama fallback.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import requests
from dotenv import load_dotenv

from core.vector_store import AstraVectorStore, RetrievedChunk

load_dotenv()


# ── Data Structures ─────────────────────────────────────────────────────────

@dataclass
class Citation:
    """A single page-level citation attached to an answer."""
    page: int
    snippet: str


@dataclass
class AnswerResult:
    """Full result object returned by the Q&A engine."""
    answer: str
    citations: List[Citation] = field(default_factory=list)
    raw_chunks: List[RetrievedChunk] = field(default_factory=list)


# ── LLM Provider Abstraction ───────────────────────────────────────────────

def _get_provider_config() -> Tuple[str, str, str]:
    """
    Detect which LLM provider is configured.

    Returns (provider, api_key, model_name).
    Priority: GROQ > GOOGLE GEMINI > OPENAI > OLLAMA (local).
    """
    groq_key = os.getenv("GROQ_API_KEY", "").strip()
    if groq_key:
        model = os.getenv("GROQ_MODEL", "llama-3.1-70b-versatile")
        return ("groq", groq_key, model)

    gemini_key = os.getenv("GOOGLE_API_KEY", "").strip()
    if gemini_key:
        model = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
        return ("gemini", gemini_key, model)

    openai_key = os.getenv("OPENAI_API_KEY", "").strip()
    if openai_key:
        model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        return ("openai", openai_key, model)

    ollama_model = os.getenv("OLLAMA_MODEL", "llama3")
    return ("ollama", "", ollama_model)


def get_active_provider_info() -> Dict[str, str]:
    """Return a dict describing the active provider (for UI display)."""
    provider, key, model = _get_provider_config()
    masked = ""
    if key:
        masked = key[:6] + "..." + key[-4:] if len(key) > 10 else "***"
    return {
        "provider": provider.upper(),
        "model": model,
        "api_key_preview": masked,
        "status": "Ready" if (key or provider == "ollama") else "No Key",
    }


def _call_llm(system_prompt: str, user_prompt: str) -> str:
    """
    Call the configured LLM and return its text response.

    Raises RuntimeError on API failure.
    """
    provider, api_key, model = _get_provider_config()

    try:
        if provider == "groq":
            return _call_groq(api_key, model, system_prompt, user_prompt)
        elif provider == "gemini":
            return _call_gemini(api_key, model, system_prompt, user_prompt)
        elif provider == "openai":
            return _call_openai(api_key, model, system_prompt, user_prompt)
        else:
            return _call_ollama(model, system_prompt, user_prompt)
    except Exception as exc:
        raise RuntimeError(
            f"LLM call failed ({provider}/{model}): {exc}"
        ) from exc


# ── Provider Implementations ───────────────────────────────────────────────

def _call_groq(key: str, model: str, sys: str, usr: str) -> str:
    resp = requests.post(
        "https://api.groq.com/openai/v1/chat/completions",
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        json={
            "model": model,
            "messages": [
                {"role": "system", "content": sys},
                {"role": "user", "content": usr},
            ],
            "temperature": 0.2,
            "max_tokens": 2048,
        },
        timeout=60,
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]


def _call_gemini(key: str, model: str, sys: str, usr: str) -> str:
    url = (
        f"https://generativelanguage.googleapis.com/v1beta/models/"
        f"{model}:generateContent?key={key}"
    )
    payload = {
        "system_instruction": {"parts": [{"text": sys}]},
        "contents": [{"parts": [{"text": usr}]}],
        "generationConfig": {"temperature": 0.2, "maxOutputTokens": 2048},
    }
    resp = requests.post(url, json=payload, timeout=60)
    resp.raise_for_status()
    data = resp.json()
    return data["candidates"][0]["content"]["parts"][0]["text"]


def _call_openai(key: str, model: str, sys: str, usr: str) -> str:
    resp = requests.post(
        "https://api.openai.com/v1/chat/completions",
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        json={
            "model": model,
            "messages": [
                {"role": "system", "content": sys},
                {"role": "user", "content": usr},
            ],
            "temperature": 0.2,
            "max_tokens": 2048,
        },
        timeout=60,
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]


def _call_ollama(model: str, sys: str, usr: str) -> str:
    base = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    resp = requests.post(
        f"{base}/api/chat",
        json={
            "model": model,
            "messages": [
                {"role": "system", "content": sys},
                {"role": "user", "content": usr},
            ],
            "stream": False,
            "options": {"temperature": 0.2},
        },
        timeout=120,
    )
    resp.raise_for_status()
    return resp.json()["message"]["content"]


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
    max_chars: int = 12000,
) -> str:
    """
    Generate an executive summary from the document text.

    *all_text* is truncated to *max_chars* to fit typical context windows.
    """
    truncated = all_text[:max_chars]
    if len(all_text) > max_chars:
        truncated += "\n\n[... document text truncated for summary generation ...]"

    user_prompt = (
        "Analyse the following document text and produce the Executive Defence "
        "Intelligence Summary as instructed.\n\n"
        "--- DOCUMENT TEXT START ---\n"
        f"{truncated}\n"
        "--- DOCUMENT TEXT END ---"
    )

    return _call_llm(SUMMARY_SYSTEM_PROMPT, user_prompt)


# ── Grounded Q&A Engine ────────────────────────────────────────────────────

QA_SYSTEM_PROMPT = """You are ASTRA INTEL, a defence-grade document intelligence analyst.

STRICT RULES — you must follow ALL of them:
1. Answer the user's question ONLY using the CONTEXT SNIPPETS provided below.
2. You must NEVER use outside knowledge, training data, or assumptions.
3. For every factual claim in your answer, include an inline citation in the format [Page X] referencing the page number from which the fact was retrieved.
4. If the provided context does not contain enough information to answer the question, reply EXACTLY:
   "The uploaded document does not contain sufficient information to answer this question."
5. Be concise, precise, and professional in a defence/intelligence briefing style.
6. If multiple pages support the same fact, cite all relevant pages: [Page 3, Page 7].
"""


def answer_question(
    question: str,
    vector_store: AstraVectorStore,
    k: int = 4,
) -> AnswerResult:
    """
    Retrieve relevant chunks and answer the question with citations.
    """
    # 1. Retrieve
    chunks = vector_store.query(query_text=question, k=k)

    if not chunks:
        return AnswerResult(
            answer="No relevant context was found in the document to answer this question.",
            citations=[],
            raw_chunks=[],
        )

    # 2. Build context block
    context_parts: List[str] = []
    for i, ch in enumerate(chunks, 1):
        context_parts.append(
            f"[Context Snippet {i} | Page {ch.page} | Relevance: {ch.relevance_score:.0%}]\n"
            f"{ch.text}\n"
        )
    context_block = "\n---\n".join(context_parts)

    user_prompt = (
        f"CONTEXT SNIPPETS:\n{context_block}\n\n"
        f"USER QUESTION:\n{question}"
    )

    # 3. Call LLM
    raw_answer = _call_llm(QA_SYSTEM_PROMPT, user_prompt)

    # 4. Parse citations from the answer + retrieved chunks
    citations = _extract_citations(raw_answer, chunks)

    return AnswerResult(
        answer=raw_answer,
        citations=citations,
        raw_chunks=chunks,
    )


# ── Citation Parser ─────────────────────────────────────────────────────────

def _extract_citations(
    answer_text: str,
    chunks: List[RetrievedChunk],
) -> List[Citation]:
    """
    Parse [Page X] markers from *answer_text* and match them to
    retrieved chunks to build Citation objects with snippet excerpts.
    """
    # Find all page numbers cited in the answer
    cited_pages = set()
    patterns = [
        r"\[Page\s+(\d+)\]",
        r"\[Pages?\s+([\d,\s]+)\]",
    ]
    for pattern in patterns:
        for match in re.finditer(pattern, answer_text, re.IGNORECASE):
            nums = match.group(1)
            for num_str in re.findall(r"\d+", nums):
                cited_pages.add(int(num_str))

    # Build citations from chunks whose page numbers were cited
    citations: List[Citation] = []
    seen_pages = set()

    for page_num in sorted(cited_pages):
        matching = [c for c in chunks if c.page == page_num]
        if matching:
            best = matching[0]
            # Trim snippet for display
            snippet = best.text[:300].strip()
            if len(best.text) > 300:
                snippet += "..."
            citations.append(Citation(page=page_num, snippet=snippet))
            seen_pages.add(page_num)

    # Also add any retrieved chunks not explicitly cited (for transparency)
    for ch in chunks:
        if ch.page not in seen_pages:
            snippet = ch.text[:300].strip()
            if len(ch.text) > 300:
                snippet += "..."
            citations.append(Citation(page=ch.page, snippet=snippet))
            seen_pages.add(ch.page)

    return citations
