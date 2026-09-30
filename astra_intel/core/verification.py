"""
ASTRA INTEL — Verification Engine (Day 2)
============================================
Anti-hallucination guardrail that evaluates whether generated answers
are properly grounded in retrieved source chunks.

Uses a hybrid approach:
  1. Fast lexical overlap (n-gram / keyword matching) — zero-cost baseline.
  2. Optional LLM-based verification for deeper semantic checking.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Set, TYPE_CHECKING

if TYPE_CHECKING:
    from core.llm_manager import LLMClient

from core.vector_store import RetrievedChunk


# ── Data Structures ─────────────────────────────────────────────────────────

class GroundingLevel(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    UNGROUNDED = "ungrounded"


@dataclass
class GroundingResult:
    """Result of a grounding verification check."""
    is_grounded: bool
    confidence: float               # 0.0 – 1.0
    level: GroundingLevel
    flagged_claims: List[str] = field(default_factory=list)
    explanation: str = ""

    @property
    def badge_emoji(self) -> str:
        return {
            GroundingLevel.HIGH: "\U0001f7e2",       # 🟢
            GroundingLevel.MEDIUM: "\U0001f7e1",     # 🟡
            GroundingLevel.LOW: "\U0001f7e0",        # 🟠
            GroundingLevel.UNGROUNDED: "\U0001f534",  # 🔴
        }.get(self.level, "\u26aa")

    @property
    def badge_label(self) -> str:
        return {
            GroundingLevel.HIGH: "Grounded",
            GroundingLevel.MEDIUM: "Partial Grounding",
            GroundingLevel.LOW: "Low Grounding",
            GroundingLevel.UNGROUNDED: "Unsupported Claim",
        }.get(self.level, "Unknown")


# ── Core Verification ──────────────────────────────────────────────────────

def verify_grounding(
    answer: str,
    context_chunks: List[RetrievedChunk],
    llm_client: Optional["LLMClient"] = None,
) -> GroundingResult:
    """
    Evaluate how well *answer* is grounded in the *context_chunks*.

    Strategy
    --------
    1. Extract assertion sentences from the answer (skip refusal responses).
    2. For each assertion, compute keyword overlap with all context chunks.
    3. Aggregate into a confidence score and grounding level.
    4. If an LLM client is provided, run an optional semantic verification pass.

    Returns a ``GroundingResult``.
    """
    # Short-circuit: refusal responses are inherently grounded
    refusal_phrases = [
        "does not contain sufficient information",
        "no relevant context was found",
        "cannot be determined from the text",
        "not mentioned in the document",
    ]
    answer_lower = answer.lower()
    if any(phrase in answer_lower for phrase in refusal_phrases):
        return GroundingResult(
            is_grounded=True,
            confidence=1.0,
            level=GroundingLevel.HIGH,
            explanation="The model correctly declined to answer based on available context.",
        )

    if not context_chunks:
        return GroundingResult(
            is_grounded=False,
            confidence=0.0,
            level=GroundingLevel.UNGROUNDED,
            explanation="No context chunks were available for verification.",
        )

    # ── 1. Extract assertion sentences from the answer ──────────────────────
    assertions = _extract_assertions(answer)
    if not assertions:
        return GroundingResult(
            is_grounded=True,
            confidence=0.8,
            level=GroundingLevel.MEDIUM,
            explanation="Could not extract verifiable assertions from the answer.",
        )

    # ── 2. Build combined context corpus ────────────────────────────────────
    context_text = " ".join(ch.text for ch in context_chunks).lower()
    context_keywords = _extract_keywords(context_text)

    # ── 3. Score each assertion ─────────────────────────────────────────────
    scores: List[float] = []
    flagged: List[str] = []

    for assertion in assertions:
        score = _compute_overlap_score(assertion, context_keywords, context_text)
        scores.append(score)
        if score < 0.3:
            # Trim to reasonable display length
            trimmed = assertion[:120] + "..." if len(assertion) > 120 else assertion
            flagged.append(trimmed)

    # ── 4. Aggregate ────────────────────────────────────────────────────────
    avg_score = sum(scores) / len(scores) if scores else 0.0
    grounded_count = sum(1 for s in scores if s >= 0.3)
    grounded_ratio = grounded_count / len(scores) if scores else 0.0

    # Combine overlap score and grounded ratio
    confidence = round(0.6 * avg_score + 0.4 * grounded_ratio, 4)

    if confidence >= 0.65:
        level = GroundingLevel.HIGH
    elif confidence >= 0.45:
        level = GroundingLevel.MEDIUM
    elif confidence >= 0.25:
        level = GroundingLevel.LOW
    else:
        level = GroundingLevel.UNGROUNDED

    # ── 5. Optional LLM semantic check (if available & score is borderline) ─
    explanation = f"Lexical grounding score: {confidence:.0%} ({grounded_count}/{len(scores)} assertions verified)."
    if llm_client and confidence < 0.65:
        try:
            llm_result = _llm_verify(answer, context_chunks, llm_client)
            if llm_result:
                # Blend LLM assessment with lexical score
                llm_conf = llm_result.get("confidence", confidence)
                confidence = round(0.4 * confidence + 0.6 * llm_conf, 4)
                if llm_result.get("flagged"):
                    flagged.extend(llm_result["flagged"])
                explanation += f" LLM verification: {llm_result.get('reasoning', 'N/A')}"

                # Re-assess level after blending
                if confidence >= 0.65:
                    level = GroundingLevel.HIGH
                elif confidence >= 0.45:
                    level = GroundingLevel.MEDIUM
                elif confidence >= 0.25:
                    level = GroundingLevel.LOW
                else:
                    level = GroundingLevel.UNGROUNDED
        except Exception:
            explanation += " (LLM verification skipped due to error.)"

    return GroundingResult(
        is_grounded=level in (GroundingLevel.HIGH, GroundingLevel.MEDIUM),
        confidence=confidence,
        level=level,
        flagged_claims=flagged,
        explanation=explanation,
    )


# ── Helpers ─────────────────────────────────────────────────────────────────

def _extract_assertions(text: str) -> List[str]:
    """
    Split the answer into individual assertion sentences.
    Strips citation markers, bullet points, and markdown formatting.
    """
    # Remove citation markers like [Page 3] or [Page 2, Page 5]
    cleaned = re.sub(r"\[Page[s]?\s*[\d,\s]+\]", "", text, flags=re.IGNORECASE)
    # Remove markdown bold/italic
    cleaned = re.sub(r"[*_]{1,3}", "", cleaned)
    # Remove markdown headers
    cleaned = re.sub(r"^#+\s*", "", cleaned, flags=re.MULTILINE)
    # Split on sentence boundaries
    sentences = re.split(r"(?<=[.!?])\s+", cleaned)
    # Filter out very short fragments and list markers
    assertions = []
    for s in sentences:
        s = re.sub(r"^[-*\u2022]\s*", "", s).strip()
        if len(s.split()) >= 4:  # at least 4 words to be a meaningful assertion
            assertions.append(s)
    return assertions


def _extract_keywords(text: str) -> Set[str]:
    """Extract meaningful keywords (3+ chars, no stopwords)."""
    stopwords = {
        "the", "and", "for", "are", "but", "not", "you", "all", "can", "had",
        "her", "was", "one", "our", "out", "has", "its", "that", "this", "with",
        "will", "from", "they", "been", "have", "some", "than", "into", "also",
        "such", "which", "their", "these", "about", "would", "there", "could",
        "other", "more", "very", "when", "what", "your", "each", "does", "those",
        "should", "being", "between", "after", "before", "where", "over",
        "page", "document", "section",
    }
    words = re.findall(r"[a-z]{3,}", text.lower())
    return {w for w in words if w not in stopwords}


def _compute_overlap_score(
    assertion: str,
    context_keywords: Set[str],
    context_text: str,
) -> float:
    """
    Compute a 0-1 score for how well an assertion is supported by context.

    Uses a blend of:
      - Keyword overlap (what fraction of assertion keywords appear in context)
      - Bigram overlap (consecutive word pairs)
      - Substring containment (for exact phrases)
    """
    assertion_lower = assertion.lower()
    assertion_words = re.findall(r"[a-z]{3,}", assertion_lower)

    if not assertion_words:
        return 0.5  # can't evaluate, assume partial

    # Keyword overlap
    assertion_keywords = _extract_keywords(assertion_lower)
    if assertion_keywords:
        kw_overlap = len(assertion_keywords & context_keywords) / len(assertion_keywords)
    else:
        kw_overlap = 0.0

    # Bigram overlap
    def bigrams(words):
        return [f"{words[i]} {words[i+1]}" for i in range(len(words) - 1)]

    a_bigrams = set(bigrams(assertion_words))
    c_bigrams = set(bigrams(re.findall(r"[a-z]{3,}", context_text)))
    if a_bigrams:
        bg_overlap = len(a_bigrams & c_bigrams) / len(a_bigrams)
    else:
        bg_overlap = 0.0

    # Exact 4-word phrase containment
    phrase_score = 0.0
    four_grams = [" ".join(assertion_words[i:i+4]) for i in range(len(assertion_words) - 3)]
    if four_grams:
        matches = sum(1 for fg in four_grams if fg in context_text)
        phrase_score = matches / len(four_grams)

    # Weighted blend
    return 0.4 * kw_overlap + 0.3 * bg_overlap + 0.3 * phrase_score


def _llm_verify(
    answer: str,
    chunks: List[RetrievedChunk],
    llm_client: "LLMClient",
) -> Optional[Dict]:
    """
    Use a quick LLM call to semantically verify answer grounding.
    Returns a dict with confidence, reasoning, and flagged claims.
    """
    import json

    context_block = "\n---\n".join(
        f"[Chunk from Page {ch.page}]\n{ch.text[:500]}" for ch in chunks[:4]
    )

    system = (
        "You are a verification analyst. Given an ANSWER and SOURCE CONTEXT, "
        "evaluate whether every claim in the ANSWER is supported by the CONTEXT.\n\n"
        "Respond ONLY with a JSON object (no markdown fences):\n"
        '{"confidence": 0.0-1.0, "reasoning": "brief explanation", "flagged": ["unsupported claim 1", ...]}'
    )
    user = f"ANSWER:\n{answer}\n\nSOURCE CONTEXT:\n{context_block}"

    raw = llm_client.call(system, user)

    # Strip markdown code fences if present
    raw = re.sub(r"```json\s*", "", raw)
    raw = re.sub(r"```\s*$", "", raw)

    try:
        result = json.loads(raw.strip())
        if "confidence" in result:
            result["confidence"] = max(0.0, min(1.0, float(result["confidence"])))
        return result
    except (json.JSONDecodeError, ValueError):
        return None
