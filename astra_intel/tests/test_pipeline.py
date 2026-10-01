"""
ASTRA INTEL — Automated Pipeline Test Suite
=============================================
Standalone test script that validates the full ingestion → chunking → embedding →
retrieval → generation pipeline *without* launching the Streamlit UI.

Run with:
    python -m pytest tests/test_pipeline.py -v --tb=short

Test Matrix
-----------
TC-1: Standard technical PDF ingestion, summarisation, and query extraction.
TC-2: Corrupted / empty PDF handling (graceful error capture).
TC-3: Image-only / scanned PDF detection (low-character-count warning).
TC-4: Hallucination / out-of-domain query (anti-hallucination guardrail).
TC-5: Citation accuracy (page metadata matches the source page range).
"""

from __future__ import annotations

import os
import sys
import textwrap

import pytest

# ── Ensure project root is on sys.path so `core.*` imports resolve ──────────
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from core.document_loader import (
    DocumentChunk,
    DocumentLoadError,
    LoadedDocument,
    ScannedOrEmptyPDFError,
    load_pdf,
    get_doc_statistics,
)
from core.vector_store import AstraVectorStore, RetrievedChunk
from core.verification import GroundingLevel, GroundingResult, verify_grounding
from core.rag_pipeline import (
    _extract_citations,
    AnswerResult,
    Citation,
)


# ═══════════════════════════════════════════════════════════════════════════════
# Helper — Build synthetic PDFs entirely in-memory via PyMuPDF
# ═══════════════════════════════════════════════════════════════════════════════

def _create_pdf_bytes(pages: list[str]) -> bytes:
    """
    Create a valid in-memory PDF with the given list of page texts.
    Each string in *pages* becomes the text content of one page.
    """
    import pymupdf

    doc = pymupdf.open()
    for text in pages:
        page = doc.new_page(width=612, height=792)
        page.insert_text((72, 72), text, fontsize=10)
    data = doc.tobytes()
    doc.close()
    return data


SAMPLE_DEFENCE_TEXT_P1 = textwrap.dedent("""\
    MQ-9 Reaper Unmanned Aerial Vehicle — Technical Overview (Unclassified)

    1. System Description
    The MQ-9 Reaper is a remotely-piloted aircraft system manufactured by
    General Atomics Aeronautical Systems Inc. It is designed for long-endurance,
    high-altitude surveillance and precision strike missions. The aircraft has
    a wingspan of 20 metres, a maximum altitude ceiling of 50,000 feet, and is
    powered by a Honeywell TPE331-10 turboprop engine producing 900 shaft
    horsepower. Maximum endurance exceeds 27 hours with a combat radius of
    1,150 nautical miles.

    2. Sensor Suite
    The Reaper carries the Raytheon Multi-Spectral Targeting System (MTS-B),
    which integrates an infrared sensor, a colour/monochrome daylight TV camera,
    an image-intensified TV camera, a laser designator, and a laser illuminator.
    This sensor ball enables full-motion video (FMV) in day or night conditions.
""")

SAMPLE_DEFENCE_TEXT_P2 = textwrap.dedent("""\
    3. Armament
    The MQ-9 can carry up to 1,746 kg of external stores across seven hardpoints.
    Typical loadouts include four AGM-114 Hellfire missiles and two GBU-12
    Paveway II laser-guided bombs. The GBU-38 Joint Direct Attack Munition (JDAM)
    is also cleared for employment. All ordnance is released through the
    Raytheon Common Ground Control Station (GCS).

    4. Communications Architecture
    Command and control uplinks use Ku-band SATCOM for beyond-line-of-sight
    operations and C-band line-of-sight data links for takeoff and landing.
    Intelligence dissemination leverages the Tactical Common Data Link (TCDL).

    5. Operational Constraints
    The MQ-9 is not designed for contested airspace and lacks self-protection
    jamming or stealth characteristics. It relies on air superiority being
    established prior to employment.
""")


# ═══════════════════════════════════════════════════════════════════════════════
# TC-1: Standard Technical PDF — Ingestion, Summarisation, Query Extraction
# ═══════════════════════════════════════════════════════════════════════════════

class TestStandardPDFIngestion:
    """Verify the full pipeline from PDF bytes → chunks with metadata."""

    @pytest.fixture(autouse=True)
    def setup(self):
        self.pdf_bytes = _create_pdf_bytes([SAMPLE_DEFENCE_TEXT_P1, SAMPLE_DEFENCE_TEXT_P2])

    def test_load_returns_loaded_document(self):
        doc = load_pdf(self.pdf_bytes, doc_name="MQ9_Reaper_Overview.pdf")
        assert isinstance(doc, LoadedDocument)
        assert doc.doc_name == "MQ9_Reaper_Overview.pdf"

    def test_correct_page_count(self):
        doc = load_pdf(self.pdf_bytes, doc_name="MQ9_Reaper_Overview.pdf")
        assert doc.total_pages == 2, f"Expected 2 pages, got {doc.total_pages}"

    def test_pages_have_text(self):
        doc = load_pdf(self.pdf_bytes, doc_name="MQ9_Reaper_Overview.pdf")
        for page in doc.pages:
            assert page.char_count > 0, f"Page {page.page_number} has no text"

    def test_chunks_created(self):
        doc = load_pdf(self.pdf_bytes, doc_name="MQ9_Reaper_Overview.pdf")
        assert len(doc.chunks) >= 1, "No chunks were produced"

    def test_chunk_metadata_integrity(self):
        doc = load_pdf(self.pdf_bytes, doc_name="MQ9_Reaper_Overview.pdf")
        for chunk in doc.chunks:
            assert chunk.source == "MQ9_Reaper_Overview.pdf"
            assert chunk.page in (1, 2), f"Unexpected page number {chunk.page}"
            assert chunk.chunk_id.startswith("MQ9_Reaper_Overview.pdf::")

    def test_word_count_reasonable(self):
        doc = load_pdf(self.pdf_bytes, doc_name="MQ9_Reaper_Overview.pdf")
        assert doc.word_count >= 50, f"Only {doc.word_count} words extracted"

    def test_get_doc_statistics(self):
        doc = load_pdf(self.pdf_bytes, doc_name="MQ9_Reaper_Overview.pdf")
        stats = get_doc_statistics(doc)
        assert stats["doc_name"] == "MQ9_Reaper_Overview.pdf"
        assert stats["total_pages"] == 2
        assert stats["total_chunks"] == len(doc.chunks)
        assert stats["total_words"] == doc.word_count
        assert stats["avg_chunk_len"] > 0

    def test_vector_store_indexing_and_retrieval(self):
        """End-to-end: ingest → embed → query → retrieve relevant chunks."""
        doc = load_pdf(self.pdf_bytes, doc_name="MQ9_Reaper_Overview.pdf")
        vs = AstraVectorStore()
        count = vs.add_chunks(doc.chunks)
        assert count == len(doc.chunks)
        assert vs.count == count

        results = vs.query("What is the maximum altitude of the MQ-9?", k=3)
        assert len(results) > 0
        assert any("altitude" in r.text.lower() or "50,000" in r.text for r in results), \
            "Expected altitude-related content in top retrieval results"

        # Relevance scores should be between 0 and 1
        for r in results:
            assert 0.0 <= r.relevance_score <= 1.0


# ═══════════════════════════════════════════════════════════════════════════════
# TC-2: Corrupted / Empty PDF Handling
# ═══════════════════════════════════════════════════════════════════════════════

class TestCorruptedAndEmptyPDF:
    """Verify graceful error handling for malformed inputs."""

    def test_random_bytes_raises_document_load_error(self):
        """Completely garbage bytes should raise DocumentLoadError."""
        with pytest.raises(DocumentLoadError, match="corrupt|password-protected|Failed"):
            load_pdf(b"THIS IS NOT A PDF AT ALL", doc_name="garbage.pdf")

    def test_empty_bytes_raises_error(self):
        """Zero-length input should raise DocumentLoadError."""
        with pytest.raises(DocumentLoadError):
            load_pdf(b"", doc_name="empty.pdf")

    def test_truncated_pdf_header_raises_error(self):
        """A PDF header fragment without a valid body."""
        with pytest.raises(DocumentLoadError):
            load_pdf(b"%PDF-1.4\n%%EOF", doc_name="truncated.pdf")

    def test_valid_pdf_with_blank_pages(self):
        """A valid PDF whose pages contain no text should raise ScannedOrEmptyPDFError."""
        pdf_bytes = _create_pdf_bytes(["", "", ""])
        with pytest.raises((ScannedOrEmptyPDFError, DocumentLoadError)):
            load_pdf(pdf_bytes, doc_name="blank_pages.pdf")


# ═══════════════════════════════════════════════════════════════════════════════
# TC-3: Image-Only / Scanned PDF Detection
# ═══════════════════════════════════════════════════════════════════════════════

class TestScannedPDFDetection:
    """
    Verify that PDFs with extremely low text content (simulating scanned
    image-only documents) trigger the ScannedOrEmptyPDFError with the
    appropriate low-character-count warning.
    """

    def test_near_empty_text_triggers_warning(self):
        """A PDF with fewer than min_word_threshold words should be rejected."""
        # Insert only a few words — below the default threshold of 50
        pdf_bytes = _create_pdf_bytes(["OK", "Yes"])
        with pytest.raises(ScannedOrEmptyPDFError, match="Insufficient extractable text"):
            load_pdf(pdf_bytes, doc_name="scanned_document.pdf")

    def test_error_message_contains_word_count(self):
        """Error message should report the actual word count for diagnosis."""
        pdf_bytes = _create_pdf_bytes(["Three words only"])
        try:
            load_pdf(pdf_bytes, doc_name="low_text.pdf")
            pytest.fail("Expected ScannedOrEmptyPDFError was not raised")
        except ScannedOrEmptyPDFError as exc:
            msg = str(exc)
            assert "word" in msg.lower(), "Error should mention word count"
            assert "low_text.pdf" in msg, "Error should mention the document name"

    def test_custom_threshold_tighter(self):
        """Raising the threshold should reject documents that would otherwise pass."""
        # 10 words of content — passes default (50) only if we set threshold higher
        text = "Alpha Bravo Charlie Delta Echo Foxtrot Golf Hotel India Juliet Kilo Lima Mike November"
        pdf_bytes = _create_pdf_bytes([text])
        # With a very high threshold, even this should be rejected
        with pytest.raises(ScannedOrEmptyPDFError):
            load_pdf(pdf_bytes, doc_name="borderline.pdf", min_word_threshold=500)

    def test_sufficient_text_passes(self):
        """A document clearly above the threshold should load successfully."""
        pdf_bytes = _create_pdf_bytes([SAMPLE_DEFENCE_TEXT_P1])
        doc = load_pdf(pdf_bytes, doc_name="good_doc.pdf")
        assert doc.word_count >= 50


# ═══════════════════════════════════════════════════════════════════════════════
# TC-4: Hallucination / Out-of-Domain Query — Anti-Hallucination Guardrail
# ═══════════════════════════════════════════════════════════════════════════════

class TestAntiHallucination:
    """
    Verify that the grounding verification engine correctly flags or rejects
    answers that are not supported by the source context.
    """

    @pytest.fixture
    def defence_chunks(self) -> list[RetrievedChunk]:
        """Simulated retrieval results from a drone whitepaper."""
        return [
            RetrievedChunk(
                text="The MQ-9 Reaper has a wingspan of 20 metres and a maximum altitude of 50,000 feet.",
                source="MQ9_Overview.pdf",
                page=1,
                chunk_index=0,
                distance=0.15,
                relevance_score=0.85,
            ),
            RetrievedChunk(
                text="The aircraft is powered by a Honeywell TPE331-10 turboprop engine producing 900 shaft horsepower.",
                source="MQ9_Overview.pdf",
                page=1,
                chunk_index=1,
                distance=0.20,
                relevance_score=0.80,
            ),
        ]

    def test_grounded_answer_passes(self, defence_chunks):
        """An answer that uses only context content should score HIGH."""
        grounded_answer = (
            "The MQ-9 Reaper has a wingspan of 20 metres and operates at a "
            "maximum altitude of 50,000 feet [Page 1]. It is powered by a "
            "Honeywell TPE331-10 turboprop engine [Page 1]."
        )
        result = verify_grounding(grounded_answer, defence_chunks)
        assert result.is_grounded, f"Expected grounded, got level={result.level}"
        assert result.confidence >= 0.4

    def test_out_of_domain_answer_flagged(self, defence_chunks):
        """An answer about baking recipes should be flagged as ungrounded."""
        hallucinated_answer = (
            "To bake the perfect sourdough bread, preheat your oven to 230°C. "
            "Combine 500g of bread flour with 350ml of warm water and 10g of salt. "
            "Allow the dough to rise for 12 hours at room temperature."
        )
        result = verify_grounding(hallucinated_answer, defence_chunks)
        # Must not be rated HIGH
        assert result.level in (GroundingLevel.LOW, GroundingLevel.UNGROUNDED), \
            f"Hallucinated answer should be LOW or UNGROUNDED, got {result.level}"
        assert result.confidence < 0.65

    def test_refusal_response_is_grounded(self, defence_chunks):
        """A proper refusal should be recognized as correctly grounded."""
        refusal = "The uploaded document does not contain sufficient information to answer this question."
        result = verify_grounding(refusal, defence_chunks)
        assert result.is_grounded is True
        assert result.level == GroundingLevel.HIGH
        assert result.confidence == 1.0

    def test_partial_hallucination_detected(self, defence_chunks):
        """An answer mixing real facts with fabricated ones should score lower."""
        mixed_answer = (
            "The MQ-9 Reaper has a wingspan of 20 metres [Page 1]. "
            "It was first deployed in the Falklands War in 1982 and can carry "
            "nuclear warheads with a yield of 50 kilotons."
        )
        result = verify_grounding(mixed_answer, defence_chunks)
        # Should have flagged claims
        assert len(result.flagged_claims) > 0 or result.confidence < 0.65, \
            "Partially hallucinated answer should have flagged claims or low confidence"


# ═══════════════════════════════════════════════════════════════════════════════
# TC-5: Citation Accuracy — Page Metadata Matches Source Range
# ═══════════════════════════════════════════════════════════════════════════════

class TestCitationAccuracy:
    """
    Verify that the citation extraction logic correctly parses page references
    from LLM answers and maps them back to the correct source chunks.
    """

    @pytest.fixture
    def multi_page_chunks(self) -> list[RetrievedChunk]:
        return [
            RetrievedChunk(
                text="The radar operates at X-band frequency with a detection range of 400 km.",
                source="radar_specs.pdf",
                page=3,
                chunk_index=0,
                distance=0.12,
                relevance_score=0.88,
            ),
            RetrievedChunk(
                text="The electronic countermeasures suite includes a digital RF memory jammer.",
                source="radar_specs.pdf",
                page=7,
                chunk_index=0,
                distance=0.18,
                relevance_score=0.82,
            ),
            RetrievedChunk(
                text="The fire control system integrates with the AN/APG-83 AESA radar.",
                source="fire_control.pdf",
                page=2,
                chunk_index=0,
                distance=0.25,
                relevance_score=0.75,
            ),
        ]

    def test_single_page_citation_parsed(self, multi_page_chunks):
        answer = "The radar operates at X-band frequency [Page 3]."
        citations = _extract_citations(answer, multi_page_chunks)
        page_3_cits = [c for c in citations if c.page == 3]
        assert len(page_3_cits) >= 1, "Page 3 citation not extracted"
        assert page_3_cits[0].source == "radar_specs.pdf"

    def test_multi_page_citation_parsed(self, multi_page_chunks):
        answer = "The radar system [Page 3] includes ECM capabilities [Page 7]."
        citations = _extract_citations(answer, multi_page_chunks)
        cited_pages = {c.page for c in citations}
        assert 3 in cited_pages, "Page 3 missing from citations"
        assert 7 in cited_pages, "Page 7 missing from citations"

    def test_doc_specific_citation_parsed(self, multi_page_chunks):
        answer = "The fire control integration [Page 2, fire_control.pdf] is notable."
        citations = _extract_citations(answer, multi_page_chunks)
        fc_cits = [c for c in citations if c.source == "fire_control.pdf"]
        assert len(fc_cits) >= 1
        assert fc_cits[0].page == 2

    def test_uncited_chunks_still_appear_for_transparency(self, multi_page_chunks):
        """Chunks not explicitly cited should still appear in the citation list."""
        answer = "The radar works at X-band [Page 3]."  # only cites page 3
        citations = _extract_citations(answer, multi_page_chunks)
        # All three source chunks should appear (page 3 + page 7 + page 2)
        all_pages = {c.page for c in citations}
        assert len(all_pages) >= 2, "Unreferenced chunks should be included for transparency"

    def test_citation_snippets_not_empty(self, multi_page_chunks):
        answer = "The detection range is 400 km [Page 3]."
        citations = _extract_citations(answer, multi_page_chunks)
        for c in citations:
            assert c.snippet, f"Citation for page {c.page} has empty snippet"
            assert len(c.snippet) > 10, f"Citation snippet too short: '{c.snippet}'"

    def test_citation_relevance_scores_populated(self, multi_page_chunks):
        answer = "The radar [Page 3] and ECM [Page 7] work together."
        citations = _extract_citations(answer, multi_page_chunks)
        for c in citations:
            assert 0.0 <= c.relevance <= 1.0, \
                f"Relevance score {c.relevance} out of range for page {c.page}"

    def test_no_duplicate_citations(self, multi_page_chunks):
        """Repeated page references should not create duplicate citation entries."""
        answer = (
            "The radar [Page 3] operates at X-band. "
            "Again, the X-band radar [Page 3] detects at 400 km."
        )
        citations = _extract_citations(answer, multi_page_chunks)
        keys = [(c.source, c.page) for c in citations]
        assert len(keys) == len(set(keys)), "Duplicate citations detected"


# ═══════════════════════════════════════════════════════════════════════════════
# TC-Bonus: Vector Store Operations
# ═══════════════════════════════════════════════════════════════════════════════

class TestVectorStoreOperations:
    """Additional integration tests for the vector store."""

    @pytest.fixture
    def populated_store(self) -> tuple[AstraVectorStore, LoadedDocument]:
        pdf_bytes = _create_pdf_bytes([SAMPLE_DEFENCE_TEXT_P1, SAMPLE_DEFENCE_TEXT_P2])
        doc = load_pdf(pdf_bytes, doc_name="test_doc.pdf")
        vs = AstraVectorStore()
        vs.add_chunks(doc.chunks)
        return vs, doc

    def test_document_tracking(self, populated_store):
        vs, doc = populated_store
        assert "test_doc.pdf" in vs.indexed_documents

    def test_reset_clears_everything(self, populated_store):
        vs, _ = populated_store
        assert vs.count > 0
        vs.reset()
        assert vs.count == 0
        assert len(vs.indexed_documents) == 0

    def test_empty_query_returns_empty(self):
        vs = AstraVectorStore()
        results = vs.query("anything", k=3)
        assert results == []

    def test_source_filter(self, populated_store):
        vs, _ = populated_store
        results = vs.query("engine", k=3, source_filter="test_doc.pdf")
        assert all(r.source == "test_doc.pdf" for r in results)

    def test_multi_document_indexing(self):
        """Index two documents and verify cross-document query."""
        pdf1 = _create_pdf_bytes([SAMPLE_DEFENCE_TEXT_P1])
        pdf2 = _create_pdf_bytes([SAMPLE_DEFENCE_TEXT_P2])
        doc1 = load_pdf(pdf1, doc_name="doc_A.pdf")
        doc2 = load_pdf(pdf2, doc_name="doc_B.pdf")

        vs = AstraVectorStore()
        vs.reset()  # ensure clean state (no cross-test leakage)
        vs.add_chunks(doc1.chunks)
        vs.add_chunks(doc2.chunks)

        assert len(vs.indexed_documents) == 2
        cross_results = vs.query_cross_document("armament", k_per_doc=2)
        assert isinstance(cross_results, dict)


# ═══════════════════════════════════════════════════════════════════════════════
# Run with: python -m pytest tests/test_pipeline.py -v --tb=short
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
