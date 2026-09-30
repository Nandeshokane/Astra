"""
ASTRA INTEL — Vector Store (Day 2)
=====================================
ChromaDB-backed vector store with sentence-transformer embeddings.
Supports multi-document indexing, per-document metadata filtering,
session cleanup, and semantic retrieval with similarity scores.
"""

from __future__ import annotations

import os
import uuid
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set

import chromadb
from chromadb.utils import embedding_functions

from core.document_loader import DocumentChunk


# ── Data Structures ─────────────────────────────────────────────────────────

@dataclass
class RetrievedChunk:
    """A chunk returned by a similarity query."""
    text: str
    source: str
    page: int
    chunk_index: int
    distance: float         # lower = more similar
    relevance_score: float  # normalised 0-1 (1 = most relevant)


# ── Vector Store ────────────────────────────────────────────────────────────

COLLECTION_NAME = "astra_intel_docs"
DEFAULT_LOCAL_MODEL = "all-MiniLM-L6-v2"


class AstraVectorStore:
    """
    ChromaDB collection manager for ASTRA INTEL.

    Supports multi-document workspaces with per-source filtering.
    """

    def __init__(
        self,
        persist_directory: Optional[str] = None,
        embedding_model: str = DEFAULT_LOCAL_MODEL,
    ):
        self.embedding_model_name = embedding_model

        self._ef = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=embedding_model,
        )

        if persist_directory:
            os.makedirs(persist_directory, exist_ok=True)
            self._client = chromadb.PersistentClient(path=persist_directory)
        else:
            self._client = chromadb.EphemeralClient()

        self._collection = self._client.get_or_create_collection(
            name=COLLECTION_NAME,
            embedding_function=self._ef,
            metadata={"hnsw:space": "cosine"},
        )

        # Track which documents are indexed
        self._indexed_docs: Set[str] = set()
        self._refresh_indexed_docs()

    # ── Internal Helpers ────────────────────────────────────────────────────

    def _refresh_indexed_docs(self) -> None:
        """Scan collection metadata to rebuild the set of indexed doc names."""
        if self._collection.count() == 0:
            self._indexed_docs = set()
            return
        try:
            result = self._collection.get(
                limit=self._collection.count(),
                include=["metadatas"],
            )
            metas = result.get("metadatas", [])
            self._indexed_docs = {m.get("source", "") for m in metas if m}
        except Exception:
            self._indexed_docs = set()

    # ── Public API ──────────────────────────────────────────────────────────

    def reset(self) -> None:
        """Full reset — delete and recreate the collection."""
        self._client.delete_collection(COLLECTION_NAME)
        self._collection = self._client.get_or_create_collection(
            name=COLLECTION_NAME,
            embedding_function=self._ef,
            metadata={"hnsw:space": "cosine"},
        )
        self._indexed_docs = set()

    def remove_document(self, doc_name: str) -> int:
        """
        Remove all chunks belonging to *doc_name* from the collection.

        Returns the number of chunks removed.
        """
        if self._collection.count() == 0:
            return 0

        # Fetch IDs for this document
        result = self._collection.get(
            where={"source": doc_name},
            include=[],
        )
        ids = result.get("ids", [])
        if ids:
            self._collection.delete(ids=ids)
        self._indexed_docs.discard(doc_name)
        return len(ids)

    def add_chunks(self, chunks: List[DocumentChunk]) -> int:
        """Embed and index a list of DocumentChunks. Returns count added."""
        if not chunks:
            return 0

        ids: List[str] = []
        documents: List[str] = []
        metadatas: List[Dict[str, Any]] = []

        for chunk in chunks:
            uid = f"{chunk.chunk_id}::{uuid.uuid4().hex[:8]}"
            ids.append(uid)
            documents.append(chunk.text)
            metadatas.append({
                "source": chunk.source,
                "page": chunk.page,
                "chunk_index": chunk.chunk_index,
            })

        self._collection.add(
            ids=ids,
            documents=documents,
            metadatas=metadatas,
        )

        # Track the document name(s)
        for chunk in chunks:
            self._indexed_docs.add(chunk.source)

        return len(ids)

    def query(
        self,
        query_text: str,
        k: int = 4,
        source_filter: Optional[str] = None,
    ) -> List[RetrievedChunk]:
        """
        Retrieve the top-*k* most relevant chunks.

        Parameters
        ----------
        source_filter : str or None
            If set, only return chunks from this specific document.
        """
        total = self._collection.count()
        if total == 0:
            return []

        where = None
        if source_filter:
            where = {"source": source_filter}

        results = self._collection.query(
            query_texts=[query_text],
            n_results=min(k, total),
            include=["documents", "metadatas", "distances"],
            where=where,
        )

        retrieved: List[RetrievedChunk] = []
        docs = results.get("documents", [[]])[0]
        metas = results.get("metadatas", [[]])[0]
        dists = results.get("distances", [[]])[0]

        for doc, meta, dist in zip(docs, metas, dists):
            relevance = max(0.0, 1.0 - dist)
            retrieved.append(RetrievedChunk(
                text=doc,
                source=meta.get("source", "unknown"),
                page=int(meta.get("page", 0)),
                chunk_index=int(meta.get("chunk_index", 0)),
                distance=dist,
                relevance_score=round(relevance, 4),
            ))

        retrieved.sort(key=lambda r: r.relevance_score, reverse=True)
        return retrieved

    def query_cross_document(
        self,
        query_text: str,
        k_per_doc: int = 3,
    ) -> Dict[str, List[RetrievedChunk]]:
        """
        Retrieve top chunks from each indexed document separately.

        Returns a dict mapping document name -> list of RetrievedChunks.
        Useful for comparative / cross-document analysis.
        """
        results: Dict[str, List[RetrievedChunk]] = {}
        for doc_name in self._indexed_docs:
            chunks = self.query(query_text, k=k_per_doc, source_filter=doc_name)
            if chunks:
                results[doc_name] = chunks
        return results

    @property
    def count(self) -> int:
        return self._collection.count()

    @property
    def indexed_documents(self) -> List[str]:
        """List of document names currently indexed."""
        return sorted(self._indexed_docs)

    def get_all_texts(self, limit: int = 100) -> List[str]:
        if self.count == 0:
            return []
        result = self._collection.get(
            limit=limit,
            include=["documents"],
        )
        return result.get("documents", [])
