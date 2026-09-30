"""
ASTRA INTEL — Vector Store
============================
ChromaDB-backed vector store with sentence-transformer embeddings.
Handles indexing, reset, and semantic retrieval with similarity scores.
"""

from __future__ import annotations

import os
import uuid
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

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
    distance: float       # lower = more similar (L2 / cosine depending on config)
    relevance_score: float  # normalised 0-1 score (1 = most relevant)


# ── Vector Store ────────────────────────────────────────────────────────────

COLLECTION_NAME = "astra_intel_docs"

# Embedding model used when running locally
DEFAULT_LOCAL_MODEL = "all-MiniLM-L6-v2"


class AstraVectorStore:
    """
    Wraps a ChromaDB collection for ASTRA INTEL.

    Parameters
    ----------
    persist_directory : str or None
        Path for persistent storage.  ``None`` = in-memory only.
    embedding_model : str
        HuggingFace sentence-transformers model name.
    """

    def __init__(
        self,
        persist_directory: Optional[str] = None,
        embedding_model: str = DEFAULT_LOCAL_MODEL,
    ):
        self.embedding_model_name = embedding_model

        # Build the embedding function
        self._ef = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=embedding_model,
        )

        # Create client
        if persist_directory:
            os.makedirs(persist_directory, exist_ok=True)
            self._client = chromadb.PersistentClient(path=persist_directory)
        else:
            self._client = chromadb.EphemeralClient()

        # Get-or-create the collection
        self._collection = self._client.get_or_create_collection(
            name=COLLECTION_NAME,
            embedding_function=self._ef,
            metadata={"hnsw:space": "cosine"},
        )

    # ── Public API ──────────────────────────────────────────────────────────

    def reset(self) -> None:
        """Delete the existing collection and recreate it (new-doc upload)."""
        self._client.delete_collection(COLLECTION_NAME)
        self._collection = self._client.get_or_create_collection(
            name=COLLECTION_NAME,
            embedding_function=self._ef,
            metadata={"hnsw:space": "cosine"},
        )

    def add_chunks(self, chunks: List[DocumentChunk]) -> int:
        """
        Embed and index a list of DocumentChunks.

        Returns the number of chunks successfully added.
        """
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
        return len(ids)

    def query(
        self,
        query_text: str,
        k: int = 4,
    ) -> List[RetrievedChunk]:
        """
        Retrieve the top-*k* most relevant chunks for *query_text*.

        Returns a list of ``RetrievedChunk`` objects sorted by relevance
        (most relevant first).
        """
        results = self._collection.query(
            query_texts=[query_text],
            n_results=min(k, self._collection.count() or 1),
            include=["documents", "metadatas", "distances"],
        )

        retrieved: List[RetrievedChunk] = []

        docs = results.get("documents", [[]])[0]
        metas = results.get("metadatas", [[]])[0]
        dists = results.get("distances", [[]])[0]

        for doc, meta, dist in zip(docs, metas, dists):
            # Cosine distance in [0, 2]; convert to a 0-1 relevance score
            relevance = max(0.0, 1.0 - dist)
            retrieved.append(RetrievedChunk(
                text=doc,
                source=meta.get("source", "unknown"),
                page=int(meta.get("page", 0)),
                chunk_index=int(meta.get("chunk_index", 0)),
                distance=dist,
                relevance_score=round(relevance, 4),
            ))

        # Sort by relevance descending (highest first)
        retrieved.sort(key=lambda r: r.relevance_score, reverse=True)
        return retrieved

    @property
    def count(self) -> int:
        """Number of documents currently indexed."""
        return self._collection.count()

    def get_all_texts(self, limit: int = 100) -> List[str]:
        """Return up to *limit* stored document texts (for summarisation)."""
        if self.count == 0:
            return []
        result = self._collection.get(
            limit=limit,
            include=["documents"],
        )
        return result.get("documents", [])
