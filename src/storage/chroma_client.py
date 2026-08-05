"""
ChromaDB Client — wrapper for the persistent vector store.

Manages the vault_chunks collection with cosine similarity.
We provide our own embeddings from the embedding service —
ChromaDB's built-in embedding function is NOT used.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import chromadb

logger = logging.getLogger(__name__)


class ChromaClient:
    """Wrapper around ChromaDB PersistentClient for vault chunk storage."""

    def __init__(self, persist_path: str | Path, collection_name: str = "vault_chunks"):
        self._persist_path = str(persist_path)
        self._collection_name = collection_name
        self._client: chromadb.ClientAPI | None = None
        self._collection: chromadb.Collection | None = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def initialize(self) -> None:
        """Create the persistent client and get/create the collection."""
        Path(self._persist_path).mkdir(parents=True, exist_ok=True)

        self._client = chromadb.PersistentClient(path=self._persist_path)
        self._collection = self._client.get_or_create_collection(
            name=self._collection_name,
            metadata={"hnsw:space": "cosine"},  # cosine distance metric
        )
        count = self._collection.count()
        logger.info(
            "ChromaDB collection '%s' ready (%d vectors) at %s",
            self._collection_name, count, self._persist_path,
        )

    @property
    def collection(self) -> chromadb.Collection:
        """Get the active collection. Raises if not initialized."""
        if self._collection is None:
            raise RuntimeError("ChromaClient not initialized. Call .initialize() first.")
        return self._collection

    # ------------------------------------------------------------------
    # CRUD operations
    # ------------------------------------------------------------------

    def upsert_chunks(
        self,
        ids: list[str],
        documents: list[str],
        embeddings: list[list[float]],
        metadatas: list[dict[str, Any]] | None = None,
    ) -> None:
        """
        Insert or update chunks in the vector store.

        Args:
            ids: Chunk IDs (must match SQLite chunk_ids).
            documents: Raw text content for each chunk.
            embeddings: Pre-computed 384-dim embedding vectors.
            metadatas: Optional metadata dicts (file_path, heading, etc.).
        """
        if not ids:
            return

        self.collection.upsert(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas or [{}] * len(ids),
        )
        logger.debug("Upserted %d chunks into ChromaDB", len(ids))

    def delete_chunks(self, ids: list[str]) -> None:
        """Delete chunks by their IDs."""
        if not ids:
            return
        self.collection.delete(ids=ids)
        logger.debug("Deleted %d chunks from ChromaDB", len(ids))

    def query(
        self,
        query_embedding: list[float],
        n_results: int = 10,
        where_filter: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Query the collection with a pre-computed embedding vector.

        Args:
            query_embedding: 384-dim normalized vector for the query.
            n_results: Maximum number of results to return.
            where_filter: Optional ChromaDB where filter on metadata.

        Returns:
            ChromaDB results dict with keys: ids, distances, documents, metadatas.
        """
        count = self.collection.count()
        if count == 0:
            return {"ids": [[]], "distances": [[]], "documents": [[]], "metadatas": [[]]}

        kwargs: dict[str, Any] = {
            "query_embeddings": [query_embedding],
            "n_results": min(n_results, count),
        }
        if where_filter:
            kwargs["where"] = where_filter

        results = self.collection.query(**kwargs)
        return results

    def get_by_ids(self, ids: list[str]) -> dict[str, Any]:
        """Fetch specific chunks by their IDs."""
        if not ids:
            return {"ids": [], "documents": [], "metadatas": []}
        return self.collection.get(ids=ids, include=["documents", "metadatas"])

    def count(self) -> int:
        """Return the total number of vectors in the collection."""
        return self.collection.count()

    def reset_collection(self) -> None:
        """Delete and recreate the collection (destructive!)."""
        if self._client:
            self._client.delete_collection(self._collection_name)
            self._collection = self._client.get_or_create_collection(
                name=self._collection_name,
                metadata={"hnsw:space": "cosine"},
            )
            logger.warning("ChromaDB collection '%s' was reset.", self._collection_name)
