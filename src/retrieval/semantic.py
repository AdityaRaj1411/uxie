"""
Semantic Search — ChromaDB vector search arm.

Queries the vector store with a pre-computed embedding and normalizes
cosine distances to similarity scores in [0, 1].
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from src.storage.chroma_client import ChromaClient

logger = logging.getLogger(__name__)


@dataclass
class SemanticResult:
    """A single result from semantic search."""
    chunk_id: str
    score: float       # Normalized similarity score 0–1
    distance: float    # Raw cosine distance from ChromaDB
    content: str
    file_path: str
    heading: str


async def semantic_search(
    query_embedding: list[float],
    chroma: ChromaClient,
    n_results: int = 20,
    distance_threshold: float = 0.5,
) -> list[SemanticResult]:
    """
    Search ChromaDB for semantically similar chunks.

    ChromaDB cosine distance: 0 = identical, 2 = opposite.
    We convert to similarity: score = 1 - (distance / 2), giving [0, 1].
    Then filter by the distance threshold.

    Args:
        query_embedding: Normalized 384-dim query vector.
        chroma: ChromaDB client instance.
        n_results: Max candidates to retrieve from ChromaDB.
        distance_threshold: Max cosine distance to accept (lower = stricter).

    Returns:
        List of SemanticResult sorted by score descending.
    """
    if chroma.count() == 0:
        return []

    results = chroma.query(
        query_embedding=query_embedding,
        n_results=min(n_results, chroma.count()),
    )

    # Unpack ChromaDB results (they come as lists-of-lists)
    ids = results.get("ids", [[]])[0]
    distances = results.get("distances", [[]])[0]
    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]

    semantic_results: list[SemanticResult] = []

    for chunk_id, distance, doc, meta in zip(ids, distances, documents, metadatas):
        # Filter by distance threshold
        if distance > distance_threshold:
            continue

        # Convert cosine distance to similarity score [0, 1]
        score = 1.0 - (distance / 2.0)

        semantic_results.append(SemanticResult(
            chunk_id=chunk_id,
            score=score,
            distance=distance,
            content=doc or "",
            file_path=meta.get("file_path", ""),
            heading=meta.get("heading", ""),
        ))

    # Sort by score descending
    semantic_results.sort(key=lambda r: r.score, reverse=True)

    logger.debug(
        "Semantic search: %d results (threshold=%.2f, candidates=%d)",
        len(semantic_results), distance_threshold, len(ids),
    )
    return semantic_results
