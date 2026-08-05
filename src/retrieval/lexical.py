"""
Lexical Search — BM25 keyword search arm.

Builds an in-memory BM25 index from all chunks and supports
fast keyword-based retrieval. The index is rebuilt after each
ingestion cycle.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass

from rank_bm25 import BM25Okapi

logger = logging.getLogger(__name__)


@dataclass
class LexicalResult:
    """A single result from BM25 search."""
    chunk_id: str
    score: float   # Normalized BM25 score [0, 1]
    content: str


# ---------------------------------------------------------------------------
# Tokenizer — simple but effective for personal notes
# ---------------------------------------------------------------------------
_SPLIT_PATTERN = re.compile(r"[^\w]+")

# Common English stopwords to filter out
_STOPWORDS = frozenset({
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "is", "are", "was", "were", "be", "been",
    "being", "have", "has", "had", "do", "does", "did", "will", "would",
    "could", "should", "may", "might", "can", "shall", "this", "that",
    "these", "those", "it", "its", "i", "me", "my", "we", "our", "you",
    "your", "he", "she", "they", "them", "their", "not", "no", "if",
    "then", "than", "so", "as", "just", "also", "about", "up", "out",
})


def _tokenize(text: str) -> list[str]:
    """Lowercase, split on non-word chars, remove stopwords."""
    tokens = _SPLIT_PATTERN.split(text.lower())
    return [t for t in tokens if t and t not in _STOPWORDS and len(t) > 1]


# ---------------------------------------------------------------------------
# BM25 Index
# ---------------------------------------------------------------------------
class LexicalIndex:
    """
    In-memory BM25 index over all vault chunks.

    Must be rebuilt after ingestion changes (add/update/remove chunks).
    """

    def __init__(self):
        self._bm25: BM25Okapi | None = None
        self._chunk_ids: list[str] = []
        self._chunk_contents: list[str] = []
        self._corpus_tokens: list[list[str]] = []

    @property
    def size(self) -> int:
        """Number of documents in the index."""
        return len(self._chunk_ids)

    def build_index(self, chunks: list[dict]) -> None:
        """
        Build the BM25 index from a list of chunk dicts.

        Each dict must have at minimum: chunk_id, content.
        """
        self._chunk_ids = []
        self._chunk_contents = []
        self._corpus_tokens = []

        for chunk in chunks:
            chunk_id = chunk.get("chunk_id", "")
            content = chunk.get("content", "")
            if not chunk_id or not content:
                continue

            tokens = _tokenize(content)
            self._chunk_ids.append(chunk_id)
            self._chunk_contents.append(content)
            self._corpus_tokens.append(tokens)

        if self._corpus_tokens:
            self._bm25 = BM25Okapi(self._corpus_tokens)
        else:
            self._bm25 = None

        logger.info("BM25 index built with %d documents", len(self._chunk_ids))

    def search(self, query: str, top_k: int = 10) -> list[LexicalResult]:
        """
        Search the BM25 index.

        Scores are normalized to [0, 1] by dividing by the max score.
        Returns top-K results sorted by score descending.
        """
        if self._bm25 is None or not self._chunk_ids:
            return []

        query_tokens = _tokenize(query)
        if not query_tokens:
            return []

        scores = self._bm25.get_scores(query_tokens)

        # Normalize scores to [0, 1]
        max_score = max(scores) if max(scores) > 0 else 1.0
        normalized = [s / max_score for s in scores]

        # Get top-K results
        scored_indices = sorted(
            range(len(normalized)),
            key=lambda i: normalized[i],
            reverse=True,
        )[:top_k]

        results: list[LexicalResult] = []
        for idx in scored_indices:
            if normalized[idx] <= 0:
                break  # No point including zero-score results
            results.append(LexicalResult(
                chunk_id=self._chunk_ids[idx],
                score=normalized[idx],
                content=self._chunk_contents[idx],
            ))

        logger.debug("BM25 search: '%s' → %d results", query, len(results))
        return results
