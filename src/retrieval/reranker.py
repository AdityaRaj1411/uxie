"""
Reranker — merges results from all retrieval arms and applies
the master scoring equation.

Final Score = (S_semantic × w_s) + (S_bm25 × w_b) + (I_importance × w_i) + (R_recency × w_r)

Default weights: 0.5 / 0.3 / 0.1 / 0.1  (configurable).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from src.retrieval.metadata import get_importance_scores
from src.retrieval.recency import compute_recency_scores
from src.retrieval.semantic import SemanticResult
from src.retrieval.lexical import LexicalResult
from src.storage.sqlite_client import SQLiteClient

logger = logging.getLogger(__name__)


@dataclass
class RankedChunk:
    """A chunk with its final composite score from reranking."""
    chunk_id: str
    final_score: float
    semantic_score: float = 0.0
    bm25_score: float = 0.0
    importance_score: float = 0.5
    recency_score: float = 0.0
    content: str = ""
    file_path: str = ""
    heading: str = ""
    # Score breakdown for logging / debugging
    breakdown: dict[str, float] = field(default_factory=dict)


@dataclass
class RerankerWeights:
    """Scoring weights — must sum to 1.0."""
    semantic: float = 0.5
    bm25: float = 0.3
    importance: float = 0.1
    recency: float = 0.1


async def rerank(
    semantic_results: list[SemanticResult],
    bm25_results: list[LexicalResult],
    sqlite: SQLiteClient,
    top_k: int = 10,
    weights: RerankerWeights | None = None,
    decay_lambda: float = 0.1,
) -> list[RankedChunk]:
    """
    Merge and rerank results from all retrieval arms.

    Steps:
      1. Collect all unique candidate chunk IDs
      2. Build score maps from semantic + BM25 results
      3. Fetch importance scores from SQLite
      4. Compute recency scores from access timestamps
      5. Apply weighted scoring equation
      6. Return top-K sorted by final score

    Args:
        semantic_results: Results from the semantic (ChromaDB) arm.
        bm25_results: Results from the lexical (BM25) arm.
        sqlite: SQLite client for metadata lookups.
        top_k: Maximum results to return.
        weights: Scoring weights (defaults to 0.5/0.3/0.1/0.1).
        decay_lambda: Recency decay parameter.

    Returns:
        Top-K RankedChunk objects sorted by final_score descending.
    """
    w = weights or RerankerWeights()

    # Step 1: Collect all candidate chunk IDs
    all_chunk_ids: set[str] = set()
    semantic_map: dict[str, SemanticResult] = {}
    bm25_map: dict[str, LexicalResult] = {}

    for sr in semantic_results:
        all_chunk_ids.add(sr.chunk_id)
        semantic_map[sr.chunk_id] = sr

    for br in bm25_results:
        all_chunk_ids.add(br.chunk_id)
        bm25_map[br.chunk_id] = br

    if not all_chunk_ids:
        return []

    chunk_id_list = list(all_chunk_ids)

    # Step 2–3: Fetch importance + access info in parallel
    importance_scores = await get_importance_scores(chunk_id_list, sqlite)
    access_info = await sqlite.get_access_info(chunk_id_list)

    # Step 4: Compute recency scores
    recency_scores = compute_recency_scores(access_info, decay_lambda)

    # Step 5: Compute final scores
    ranked: list[RankedChunk] = []

    for cid in chunk_id_list:
        s_sem = semantic_map[cid].score if cid in semantic_map else 0.0
        s_bm25 = bm25_map[cid].score if cid in bm25_map else 0.0
        s_imp = importance_scores.get(cid, 0.5)
        s_rec = recency_scores.get(cid, 0.0)

        final = (s_sem * w.semantic) + (s_bm25 * w.bm25) + (s_imp * w.importance) + (s_rec * w.recency)

        # Get content and metadata from whichever arm found it
        content = ""
        file_path = ""
        heading = ""
        if cid in semantic_map:
            content = semantic_map[cid].content
            file_path = semantic_map[cid].file_path
            heading = semantic_map[cid].heading
        elif cid in bm25_map:
            content = bm25_map[cid].content

        ranked.append(RankedChunk(
            chunk_id=cid,
            final_score=final,
            semantic_score=s_sem,
            bm25_score=s_bm25,
            importance_score=s_imp,
            recency_score=s_rec,
            content=content,
            file_path=file_path,
            heading=heading,
            breakdown={
                "semantic": round(s_sem * w.semantic, 4),
                "bm25": round(s_bm25 * w.bm25, 4),
                "importance": round(s_imp * w.importance, 4),
                "recency": round(s_rec * w.recency, 4),
            },
        ))

    # Step 6: Sort and return top-K
    ranked.sort(key=lambda r: r.final_score, reverse=True)
    top = ranked[:top_k]

    logger.info(
        "Reranker: %d candidates → top-%d (best=%.4f, worst=%.4f)",
        len(ranked), len(top),
        top[0].final_score if top else 0,
        top[-1].final_score if top else 0,
    )

    # Update access stats for returned chunks
    for rc in top:
        await sqlite.update_access_stats(rc.chunk_id)

    return top
