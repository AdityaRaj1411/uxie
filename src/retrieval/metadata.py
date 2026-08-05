"""
Metadata Search — SQLite importance score arm.

Retrieves the importance scores for candidate chunks from the
metadata layer. This feeds into the reranker as one of the four signals.
"""

from __future__ import annotations

import logging

from src.storage.sqlite_client import SQLiteClient

logger = logging.getLogger(__name__)


async def get_importance_scores(
    chunk_ids: list[str],
    sqlite: SQLiteClient,
) -> dict[str, float]:
    """
    Fetch importance scores for a set of candidate chunk IDs.

    Returns a dict mapping chunk_id → importance (0.0–1.0).
    Missing chunks default to 0.5 (neutral importance).
    """
    if not chunk_ids:
        return {}

    scores = await sqlite.get_importance_scores(chunk_ids)

    # Fill in defaults for any chunks not found
    for cid in chunk_ids:
        if cid not in scores:
            scores[cid] = 0.5

    return scores
