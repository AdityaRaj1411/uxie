"""
Recency Scoring — timestamp-based decay arm.

Computes a recency score for chunks based on how recently they were
accessed. Uses exponential decay: score = exp(-λ × days_since_access).
"""

from __future__ import annotations

import logging
import math
from datetime import datetime, timezone

logger = logging.getLogger(__name__)


def compute_recency_score(
    last_accessed: str | datetime | None,
    decay_lambda: float = 0.1,
) -> float:
    """
    Compute a recency score based on time since last access.

    Uses exponential decay: score = exp(-λ × days_since_access)
    - Recently accessed chunks score close to 1.0
    - Old chunks decay toward 0.0
    - Never-accessed chunks get 0.0

    Args:
        last_accessed: ISO timestamp string, datetime, or None.
        decay_lambda: Decay rate. Higher = faster decay.
            0.1  → half-life ~7 days
            0.05 → half-life ~14 days
            0.01 → half-life ~69 days

    Returns:
        Score in [0.0, 1.0].
    """
    if last_accessed is None:
        return 0.0

    # Parse string timestamps
    if isinstance(last_accessed, str):
        try:
            last_accessed = datetime.fromisoformat(last_accessed)
        except (ValueError, TypeError):
            return 0.0

    # Ensure timezone awareness
    if last_accessed.tzinfo is None:
        last_accessed = last_accessed.replace(tzinfo=timezone.utc)

    now = datetime.now(timezone.utc)
    delta = now - last_accessed
    days = max(0.0, delta.total_seconds() / 86400.0)

    score = math.exp(-decay_lambda * days)
    return max(0.0, min(1.0, score))


def compute_recency_scores(
    access_info: dict[str, dict],
    decay_lambda: float = 0.1,
) -> dict[str, float]:
    """
    Compute recency scores for multiple chunks.

    Args:
        access_info: Dict of chunk_id → {"access_count": int, "last_accessed": str|None}
        decay_lambda: Exponential decay rate.

    Returns:
        Dict of chunk_id → recency score [0.0, 1.0].
    """
    scores = {}
    for chunk_id, info in access_info.items():
        last = info.get("last_accessed")
        scores[chunk_id] = compute_recency_score(last, decay_lambda)
    return scores
