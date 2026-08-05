"""
Graph Neighbor Expander — pulls linked chunks into the context pool.

If chunk A is retrieved, its linked neighbors B and C may be relevant
even without matching the query directly. Bypasses standard similarity thresholds.
"""

from __future__ import annotations

import logging

from src.storage.sqlite_client import SQLiteClient

logger = logging.getLogger(__name__)


async def expand_neighbors(
    base_chunk_ids: list[str],
    sqlite: SQLiteClient,
    max_hops: int = 1,
    min_strength: float = 0.6,
) -> set[str]:
    """
    Find linked neighbors up to `max_hops` deep.

    Args:
        base_chunk_ids: Initial set of chunk IDs retrieved by search.
        sqlite: Metadata database client.
        max_hops: Depth of BFS traversal. 0 = no expansion.
        min_strength: Minimum link strength to traverse.

    Returns:
        Set of all chunk IDs (including the base IDs and any expanded neighbors).
    """
    if max_hops <= 0 or not base_chunk_ids:
        return set(base_chunk_ids)

    visited = set(base_chunk_ids)
    current_frontier = set(base_chunk_ids)

    for hop in range(max_hops):
        next_frontier = set()
        for chunk_id in current_frontier:
            # Get neighbors from SQLite
            neighbors = await sqlite.get_neighbors(chunk_id, min_strength=min_strength)

            for n in neighbors:
                n_id = n["neighbor_id"]
                if n_id not in visited:
                    visited.add(n_id)
                    next_frontier.add(n_id)

        current_frontier = next_frontier
        if not current_frontier:
            break

    expanded_count = len(visited) - len(base_chunk_ids)
    if expanded_count > 0:
        logger.debug("Graph expanded %d base chunks by %d neighbors (hops=%d)", len(base_chunk_ids), expanded_count, max_hops)

    return visited
