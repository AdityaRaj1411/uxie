"""
SQLite Client — async wrapper for all chunk and link operations.

All read/write operations against the metadata.sqlite database go through
this class. No other module should directly execute SQL.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

import aiosqlite

from src.chunker.chunker import Chunk

logger = logging.getLogger(__name__)


class SQLiteClient:
    """Async wrapper around the SQLite metadata database."""

    def __init__(self, conn: aiosqlite.Connection):
        self._conn = conn

    # ==================================================================
    # Chunk operations
    # ==================================================================

    async def upsert_chunk(self, chunk: Chunk) -> None:
        """Insert or update a chunk in the database."""
        tags_str = ",".join(chunk.tags) if chunk.tags else None
        await self._conn.execute(
            """
            INSERT INTO chunks (chunk_id, file_path, heading, content, content_hash, tags, modified_at)
            VALUES (?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            ON CONFLICT(chunk_id) DO UPDATE SET
                file_path    = excluded.file_path,
                heading      = excluded.heading,
                content      = excluded.content,
                content_hash = excluded.content_hash,
                tags         = excluded.tags,
                modified_at  = CURRENT_TIMESTAMP
            """,
            (chunk.chunk_id, chunk.file_path, chunk.heading, chunk.content, chunk.content_hash, tags_str),
        )
        await self._conn.commit()

    async def upsert_chunks_batch(self, chunks: list[Chunk]) -> None:
        """Insert or update multiple chunks in a single transaction."""
        rows = [
            (c.chunk_id, c.file_path, c.heading, c.content, c.content_hash, ",".join(c.tags) if c.tags else None)
            for c in chunks
        ]
        await self._conn.executemany(
            """
            INSERT INTO chunks (chunk_id, file_path, heading, content, content_hash, tags, modified_at)
            VALUES (?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            ON CONFLICT(chunk_id) DO UPDATE SET
                file_path    = excluded.file_path,
                heading      = excluded.heading,
                content      = excluded.content,
                content_hash = excluded.content_hash,
                tags         = excluded.tags,
                modified_at  = CURRENT_TIMESTAMP
            """,
            rows,
        )
        await self._conn.commit()

    async def get_chunk(self, chunk_id: str) -> dict[str, Any] | None:
        """Fetch a single chunk by ID. Returns None if not found."""
        cursor = await self._conn.execute(
            "SELECT * FROM chunks WHERE chunk_id = ?", (chunk_id,)
        )
        row = await cursor.fetchone()
        if row is None:
            return None
        return await self._row_to_dict(cursor, row)

    async def get_chunks_by_file(self, file_path: str) -> list[dict[str, Any]]:
        """Fetch all chunks belonging to a specific file."""
        cursor = await self._conn.execute(
            "SELECT * FROM chunks WHERE file_path = ?", (file_path,)
        )
        rows = await cursor.fetchall()
        return [await self._row_to_dict(cursor, row) for row in rows]

    async def get_content_hash(self, chunk_id: str) -> str | None:
        """Fetch only the content_hash for a chunk. Used for change detection."""
        cursor = await self._conn.execute(
            "SELECT content_hash FROM chunks WHERE chunk_id = ?", (chunk_id,)
        )
        row = await cursor.fetchone()
        return row[0] if row else None

    async def get_all_chunks(self) -> list[dict[str, Any]]:
        """Fetch all chunks in the database."""
        cursor = await self._conn.execute("SELECT * FROM chunks")
        rows = await cursor.fetchall()
        return [await self._row_to_dict(cursor, row) for row in rows]

    async def get_all_chunk_ids_for_file(self, file_path: str) -> set[str]:
        """Get just the chunk IDs for a file (fast, for diffing)."""
        cursor = await self._conn.execute(
            "SELECT chunk_id FROM chunks WHERE file_path = ?", (file_path,)
        )
        rows = await cursor.fetchall()
        return {row[0] for row in rows}

    async def get_all_file_paths(self) -> set[str]:
        """Get all unique file paths currently in the database."""
        cursor = await self._conn.execute("SELECT DISTINCT file_path FROM chunks")
        rows = await cursor.fetchall()
        return {row[0] for row in rows}

    async def delete_chunks(self, chunk_ids: list[str]) -> None:
        """Delete specific chunks by ID."""
        if not chunk_ids:
            return
        placeholders = ",".join("?" for _ in chunk_ids)
        await self._conn.execute(
            f"DELETE FROM chunks WHERE chunk_id IN ({placeholders})", chunk_ids
        )
        await self._conn.commit()

    async def delete_chunks_by_file(self, file_path: str) -> int:
        """Delete all chunks for a file. Returns count of deleted rows."""
        cursor = await self._conn.execute(
            "DELETE FROM chunks WHERE file_path = ?", (file_path,)
        )
        await self._conn.commit()
        return cursor.rowcount

    # ==================================================================
    # Access tracking & importance
    # ==================================================================

    async def update_access_stats(self, chunk_id: str) -> None:
        """Increment access count and update last_accessed timestamp."""
        await self._conn.execute(
            """
            UPDATE chunks
            SET access_count = access_count + 1,
                last_accessed = ?
            WHERE chunk_id = ?
            """,
            (datetime.now(timezone.utc).isoformat(), chunk_id),
        )
        await self._conn.commit()

    async def update_importance(self, chunk_id: str, importance: float) -> None:
        """Set the importance score (clamped 0.0–1.0)."""
        clamped = max(0.0, min(1.0, importance))
        await self._conn.execute(
            "UPDATE chunks SET importance = ? WHERE chunk_id = ?",
            (clamped, chunk_id),
        )
        await self._conn.commit()

    async def get_importance_scores(self, chunk_ids: list[str]) -> dict[str, float]:
        """Fetch importance scores for a list of chunk IDs."""
        if not chunk_ids:
            return {}
        placeholders = ",".join("?" for _ in chunk_ids)
        cursor = await self._conn.execute(
            f"SELECT chunk_id, importance FROM chunks WHERE chunk_id IN ({placeholders})",
            chunk_ids,
        )
        rows = await cursor.fetchall()
        return {row[0]: row[1] for row in rows}

    async def get_access_info(self, chunk_ids: list[str]) -> dict[str, dict]:
        """Fetch access_count and last_accessed for a list of chunk IDs."""
        if not chunk_ids:
            return {}
        placeholders = ",".join("?" for _ in chunk_ids)
        cursor = await self._conn.execute(
            f"SELECT chunk_id, access_count, last_accessed FROM chunks WHERE chunk_id IN ({placeholders})",
            chunk_ids,
        )
        rows = await cursor.fetchall()
        return {
            row[0]: {"access_count": row[1], "last_accessed": row[2]}
            for row in rows
        }

    # ==================================================================
    # Link operations
    # ==================================================================

    async def upsert_link(
        self, source_id: str, target_id: str, relation_type: str, strength: float
    ) -> None:
        """Insert or update a link between two chunks."""
        clamped_strength = max(0.6, min(1.0, strength))
        await self._conn.execute(
            """
            INSERT INTO links (source_chunk_id, target_chunk_id, relation_type, strength)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(source_chunk_id, target_chunk_id) DO UPDATE SET
                relation_type = excluded.relation_type,
                strength      = excluded.strength
            """,
            (source_id, target_id, relation_type, clamped_strength),
        )
        await self._conn.commit()

    async def get_neighbors(
        self, chunk_id: str, min_strength: float = 0.6
    ) -> list[dict[str, Any]]:
        """
        Get linked neighbors for a chunk with strength >= threshold.
        Returns both outgoing and incoming links.
        """
        cursor = await self._conn.execute(
            """
            SELECT target_chunk_id AS neighbor_id, relation_type, strength
            FROM links
            WHERE source_chunk_id = ? AND strength >= ?
            UNION
            SELECT source_chunk_id AS neighbor_id, relation_type, strength
            FROM links
            WHERE target_chunk_id = ? AND strength >= ?
            ORDER BY strength DESC
            """,
            (chunk_id, min_strength, chunk_id, min_strength),
        )
        rows = await cursor.fetchall()
        return [
            {"neighbor_id": row[0], "relation_type": row[1], "strength": row[2]}
            for row in rows
        ]

    async def delete_links_for_chunk(self, chunk_id: str) -> None:
        """Delete all links where this chunk is either source or target."""
        await self._conn.execute(
            "DELETE FROM links WHERE source_chunk_id = ? OR target_chunk_id = ?",
            (chunk_id, chunk_id),
        )
        await self._conn.commit()

    async def get_link_count(self, chunk_id: str) -> int:
        """Count total links for a chunk (both directions)."""
        cursor = await self._conn.execute(
            """
            SELECT COUNT(*) FROM (
                SELECT 1 FROM links WHERE source_chunk_id = ?
                UNION ALL
                SELECT 1 FROM links WHERE target_chunk_id = ?
            )
            """,
            (chunk_id, chunk_id),
        )
        row = await cursor.fetchone()
        return row[0] if row else 0

    # ==================================================================
    # Helpers
    # ==================================================================

    @staticmethod
    async def _row_to_dict(cursor: aiosqlite.Cursor, row: tuple) -> dict[str, Any]:
        """Convert a row tuple to a dictionary using column names."""
        columns = [desc[0] for desc in cursor.description]
        return dict(zip(columns, row))

    async def close(self) -> None:
        """Close the database connection."""
        await self._conn.close()
