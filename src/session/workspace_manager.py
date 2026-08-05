"""
Workspace Manager — Git-style context stashing for chat sessions.

Serializes and deserializes active conversation state (chat history + mode)
to/from the SQLite ``workspaces`` table. The full history is always preserved
in the database; token pruning is handled separately in SessionManager.
"""

from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime
from typing import Any

import aiosqlite

from src.buddy.modes import BuddyMode
from src.session.session_manager import Message

logger = logging.getLogger(__name__)


class WorkspaceManager:
    """Manages workspace persistence in the SQLite workspaces table."""

    def __init__(self, conn: aiosqlite.Connection):
        self._conn = conn

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------
    async def save(
        self,
        name: str,
        history: list[Message],
        mode: BuddyMode,
    ) -> None:
        """Serialize active session state and upsert into the workspaces table.

        If a workspace with the given *name* already exists it is overwritten
        (the ``last_updated`` timestamp is refreshed).
        """
        # Serialize history to JSON
        history_json = json.dumps(
            [{"role": m.role, "content": m.content} for m in history],
            ensure_ascii=False,
        )

        # Check if workspace already exists by name
        cursor = await self._conn.execute(
            "SELECT id FROM workspaces WHERE name = ?", (name,)
        )
        existing = await cursor.fetchone()

        if existing:
            workspace_id = existing[0]
            await self._conn.execute(
                """
                UPDATE workspaces
                SET chat_history = ?, mode = ?, last_updated = CURRENT_TIMESTAMP
                WHERE id = ?
                """,
                (history_json, mode.value, workspace_id),
            )
        else:
            workspace_id = str(uuid.uuid4())
            await self._conn.execute(
                """
                INSERT INTO workspaces (id, name, chat_history, mode)
                VALUES (?, ?, ?, ?)
                """,
                (workspace_id, name, history_json, mode.value),
            )

        await self._conn.commit()
        logger.info("Workspace '%s' saved (id=%s, messages=%d)", name, workspace_id[:8], len(history))

    # ------------------------------------------------------------------
    # Load
    # ------------------------------------------------------------------
    async def load(self, name_or_id: str) -> tuple[list[Message], BuddyMode] | None:
        """Load a workspace by name or UUID prefix.

        Returns ``(history, mode)`` on success, or ``None`` if not found.
        """
        # Try exact name match first
        cursor = await self._conn.execute(
            "SELECT chat_history, mode FROM workspaces WHERE name = ?",
            (name_or_id,),
        )
        row = await cursor.fetchone()

        # Fallback: try ID prefix match
        if row is None:
            cursor = await self._conn.execute(
                "SELECT chat_history, mode FROM workspaces WHERE id LIKE ?",
                (name_or_id + "%",),
            )
            row = await cursor.fetchone()

        if row is None:
            return None

        history_raw = json.loads(row[0])
        history = [Message(role=m["role"], content=m["content"]) for m in history_raw]

        # Parse mode (fallback to CASUAL if stored value is invalid)
        try:
            mode = BuddyMode(row[1])
        except ValueError:
            mode = BuddyMode.CASUAL

        return history, mode

    # ------------------------------------------------------------------
    # List
    # ------------------------------------------------------------------
    async def list_all(self) -> list[dict[str, Any]]:
        """Return all workspaces ordered by most recently updated.

        Each dict contains: id, name, message_count, mode, created_at, last_updated,
        and last_message (truncated preview of the final message).
        """
        cursor = await self._conn.execute(
            "SELECT id, name, chat_history, mode, created_at, last_updated "
            "FROM workspaces ORDER BY last_updated DESC"
        )
        rows = await cursor.fetchall()

        results = []
        for row in rows:
            history_raw = json.loads(row[2])
            msg_count = len(history_raw)

            # Extract a short preview of the last message
            last_msg = ""
            if history_raw:
                last_content = history_raw[-1].get("content", "")
                last_msg = last_content[:60] + ("…" if len(last_content) > 60 else "")

            results.append({
                "id": row[0],
                "name": row[1],
                "message_count": msg_count,
                "mode": row[3],
                "created_at": row[4],
                "last_updated": row[5],
                "last_message": last_msg,
            })

        return results

    # ------------------------------------------------------------------
    # Delete
    # ------------------------------------------------------------------
    async def delete(self, name_or_id: str) -> bool:
        """Delete a workspace by name or UUID prefix. Returns True if a row was removed."""
        # Try exact name first
        cursor = await self._conn.execute(
            "DELETE FROM workspaces WHERE name = ?", (name_or_id,)
        )
        if cursor.rowcount > 0:
            await self._conn.commit()
            return True

        # Fallback: ID prefix
        cursor = await self._conn.execute(
            "DELETE FROM workspaces WHERE id LIKE ?", (name_or_id + "%",)
        )
        deleted = cursor.rowcount > 0
        await self._conn.commit()
        return deleted

    # ------------------------------------------------------------------
    # Existence check
    # ------------------------------------------------------------------
    async def exists(self, name: str) -> bool:
        """Check if a workspace with the given name already exists."""
        cursor = await self._conn.execute(
            "SELECT 1 FROM workspaces WHERE name = ?", (name,)
        )
        return await cursor.fetchone() is not None
