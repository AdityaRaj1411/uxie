"""
Session Manager — tracks conversation history and auto-drafts.

Manages the lifecycle of a chat session, automatically saving the raw
unstructured conversation to /drafts/session_draft.md every N messages.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

from src.buddy.modes import BuddyMode

logger = logging.getLogger(__name__)


@dataclass
class Message:
    role: str  # "user" or "assistant"
    content: str


@dataclass
class Session:
    """Tracks state for an ongoing chat session."""
    history: list[Message] = field(default_factory=list)
    current_mode: BuddyMode = BuddyMode.CASUAL
    message_count: int = 0
    new_chunk_ids: list[str] = field(default_factory=list)  # Chunks created this session
    pending_mode_switch: BuddyMode | None = None
    pivot_index: int = 0  # Index in history where the last context pivot occurred

class SessionManager:
    def __init__(
        self,
        drafts_path: str | Path,
        auto_draft_interval: int = 5,
        prune_threshold: int = 20,
        prune_keep: int = 18,
        router: Any = None,
    ):
        self._drafts_path = Path(drafts_path)
        self._draft_file = self._drafts_path / "session_draft.md"
        self._interval = auto_draft_interval
        self._prune_threshold = prune_threshold
        self._prune_keep = prune_keep
        self.router = router
        self.session = Session()

        # Ensure drafts dir exists
        self._drafts_path.mkdir(parents=True, exist_ok=True)
        # Clear previous draft
        if self._draft_file.exists():
            self._draft_file.unlink()

    def add_message(
        self,
        role: str,
        content: str,
        embedding: list[float] | None = None
    ) -> None:
        """Add a message to the session, index into semantic router, and auto-draft if needed."""
        self.session.history.append(Message(role=role, content=content))
        self.session.message_count += 1

        if self.router is not None:
            try:
                self.router.add_turn(role, content, embedding)
            except Exception as e:
                logger.warning("Failed to index message in semantic router: %s", e)

        if self.session.message_count % self._interval == 0:
            self._save_auto_draft()

    def add_new_chunk_id(self, chunk_id: str) -> None:
        """Track chunks created during this session (for graph linking later)."""
        if chunk_id not in self.session.new_chunk_ids:
            self.session.new_chunk_ids.append(chunk_id)

    def replace_session(self, history: list[Message], mode: BuddyMode) -> None:
        """Replace the active session state with a loaded workspace.

        Used by ``/load`` to swap in a previously stashed workspace.
        The message_count is reset to the loaded history length so that
        auto-drafting intervals remain accurate.
        """
        self.session = Session(
            history=list(history),
            current_mode=mode,
            message_count=len(history),
        )
        logger.info("Session replaced: %d messages, mode=%s", len(history), mode.value)

    def pivot_context(self) -> None:
        """Mark a context pivot to stop sending old history to the LLM (prevents topic bleed)."""
        self.session.pivot_index = len(self.session.history)
        logger.info("Context pivot marked at message index %d", self.session.pivot_index)

    def get_history_for_llm(self, route_payload: Any = None) -> list[dict[str, str]]:
        """Format history for the LLM client using semantic routing or sliding-window pruning.

        If route_payload is provided and contains topic-matched messages, returns:
          1. Head (continuity memory pair, if present)
          2. Matched topic block messages
          3. N-1 Safety Net (if from a different topic)
        Otherwise falls back to sliding-window pruning.
        """
        full = [{"role": m.role, "content": m.content} for m in self.session.history]
        head = full[:2] if len(full) >= 2 and full[0]["role"] == "user" and "Continuity Memory" in full[0]["content"] else []

        if route_payload is not None and hasattr(route_payload, "matched_messages"):
            routed: list[dict[str, str]] = list(head)
            for m in route_payload.matched_messages:
                # Avoid duplicate with head
                if not any(h["content"] == m.content for h in routed):
                    routed.append({"role": m.role, "content": m.content})
            if route_payload.safety_net:
                sn = route_payload.safety_net
                if not any(r["content"] == sn.content for r in routed):
                    routed.append({"role": sn.role, "content": sn.content})
            return routed

        # Apply pivot cut (keep system prompt / continuity head, but drop pre-pivot messages)
        if self.session.pivot_index > 0:
            # Assumes first 2 messages are the continuity setup if it exists. 
            # If continuity wasn't loaded, there is no head, so we just use the pivot cut.
            # A safe way is to take up to 2 if they exist, but actually it's easier to just 
            # take from pivot_index onwards.
            # Wait, if continuity was loaded, they are at index 0 and 1.
            head = full[:2] if len(full) >= 2 and full[0]["role"] == "user" and "Continuity Memory" in full[0]["content"] else []
            rest = full[self.session.pivot_index:]
            full = head + rest

        if len(full) <= self._prune_threshold:
            return full

        # Keep the opening pair (continuity injection) + recent tail
        head = full[:2]
        tail = full[-self._prune_keep:]
        return head + tail

    def get_last_n_exchanges(self, n: int = 3) -> list[dict[str, str]]:
        """Return the last *n* user+model exchange pairs for terminal display.

        Walks backwards through history collecting up to *n* pairs.
        Returns them in chronological order.
        """
        pairs: list[dict[str, str]] = []
        history = self.session.history
        i = len(history) - 1

        while i >= 0 and len(pairs) < n * 2:
            pairs.append({"role": history[i].role, "content": history[i].content})
            i -= 1

        pairs.reverse()
        return pairs

    def _save_auto_draft(self) -> None:
        """Dump raw history to the auto-draft file."""
        try:
            content = "# Session Auto-Draft\n\n*This is a raw safety net. Overwritten each session.*\n\n---\n\n"
            for msg in self.session.history:
                role = "USER" if msg.role == "user" else "BUDDY"
                content += f"**{role}:**\n{msg.content}\n\n"
                
            self._draft_file.write_text(content, encoding="utf-8")
            logger.debug("Auto-draft saved to %s", self._draft_file)
        except Exception as e:
            logger.error("Failed to save auto-draft: %s", e)

    def end_session(self) -> list[str]:
        """
        End the session. 
        Returns the list of new chunk IDs created during this session so the 
        orchestrator can trigger graph link generation.
        """
        new_ids = list(self.session.new_chunk_ids)
        
        # Clear draft
        if self._draft_file.exists():
            try:
                self._draft_file.unlink()
            except Exception as e:
                logger.error("Failed to clear draft file: %s", e)
                
        # Reset state
        self.session = Session()
        return new_ids
