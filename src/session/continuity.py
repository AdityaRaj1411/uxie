"""
Continuity Memory — persistent "working memory" across sessions.

On shutdown, the LLM distills the session into a dense, intent-filtered
summary that is persisted to disk. On startup, the summary is injected
into the system prompt so the buddy can resume context seamlessly.

The memory file lives at  <drafts_path>/.continuity_memory.md
and is overwritten every session. A TTL mechanism auto-expires stale memories.
"""

from __future__ import annotations

import logging
import time
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)

# Memory older than this (in seconds) is considered stale and discarded.
_DEFAULT_TTL_SECONDS = 48 * 60 * 60  # 48 hours

_CONTINUITY_FILENAME = ".continuity_memory.md"

# ── System prompt sent to the LLM at shutdown ──────────────────────────
CONTINUITY_EXTRACTION_PROMPT = """\
You are a context-preservation engine. The user is ending their session with a Personal Second Brain assistant.

Review the following conversation history and extract ONLY dense, technical, and conceptual objectives.

CRITICAL RULES:
- IGNORE all casual chat, greetings, pleasantries, jokes, and minor conversational tangents.
- IGNORE resolved minor questions that do not impact future work.
- IGNORE any system messages about mode switches, confirmations, or UI commands.
- FOCUS entirely on active projects, overarching goals, and unresolved problems.
- Be extremely concise. Each bullet should be a single dense sentence.

OUTPUT FORMAT (use this EXACTLY, do NOT add any wrapper text):

### Active Core Objectives
- [Primary conceptual or technical goal the user was working on — include specific file names, frameworks, or architectural decisions if discussed]
- [Second objective if applicable]

### Key Decisions & Insights
- [Important decisions made or insights reached during the session. If none, write "None."]

### Pending / Left-over Tasks
- [Task description] — Status: [In Progress / Blocked by X / Needs Review / Not Started]
- [Another task if applicable] — Status: [...]
(If there are no explicit pending or left-over tasks, write exactly: "None. Do not invent tasks.")

### Session Final State
- Mode: [the mode the user was in at the end]
- Primary topic depth: [shallow / moderate / deep]
- [One sentence capturing what the user would want to pick up on next time]

If the conversation was purely casual with no substantive work, output ONLY:
### No substantive context to preserve.
"""


def get_memory_path(drafts_path: str | Path) -> Path:
    """Return the path to the continuity memory file."""
    return Path(drafts_path) / _CONTINUITY_FILENAME


def load_continuity_memory(
    drafts_path: str | Path,
    ttl_seconds: int = _DEFAULT_TTL_SECONDS,
) -> str:
    """
    Load the previous session's continuity memory if it exists and is fresh.

    Returns the memory content as a string, or empty string if absent/stale.
    """
    memory_file = get_memory_path(drafts_path)

    if not memory_file.exists():
        logger.info("No continuity memory found — fresh start.")
        return ""

    # Check TTL
    file_age = time.time() - memory_file.stat().st_mtime
    if file_age > ttl_seconds:
        logger.info(
            "Continuity memory is stale (%.1f hours old, TTL=%.1f hours). Discarding.",
            file_age / 3600,
            ttl_seconds / 3600,
        )
        memory_file.unlink(missing_ok=True)
        return ""

    try:
        content = memory_file.read_text(encoding="utf-8").strip()
        if content:
            logger.info("Loaded continuity memory (%.1f hours old).", file_age / 3600)
            return content
    except Exception as e:
        logger.error("Failed to read continuity memory: %s", e)

    return ""


def save_continuity_memory(drafts_path: str | Path, content: str) -> None:
    """Persist the LLM-generated continuity summary to disk."""
    memory_file = get_memory_path(drafts_path)
    memory_file.parent.mkdir(parents=True, exist_ok=True)

    try:
        # Prepend a timestamp header for human readability
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
        full_content = f"<!-- Generated: {timestamp} -->\n{content}"
        memory_file.write_text(full_content, encoding="utf-8")
        logger.info("Continuity memory saved to %s", memory_file)
    except Exception as e:
        logger.error("Failed to save continuity memory: %s", e)


def format_memory_for_injection(memory_content: str) -> str:
    """Wrap the raw memory content for injection into the system prompt."""
    return (
        "--- Previous Session Context (Continuity Memory) ---\n"
        "The following is a summary of the user's last session. Use it to maintain "
        "conversational continuity. Do NOT explicitly mention this summary unless the "
        "user asks about it. Just naturally resume context.\n\n"
        f"{memory_content}\n"
        "--- End Previous Session Context ---"
    )
