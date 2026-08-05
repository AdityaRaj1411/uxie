"""
SQLite Schema — DDL, table creation, and migrations for the metadata layer.

Tables:
  - chunks: textual attributes, properties, change tracking for each chunk
  - links: lightweight inter-chunk semantic graph
"""

from __future__ import annotations

import logging
from pathlib import Path

import aiosqlite

logger = logging.getLogger(__name__)

# Current schema version — bump when making breaking changes
SCHEMA_VERSION = 2

# ---------------------------------------------------------------------------
# DDL statements
# ---------------------------------------------------------------------------
_DDL_CHUNKS = """
CREATE TABLE IF NOT EXISTS chunks (
    chunk_id        TEXT PRIMARY KEY,
    file_path       TEXT NOT NULL,
    heading         TEXT,
    content         TEXT NOT NULL,
    content_hash    TEXT NOT NULL,
    tags            TEXT,
    importance      REAL DEFAULT 0.5,
    access_count    INTEGER DEFAULT 0,
    last_accessed   TIMESTAMP,
    created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    modified_at     TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
"""

_DDL_LINKS = """
CREATE TABLE IF NOT EXISTS links (
    source_chunk_id TEXT NOT NULL,
    target_chunk_id TEXT NOT NULL,
    relation_type   TEXT NOT NULL,
    strength        REAL NOT NULL,
    created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (source_chunk_id, target_chunk_id),
    FOREIGN KEY (source_chunk_id) REFERENCES chunks(chunk_id) ON DELETE CASCADE,
    FOREIGN KEY (target_chunk_id) REFERENCES chunks(chunk_id) ON DELETE CASCADE
);
"""

_DDL_WORKSPACES = """
CREATE TABLE IF NOT EXISTS workspaces (
    id              TEXT PRIMARY KEY,
    name            TEXT NOT NULL UNIQUE,
    chat_history    TEXT NOT NULL,
    mode            TEXT NOT NULL DEFAULT 'casual',
    created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_updated    TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
"""

_DDL_SCHEMA_VERSION = """
CREATE TABLE IF NOT EXISTS schema_version (
    version     INTEGER NOT NULL,
    applied_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
"""

_INDEXES = [
    "CREATE INDEX IF NOT EXISTS idx_chunks_file_path    ON chunks(file_path);",
    "CREATE INDEX IF NOT EXISTS idx_chunks_content_hash ON chunks(content_hash);",
    "CREATE INDEX IF NOT EXISTS idx_links_source        ON links(source_chunk_id);",
    "CREATE INDEX IF NOT EXISTS idx_workspaces_name     ON workspaces(name);",
]


# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------
async def init_db(db_path: str | Path) -> aiosqlite.Connection:
    """
    Initialize the SQLite database at the given path.

    Creates the database file and all tables/indexes if they don't exist.
    Returns an open aiosqlite connection with WAL mode and foreign keys enabled.
    """
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)

    conn = await aiosqlite.connect(str(db_path))

    # Enable WAL mode for better concurrent read/write performance
    await conn.execute("PRAGMA journal_mode=WAL;")
    # Enable foreign key enforcement
    await conn.execute("PRAGMA foreign_keys=ON;")

    # Create tables
    await conn.execute(_DDL_SCHEMA_VERSION)
    await conn.execute(_DDL_CHUNKS)
    await conn.execute(_DDL_LINKS)
    await conn.execute(_DDL_WORKSPACES)

    # Create indexes
    for idx_sql in _INDEXES:
        await conn.execute(idx_sql)

    # Track schema version
    cursor = await conn.execute("SELECT MAX(version) FROM schema_version")
    row = await cursor.fetchone()
    current_version = row[0] if row and row[0] is not None else 0

    if current_version < SCHEMA_VERSION:
        await conn.execute(
            "INSERT INTO schema_version (version) VALUES (?)",
            (SCHEMA_VERSION,),
        )
        logger.info("Schema initialized at version %d", SCHEMA_VERSION)

    await conn.commit()
    logger.info("SQLite database ready at %s", db_path)
    return conn
