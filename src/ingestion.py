"""
Ingestion Pipeline — chunk, hash, embed, and store markdown files.

Implements the change-detection loop from the spec:
  File Modified → Re-chunk → For each chunk:
     Compute SHA256 → Compare with stored hash
     ├── Unchanged → Skip
     └── Changed/new → Re-embed → Update SQLite + ChromaDB
"""

from __future__ import annotations

import logging
from pathlib import Path

from src.chunker.chunker import Chunk, parse_file
from src.embedding.embedder import Embedder
from src.storage.chroma_client import ChromaClient
from src.storage.sqlite_client import SQLiteClient

logger = logging.getLogger(__name__)


async def ingest_file(
    file_path: Path,
    vault_root: Path,
    sqlite: SQLiteClient,
    chroma: ChromaClient,
    embedder: Embedder,
    max_words: int = 800,
    min_words: int = 50,
    overlap_words: int = 50,
) -> dict[str, int]:
    """
    Ingest a single markdown file: chunk, diff, embed, store.

    Returns a summary dict with counts: added, updated, removed, unchanged.
    """
    stats = {"added": 0, "updated": 0, "removed": 0, "unchanged": 0}

    # Compute relative path for consistent storage
    try:
        rel_path = file_path.relative_to(vault_root).as_posix()
    except ValueError:
        rel_path = file_path.name

    # Parse file into chunks
    new_chunks = parse_file(
        file_path, vault_root,
        max_words=max_words, min_words=min_words, overlap_words=overlap_words,
    )
    new_chunk_map = {c.chunk_id: c for c in new_chunks}

    # Get existing chunk IDs for this file from SQLite
    existing_ids = await sqlite.get_all_chunk_ids_for_file(rel_path)

    # Determine what changed
    new_ids = set(new_chunk_map.keys())
    ids_to_remove = existing_ids - new_ids
    ids_to_check = new_ids  # Check all new IDs (could be new or updated)

    # Remove chunks that no longer exist in the file
    if ids_to_remove:
        await sqlite.delete_chunks(list(ids_to_remove))
        chroma.delete_chunks(list(ids_to_remove))
        stats["removed"] = len(ids_to_remove)
        logger.info("Removed %d orphan chunks from '%s'", len(ids_to_remove), rel_path)

    # Check each chunk for changes
    chunks_to_embed: list[Chunk] = []

    for chunk_id in ids_to_check:
        chunk = new_chunk_map[chunk_id]
        stored_hash = await sqlite.get_content_hash(chunk_id)

        if stored_hash == chunk.content_hash:
            stats["unchanged"] += 1
            continue

        if stored_hash is None:
            stats["added"] += 1
        else:
            stats["updated"] += 1

        chunks_to_embed.append(chunk)

    # Batch embed and store changed chunks
    if chunks_to_embed:
        await _embed_and_store(chunks_to_embed, sqlite, chroma, embedder)

    logger.info(
        "Ingested '%s': +%d added, ~%d updated, -%d removed, =%d unchanged",
        rel_path, stats["added"], stats["updated"], stats["removed"], stats["unchanged"],
    )
    return stats


async def _embed_and_store(
    chunks: list[Chunk],
    sqlite: SQLiteClient,
    chroma: ChromaClient,
    embedder: Embedder,
) -> None:
    # Process in smaller batches to avoid HTTP timeouts on massive files
    batch_size = 50
    total_batches = (len(chunks) + batch_size - 1) // batch_size
    
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        batch_num = (i // batch_size) + 1
        logger.info(f"Embedding batch {batch_num}/{total_batches} ({len(batch)} chunks)...")
        
        # Generate embeddings via the microservice
        texts = [c.content for c in batch]
        embeddings = await embedder.encode(texts)

        # Upsert into ChromaDB
        chroma.upsert_chunks(
            ids=[c.chunk_id for c in batch],
            documents=texts,
            embeddings=embeddings,
            metadatas=[
                {"file_path": c.file_path, "heading": c.heading}
                for c in batch
            ],
        )

        # Upsert into SQLite
        await sqlite.upsert_chunks_batch(batch)


async def purge_file(
    file_path: Path,
    vault_root: Path,
    sqlite: SQLiteClient,
    chroma: ChromaClient,
) -> int:
    """
    Remove all chunks for a deleted file from both stores.

    Returns the count of removed chunks.
    """
    try:
        rel_path = file_path.relative_to(vault_root).as_posix()
    except ValueError:
        rel_path = file_path.name

    # Get chunk IDs to remove
    chunk_ids = await sqlite.get_all_chunk_ids_for_file(rel_path)
    if not chunk_ids:
        return 0

    # Delete from both stores
    chroma.delete_chunks(list(chunk_ids))
    count = await sqlite.delete_chunks_by_file(rel_path)

    logger.info("Purged %d chunks for deleted file '%s'", count, rel_path)
    return count


async def full_vault_sync(
    vault_root: Path,
    sqlite: SQLiteClient,
    chroma: ChromaClient,
    embedder: Embedder,
    max_words: int = 800,
    min_words: int = 50,
    overlap_words: int = 50,
) -> dict[str, int]:
    """
    Full vault synchronization — scan all .md files and sync with stores.

    Steps:
      1. Scan all .md files in vault (excluding _buddy_config.md)
      2. Ingest each file (chunk/hash/embed delta)
      3. Purge orphan DB entries for files that no longer exist

    Returns aggregate stats across all files.
    """
    total_stats = {"added": 0, "updated": 0, "removed": 0, "unchanged": 0, "files": 0}

    vault_root = Path(vault_root)
    if not vault_root.exists():
        logger.warning("Vault directory does not exist: %s", vault_root)
        return total_stats

    # Scan vault for supported files (skip _buddy_config.md — it's config, not knowledge)
    supported_exts = {".md", ".pdf", ".txt", ".json", ".csv"}
    md_files = [
        f for f in vault_root.rglob("*")
        if f.suffix.lower() in supported_exts and f.name != "_buddy_config.md" and f.is_file()
    ]

    logger.info("Found %d supported files in vault", len(md_files))

    # Ingest each file
    for md_file in md_files:
        stats = await ingest_file(
            md_file, vault_root, sqlite, chroma, embedder,
            max_words=max_words, min_words=min_words, overlap_words=overlap_words,
        )
        total_stats["added"] += stats["added"]
        total_stats["updated"] += stats["updated"]
        total_stats["removed"] += stats["removed"]
        total_stats["unchanged"] += stats["unchanged"]
        total_stats["files"] += 1

    # Purge orphan entries — files in DB that no longer exist on disk
    known_files = await sqlite.get_all_file_paths()
    current_files = set()
    for md_file in md_files:
        try:
            current_files.add(md_file.relative_to(vault_root).as_posix())
        except ValueError:
            current_files.add(md_file.name)

    orphan_files = known_files - current_files
    for orphan_path in orphan_files:
        chunk_ids = await sqlite.get_all_chunk_ids_for_file(orphan_path)
        if chunk_ids:
            chroma.delete_chunks(list(chunk_ids))
            removed = await sqlite.delete_chunks_by_file(orphan_path)
            total_stats["removed"] += removed
            logger.info("Purged %d orphan chunks from deleted file '%s'", removed, orphan_path)

    logger.info(
        "Vault sync complete: %d files, +%d added, ~%d updated, -%d removed, =%d unchanged",
        total_stats["files"], total_stats["added"], total_stats["updated"],
        total_stats["removed"], total_stats["unchanged"],
    )
    return total_stats
