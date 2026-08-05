"""
Graph Link Generator — classifies relations between chunks using the cloud LLM.

At the end of a session, newly added or modified chunks are compared against
their semantic neighbors. The LLM determines if they are related, expand upon,
or act as examples of one another.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from src.storage.sqlite_client import SQLiteClient
    from src.storage.chroma_client import ChromaClient
    from src.llm_client import LLMClient

logger = logging.getLogger(__name__)

# Structured output schema for the LLM
class LinkClassification(BaseModel):
    is_linked: bool = Field(description="True if the two chunks have a meaningful conceptual link.")
    relation_type: str = Field(
        description="One of: 'related_to', 'expands_on', 'example_of', or 'none'.",
        default="none"
    )
    strength: float = Field(
        description="Strength of the relationship from 0.0 to 1.0. Minimum 0.6 to be considered linked.",
        default=0.0
    )


_SYSTEM_PROMPT = """
You are a knowledge graph builder. Your task is to analyze two text chunks and determine if they have a strong conceptual link.

Allowed relation types:
1. related_to: General conceptual similarity or shared topic.
2. expands_on: Chunk B elaborates, explains, or goes deeper into a concept mentioned in Chunk A.
3. example_of: Chunk B provides a specific instance or application of a concept in Chunk A.
4. none: No strong conceptual link.

Assign a strength score from 0.0 to 1.0. Only assign >= 0.6 if the link is clearly useful for a human exploring these notes.
Respond with the JSON format specified.
"""


async def generate_links(
    new_chunk_ids: list[str],
    sqlite: "SQLiteClient",
    chroma: "ChromaClient",
    llm: "LLMClient",
    max_neighbors: int = 5,
    min_strength: float = 0.6,
) -> int:
    """
    Generate links for newly added/modified chunks.

    For each chunk in `new_chunk_ids`:
    1. Query ChromaDB for top-N nearest neighbors.
    2. Ask the LLM to classify the relationship between the new chunk and each neighbor.
    3. Save valid links (strength >= min_strength) to SQLite.

    Returns the number of links successfully generated.
    """
    if not new_chunk_ids:
        return 0

    links_created = 0

    for chunk_id in new_chunk_ids:
        # Get the new chunk's data from SQLite
        chunk_data = await sqlite.get_chunk(chunk_id)
        if not chunk_data:
            continue

        # Get its embedding from Chroma to find neighbors
        results = chroma.get_by_ids([chunk_id])
        embeddings = results.get("embeddings")
        if not embeddings or not embeddings[0]:
            continue

        query_embedding = embeddings[0]

        # Query ChromaDB for neighbors (get a few extra to account for filtering itself out)
        search_results = chroma.query(
            query_embedding=query_embedding,
            n_results=max_neighbors + 2,
        )

        neighbor_ids = search_results.get("ids", [[]])[0]
        neighbor_docs = search_results.get("documents", [[]])[0]

        source_text = chunk_data.get("content", "")

        for n_id, n_doc in zip(neighbor_ids, neighbor_docs):
            if n_id == chunk_id:
                continue

            # Don't prompt LLM if they are near-identical textually
            if len(source_text) > 20 and source_text == n_doc:
                continue

            prompt = (
                f"Chunk A (New):\n---\n{source_text}\n---\n\n"
                f"Chunk B (Existing):\n---\n{n_doc}\n---\n"
            )

            try:
                # Call LLM to classify
                classification: LinkClassification = await llm.classify_structured(
                    system_prompt=_SYSTEM_PROMPT,
                    user_prompt=prompt,
                    response_model=LinkClassification,
                )

                if classification.is_linked and classification.strength >= min_strength and classification.relation_type != "none":
                    await sqlite.upsert_link(
                        source_id=chunk_id,
                        target_id=n_id,
                        relation_type=classification.relation_type,
                        strength=classification.strength,
                    )
                    links_created += 1

            except Exception as e:
                logger.error("Failed to generate link for %s -> %s: %s", chunk_id, n_id, e)

    if links_created > 0:
        logger.info("Generated %d new graph links", links_created)

    return links_created
