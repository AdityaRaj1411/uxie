"""
Embedder — async HTTP client that talks to the embedding microservice.

Calls the FastAPI embedding server (started as a subprocess by main.py)
over HTTP to generate text embeddings.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import httpx

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


class EmbedderError(Exception):
    """Raised when the embedding service is unreachable or returns an error."""


class Embedder:
    """Async client for the embedding microservice."""

    def __init__(self, host: str = "localhost", port: int = 9500, timeout: float = 120.0):
        self._base_url = f"http://{host}:{port}"
        self._timeout = timeout
        self._client: httpx.AsyncClient | None = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    async def start(self) -> None:
        """Create the underlying HTTP client."""
        self._client = httpx.AsyncClient(
            base_url=self._base_url,
            timeout=self._timeout,
        )

    async def close(self) -> None:
        """Close the underlying HTTP client."""
        if self._client:
            await self._client.aclose()
            self._client = None

    async def __aenter__(self) -> "Embedder":
        await self.start()
        return self

    async def __aexit__(self, *exc) -> None:
        await self.close()

    # ------------------------------------------------------------------
    # Health check
    # ------------------------------------------------------------------
    async def health_check(self) -> bool:
        """Return True if the embedding service is alive and model is loaded."""
        try:
            resp = await self._get_client().get("/health")
            resp.raise_for_status()
            data = resp.json()
            return data.get("status") == "ok"
        except Exception as e:
            logger.warning("Embedding service health check failed: %s", e)
            return False

    # ------------------------------------------------------------------
    # Embedding calls
    # ------------------------------------------------------------------
    async def encode(self, texts: list[str]) -> list[list[float]]:
        """
        Encode a batch of texts into normalized 384-dim embeddings.

        Returns a list of float vectors, one per input text.
        """
        if not texts:
            return []

        try:
            resp = await self._get_client().post("/embed", json={"texts": texts})
            resp.raise_for_status()
            data = resp.json()
            return data["embeddings"]
        except httpx.HTTPStatusError as e:
            raise EmbedderError(f"Embedding service returned {e.response.status_code}: {e.response.text}") from e
        except httpx.ConnectError as e:
            raise EmbedderError(
                f"Cannot connect to embedding service at {self._base_url}. "
                "Is it running? Start it with: python -m uvicorn src.embedding_service:app --port 9500"
            ) from e
        except Exception as e:
            raise EmbedderError(f"Embedding request failed: {e}") from e

    async def encode_query(self, query: str) -> list[float]:
        """Encode a single query string. Convenience wrapper around encode()."""
        results = await self.encode([query])
        return results[0]

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------
    def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            raise EmbedderError("Embedder not started. Call .start() or use as async context manager.")
        return self._client
