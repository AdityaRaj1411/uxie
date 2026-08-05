"""
Embedding Microservice — loads sentence-transformers model at startup.

A lightweight FastAPI server that loads all-mpnet-base-v2 once at startup
and serves embeddings over HTTP. Launched as a subprocess by main.py.

Model: all-mpnet-base-v2
  - 768-dim embeddings
  - 512-token max sequence length (~380 words)
  - Stronger retrieval quality than MiniLM

Usage:
    python -m uvicorn src.embedding_service:app --port 9500
"""

import logging
import torch
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s [EMB] %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Model singleton — loaded once at startup
# ---------------------------------------------------------------------------
_model = None
_MODEL_NAME = "all-mpnet-base-v2"


def _load_model():
    """Load the sentence-transformer model (lazy, first call only)."""
    global _model
    if _model is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info("Loading model '%s' on device: %s …", _MODEL_NAME, device)
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer(_MODEL_NAME, device=device)
        logger.info("Model loaded successfully (dim=%d, device=%s).", _model.get_embedding_dimension(), device)
    return _model


# ---------------------------------------------------------------------------
# Lifespan — preload model on startup
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    _load_model()
    yield


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
app = FastAPI(title="Second Brain Embedding Service", lifespan=lifespan)


# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------
class EmbedRequest(BaseModel):
    texts: list[str]


class EmbedResponse(BaseModel):
    embeddings: list[list[float]]
    model: str
    dimension: int


class HealthResponse(BaseModel):
    status: str
    model: str
    dimension: int


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get("/health", response_model=HealthResponse)
async def health():
    """Health check — confirms the model is loaded and ready."""
    model = _load_model()
    return HealthResponse(
        status="ok",
        model=_MODEL_NAME,
        dimension=model.get_embedding_dimension(),
    )


@app.post("/embed", response_model=EmbedResponse)
async def embed(request: EmbedRequest):
    """Generate normalized embeddings for a list of texts."""
    if not request.texts:
        raise HTTPException(status_code=400, detail="'texts' list must not be empty.")

    model = _load_model()
    try:
        vectors = model.encode(
            request.texts,
            normalize_embeddings=True,
            show_progress_bar=False,
            batch_size=64,
        )
        return EmbedResponse(
            embeddings=vectors.tolist(),
            model=_MODEL_NAME,
            dimension=vectors.shape[1],
        )
    except Exception as e:
        logger.exception("Embedding failed")
        raise HTTPException(status_code=500, detail=str(e))
