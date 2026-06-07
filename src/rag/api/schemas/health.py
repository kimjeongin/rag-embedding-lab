"""Response model for the /health endpoint."""
from __future__ import annotations

from pydantic import BaseModel


class HealthResponse(BaseModel):
    status: str
    embedder: str           # "ollama" | "sentence-transformers"
    embed_model: str        # the active model (Ollama name or ST model path)
    embed_dim: int
    vector_store: str
    document_count: int | None = None
