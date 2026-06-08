"""GET /api/models?embedder=… — the models the Evaluate/Train screens can pick.

For Ollama that's the served tags; for sentence-transformers it's the saved models
under ``outputs/``. ``default`` is a sensible pre-selection so the UI never starts on
a stale choice. Read-only → sync ``def`` (threadpool) since the Ollama list is a
blocking HTTP probe.
"""
from __future__ import annotations

from fastapi import APIRouter, Depends

from rag import lab
from rag.api.deps import get_settings
from rag.api.schemas.lab import Embedder, ModelsResponse
from rag.config import Settings

router = APIRouter()


@router.get("/models", response_model=ModelsResponse)
def models(
    embedder: Embedder = "ollama",
    settings: Settings = Depends(get_settings),
) -> ModelsResponse:
    choices = lab.list_models(embedder, settings.ollama_url)
    return ModelsResponse(
        embedder=embedder,
        models=choices,
        default=lab.default_model(embedder, choices),
    )
