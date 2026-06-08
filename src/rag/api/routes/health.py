"""GET /health — liveness + config introspection.

Always 200 and never raises: it *reports* vector-store reachability rather than
depending on it, so it stays useful even when Qdrant is down (status "degraded").
"""
from __future__ import annotations

from fastapi import APIRouter, Depends

from rag.api.deps import get_settings, get_store_optional
from rag.api.schemas.health import HealthResponse
from rag.config import Settings
from rag.core.ports import VectorStore

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
async def health(
    settings: Settings = Depends(get_settings),
    store: VectorStore | None = Depends(get_store_optional),
) -> HealthResponse:
    """Report model/dim and vector-store reachability. Never raises."""
    if store is None:
        store_status = "disconnected"
        count: int | None = None
    else:
        count = None
        try:
            count = await store.count()
            store_status = "connected"
        except Exception as exc:  # noqa: BLE001 - health should never raise
            store_status = f"error: {exc}"

    return HealthResponse(
        status="ok" if store_status == "connected" else "degraded",
        embedder=settings.embedder,
        embed_model=settings.active_model,
        embed_dim=settings.embed_dim,
        vector_store=store_status,
        document_count=count,
    )
