"""Request-scoped dependencies.

``Settings`` is built once at the composition root (``rag.api.app.create_app``) and
stashed on ``app.state``; this accessor hands it to the lab routes via ``Depends`` so
handlers never reach into ``app.state`` directly.

The serving deps (embedder + Qdrant store) are process-wide singletons on
``app.state``: the ST embedder holds a loaded torch model (seconds to construct —
loading it per request would dwarf the search itself), and the store wraps a reusable
HTTP client. Lifecycle: created here on first use / at startup, closed by the app's
lifespan.
"""
from __future__ import annotations

from fastapi import Request

from rag.config import Settings
from rag.core.ports import Embedder
from rag.vectorstore.qdrant import QdrantStore


def get_settings(request: Request) -> Settings:
    return request.app.state.settings


async def get_embedder(request: Request) -> Embedder:
    """The process-wide serving embedder, built lazily on the first request that
    needs it (startup stays fast, and the training stack is only required if search
    is actually used). The one-time ST model load blocks the loop for a few seconds —
    acceptable once, which is exactly what the lock guarantees."""
    from rag.embeddings import build_embedder

    state = request.app.state
    async with state.embedder_lock:
        if getattr(state, "embedder", None) is None:
            state.embedder = await state.embedder_stack.enter_async_context(
                build_embedder(state.settings)
            )
    return state.embedder


def get_store(request: Request) -> QdrantStore:
    return request.app.state.store
