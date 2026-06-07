"""FastAPI application + composition root.

This is the ONE place that knows the concrete adapters: it builds Settings, opens
the Qdrant client / HTTP client, constructs the OllamaEmbedder and
QdrantVectorStore, and wires them into the use cases. Everything downstream
depends only on ports.

ASGI target for uvicorn is ``rag.api.app:app``.
"""
from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager

from fastapi import FastAPI
from qdrant_client import AsyncQdrantClient

from rag.api.errors import register_error_handlers
from rag.api.routes import documents, health, search
from rag.config import Settings
from rag.embeddings import build_embedder
from rag.stores import QdrantVectorStore
from rag.usecases import IndexDocuments, SearchDocuments


async def _ensure_ready(store: QdrantVectorStore, attempts: int = 30, delay: float = 1.0) -> None:
    """Wait for Qdrant to accept connections, then ensure the collection exists.

    The compose service has no healthcheck (the image lacks curl/bash), so the app
    tolerates a not-yet-ready Qdrant by retrying.
    """
    last: Exception | None = None
    for _ in range(attempts):
        try:
            await store.ensure_collection()
            return
        except Exception as exc:  # noqa: BLE001 - retry any startup/connection error
            last = exc
            await asyncio.sleep(delay)
    raise RuntimeError(f"Qdrant not reachable after {attempts} attempts: {last}")


def create_app(settings: Settings | None = None) -> FastAPI:
    """Build the app. `settings` is injectable (tests can pass their own)."""
    settings = settings or Settings.from_env()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        client = AsyncQdrantClient(url=settings.qdrant_url)
        store = QdrantVectorStore(client, settings.qdrant_collection, settings.embed_dim)
        await _ensure_ready(store)

        # build_embedder picks the backend (ollama/ST) and manages its resources.
        async with build_embedder(settings) as embedder:
            app.state.settings = settings
            app.state.store = store
            app.state.indexer = IndexDocuments(embedder, store)
            app.state.searcher = SearchDocuments(embedder, store)
            try:
                yield
            finally:
                await client.close()

    app = FastAPI(title="qdrant dense-retrieval RAG", lifespan=lifespan)
    register_error_handlers(app)
    app.include_router(health.router)
    app.include_router(documents.router)
    app.include_router(search.router)
    return app


# Module-level ASGI app for `uvicorn rag.api.app:app`.
app = create_app()
