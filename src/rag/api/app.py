"""FastAPI application + composition root.

This is the ONE place that knows the concrete adapters: it builds Settings, probes
Qdrant, constructs the embedder + QdrantVectorStore, and wires them into the use
cases. Everything downstream depends only on ports.

The app serves two surfaces from one process:
  - **serving** (/health, /documents, /search) — needs Qdrant + an embedder.
  - **lab** (/api/*) — generate data → train → evaluate → compare; needs neither.

Qdrant is therefore **optional**: if it isn't reachable at startup the app still comes
up, the lab API works, and the serving routes return 503. This keeps the React lab
usable without standing up a vector store. Set ``RAG_REQUIRE_STORE=1`` to instead fail
fast when Qdrant is down (for serving-first deployments).

ASGI target for uvicorn is ``rag.api.app:app``.
"""
from __future__ import annotations

import asyncio
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from qdrant_client import AsyncQdrantClient
from starlette.exceptions import HTTPException as StarletteHTTPException

from rag.api.errors import register_error_handlers
from rag.api.routes import documents, health, search
from rag.api.routes.lab import router as lab_router
from rag.config import Settings
from rag.embeddings import build_embedder
from rag.stores import QdrantVectorStore
from rag.usecases import IndexDocuments, SearchDocuments

log = logging.getLogger("rag.api")

# Built React app (Vite `npm run build` → frontend/dist). Served by `rag-serve` so the
# API and UI share one origin/port (no CORS, no separate static host). Overridable for
# non-editable installs where the repo layout isn't reachable from this file.
_FRONTEND_DIST = Path(
    os.getenv("RAG_FRONTEND_DIST", str(Path(__file__).resolve().parents[3] / "frontend" / "dist"))
)


class _SPAStaticFiles(StaticFiles):
    """Static files for a single-page app: fall back to index.html for client-side
    routes (so refreshing /eval works) without masking the JSON API — API/doc paths
    that 404 stay 404 (they never reach here unless unmatched)."""

    _passthrough = ("api", "health", "documents", "search", "docs", "redoc", "openapi.json")

    def _is_spa_route(self, path: str) -> bool:
        return path.split("/", 1)[0] not in self._passthrough

    async def get_response(self, path: str, scope):
        # StaticFiles signals "not found" by *raising* HTTPException(404) (with html=True
        # it first tries index.html / 404.html), so the fallback must catch it too.
        try:
            response = await super().get_response(path, scope)
        except StarletteHTTPException as exc:
            if exc.status_code == 404 and self._is_spa_route(path):
                return await super().get_response("index.html", scope)
            raise
        if response.status_code == 404 and self._is_spa_route(path):
            return await super().get_response("index.html", scope)
        return response


def _mount_frontend(app: FastAPI) -> None:
    """Mount the built SPA at / when it exists; otherwise the app is API-only."""
    if _FRONTEND_DIST.is_dir():
        app.mount("/", _SPAStaticFiles(directory=_FRONTEND_DIST, html=True), name="frontend")
        log.info("serving built frontend from %s", _FRONTEND_DIST)


async def _await_store(store: QdrantVectorStore, attempts: int, delay: float) -> bool:
    """Best-effort: wait for Qdrant and ensure the collection exists.

    Returns True once ready, False if it never answered within ``attempts``. The compose
    service has no healthcheck (the image lacks curl/bash), so a not-yet-ready Qdrant is
    tolerated by retrying. A permanently-absent Qdrant just disables serving.
    """
    last: Exception | None = None
    for _ in range(attempts):
        try:
            await store.ensure_collection()
            return True
        except Exception as exc:  # noqa: BLE001 - retry any startup/connection error
            last = exc
            await asyncio.sleep(delay)
    log.warning("Qdrant not reachable after %d attempts (%s) — serving disabled, lab API still works", attempts, last)
    return False


def create_app(settings: Settings | None = None) -> FastAPI:
    """Build the app. `settings` is injectable (tests can pass their own)."""
    settings = settings or Settings.from_env()
    require_store = os.getenv("RAG_REQUIRE_STORE", "").lower() in {"1", "true", "yes"}
    attempts = int(os.getenv("RAG_STORE_WAIT_ATTEMPTS", "30" if require_store else "5"))
    delay = float(os.getenv("RAG_STORE_WAIT_DELAY", "1.0"))

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # Lab routes only need settings; serving wiring is added below iff Qdrant is up.
        app.state.settings = settings
        app.state.store = None
        app.state.indexer = None
        app.state.searcher = None

        client = AsyncQdrantClient(url=settings.qdrant_url)
        store = QdrantVectorStore(client, settings.qdrant_collection, settings.embed_dim)
        try:
            ready = await _await_store(store, attempts, delay)
            if not ready and require_store:
                raise RuntimeError(f"Qdrant not reachable at {settings.qdrant_url} (RAG_REQUIRE_STORE=1)")
            if ready:
                # build_embedder picks the backend (ollama/ST) and manages its resources.
                async with build_embedder(settings) as embedder:
                    app.state.store = store
                    app.state.indexer = IndexDocuments(embedder, store)
                    app.state.searcher = SearchDocuments(embedder, store)
                    yield
            else:
                yield
        finally:
            await client.close()

    app = FastAPI(title="RAG Embedding Lab", lifespan=lifespan)
    register_error_handlers(app)
    app.include_router(health.router)
    app.include_router(documents.router)
    app.include_router(search.router)
    app.include_router(lab_router)
    _mount_frontend(app)  # last: the SPA catch-all must not shadow the API routes above
    return app


# Module-level ASGI app for `uvicorn rag.api.app:app`.
app = create_app()
