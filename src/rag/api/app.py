"""FastAPI application for the embedding lab — the composition root.

Serves the **lab API** (``/api/*``): generate data → train → evaluate → compare, plus
(when built) the React front-end from ``frontend/dist`` at ``/`` so the API and UI share
one origin. Evaluation needs no vector store (it ranks the corpus in-memory, numpy
cosine); the **serving path** (``/api/search``) is the exception — it reads the Qdrant
index built by ``rag-index`` (see rag.serving, docs/serving.md).

ASGI target for uvicorn is ``rag.api.app:app``.
"""
from __future__ import annotations

import asyncio
import logging
import os
from contextlib import AsyncExitStack, asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from starlette.exceptions import HTTPException as StarletteHTTPException

from rag.api.errors import register_error_handlers
from rag.api.routes.lab import router as lab_router
from rag.config import Settings

log = logging.getLogger("rag.api")

# Built React app (Vite `npm run build` → frontend/dist). Served by `rag-serve` so the
# API and UI share one origin/port. Overridable for non-editable installs.
_FRONTEND_DIST = Path(
    os.getenv("RAG_FRONTEND_DIST", str(Path(__file__).resolve().parents[3] / "frontend" / "dist"))
)


class _SPAStaticFiles(StaticFiles):
    """Static files for a single-page app: fall back to index.html for client-side
    routes (so refreshing /eval works) without masking the JSON API — ``/api`` and the
    docs paths that 404 stay 404 (they never reach here unless unmatched)."""

    _passthrough = ("api", "docs", "redoc", "openapi.json")

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


def create_app(settings: Settings | None = None) -> FastAPI:
    """Build the app. `settings` is injectable (tests can pass their own)."""
    settings = settings or Settings.from_env()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        from rag.vectorstore.qdrant import QdrantStore

        app.state.settings = settings
        # Serving singletons (see rag.api.deps): the store is cheap to construct
        # (no connection until used) so it's eager; the embedder is lazy — the stack
        # + lock exist so `get_embedder` can build-once and shutdown can close it.
        app.state.store = QdrantStore(settings.qdrant_url)
        app.state.embedder = None
        app.state.embedder_lock = asyncio.Lock()
        app.state.embedder_stack = AsyncExitStack()
        yield
        await app.state.embedder_stack.aclose()
        app.state.store.close()

    app = FastAPI(title="RAG Embedding Lab", lifespan=lifespan)
    register_error_handlers(app)
    app.include_router(lab_router)
    _mount_frontend(app)  # last: the SPA catch-all must not shadow the API routes above
    return app


# Module-level ASGI app for `uvicorn rag.api.app:app`.
app = create_app()
