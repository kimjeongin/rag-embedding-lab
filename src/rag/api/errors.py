"""Map domain exceptions to HTTP responses in one place.

Adapters raise domain errors (rag.core.errors); the web layer decides the status
code here. This keeps transport concerns out of the use cases and removes the
duplicated try/except that would otherwise live in every route.
"""
from __future__ import annotations

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from rag.core.errors import EmbeddingError


def register_error_handlers(app: FastAPI) -> None:
    @app.exception_handler(EmbeddingError)
    async def _on_embedding_error(_: Request, exc: EmbeddingError) -> JSONResponse:
        # Upstream embedding service problem -> 502 Bad Gateway.
        return JSONResponse(status_code=502, content={"detail": f"embedding service error: {exc}"})
