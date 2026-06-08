"""Request-scoped dependencies.

Settings, the store, and the use cases are constructed once in the app lifespan
(the composition root) and stashed on ``app.state``. These accessors expose them to
routes via ``Depends`` so handlers never reach into ``app.state`` directly.

The vector store is **optional**: the lab side (generate data → train → evaluate →
compare) needs no Qdrant, so the app starts even when Qdrant is down. The serving
dependencies below therefore 503 when the store wasn't wired, instead of handing back
a ``None`` that would explode deep in a handler.
"""
from __future__ import annotations

from fastapi import HTTPException, Request

from rag.config import Settings
from rag.core.ports import VectorStore
from rag.usecases import IndexDocuments, SearchDocuments

_NO_STORE = "vector store unavailable — start Qdrant to use /documents and /search (the lab API works without it)"


def get_settings(request: Request) -> Settings:
    return request.app.state.settings


def get_store(request: Request) -> VectorStore:
    store = request.app.state.store
    if store is None:
        raise HTTPException(status_code=503, detail=_NO_STORE)
    return store


def get_store_optional(request: Request) -> VectorStore | None:
    """The store if Qdrant came up, else None — for /health, which reports rather than
    refuses when the store is absent."""
    return request.app.state.store


def get_indexer(request: Request) -> IndexDocuments:
    indexer = request.app.state.indexer
    if indexer is None:
        raise HTTPException(status_code=503, detail=_NO_STORE)
    return indexer


def get_searcher(request: Request) -> SearchDocuments:
    searcher = request.app.state.searcher
    if searcher is None:
        raise HTTPException(status_code=503, detail=_NO_STORE)
    return searcher
