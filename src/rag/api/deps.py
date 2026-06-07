"""Request-scoped dependencies.

Settings, the store, and the use cases are constructed once in the app lifespan
(the composition root) and stashed on ``app.state``. These accessors expose them
to routes via ``Depends`` so handlers never reach into ``app.state`` directly.
"""
from __future__ import annotations

from fastapi import Request

from rag.config import Settings
from rag.core.ports import VectorStore
from rag.usecases import IndexDocuments, SearchDocuments


def get_settings(request: Request) -> Settings:
    return request.app.state.settings


def get_store(request: Request) -> VectorStore:
    return request.app.state.store


def get_indexer(request: Request) -> IndexDocuments:
    return request.app.state.indexer


def get_searcher(request: Request) -> SearchDocuments:
    return request.app.state.searcher
