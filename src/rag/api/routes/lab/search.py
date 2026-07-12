"""/api/search — query the Qdrant serving index with the process's embedder.

This is the serving surface: what production integration actually calls. The heavy
lifting lives in ``rag.serving`` (framework-free, shared with rag-index/rag-search);
these handlers only wire the process singletons (settings / embedder / store) to it.

Serving embeds ST in-process (the default backend); point ST_MODEL at the
handed-off model:

    ST_MODEL=outputs/embedding-ft uv run rag-serve
"""
from __future__ import annotations

import asyncio

from fastapi import APIRouter, Depends, HTTPException

from rag import serving
from rag.api import indexjob
from rag.api.deps import get_embedder, get_settings, get_store
from rag.api.schemas.lab import (
    AliasRequest,
    IndexJobStatus,
    IndexRequest,
    PruneResponse,
    SearchRequest,
    SearchResponse,
    SearchStatusResponse,
)
from rag.config import Settings
from rag.core.ports import Embedder
from rag.vectorstore.qdrant import QdrantStore

router = APIRouter()


@router.post("/search", response_model=SearchResponse)
async def search(
    req: SearchRequest,
    settings: Settings = Depends(get_settings),
    embedder: Embedder = Depends(get_embedder),
    store: QdrantStore = Depends(get_store),
) -> SearchResponse:
    result = await serving.search(settings, embedder, store, req.query, req.top_k)
    return SearchResponse(**result)


@router.get("/search/status", response_model=SearchStatusResponse)
async def search_status(
    settings: Settings = Depends(get_settings),
    store: QdrantStore = Depends(get_store),
) -> SearchStatusResponse:
    overview = await asyncio.to_thread(serving.index_status, settings, store)
    return SearchStatusResponse(
        **overview, embedder=settings.embedder, model=settings.active_model
    )


@router.post("/index", response_model=IndexJobStatus)
def start_index(
    req: IndexRequest, settings: Settings = Depends(get_settings)
) -> IndexJobStatus:
    """Start a background reindex (409 while one runs — one device, one embed pass)."""
    model = req.model or settings.st_model
    try:
        state = indexjob.start(
            model, corpus_file=req.corpus_file,
            recreate=req.recreate, truncate_dim=req.truncate_dim,
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return IndexJobStatus(**state)


@router.get("/index/status", response_model=IndexJobStatus)
def index_job_status() -> IndexJobStatus:
    return IndexJobStatus(**indexjob.status())


@router.post("/index/alias", response_model=SearchStatusResponse)
async def set_alias(
    req: AliasRequest,
    settings: Settings = Depends(get_settings),
    store: QdrantStore = Depends(get_store),
) -> SearchStatusResponse:
    """Repoint the live alias at an existing collection — instant rollback, no re-embed."""
    overview = await asyncio.to_thread(serving.set_live, settings, store, req.collection)
    return SearchStatusResponse(
        **overview, embedder=settings.embedder, model=settings.active_model
    )


@router.post("/index/prune", response_model=PruneResponse)
async def prune(
    settings: Settings = Depends(get_settings),
    store: QdrantStore = Depends(get_store),
) -> PruneResponse:
    """Delete every family collection except the live target (the 'new index is good' call)."""
    if indexjob.status()["status"] == "running":
        # a running reindex is building a non-live collection — pruning now would eat it
        raise HTTPException(status_code=409, detail="재색인이 실행 중입니다 — 끝난 뒤 정리하세요")
    pruned = await asyncio.to_thread(serving.prune_collections, settings, store)
    return PruneResponse(pruned=pruned)
