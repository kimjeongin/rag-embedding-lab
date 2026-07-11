"""/api/search — query the Qdrant serving index with the process's embedder.

This is the serving surface: what production integration actually calls. The heavy
lifting lives in ``rag.serving`` (framework-free, shared with rag-index/rag-search);
these handlers only wire the process singletons (settings / embedder / store) to it.

Run the server with the serving embedder configured (the lab default is Ollama, but
serving is ST in-process by decision):

    EMBEDDER=sentence-transformers ST_MODEL=outputs/embedding-ft uv run rag-serve
"""
from __future__ import annotations

import asyncio

from fastapi import APIRouter, Depends, HTTPException

from rag import serving
from rag.api import indexjob
from rag.api.deps import get_embedder, get_settings, get_store
from rag.api.schemas.lab import (
    IndexJobStatus,
    IndexRequest,
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
