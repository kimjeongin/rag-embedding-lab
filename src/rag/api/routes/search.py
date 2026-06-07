"""POST /search — page or site mode.

Thin: pick the mode, delegate to the use case, map domain entities to DTOs.
"""
from __future__ import annotations

from fastapi import APIRouter, Depends

from rag.api.deps import get_searcher
from rag.api.schemas.search import (
    PageHit,
    PageSearchResponse,
    SearchRequest,
    SiteHit,
    SiteSearchResponse,
)
from rag.usecases import SearchDocuments

router = APIRouter()


# response_model=None: the response shape depends on `mode`, so we return one of
# two distinct models directly (FastAPI still serializes the pydantic instance).
@router.post("/search", response_model=None)
async def search(
    req: SearchRequest,
    searcher: SearchDocuments = Depends(get_searcher),
) -> PageSearchResponse | SiteSearchResponse:
    if req.mode == "site":
        sites = await searcher.sites(req.query, req.top_k, req.fetch_k)
        return SiteSearchResponse(
            query=req.query,
            top_k=req.top_k,
            fetch_k=req.fetch_k,
            results=[SiteHit.from_site(site) for site in sites],
        )

    hits = await searcher.pages(req.query, req.top_k, req.max_per_domain, req.fetch_k)
    return PageSearchResponse(
        query=req.query,
        top_k=req.top_k,
        results=[PageHit.from_hit(hit) for hit in hits],
    )
