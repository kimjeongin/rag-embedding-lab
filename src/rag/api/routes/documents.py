"""POST /documents — index a batch of pages.

Thin: validate, map request DTOs to Document entities, delegate to the use case.
Embedding/store failures surface as domain errors handled centrally (rag.api.errors).
"""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from rag.api.deps import get_indexer
from rag.api.schemas.documents import DocumentsRequest, DocumentsResponse
from rag.core.entities import Document
from rag.usecases import IndexDocuments

router = APIRouter()


@router.post("/documents", response_model=DocumentsResponse)
async def add_documents(
    req: DocumentsRequest,
    indexer: IndexDocuments = Depends(get_indexer),
) -> DocumentsResponse:
    if not req.documents:
        raise HTTPException(status_code=400, detail="documents must not be empty")

    documents = [
        Document(content=d.content, url=d.url, title=d.title, metadata=d.metadata)
        for d in req.documents
    ]
    ids = await indexer.execute(documents)
    return DocumentsResponse(ids=ids, count=len(ids))
