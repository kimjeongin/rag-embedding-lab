"""Application layer — use cases that orchestrate the domain via ports.

These depend ONLY on rag.core (entities, ports, ranking). They never import
FastAPI, httpx, or qdrant_client, so the business operations (index, search) can be
exercised with in-memory fakes and contain no framework concerns.
"""
from rag.usecases.indexing import IndexDocuments
from rag.usecases.search import SearchDocuments

__all__ = ["IndexDocuments", "SearchDocuments"]
