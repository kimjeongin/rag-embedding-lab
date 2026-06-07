"""Ports — the abstractions the use cases depend on (Dependency Inversion).

Use cases talk to these Protocols, never to Ollama or Qdrant directly. The
concrete adapters (rag.embeddings.OllamaEmbedder, rag.stores.QdrantVectorStore)
implement them structurally; tests can substitute in-memory fakes. This is what
keeps the application layer free of any framework or driver.
"""
from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

from rag.core.entities import Document, EmbeddedDocument, Hit


class Embedder(Protocol):
    """Turns documents/queries into vectors (asymmetric — see rag.core.formatting)."""

    async def embed_documents(self, documents: Sequence[Document]) -> list[list[float]]: ...

    async def embed_query(self, query: str) -> list[float]: ...


class VectorStore(Protocol):
    """Persists embedded documents and runs nearest-neighbour search."""

    async def add(self, documents: Sequence[EmbeddedDocument]) -> list[int]: ...

    async def search(self, embedding: Sequence[float], limit: int) -> list[Hit]: ...

    async def count(self) -> int: ...
