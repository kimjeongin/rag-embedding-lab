"""Ports — the abstraction the lab depends on (Dependency Inversion).

The embedding evaluation talks to the ``Embedder`` Protocol, never to Ollama or
sentence-transformers directly; the concrete adapters (``rag.embeddings.*``) implement
it structurally, and tests can substitute an in-memory fake. This keeps the offline
pipeline free of any driver.
"""
from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

from rag.core.entities import Document


class Embedder(Protocol):
    """Turns documents/queries into vectors (asymmetric — see rag.core.formatting).

    Both sides are batch (one call per collection) so adapters can issue a single
    Ollama request / on-device encode instead of one round-trip per item.
    """

    async def embed_documents(self, documents: Sequence[Document]) -> list[list[float]]: ...

    async def embed_queries(self, queries: Sequence[str]) -> list[list[float]]: ...
