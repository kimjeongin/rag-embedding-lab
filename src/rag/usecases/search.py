"""Search documents: embed the query, retrieve via the store, post-process.

`pages` and `sites` share the same retrieval (embed query -> store.search); they
differ only in the pure post-processing applied to the hits (rag.core.ranking).
The use case depends on the Embedder/VectorStore ports, never on Ollama/Qdrant.
"""
from __future__ import annotations

from rag.core.entities import Hit, Site
from rag.core.ports import Embedder, VectorStore
from rag.core.ranking import apply_max_per_domain, group_by_site


class SearchDocuments:
    def __init__(self, embedder: Embedder, store: VectorStore) -> None:
        self._embedder = embedder
        self._store = store

    async def pages(
        self,
        query: str,
        top_k: int,
        max_per_domain: int | None = None,
        fetch_k: int = 50,
    ) -> list[Hit]:
        """Page-level results. Without a cap this is plain cosine top-k; with a
        cap we pull a larger pool (`fetch_k`) so we can still fill `top_k`."""
        embedding = await self._embedder.embed_query(query)
        limit = top_k if max_per_domain is None else max(fetch_k, top_k)
        hits = await self._store.search(embedding, limit)
        return apply_max_per_domain(hits, max_per_domain)[:top_k]

    async def sites(self, query: str, top_k: int, fetch_k: int = 50) -> list[Site]:
        """Site-level results: a `fetch_k` pool grouped by domain into top_k sites."""
        embedding = await self._embedder.embed_query(query)
        hits = await self._store.search(embedding, fetch_k)
        return group_by_site(hits, top_k)
