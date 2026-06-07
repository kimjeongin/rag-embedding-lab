"""Qdrant adapter (implements rag.core.ports.VectorStore).

A dedicated vector database — no extension to install, no SQL. The collection is
created on demand (cosine distance, embed_dim sized). Documents are upserted as
points whose payload carries the body + metadata; search maps scored points back
to domain `Hit` entities. The use cases never see Qdrant types.
"""
from __future__ import annotations

from collections.abc import Sequence

from qdrant_client import AsyncQdrantClient, models

from rag.core.entities import EmbeddedDocument, Hit


def _point_to_hit(point) -> Hit:
    """Map a scored Qdrant point back to a domain Hit.

    With Cosine distance the point `score` IS the similarity (larger = closer),
    matching our convention. Body + metadata come from the payload.
    """
    payload = point.payload or {}
    meta = payload.get("metadata") or {}
    return Hit(
        id=int(point.id),
        content=payload.get("content", ""),
        similarity=float(point.score),
        url=meta.get("url"),
        title=meta.get("title"),
        domain=meta.get("domain"),
    )


class QdrantVectorStore:
    def __init__(self, client: AsyncQdrantClient, collection: str, vector_size: int) -> None:
        self._client = client
        self._collection = collection
        self._size = vector_size

    async def ensure_collection(self) -> None:
        """Create the collection if it doesn't exist (cosine, embed_dim sized)."""
        if not await self._client.collection_exists(self._collection):
            await self._client.create_collection(
                collection_name=self._collection,
                vectors_config=models.VectorParams(
                    size=self._size, distance=models.Distance.COSINE
                ),
            )

    async def add(self, documents: Sequence[EmbeddedDocument]) -> list[int]:
        """Upsert documents as points. Point ids are assigned sequentially from the
        current count (append-only; fine for this demo's indexing flow)."""
        base = await self.count()
        points: list[models.PointStruct] = []
        ids: list[int] = []
        for offset, doc in enumerate(documents, start=1):
            point_id = base + offset
            ids.append(point_id)
            points.append(
                models.PointStruct(
                    id=point_id,
                    vector=list(doc.embedding),
                    payload={"content": doc.content, "metadata": doc.metadata},
                )
            )
        await self._client.upsert(collection_name=self._collection, points=points)
        return ids

    async def search(self, embedding: Sequence[float], limit: int) -> list[Hit]:
        response = await self._client.query_points(
            collection_name=self._collection,
            query=list(embedding),
            limit=limit,
            with_payload=True,
        )
        return [_point_to_hit(point) for point in response.points]

    async def count(self) -> int:
        return (await self._client.count(collection_name=self._collection)).count
