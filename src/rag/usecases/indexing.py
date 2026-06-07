"""Index documents: embed each page and persist it with derived metadata."""
from __future__ import annotations

from collections.abc import Sequence

from rag.core.entities import Document, EmbeddedDocument
from rag.core.ports import Embedder, VectorStore
from rag.core.urls import build_document_metadata


class IndexDocuments:
    def __init__(self, embedder: Embedder, store: VectorStore) -> None:
        self._embedder = embedder
        self._store = store

    async def execute(self, documents: Sequence[Document]) -> list[int]:
        """Embed the pages (title + body) and store them, deriving url/domain/path
        into the metadata. Returns the inserted ids."""
        embeddings = await self._embedder.embed_documents(documents)
        embedded = [
            EmbeddedDocument(
                content=doc.content,
                metadata=build_document_metadata(doc.url, doc.title, doc.metadata),
                embedding=embedding,
            )
            for doc, embedding in zip(documents, embeddings)
        ]
        return await self._store.add(embedded)
