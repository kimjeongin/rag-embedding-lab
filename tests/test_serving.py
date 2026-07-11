"""rag.serving flows — indexing (versioned collection + alias swap + idempotency)
and search (dim guard, payload mapping) against in-memory fakes."""
from __future__ import annotations

import pytest

from rag import serving
from rag.config import Settings
from rag.core.entities import Document
from rag.core.errors import VectorStoreError
from rag.dataset import write_jsonl


class FakeEmbedder:
    """Deterministic 4-dim vectors; records what was embedded."""

    def __init__(self) -> None:
        self.documents: list[Document] = []
        self.queries: list[str] = []

    async def embed_documents(self, documents):
        self.documents.extend(documents)
        return [[1.0, 0.0, 0.0, 0.0] for _ in documents]

    async def embed_queries(self, queries):
        self.queries.extend(queries)
        return [[0.0, 1.0, 0.0, 0.0] for _ in queries]


class FakeStore:
    """In-memory stand-in mirroring QdrantStore's surface."""

    def __init__(self) -> None:
        self.collections: dict[str, dict] = {}   # name -> {"dim": int, "points": {id: point}}
        self.aliases: dict[str, str] = {}
        self.query_results: list[dict] = []

    def ping(self):
        return True

    def list_collections(self):
        return sorted(self.collections)

    def collection_info(self, name):
        name = self.aliases.get(name, name)
        if name not in self.collections:
            return None
        c = self.collections[name]
        return {"points": len(c["points"]), "dim": c["dim"]}

    def create_collection(self, name, dim):
        self.collections[name] = {"dim": dim, "points": {}}

    def delete_collection(self, name):
        self.collections.pop(name, None)

    def upsert(self, name, points):
        self.collections[name]["points"].update({p["id"]: p for p in points})

    def query(self, target, vector, top_k):
        return self.query_results[:top_k]

    def alias_target(self, alias):
        return self.aliases.get(alias)

    def swap_alias(self, alias, collection):
        self.aliases[alias] = collection


def settings(**kw) -> Settings:
    defaults = dict(embedder="sentence-transformers", st_model="outputs/ft",
                    embed_dim=4, qdrant_collection="docs")
    return Settings(**{**defaults, **kw})


def corpus_file(tmp_path, records=None) -> str:
    records = records if records is not None else [
        {"url": "https://x/a", "title": "A", "content": "본문 a"},
        {"url": "https://x/b", "title": "B", "content": "본문 b"},
        {"url": "https://x/c", "title": None, "content": "본문 c"},
    ]
    path = str(tmp_path / "corpus.jsonl")
    write_jsonl(path, records)
    return path


# ── naming / identity ────────────────────────────────────────────────────────────
def test_collection_name_is_deterministic_and_safe():
    name = serving.collection_name("docs", "outputs/embedding-ft", 1024, "ab12")
    assert name == "docs__outputs-embedding-ft__1024d__ab12"
    assert name == serving.collection_name("docs", "outputs/embedding-ft", 1024, "ab12")


def test_point_id_is_stable_per_url():
    assert serving.point_id("https://x/a") == serving.point_id("https://x/a")
    assert serving.point_id("https://x/a") != serving.point_id("https://x/b")


# ── index_corpus ─────────────────────────────────────────────────────────────────
async def test_index_corpus_builds_collection_and_swaps_alias(tmp_path):
    store, embedder = FakeStore(), FakeEmbedder()
    summary = await serving.index_corpus(settings(), embedder, store, corpus_file(tmp_path))

    assert summary["skipped"] is False and summary["docs"] == 3
    assert store.aliases["docs-live"] == summary["collection"]
    assert store.collection_info(summary["collection"]) == {"points": 3, "dim": 4}
    # title flows into the embedded Document (formatting parity happens in the embedder)
    assert [d.title for d in embedder.documents] == ["A", "B", None]
    point = store.collections[summary["collection"]]["points"][serving.point_id("https://x/a")]
    assert point["payload"] == {"url": "https://x/a", "title": "A", "content": "본문 a"}


async def test_index_corpus_is_idempotent(tmp_path):
    store, path = FakeStore(), corpus_file(tmp_path)
    await serving.index_corpus(settings(), FakeEmbedder(), store, path)

    second = FakeEmbedder()
    summary = await serving.index_corpus(settings(), second, store, path)
    assert summary["skipped"] is True
    assert second.documents == []          # no re-embedding


async def test_index_corpus_recreate_rebuilds(tmp_path):
    store, path = FakeStore(), corpus_file(tmp_path)
    await serving.index_corpus(settings(), FakeEmbedder(), store, path)

    second = FakeEmbedder()
    summary = await serving.index_corpus(settings(), second, store, path, recreate=True)
    assert summary["skipped"] is False and len(second.documents) == 3


async def test_index_corpus_new_model_gets_new_collection_old_kept(tmp_path):
    store, path = FakeStore(), corpus_file(tmp_path)
    first = await serving.index_corpus(settings(), FakeEmbedder(), store, path)
    second = await serving.index_corpus(
        settings(st_model="outputs/ft-v2"), FakeEmbedder(), store, path
    )

    assert first["collection"] != second["collection"]
    assert store.aliases["docs-live"] == second["collection"]
    assert first["collection"] in store.collections     # rollback copy survives

    stale = serving.prune_collections(settings(), store)
    assert stale == [first["collection"]]
    assert first["collection"] not in store.collections


async def test_index_corpus_empty_corpus_fails_clearly(tmp_path):
    with pytest.raises(VectorStoreError, match="비어"):
        await serving.index_corpus(
            settings(), FakeEmbedder(), FakeStore(), corpus_file(tmp_path, records=[])
        )


# ── search ───────────────────────────────────────────────────────────────────────
async def test_search_returns_payload_hits(tmp_path):
    store, embedder, path = FakeStore(), FakeEmbedder(), corpus_file(tmp_path)
    await serving.index_corpus(settings(), embedder, store, path)
    store.query_results = [
        {"id": "1", "score": 0.9, "payload": {"url": "https://x/a", "title": "A", "content": "본문 a"}},
    ]

    result = await serving.search(settings(), embedder, store, "vpn 안됨", top_k=5)
    assert result["hits"] == [{"score": 0.9, "url": "https://x/a", "title": "A", "content": "본문 a"}]
    assert embedder.queries == ["vpn 안됨"]


async def test_search_without_index_or_with_wrong_dim_fails(tmp_path):
    store, embedder = FakeStore(), FakeEmbedder()
    with pytest.raises(VectorStoreError, match="색인"):
        await serving.search(settings(), embedder, store, "q")

    await serving.index_corpus(settings(), embedder, store, corpus_file(tmp_path))
    with pytest.raises(VectorStoreError, match="차원"):
        await serving.search(settings(embed_dim=8), embedder, store, "q")


def test_index_status_reports_live_state(tmp_path):
    store = FakeStore()
    empty = serving.index_status(settings(), store)
    assert empty["reachable"] is True and empty["collection"] is None


async def test_index_status_after_indexing(tmp_path):
    store = FakeStore()
    summary = await serving.index_corpus(settings(), FakeEmbedder(), store, corpus_file(tmp_path))
    overview = serving.index_status(settings(), store)
    assert overview["collection"] == summary["collection"]
    assert overview["points"] == 3 and overview["dim_matches"] is True
    assert overview["collections"] == [summary["collection"]]
