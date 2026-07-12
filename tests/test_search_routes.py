"""/api/search routes — serving wiring over fakes (no Qdrant, no torch)."""
from __future__ import annotations

from fastapi.testclient import TestClient

from rag.api.app import create_app
from rag.api.deps import get_embedder, get_store
from rag.config import Settings

from test_serving import FakeEmbedder, FakeStore  # pytest puts tests/ on sys.path


def make_client(store: FakeStore, embedder: FakeEmbedder) -> TestClient:
    app = create_app(Settings(embedder="sentence-transformers", st_model="outputs/ft",
                              embed_dim=4, qdrant_collection="docs"))
    app.dependency_overrides[get_store] = lambda: store
    app.dependency_overrides[get_embedder] = lambda: embedder
    return TestClient(app)


def indexed_store() -> FakeStore:
    store = FakeStore()
    store.create_collection("docs__outputs-ft__4d__f", dim=4)
    store.swap_alias("docs-live", "docs__outputs-ft__4d__f")
    store.query_results = [
        {"id": "1", "score": 0.87, "payload": {"url": "https://x/a", "title": "A", "content": "본문"}},
    ]
    return store


def test_search_returns_hits():
    with make_client(indexed_store(), FakeEmbedder()) as client:
        resp = client.post("/api/search", json={"query": "vpn 안됨", "top_k": 3})
    assert resp.status_code == 200
    body = resp.json()
    assert body["collection"] == "docs__outputs-ft__4d__f"
    assert body["model"] == "outputs/ft"
    assert body["hits"] == [{"score": 0.87, "url": "https://x/a", "title": "A", "content": "본문"}]


def test_search_without_index_is_503_with_hint():
    with make_client(FakeStore(), FakeEmbedder()) as client:
        resp = client.post("/api/search", json={"query": "q"})
    assert resp.status_code == 503
    assert "rag-index" in resp.json()["detail"]


def test_search_status_reports_index():
    with make_client(indexed_store(), FakeEmbedder()) as client:
        resp = client.get("/api/search/status")
    assert resp.status_code == 200
    body = resp.json()
    assert body["reachable"] is True
    assert body["collection"] == "docs__outputs-ft__4d__f"
    assert body["dim_matches"] is True
    assert body["model_matches"] is True
    assert body["embedder"] == "sentence-transformers"
    [entry] = body["collections"]
    assert entry["live"] is True and entry["model_slug"] == "outputs-ft"


def test_search_status_flags_model_mismatch_with_same_dim():
    store = FakeStore()
    store.create_collection("docs__other-model__4d__f", dim=4)   # same dim, different model
    store.swap_alias("docs-live", "docs__other-model__4d__f")
    with make_client(store, FakeEmbedder()) as client:
        body = client.get("/api/search/status").json()
    assert body["dim_matches"] is True
    assert body["model_matches"] is False


def test_alias_rollback_and_prune():
    store = indexed_store()
    store.create_collection("docs__old-model__4d__f", dim=4)     # a rollback copy
    with make_client(store, FakeEmbedder()) as client:
        # roll back to the old collection
        resp = client.post("/api/index/alias", json={"collection": "docs__old-model__4d__f"})
        assert resp.status_code == 200
        assert resp.json()["collection"] == "docs__old-model__4d__f"
        # foreign name is refused via the domain error → 503 with the reason
        bad = client.post("/api/index/alias", json={"collection": "other__x__4d__f"})
        assert bad.status_code == 503 and "패밀리" in bad.json()["detail"]
        # prune keeps only the (new) live target
        resp = client.post("/api/index/prune")
        assert resp.status_code == 200
        assert resp.json()["pruned"] == ["docs__outputs-ft__4d__f"]
    assert "docs__old-model__4d__f" in store.collections
    assert "docs__outputs-ft__4d__f" not in store.collections
