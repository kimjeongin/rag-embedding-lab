"""QdrantStore adapter — each method's request shape + error translation, verified
against a mock transport (no Qdrant needed)."""
from __future__ import annotations

import json

import httpx
import pytest

from rag.core.errors import VectorStoreError
from rag.vectorstore.qdrant import QdrantStore


def make_store(handler) -> QdrantStore:
    client = httpx.Client(transport=httpx.MockTransport(handler), base_url="http://q")
    return QdrantStore("http://q", client=client)


def ok(result) -> httpx.Response:
    return httpx.Response(200, json={"result": result, "status": "ok"})


def test_create_collection_sends_cosine_schema():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["method"], seen["path"] = request.method, request.url.path
        seen["body"] = json.loads(request.content)
        return ok(True)

    make_store(handler).create_collection("docs__m__4d__f", dim=4)
    assert (seen["method"], seen["path"]) == ("PUT", "/collections/docs__m__4d__f")
    assert seen["body"] == {"vectors": {"size": 4, "distance": "Cosine"}}


def test_collection_info_present_and_missing():
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/missing"):
            return httpx.Response(404, json={"status": {"error": "not found"}})
        return ok({"points_count": 7, "config": {"params": {"vectors": {"size": 4, "distance": "Cosine"}}}})

    store = make_store(handler)
    assert store.collection_info("docs") == {"points": 7, "dim": 4}
    assert store.collection_info("missing") is None


def test_upsert_waits_and_query_parses_hits():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "PUT":
            seen["wait"] = request.url.params.get("wait")
            return ok({"status": "completed"})
        seen["query_body"] = json.loads(request.content)
        return ok({"points": [{"id": "a", "score": 0.9, "payload": {"title": "t"}},
                              {"id": "b", "score": 0.5}]})

    store = make_store(handler)
    store.upsert("c", [{"id": "a", "vector": [0.1], "payload": {}}])
    assert seen["wait"] == "true"

    hits = store.query("docs-live", [0.1, 0.2], top_k=2)
    assert seen["query_body"] == {"query": [0.1, 0.2], "limit": 2, "with_payload": True}
    assert hits == [{"id": "a", "score": 0.9, "payload": {"title": "t"}},
                    {"id": "b", "score": 0.5, "payload": {}}]


def test_swap_alias_is_one_atomic_request():
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append((request.method, request.url.path))
        if request.url.path == "/aliases":
            return ok({"aliases": [{"alias_name": "docs-live", "collection_name": "old"}]})
        calls.append(json.loads(request.content))
        return ok(True)

    make_store(handler).swap_alias("docs-live", "new")
    # delete+create travel together in the single POST — never a gap with no alias
    assert calls[-1] == {"actions": [
        {"delete_alias": {"alias_name": "docs-live"}},
        {"create_alias": {"alias_name": "docs-live", "collection_name": "new"}},
    ]}


def test_swap_alias_skips_delete_when_alias_is_new():
    body = {}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/aliases":
            return ok({"aliases": []})
        body.update(json.loads(request.content))
        return ok(True)

    make_store(handler).swap_alias("docs-live", "first")
    assert body["actions"] == [{"create_alias": {"alias_name": "docs-live", "collection_name": "first"}}]


def test_connection_failure_becomes_domain_error_and_ping_false():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused")

    store = make_store(handler)
    with pytest.raises(VectorStoreError, match="연결할 수 없습니다"):
        store.list_collections()
    assert store.ping() is False
