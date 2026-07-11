"""rag.api.indexjob — the one background reindex slot (state machine + routes),
with the heavy _execute substituted so no torch/Qdrant is involved."""
from __future__ import annotations

import threading

import pytest
from fastapi.testclient import TestClient

from rag.api import indexjob
from rag.api.app import create_app
from rag.config import Settings


@pytest.fixture(autouse=True)
def reset_state():
    indexjob._state.clear()
    indexjob._state["status"] = "idle"
    yield
    indexjob._state.clear()
    indexjob._state["status"] = "idle"


def instant_execute(model, corpus_file, recreate, truncate_dim, progress):
    progress(3, 3)
    return {"collection": f"docs__{model}__4d__f", "alias": "docs-live",
            "docs": 3, "dim": 4, "model": model, "skipped": False}


def wait_done(timeout=5.0):
    import time
    deadline = time.monotonic() + timeout
    while indexjob.status()["status"] == "running" and time.monotonic() < deadline:
        time.sleep(0.01)
    return indexjob.status()


def test_start_runs_to_done_with_progress(monkeypatch):
    monkeypatch.setattr(indexjob, "_execute", instant_execute)
    started = indexjob.start("outputs/ft")
    assert started["status"] == "running" and started["model"] == "outputs/ft"

    state = wait_done()
    assert state["status"] == "done"
    assert (state["done"], state["total"]) == (3, 3)
    assert state["summary"]["docs"] == 3


def test_failure_lands_in_state_not_a_dead_thread(monkeypatch):
    def boom(*args):
        raise ConnectionError("qdrant down")

    monkeypatch.setattr(indexjob, "_execute", boom)
    indexjob.start("outputs/ft")
    state = wait_done()
    assert state["status"] == "failed"
    assert "qdrant down" in state["error"]


def test_second_start_while_running_is_refused(monkeypatch):
    release = threading.Event()

    def blocking(*args):
        release.wait(5)
        return {}

    monkeypatch.setattr(indexjob, "_execute", blocking)
    indexjob.start("outputs/ft")
    with pytest.raises(RuntimeError, match="이미 실행 중"):
        indexjob.start("outputs/other")
    release.set()
    wait_done()

    # done → a new job may start again
    monkeypatch.setattr(indexjob, "_execute", instant_execute)
    assert indexjob.start("outputs/other")["status"] == "running"
    wait_done()


# ── routes ───────────────────────────────────────────────────────────────────────
def make_client() -> TestClient:
    return TestClient(create_app(Settings(embedder="sentence-transformers",
                                          st_model="outputs/ft", embed_dim=4)))


def test_index_routes_start_and_poll(monkeypatch):
    monkeypatch.setattr(indexjob, "_execute", instant_execute)
    with make_client() as client:
        resp = client.post("/api/index", json={})    # model "" → falls back to ST_MODEL
        assert resp.status_code == 200
        assert resp.json()["model"] == "outputs/ft"

        wait_done()
        polled = client.get("/api/index/status").json()
    assert polled["status"] == "done"
    assert polled["summary"]["model"] == "outputs/ft"


def test_index_route_conflict_while_running(monkeypatch):
    release = threading.Event()

    def blocking(*args):
        release.wait(5)
        return {}

    monkeypatch.setattr(indexjob, "_execute", blocking)
    with make_client() as client:
        assert client.post("/api/index", json={}).status_code == 200
        resp = client.post("/api/index", json={"model": "outputs/other"})
    assert resp.status_code == 409
    release.set()
    wait_done()
