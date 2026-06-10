"""POST /api/train streams the fine-tune as Server-Sent Events.

The subprocess (rag-train) is faked, so this exercises the route's stream + the
log→event parsing without torch or a real training run.
"""
import asyncio

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402


class _FakeStdout:
    """An async-iterable over canned stdout lines, like asyncio's StreamReader."""

    def __init__(self, lines):
        self._lines = list(lines)

    def __aiter__(self):
        return self

    async def __anext__(self):
        if not self._lines:
            raise StopAsyncIteration
        return self._lines.pop(0)


class _FakeProc:
    def __init__(self, lines):
        self.stdout = _FakeStdout(lines)
        self.returncode = None          # like asyncio's Process: None until wait()

    async def wait(self):
        self.returncode = 0
        return 0

    def kill(self):                     # the route kills on client disconnect
        self.returncode = -9


_LINES = [
    b"loading base model\n",
    b"baseline eval: ndcg@10 = 0.8000\n",
    b"{'loss': 0.5, 'epoch': 1.0}\n",
    b"{'loss': 0.3, 'epoch': 2.0}\n",
    b"after fine-tuning: ndcg@10 = 0.9200\n",
    b"saved model\n",
]


def _build_app(monkeypatch):
    async def _fake_exec(*_args, **_kwargs):
        return _FakeProc(_LINES)

    monkeypatch.setattr(asyncio, "create_subprocess_exec", _fake_exec)

    from rag.api.app import create_app

    return create_app()


def test_train_streams_start_loss_metrics_done(monkeypatch):
    app = _build_app(monkeypatch)
    with TestClient(app) as client:
        resp = client.post("/api/train", json={"epochs": 1, "batch_size": 4})

    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/event-stream")
    body = resp.text

    # start → log → two loss points (newly-seen only) → metrics → done, in order.
    assert "event: start" in body
    assert body.count("event: loss") == 2
    assert '"loss": 0.5' in body and '"loss": 0.3' in body
    assert '"before": 0.8' in body and '"after": 0.92' in body
    assert '"exit_code": 0' in body
    assert body.index("event: start") < body.index("event: loss") < body.index("event: done")


def test_train_rejects_concurrent_run_with_409(monkeypatch):
    """Two trainers saving the same output_dir would corrupt the artifact — while one
    run holds the lock, a second POST must be refused."""
    from rag.api.routes.lab import train as train_route

    app = _build_app(monkeypatch)
    with TestClient(app) as client:
        asyncio.run(train_route._run_lock.acquire())   # simulate an in-flight run
        try:
            resp = client.post("/api/train", json={})
            assert resp.status_code == 409
            assert "이미 실행 중" in resp.json()["detail"]
        finally:
            train_route._run_lock.release()
