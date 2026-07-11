"""Server-owned background reindex — one at a time, in-process.

Indexing embeds the whole corpus (minutes on a laptop GPU), so a request can't wait
for it: POST /api/index starts a daemon thread, GET /api/index/status polls. Module-
global state is the same single-worker trade jobs.py makes (documented there). Unlike
training it does NOT need subprocess isolation — the search path already runs torch in
this process, and indexing uses the very same stack.

The thread owns its own event loop (asyncio.run) and drives the shared flow
(rag.serving.index_corpus), so CLI and server indexing cannot drift apart. The model
is embedded per-job (a handoff may index a model that is not the process default);
Qdrant location comes from the process env.
"""
from __future__ import annotations

import asyncio
import threading
from datetime import datetime

from rag import lab, serving
from rag.vectorstore.qdrant import QdrantStore

_state: dict = {"status": "idle"}   # idle | running | done | failed
_lock = threading.Lock()


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def status() -> dict:
    return dict(_state)


def start(model: str, corpus_file: str = "data/corpus.jsonl",
          recreate: bool = False, truncate_dim: int | None = None) -> dict:
    """Kick off one background reindex. Raises RuntimeError while one is running —
    two concurrent embed passes would fight over the one device."""
    with _lock:
        if _state.get("status") == "running":
            raise RuntimeError(f"재색인이 이미 실행 중입니다 ({_state.get('model')})")
        _state.clear()
        _state.update({"status": "running", "model": model, "done": 0, "total": None,
                       "started_at": _now(), "error": None, "summary": None})
        snapshot = dict(_state)  # before the thread runs — it may finish instantly
        threading.Thread(
            target=_run, args=(model, corpus_file, recreate, truncate_dim), daemon=True
        ).start()
    return snapshot


def _run(model: str, corpus_file: str, recreate: bool, truncate_dim: int | None) -> None:
    try:
        summary = _execute(model, corpus_file, recreate, truncate_dim, _progress)
        _state.update({"status": "done", "summary": summary, "finished_at": _now()})
    except Exception as exc:  # noqa: BLE001 — a failed job must land in state, not a dead thread
        _state.update({"status": "failed", "error": f"{type(exc).__name__}: {exc}",
                       "finished_at": _now()})


def _progress(done: int, total: int) -> None:
    _state.update({"done": done, "total": total})


def _execute(model: str, corpus_file: str, recreate: bool,
             truncate_dim: int | None, progress) -> dict:
    """The actual work, separated so tests can substitute it without threads/torch."""
    from rag.embeddings import build_embedder

    dim = lab.infer_dim("sentence-transformers", model, "", truncate_dim)
    settings = lab.build_eval_settings("sentence-transformers", model, dim, "", truncate_dim)

    async def run() -> dict:
        async with build_embedder(settings) as embedder:
            return await serving.index_corpus(
                settings, embedder, store, corpus_file,
                recreate=recreate, progress=progress,
            )

    with QdrantStore(settings.qdrant_url) as store:
        return asyncio.run(run())
