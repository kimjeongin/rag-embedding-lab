"""Eval-run registry — append each evaluation result so the API (and CLI) can list and
compare models.

A run is one row: which model was measured, over which eval set, and the metrics it
scored. Stored append-only as JSONL under `runs/`, newest-first on read. Stdlib only —
keeps it unit-testable and free of the API/training stacks.
"""
from __future__ import annotations

import json
import logging
import os
import uuid
from datetime import datetime
from pathlib import Path

log = logging.getLogger(__name__)

DEFAULT_RUNS_FILE = "runs/evals.jsonl"

# Metric columns we track, in display order (matches rag.evaluation.metrics output).
# recall@50 is the candidate-generation headline — in a hybrid+rerank pipeline the
# dense model's job is getting the answer into the fused candidate set, not top-1.
METRIC_KEYS: tuple[str, ...] = (
    "recall@1", "recall@3", "recall@5", "recall@10", "recall@50", "mrr@10", "ndcg@10",
)


def runs_file() -> str:
    """Path to the registry file (RUNS_FILE), defaulting to runs/evals.jsonl."""
    return os.getenv("RUNS_FILE", DEFAULT_RUNS_FILE)


def append_run(
    label: str,
    embedder: str,
    model: str,
    eval_dir: str,
    metrics: dict[str, float],
    path: str | None = None,
    *,
    eval_fingerprint: str | None = None,
    n_queries: int | None = None,
    ci95: dict[str, tuple[float, float]] | None = None,
    per_query: dict[str, dict[str, float]] | None = None,
    rankings: dict[str, list[str]] | None = None,
    split: str | None = None,
    note: str | None = None,
) -> dict:
    """Append one eval result and return the stored record (label falls back to model).

    ``eval_fingerprint`` identifies the eval set's *contents* (the dir path can't —
    regenerating the set reuses the path), so scores stay comparable only within a
    fingerprint. ``per_query``/``ci95`` keep the raw scores behind the averages for
    confidence intervals and paired run comparisons; ``rankings`` keeps what was
    actually retrieved (top-10 per query) so a diff can show results side by side.
    ``split`` records which qrels split scored this (dev = tuning, final = one-shot
    confirmation); ``note`` carries the experimenter's hypothesis/memo.
    """
    record: dict = {
        "id": uuid.uuid4().hex[:8],
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "label": (label or "").strip() or model,
        "embedder": embedder,
        "model": model,
        "eval_dir": eval_dir,
        "metrics": {k: float(v) for k, v in metrics.items()},
    }
    if eval_fingerprint:
        record["eval_fingerprint"] = eval_fingerprint
    if n_queries is not None:
        record["n_queries"] = n_queries
    if ci95:
        record["ci95"] = {k: [float(lo), float(hi)] for k, (lo, hi) in ci95.items()}
    if per_query:
        record["per_query"] = per_query
    if rankings:
        record["rankings"] = rankings
    if split:
        record["split"] = split
    if note and note.strip():
        record["note"] = note.strip()
    out = Path(path or runs_file())
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
    return record


def get_run(run_id: str, path: str | None = None) -> dict | None:
    """The full stored record for one run id (None if absent)."""
    for record in load_runs(path):
        if record.get("id") == run_id:
            return record
    return None


def _read(src: Path) -> list[dict]:
    """Records in file (insertion) order. A corrupt line (e.g. an append torn by a
    crash) is skipped with a warning instead of failing the whole registry — one bad
    line must not take down every screen that lists runs."""
    records: list[dict] = []
    for lineno, line in enumerate(src.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            log.warning("runs registry: skipping corrupt line %d in %s", lineno, src)
    return records


def load_runs(path: str | None = None) -> list[dict]:
    """All stored runs, newest first ([] if the file doesn't exist yet)."""
    src = Path(path or runs_file())
    if not src.exists():
        return []
    runs = _read(src)
    runs.reverse()
    return runs


def delete_run(run_id: str, path: str | None = None) -> int:
    """Remove the run with this id; return how many remain.

    Rewrites via a temp file + atomic replace — a crash mid-delete must not wipe the
    registry (the only persistent record of every evaluation).
    """
    src = Path(path or runs_file())
    if not src.exists():
        return 0
    kept = [r for r in _read(src) if r.get("id") != run_id]
    tmp = src.with_name(src.name + ".tmp")  # same dir → same filesystem → atomic replace
    with tmp.open("w", encoding="utf-8") as f:
        for record in kept:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    os.replace(tmp, src)
    return len(kept)


def best_per_metric(path: str | None = None, fingerprint: str | None = None) -> dict[str, float]:
    """The best (max) value seen for each metric — for Δ comparisons.

    Pass ``fingerprint`` to restrict to runs measured on that exact eval-set content;
    scores from different eval sets aren't comparable, so a cross-set "best" would be
    meaningless. (Content hash, not the dir path: regenerating a set reuses the path.)
    Runs recorded before fingerprints existed never match a fingerprint filter.
    """
    best: dict[str, float] = {}
    for record in load_runs(path):
        if fingerprint is not None and record.get("eval_fingerprint") != fingerprint:
            continue
        for key, value in record.get("metrics", {}).items():
            if value is not None and (key not in best or value > best[key]):
                best[key] = float(value)
    return best
