"""Eval-run registry — append each `rag-eval` result so the UI can compare models.

A run is one row: which model was measured, over which eval set, and the metrics it
scored. Stored append-only as JSONL under `runs/` so the Compare tab just reads them
back, newest first. Stdlib only (no gradio/pandas) — keeps it unit-testable.
"""
from __future__ import annotations

import json
import os
import uuid
from datetime import datetime
from pathlib import Path

DEFAULT_RUNS_FILE = "runs/evals.jsonl"

# Metric columns we track, in display order (matches rag.evaluation.metrics output).
METRIC_KEYS: tuple[str, ...] = ("recall@1", "recall@3", "recall@5", "recall@10", "mrr@10", "ndcg@10")


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
) -> dict:
    """Append one eval result and return the stored record (label falls back to model)."""
    record = {
        "id": uuid.uuid4().hex[:8],
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "label": (label or "").strip() or model,
        "embedder": embedder,
        "model": model,
        "eval_dir": eval_dir,
        "metrics": {k: float(v) for k, v in metrics.items()},
    }
    out = Path(path or runs_file())
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
    return record


def _read(src: Path) -> list[dict]:
    """Records in file (insertion) order."""
    return [json.loads(line) for line in src.read_text(encoding="utf-8").splitlines() if line.strip()]


def load_runs(path: str | None = None) -> list[dict]:
    """All stored runs, newest first ([] if the file doesn't exist yet)."""
    src = Path(path or runs_file())
    if not src.exists():
        return []
    runs = _read(src)
    runs.reverse()
    return runs


def delete_run(run_id: str, path: str | None = None) -> int:
    """Remove the run with this id; return how many remain."""
    src = Path(path or runs_file())
    if not src.exists():
        return 0
    kept = [r for r in _read(src) if r.get("id") != run_id]
    with src.open("w", encoding="utf-8") as f:
        for record in kept:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    return len(kept)


def best_per_metric(path: str | None = None) -> dict[str, float]:
    """The best (max) value seen for each metric across all runs — for Δ comparisons."""
    best: dict[str, float] = {}
    for record in load_runs(path):
        for key, value in record.get("metrics", {}).items():
            if value is not None and (key not in best or value > best[key]):
                best[key] = float(value)
    return best
