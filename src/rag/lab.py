"""Lab support glue — environment introspection + eval-settings construction.

The "lab" (generate data → train → evaluate → compare) needs to answer a few
cross-cutting questions that aren't business logic and don't belong to any one
delivery layer: *is Ollama up and what models does it serve?*, *what device will
training use?*, *does this eval set look like the bundled sample?*, *what embedding
dimension does this model produce?*.

This module owns those, framework-free: stdlib + httpx + the rag stack, but **no
fastapi**. The HTTP API (`rag.api`) imports from here, which keeps the route handlers
thin and this logic unit-testable. (Same neutral move as `rag.runs` for the eval-run
registry.)
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import httpx

from rag.config import Settings
from rag.dataset import load_jsonl
from rag.evaluation.beir import available_splits, eval_set_fingerprint, resolve_split


# ── environment introspection ──────────────────────────────────────────────────
def ollama_status(url: str | None = None) -> tuple[bool, list[str]]:
    """(reachable, model_names). Any failure means "not reachable" with no models."""
    url = url or Settings.from_env().ollama_url
    try:
        resp = httpx.get(f"{url}/api/tags", timeout=2.5)
        resp.raise_for_status()
        return True, [m["name"] for m in resp.json().get("models", [])]
    except Exception:  # noqa: BLE001 — any failure means "not reachable"
        return False, []


def device_status() -> str:
    """The device training would pick: cuda / mps / cpu (or a note if torch is absent)."""
    try:
        import torch
    except ImportError:
        return "torch 미설치 (학습 그룹 필요)"
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def training_ready() -> bool:
    """True if the training stack (torch + sentence-transformers) is importable."""
    return all(importlib.util.find_spec(m) is not None for m in ("torch", "sentence_transformers"))


# ── model discovery ────────────────────────────────────────────────────────────
def list_st_models() -> list[str]:
    """Sub-dirs of outputs/ that look like a saved sentence-transformers model."""
    root = Path("outputs")
    if not root.exists():
        return []
    return [
        str(p)
        for p in sorted(root.iterdir())
        if p.is_dir() and ((p / "config.json").exists() or (p / "modules.json").exists())
    ]


def list_models(embedder: str, ollama_url: str) -> list[str]:
    """Available models for the chosen backend: Ollama's served tags or local ST dirs."""
    if embedder == "ollama":
        return ollama_status(ollama_url)[1]
    return list_st_models()


def default_model(embedder: str, choices: list[str], preferred: str | None = None) -> str:
    """A sensible default selection when the backend changes (so a stale model isn't kept).

    ``preferred`` is the process's own model (ST_MODEL / EMBED_MODEL) — the one the
    user is actually serving — and wins when it's available; otherwise fall back to
    a heuristic (an Ollama tag that looks like an embedder, or just the first)."""
    if not choices:
        return ""
    if preferred and preferred in choices:
        return preferred
    if embedder == "ollama":
        for choice in choices:
            if "embedding" in choice:
                return choice
    return choices[0]


# ── eval set introspection ─────────────────────────────────────────────────────
def count_lines(path: str) -> int:
    """Number of JSONL records in `path` (0 if it doesn't exist)."""
    try:
        return sum(1 for _ in load_jsonl(path))
    except FileNotFoundError:
        return 0


def eval_overview(eval_dir: str) -> dict:
    """Everything the UI shows about the bound eval set (feeds the EvalInfo DTO) —
    built here once so the status and data routes can't drift apart."""
    return {
        "dir": eval_dir,
        "is_sample": is_sample_eval(eval_dir),
        "corpus": count_lines(f"{eval_dir}/corpus.jsonl"),
        "queries": count_lines(f"{eval_dir}/queries.jsonl"),
        "fingerprint": eval_set_fingerprint(eval_dir, resolve_split(eval_dir)),
        "splits": available_splits(eval_dir),
    }


def is_sample_eval(eval_dir: str) -> bool:
    """True if EVAL_DIR looks like the bundled sample (gold-/distractor- ids)."""
    path = Path(eval_dir) / "corpus.jsonl"
    if not path.exists():
        return False
    try:
        first = next(load_jsonl(str(path)))
    except (StopIteration, FileNotFoundError, json.JSONDecodeError):
        return False  # empty/missing/corrupt corpus — just "not the sample", never a 500
    return str(first.get("_id", "")).startswith(("distractor-", "gold-"))


# ── eval settings (which model to measure) ─────────────────────────────────────
def infer_dim(embedder: str, model: str, ollama_url: str, truncate_dim: int | None = None) -> int:
    """The embedding dimension this model produces — so no manual dim field can be wrong.
    With ``truncate_dim`` (Matryoshka inference) the vectors are cut to that length, so
    that IS the dimension; Ollama has no truncation, so it's rejected upstream."""
    if truncate_dim:
        return truncate_dim
    if embedder == "ollama":
        resp = httpx.post(f"{ollama_url}/api/embed", json={"model": model, "input": "x"}, timeout=30)
        resp.raise_for_status()
        return len(resp.json()["embeddings"][0])

    # Local ST dirs record the dim in the pooling config — read it instead of loading
    # the full model (which evaluate() will load again right after anyway).
    pooling_cfg = Path(model) / "1_Pooling" / "config.json"
    if pooling_cfg.exists():
        try:
            dim = json.loads(pooling_cfg.read_text(encoding="utf-8"))["embedding_dimension"]
            return int(dim)
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            pass  # malformed config — fall back to loading the model
    from sentence_transformers import SentenceTransformer

    return int(SentenceTransformer(model).get_embedding_dimension())


def build_eval_settings(
    embedder: str, model: str, embed_dim: int, ollama_url: str, truncate_dim: int | None = None
) -> Settings:
    """Settings for evaluating one model, inheriting unrelated fields from the env."""
    base = Settings.from_env()
    if embedder == "ollama":
        if truncate_dim:
            raise ValueError("차원 절단(truncate_dim)은 sentence-transformers 모델에서만 됩니다 (Ollama는 고정 차원)")
        return Settings(
            embedder="ollama", embed_model=(model or base.embed_model), embed_dim=embed_dim,
            ollama_url=(ollama_url or base.ollama_url), query_instruction=base.query_instruction,
            model_profile=base.model_profile,
            qdrant_url=base.qdrant_url, qdrant_collection=base.qdrant_collection,
        )
    # model_profile carries the env override through: evaluating with a different input
    # format than training/serving would use is the silent failure this exists to stop.
    return Settings(
        embedder="sentence-transformers", st_model=(model or base.st_model), embed_dim=embed_dim,
        query_instruction=base.query_instruction, truncate_dim=truncate_dim,
        model_profile=base.model_profile,
        qdrant_url=base.qdrant_url, qdrant_collection=base.qdrant_collection,
    )
