"""Request/response DTOs for the lab API (``/api/*``).

The lab is the "generate data → train → evaluate → compare" workflow the React
front-end drives. These models are the wire contract for that UI. Plain dict/list
payloads (no pandas) — the API layer never imports the UI extras.
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

Embedder = Literal["ollama", "sentence-transformers"]


# ── shared ─────────────────────────────────────────────────────────────────────
class FileCount(BaseModel):
    """A dataset file and how many records it holds (0 if absent)."""
    file: str
    count: int


class EvalInfo(BaseModel):
    """The eval set bound to this process (EVAL_DIR), with record counts."""
    dir: str
    is_sample: bool
    corpus: int
    queries: int


# ── GET /api/status ─────────────────────────────────────────────────────────────
class OllamaStatus(BaseModel):
    reachable: bool
    models: list[str] = Field(default_factory=list)


class EmbedInfo(BaseModel):
    embedder: str
    model: str
    embed_dim: int


class StatusResponse(BaseModel):
    ollama: OllamaStatus
    device: str                 # "cuda" | "mps" | "cpu" | "torch 미설치 …"
    settings: EmbedInfo         # the process default embedder/model/dim
    eval: EvalInfo
    training_ready: bool
    runs: int                   # number of recorded eval runs


# ── GET /api/models ─────────────────────────────────────────────────────────────
class ModelsResponse(BaseModel):
    embedder: Embedder
    models: list[str]
    default: str                # a sensible pre-selection (e.g. an "embedding" model)


# ── GET /api/data/* ─────────────────────────────────────────────────────────────
class DataOverviewResponse(BaseModel):
    train: FileCount
    test: FileCount
    eval: EvalInfo


class PairItem(BaseModel):
    query: str | None = None
    title: str | None = None      # the positive's title
    content: str | None = None    # the positive's text (only when requested)


class PairsResponse(BaseModel):
    file: str
    total: int
    items: list[PairItem]


class CorpusDoc(BaseModel):
    id: str | None = None
    title: str | None = None
    text: str = ""


class CorpusResponse(BaseModel):
    dir: str
    total: int
    items: list[CorpusDoc]


# ── POST /api/data/pairs ────────────────────────────────────────────────────────
class GenPairsRequest(BaseModel):
    method: Literal["toy", "synthetic"] = "toy"
    # synthetic-only knobs (ignored for toy):
    corpus_file: str | None = None
    gen_model: str | None = None
    n_queries: int = Field(default=50, ge=1, le=10_000)
    hard_negatives: int = Field(default=4, ge=0, le=50)


class GenPairsResponse(BaseModel):
    message: str
    train: FileCount
    test: FileCount
    preview: list[PairItem]


# ── POST /api/data/eval ─────────────────────────────────────────────────────────
class GenEvalRequest(BaseModel):
    # None → use the generator's full distractor pool.
    n_distractors: int | None = Field(default=None, ge=0)


class GenEvalResponse(BaseModel):
    message: str
    dir: str
    corpus: int
    queries: int
    qrels: int
    preview: list[CorpusDoc]


# ── runs registry ───────────────────────────────────────────────────────────────
class RunRecord(BaseModel):
    id: str
    created_at: str
    label: str
    embedder: str
    model: str
    eval_dir: str
    metrics: dict[str, float] = Field(default_factory=dict)
    # Content hash of the eval set this run was measured on (None for legacy runs).
    # Scores are only comparable within one fingerprint.
    eval_fingerprint: str | None = None
    n_queries: int | None = None                    # how many judged queries the means cover
    ci95: dict[str, list[float]] | None = None      # {metric: [lo, hi]} bootstrap 95% CI


class RunsResponse(BaseModel):
    runs: list[RunRecord]               # newest first
    best: dict[str, float]              # max per metric on the CURRENT eval set (for Δ / highlight)
    current_fingerprint: str | None = None  # fingerprint of the eval set bound right now
    metric_keys: list[str]              # display order


class DeleteRunResponse(BaseModel):
    deleted: str
    remaining: int


# ── POST /api/eval ──────────────────────────────────────────────────────────────
class EvalRequest(BaseModel):
    embedder: Embedder = "ollama"
    model: str
    ollama_url: str | None = None
    eval_dir: str | None = None         # defaults to EVAL_DIR
    label: str = ""                     # falls back to the model name


class EvalResponse(BaseModel):
    model: str                          # the resolved active model
    embed_dim: int                      # auto-detected
    metrics: dict[str, float]
    n_queries: int                      # judged queries behind the means
    ci95: dict[str, list[float]]        # {metric: [lo, hi]} bootstrap 95% CI of each mean
    run: RunRecord                      # the appended registry record
    prior_best: dict[str, float]        # best *before* this run, same eval set only (for Δ)


# ── POST /api/train (Server-Sent Events) ────────────────────────────────────────
# The response is an SSE stream, not a JSON body — see rag.api.routes.lab.train for the
# event protocol (start / log / loss / metrics / done / error).
class TrainRequest(BaseModel):
    base_model: str = "Qwen/Qwen3-Embedding-0.6B"   # the HF checkpoint to fine-tune
    output_dir: str = "outputs/embedding-ft"
    epochs: int = Field(default=1, ge=1, le=100)
    batch_size: int = Field(default=16, ge=1, le=1024)
    learning_rate: float = Field(default=2e-5, gt=0)
    device: str = ""                                 # "" = auto (cuda → mps → cpu)
    # Fine-tuning method: "full" (all weights) or "lora" (low-rank adapters, merged on
    # save). lora_* are ignored when method="full".
    method: Literal["full", "lora"] = "full"
    lora_r: int = Field(default=16, ge=1, le=256)
    lora_alpha: int = Field(default=32, ge=1, le=512)
    lora_dropout: float = Field(default=0.05, ge=0, le=0.9)
