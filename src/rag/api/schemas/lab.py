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
    fingerprint: str | None = None      # content hash of the tuning (dev) split
    splits: list[str] = Field(default_factory=list)  # qrels splits present (dev/final/test)


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
    best_ndcg: float | None = None      # best nDCG@10 on the current eval set (context bar)
    active_job: str | None = None       # id of the running training job, if any
    handed_off: dict | None = None      # latest delivery marker {model, at}
    qdrant_reachable: bool = False      # serving vector store up? (header dot)
    indexing: bool = False              # background reindex running? (header pill)


# ── GET /api/models ─────────────────────────────────────────────────────────────
class ModelsResponse(BaseModel):
    embedder: Embedder
    models: list[str]
    default: str                # a sensible pre-selection (e.g. an "embedding" model)


# ── GET /api/data/* ─────────────────────────────────────────────────────────────
class DataOverviewResponse(BaseModel):
    # True when EVERY training record carries hard negatives — the Train form uses
    # this to warn about TripletLoss before the run fails, not after.
    train_has_negatives: bool = False
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
    # quality gates (0 disables): round-trip consistency filter / false-negative margin
    round_trip_k: int = Field(default=1, ge=0, le=10)
    neg_margin: float = Field(default=0.05, ge=0.0, le=0.5)


class GenPairsResponse(BaseModel):
    message: str
    train: FileCount
    test: FileCount
    preview: list[PairItem]


# ── POST /api/data/crawl/stream ─────────────────────────────────────────────────
class CrawlRequest(BaseModel):
    url: str = Field(min_length=8)            # site root, or a sitemap.xml directly
    max_pages: int = Field(default=300, ge=1, le=5_000)
    corpus_file: str = "data/corpus.jsonl"    # where the pages land


# ── POST /api/data/eval ─────────────────────────────────────────────────────────
class GenEvalRequest(BaseModel):
    # sample = toy gold docs + synthetic distractors; corpus = the real corpus as the
    # haystack with the held-out test split as queries (after crawl + synthetic gen).
    source: Literal["sample", "corpus"] = "sample"
    # sample-only: None → use the generator's full distractor pool.
    n_distractors: int | None = Field(default=None, ge=0)
    # corpus-only: where the crawled corpus lives.
    corpus_file: str = "data/corpus.jsonl"


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
    split: str | None = None                        # dev (tuning) | final (one-shot confirm) | test (legacy)
    note: str | None = None                         # experimenter's hypothesis/memo


class RunsResponse(BaseModel):
    runs: list[RunRecord]               # newest first
    best: dict[str, float]              # max per metric on the CURRENT eval set (for Δ / highlight)
    current_fingerprint: str | None = None  # fingerprint of the current eval set's dev split
    final_fingerprint: str | None = None    # fingerprint of its final split (None if absent)
    metric_keys: list[str]              # display order


class DeleteRunResponse(BaseModel):
    deleted: str
    remaining: int


# ── GET /api/runs/diff — paired run-vs-run comparison ────────────────────────────
class DiffResponse(BaseModel):
    a: RunRecord                        # baseline run
    b: RunRecord                        # candidate run (+delta = b better)
    metric: str                         # the headline metric the win/loss uses
    n: int                              # paired queries behind the comparison
    wins: int
    losses: int
    ties: int
    mean_a: float
    mean_b: float
    delta: float
    p_value: float                      # sign-flip permutation test on per-query deltas
    queries: list[dict]                 # {query_id, a, b, delta[, text, retrieved_a/b, relevant]}
    by_metric: dict[str, dict]          # every shared metric's paired summary
    slices: list[dict]                  # per-topic {topic, n, mean_a, mean_b, delta}
    texts_available: bool = False       # query texts + doc titles joined from the live eval set


# ── POST /api/eval ──────────────────────────────────────────────────────────────
class EvalRequest(BaseModel):
    embedder: Embedder = "sentence-transformers"
    model: str
    ollama_url: str | None = None
    eval_dir: str | None = None         # defaults to EVAL_DIR
    label: str = ""                     # falls back to the model name
    # dev = tuning split (default, all day-to-day comparisons); final = the held-out
    # one-shot confirmation for the chosen winner (never used for selection).
    split: Literal["dev", "final"] = "dev"
    note: str = ""                      # experimenter's memo, shown alongside the run
    # Matryoshka inference: truncate to this many dims before scoring (ST models only).
    # Records the run as "…@{dim}" so the dim→quality curve is comparable in Compare.
    truncate_dim: int | None = Field(default=None, ge=8, le=4096)


class EvalResponse(BaseModel):
    model: str                          # the resolved active model
    embed_dim: int                      # auto-detected
    metrics: dict[str, float]
    n_queries: int                      # judged queries behind the means
    ci95: dict[str, list[float]]        # {metric: [lo, hi]} bootstrap 95% CI of each mean
    run: RunRecord                      # the appended registry record
    prior_best: dict[str, float]        # best *before* this run, same eval set only (for Δ)
    split: str = "dev"                  # the split that actually scored this (resolved)


# ── training config (one run inside a job) ──────────────────────────────────────
class TrainRequest(BaseModel):
    base_model: str = "Qwen/Qwen3-Embedding-0.6B"   # the HF checkpoint to fine-tune
    output_dir: str = "outputs/embedding-ft"        # name prefix — see auto_name
    epochs: int = Field(default=12, ge=1, le=100)   # a CEILING — early stopping ends sooner
    batch_size: int = Field(default=16, ge=1, le=1024)
    learning_rate: float = Field(default=2e-5, gt=0)
    device: str = ""                                 # "" = auto (cuda → mps → cpu)
    # Training loss — all fit the (query, positive[, negatives]) dataset; "triplet"
    # additionally requires mined hard negatives (short records get padded).
    loss: Literal["mnrl", "cached_mnrl", "gist", "triplet"] = "mnrl"
    # Backbone dropout override; None keeps the model's own defaults. (LoRA adapters
    # have their separate lora_dropout below.)
    dropout: float | None = Field(default=None, ge=0, le=0.9)
    # Matryoshka: wrap the loss so truncated vectors (768→256→128→…) stay strong.
    # matryoshka_dims empty + matryoshka on → derived from the model's dim at train time.
    matryoshka: bool = False
    matryoshka_dims: list[int] = Field(default_factory=list)
    # Early stopping: stop after this many epochs without improvement on the monitored
    # metric and save the BEST epoch's weights. 0 = off (run all epochs, save the last).
    early_stop_patience: int = Field(default=3, ge=0, le=20)
    early_stop_metric: Literal["ndcg", "loss"] = "ndcg"
    # Append "-{loss}[-r{r}]-e{best_epoch}" to output_dir at save time so the model
    # name itself says how it was trained.
    auto_name: bool = True
    # Fixed seed = reproducible config; sweep the same config over seeds for variance.
    seed: int = Field(default=42, ge=0)
    # Experimenter's hypothesis/memo — lands in train_meta.json and next to the run.
    note: str = ""
    # Fine-tuning method: "full" (all weights) or "lora" (low-rank adapters, merged on
    # save). lora_* are ignored when method="full".
    method: Literal["full", "lora"] = "full"
    lora_r: int = Field(default=16, ge=1, le=256)
    lora_alpha: int = Field(default=32, ge=1, le=512)
    lora_dropout: float = Field(default=0.05, ge=0, le=0.9)
    lora_target: Literal["all-linear", "attention"] = "all-linear"


# ── /api/jobs — server-owned training jobs (single run or sweep) ─────────────────
class JobRunSpec(BaseModel):
    label: str = ""                     # what varied vs the base config, e.g. "lr=1e-4"
    config: TrainRequest


class JobCreateRequest(BaseModel):
    # One run = a normal training; several = a sweep, executed sequentially (one
    # device). The client expands axes/seeds into this explicit list — what you see
    # in the preview is exactly what runs.
    runs: list[JobRunSpec] = Field(min_length=1, max_length=64)
    auto_eval: bool = True              # evaluate each run on the dev split as it finishes
    # After the sweep: keep only the top-k models' folders (~1GB each); the losing
    # runs keep their eval records, just not their weights.
    keep_top_k: int | None = Field(default=None, ge=1)
    # Median pruning (sweep-only): kill a run mid-training once its best-so-far val
    # nDCG trails the median of the completed runs — saves compute on losing configs.
    prune: bool = False


class JobRunState(BaseModel):
    idx: int
    label: str = ""
    status: str                          # pending|running|trained|evaluated|failed|skipped|stopped|interrupted
    config: dict
    loss: list[dict] = Field(default_factory=list)    # per-step {step, epoch, loss}
    epochs: list[dict] = Field(default_factory=list)  # per-epoch {epoch, eval_loss, ndcg, best_epoch, elapsed}
    result: dict | None = None           # {output_dir, best_epoch, ran, early_stopped, ndcg_before/after}
    eval: dict | None = None             # {run_id, metrics, n_queries, split} from auto-eval
    error: str | None = None
    hint: str | None = None              # actionable next step for a failure
    started_at: str | None = None
    finished_at: str | None = None
    model_deleted: bool = False          # pruned by keep_top_k


class JobState(BaseModel):
    id: str
    kind: str                            # train | sweep
    status: str                          # pending|running|done|stopped|failed|interrupted
    created_at: str
    auto_eval: bool = True
    keep_top_k: int | None = None
    prune: bool = False                  # median pruning enabled for this sweep
    current: int | None = None           # idx of the run training right now
    error: str | None = None
    runs: list[JobRunState]


class JobSummary(BaseModel):
    id: str
    kind: str
    status: str
    created_at: str
    n_runs: int
    n_finished: int                      # runs in a terminal state
    labels: list[str] = Field(default_factory=list)  # first few run labels (list display)


class JobsListResponse(BaseModel):
    jobs: list[JobSummary]               # newest first
    active: str | None = None            # id of the running job, if any


# ── /api/models — saved-model shelf (detail / delete / handoff) ──────────────────
class ModelDetail(BaseModel):
    path: str
    size_bytes: int = 0
    dim: int | None = None
    created_at: str | None = None
    meta: dict | None = None             # train_meta.json (recipe, history, fingerprints)
    eval_dev: dict | None = None         # best dev-split run for this model
    eval_final: dict | None = None       # latest final-split (one-shot confirm) run
    handed_off: bool = False             # a HANDOFF.md exists in the dir


class ModelsDetailResponse(BaseModel):
    models: list[ModelDetail]
    disk_total_bytes: int = 0


class DeleteModelResponse(BaseModel):
    deleted: str
    models: list[ModelDetail]
    disk_total_bytes: int = 0


class HandoffRequest(BaseModel):
    path: str
    # Also reindex the serving index with this model in the background — handoff IS
    # the "this model goes live" decision, so the index follows it by default.
    reindex: bool = True


class HandoffResponse(BaseModel):
    path: str
    markdown: str                        # HANDOFF.md content (also written into the dir)
    handoff: dict                        # handoff.json content
    indexing: str | None = None          # "started" | why not (busy/off) — reindex hook result


# ── POST /api/data/import — real query/click logs → pairs and/or qrels ──────────
class ImportPairsRequest(BaseModel):
    content: str                         # pasted JSONL or CSV (auto-detected)
    target: Literal["train", "qrels", "both"] = "train"


class ImportPairsResponse(BaseModel):
    parsed: int
    added_train: int = 0
    added_qrels: int = 0
    skipped: list[str] = Field(default_factory=list)
    fingerprint_changed: bool = False    # qrels/queries touched → prior runs incomparable
    message: str


# ── /api/data/label — judge queries against the corpus to grow qrels ────────────
class LabelSearchRequest(BaseModel):
    query: str
    embedder: Embedder = "sentence-transformers"
    model: str


class LabelDoc(BaseModel):
    id: str
    title: str | None = None
    text: str


class LabelSearchResponse(BaseModel):
    query: str
    results: list[LabelDoc]              # current model's top-k over the eval corpus


class LabelCommitRequest(BaseModel):
    query: str
    doc_ids: list[str] = Field(min_length=1)
    also_train: bool = True              # clicked docs also become training pairs


class LabelCommitResponse(BaseModel):
    query_id: str
    added_qrels: int
    added_train: int = 0
    message: str


# ── /api/search — the Qdrant-backed serving path ────────────────────────────────
class SearchRequest(BaseModel):
    query: str = Field(min_length=1)
    top_k: int = Field(default=10, ge=1, le=100)


class SearchHit(BaseModel):
    score: float                         # cosine similarity (index stores normalised vectors)
    url: str | None = None
    title: str | None = None
    content: str = ""


class SearchResponse(BaseModel):
    query: str
    collection: str                      # the versioned collection the live alias resolved to
    model: str                           # the model that embedded THIS query (must match the index)
    embed_ms: float = 0                  # query-embedding latency (the model-bound part)
    search_ms: float = 0                 # Qdrant ANN latency
    hits: list[SearchHit]


class IndexRequest(BaseModel):
    model: str = ""                      # "" → the process's ST model (ST_MODEL)
    corpus_file: str = "data/corpus.jsonl"
    recreate: bool = False               # force a rebuild even if up to date
    truncate_dim: int | None = Field(default=None, ge=8, le=4096)  # Matryoshka index


class IndexJobStatus(BaseModel):
    """The one background reindex slot (see rag.api.indexjob)."""
    status: Literal["idle", "running", "done", "failed"] = "idle"
    model: str | None = None
    done: int = 0                        # docs embedded so far
    total: int | None = None
    error: str | None = None
    summary: dict | None = None          # index_corpus result once done
    started_at: str | None = None
    finished_at: str | None = None


class CollectionInfo(BaseModel):
    """One family collection — its name encodes (model, dim, corpus fingerprint)."""
    name: str
    model_slug: str | None = None        # which model built it (from the name)
    dim: int | None = None
    points: int = 0
    fingerprint: str | None = None       # corpus-content hash (from the name)
    live: bool = False                   # is the alias pointing here right now


class SearchStatusResponse(BaseModel):
    reachable: bool                      # Qdrant answered
    alias: str                           # the serving pointer ({prefix}-live)
    collection: str | None = None        # its current target (None → nothing indexed yet)
    points: int = 0
    dim: int | None = None               # the index's vector size
    dim_matches: bool | None = None      # index dim == this process's embedder dim
    model_matches: bool | None = None    # index model == query embedder (same-dim trap guard)
    collections: list[CollectionInfo] = Field(default_factory=list)  # family incl. rollback copies
    embedder: str                        # what /api/search would embed queries with
    model: str


class AliasRequest(BaseModel):
    """POST /api/index/alias — instant rollback/roll-forward to an existing collection."""
    collection: str = Field(min_length=1)


class PruneResponse(BaseModel):
    pruned: list[str]                    # collections deleted (everything but the live target)


# ── POST /api/runs/import-trec — an external retriever's ranking as a run ───────
class ImportTrecRequest(BaseModel):
    label: str = ""                      # e.g. "BM25 (production)"
    content: str                         # TREC run lines: qid Q0 docid rank score tag


class ImportTrecResponse(BaseModel):
    run: RunRecord
    metrics: dict[str, float]
    n_queries: int
    errors: list[str] = Field(default_factory=list)
    message: str
