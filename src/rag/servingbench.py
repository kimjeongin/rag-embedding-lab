"""Serving benchmark — what a model costs and delivers on the REAL Qdrant path.

`rag-eval` answers "which model ranks better" with an in-memory matrix product. That
is the right shape for comparing models, and the wrong shape for choosing what to
deploy: it never touches the index, never pays per-query latency, and never shows what
the model costs in memory or storage. This module measures the other half —

  정확도  the same recall/nDCG, but scored on what **Qdrant's ANN actually returned**,
          plus the brute-force ceiling over the SAME stored vectors, so the index's
          approximation loss is separated from the model's own quality.
  응답속도 per-query embed and search latency as p50/p95/p99 (not the mean — tail
          latency is what users feel), measured one query at a time because that is
          how a search box calls it. Warm-up queries are discarded.
  자원    peak GPU/accelerator memory during encoding, model size on disk, and the
          index's vector bytes projected to 1M documents.
  색인비용 wall-clock and docs/sec for a full reindex — the cost of shipping a new
          model, which a bigger backbone can make prohibitive.

**Numbers are hardware-bound.** Latency and memory measured on Apple MPS say nothing
about a CUDA server, so every record carries a hardware fingerprint and comparisons
across fingerprints are refused by the reader, not silently averaged. Accuracy is the
only part that transfers.

Framework-free (no fastapi), same stance as `rag.serving` / `rag.evalflow`: the CLI
drives it, and it can be driven from a route later without changes.
"""
from __future__ import annotations

import json
import platform
import statistics
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path

from rag import serving
from rag.config import Settings
from rag.core.ports import Embedder
from rag.evaluation.beir import load_corpus, load_qrels, load_queries, resolve_split
from rag.evaluation.metrics import mean_metrics, per_query_metrics
from rag.vectorstore.qdrant import QdrantStore

DEFAULT_TOP_K = 50          # the lab's candidate-generation depth (see evaluation.retrieval)
DEFAULT_WARMUP = 5          # first calls include lazy init/kernel autotune — never counted
BENCH_RUNS_FILE = "runs/bench.jsonl"


# ── hardware fingerprint ──────────────────────────────────────────────────────

def hardware_fingerprint(device: str, max_seq_length: int | None = None) -> dict:
    """What the timings are bound to. Latency compared across two of these is noise."""
    import torch

    info = {
        "device": device,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        # Sequence length caps how much work one document is. A long-context model
        # left at its default looks slow for reasons that have nothing to do with
        # quality, so it belongs in the fingerprint, not in a footnote.
        "max_seq_length": max_seq_length,
    }
    if device.startswith("cuda") and torch.cuda.is_available():
        info["gpu"] = torch.cuda.get_device_name(0)
        info["gpu_total_bytes"] = int(torch.cuda.get_device_properties(0).total_memory)
    elif device.startswith("mps"):
        info["gpu"] = f"apple-{platform.machine()}"
        with_budget = getattr(torch.mps, "recommended_max_memory", None)
        if with_budget:
            info["gpu_total_bytes"] = int(with_budget())
    return info


def fingerprint_key(info: dict) -> str:
    """Short stable id for a hardware fingerprint — the scope within which timings
    are comparable."""
    seed = f"{info.get('device')}|{info.get('gpu')}|{info.get('platform')}|{info.get('torch')}"
    return uuid.uuid5(uuid.NAMESPACE_OID, seed).hex[:8]


# ── accelerator memory ────────────────────────────────────────────────────────

class MemorySampler:
    """Peak accelerator memory across a block of work, on whatever backend is here.

    CUDA keeps a true high-water mark (``max_memory_allocated``); MPS exposes only an
    instantaneous counter, so there we poll on a thread and keep the max. Those are
    genuinely different measurements, so the result says which one it is instead of
    presenting a sampled maximum as if it were exact.
    """

    def __init__(self, device: str, interval: float = 0.05) -> None:
        self.device = device
        self.interval = interval
        self.peak_bytes: int | None = None
        self.source: str | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._reader = self._pick_reader()

    def _pick_reader(self):
        try:
            import torch
        except ImportError:                                    # pragma: no cover
            return None
        # Devices arrive as "mps:0" / "cuda:0" from a live model, and as "mps" / "cuda"
        # from a config string — match the prefix, or the probe silently measures nothing.
        if self.device.startswith("cuda") and torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            self.source = "cuda.max_memory_allocated"
            return None                                        # exact counter, no polling
        if self.device.startswith("mps") and getattr(torch.mps, "current_allocated_memory", None):
            self.source = "mps.current_allocated_memory (sampled)"
            return torch.mps.current_allocated_memory
        return None

    def __enter__(self) -> "MemorySampler":
        if self._reader is not None:
            self._thread = threading.Thread(target=self._poll, daemon=True)
            self._thread.start()
        return self

    def _poll(self) -> None:
        while not self._stop.is_set():
            try:
                value = int(self._reader())
            except Exception:                                  # noqa: BLE001 - probe must never break the run
                return
            if self.peak_bytes is None or value > self.peak_bytes:
                self.peak_bytes = value
            self._stop.wait(self.interval)

    def __exit__(self, *exc) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        if self.source == "cuda.max_memory_allocated":
            import torch

            self.peak_bytes = int(torch.cuda.max_memory_allocated())


# ── latency ───────────────────────────────────────────────────────────────────

def latency_stats(samples: list[float]) -> dict:
    """p50/p95/p99 (+mean/min/max) in ms. Percentiles, because a mean hides the tail
    that decides whether a search box feels instant."""
    if not samples:
        return {}
    ordered = sorted(samples)

    def pct(q: float) -> float:
        if len(ordered) == 1:
            return ordered[0]
        pos = q * (len(ordered) - 1)
        low = int(pos)
        high = min(low + 1, len(ordered) - 1)
        return ordered[low] + (ordered[high] - ordered[low]) * (pos - low)

    return {
        "n": len(ordered),
        "mean_ms": round(statistics.fmean(ordered), 2),
        "p50_ms": round(pct(0.50), 2),
        "p95_ms": round(pct(0.95), 2),
        "p99_ms": round(pct(0.99), 2),
        "min_ms": round(ordered[0], 2),
        "max_ms": round(ordered[-1], 2),
    }


# ── exact (ANN-free) ranking over the stored vectors ──────────────────────────

def exact_rankings(
    doc_ids: list[str], doc_vectors: list[list[float]],
    query_ids: list[str], query_vectors: list[list[float]], top_k: int,
) -> dict[str, list[str]]:
    """Brute-force cosine ranking — the ceiling the ANN index is approximating.

    Uses the vectors READ BACK from Qdrant, not a re-embedding, so the only thing
    that differs from the ANN run is the search algorithm.
    """
    import numpy as np

    docs = np.asarray(doc_vectors, dtype="float32")
    queries = np.asarray(query_vectors, dtype="float32")
    docs /= np.linalg.norm(docs, axis=1, keepdims=True) + 1e-12
    queries /= np.linalg.norm(queries, axis=1, keepdims=True) + 1e-12
    sims = queries @ docs.T
    depth = min(top_k, len(doc_ids))
    out: dict[str, list[str]] = {}
    for row, query_id in enumerate(query_ids):
        idx = np.argpartition(-sims[row], depth - 1)[:depth]
        idx = idx[np.argsort(-sims[row][idx])]
        out[query_id] = [doc_ids[i] for i in idx]
    return out


# ── the benchmark ─────────────────────────────────────────────────────────────

@dataclass
class BenchReport:
    model: str
    model_profile: str
    embed_dim: int
    collection: str
    hardware: dict = field(default_factory=dict)
    indexing: dict = field(default_factory=dict)
    latency: dict = field(default_factory=dict)
    accuracy: dict = field(default_factory=dict)
    footprint: dict = field(default_factory=dict)

    def as_record(self, label: str, eval_dir: str, split: str, note: str = "") -> dict:
        record = {
            "id": uuid.uuid4().hex[:8],
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "label": label or self.model,
            "model": self.model,
            "model_profile": self.model_profile,
            "embed_dim": self.embed_dim,
            "collection": self.collection,
            "eval_dir": eval_dir,
            "split": split,
            "hardware": self.hardware,
            "hardware_key": fingerprint_key(self.hardware),
            "indexing": self.indexing,
            "latency": self.latency,
            "accuracy": self.accuracy,
            "footprint": self.footprint,
        }
        if note.strip():
            record["note"] = note.strip()
        return record


def _directory_bytes(path: str) -> int | None:
    root = Path(path)
    if not root.is_dir():
        return None                     # a hub model id, not a local checkpoint
    return sum(f.stat().st_size for f in root.rglob("*") if f.is_file())


async def run_bench(
    settings: Settings,
    embedder: Embedder,
    store: QdrantStore,
    eval_dir: str,
    *,
    split: str = "dev",
    top_k: int = DEFAULT_TOP_K,
    warmup: int = DEFAULT_WARMUP,
    device: str = "",
    max_seq_length: int | None = None,
    recreate: bool = True,
    progress=None,
) -> BenchReport:
    """Index the eval haystack into Qdrant, then measure serving it.

    ``recreate`` forces a real rebuild: ``serving.index_corpus`` is idempotent, and a
    skipped reindex would report an indexing cost of ~0.
    """
    from rag.modelprofile import resolve_profile

    resolved_split = resolve_split(eval_dir, split)
    corpus_file = str(Path(eval_dir) / "corpus.jsonl")
    queries = load_queries(eval_dir)
    qrels = load_qrels(eval_dir, resolved_split)
    judged = [q for q in queries if q in qrels]
    if not judged:
        raise ValueError(f"판정된 쿼리가 없습니다 — {eval_dir}/qrels/{resolved_split}.tsv 확인")

    profile = resolve_profile(settings.active_model, settings.model_profile)
    report = BenchReport(
        model=settings.active_model, model_profile=profile.name,
        embed_dim=settings.embed_dim, collection="",
        hardware=hardware_fingerprint(device or "cpu", max_seq_length),
    )

    # ── 색인 ──────────────────────────────────────────────────────────────────
    with MemorySampler(device) as sampler:
        started = time.perf_counter()
        result = await serving.index_corpus(
            settings, embedder, store, corpus_file, recreate=recreate, progress=progress
        )
        elapsed = time.perf_counter() - started
    report.collection = result["collection"]
    report.indexing = {
        "seconds": round(elapsed, 2),
        "docs": result["docs"],
        "docs_per_sec": round(result["docs"] / elapsed, 1) if elapsed else None,
        "peak_bytes": sampler.peak_bytes,
        "peak_source": sampler.source,
    }

    # ── 검색 (실제 서빙 경로, 쿼리 1건씩) ─────────────────────────────────────
    # Deterministic point ids let an ANN hit map back to its BEIR doc id exactly —
    # no payload-text matching, which would be fragile and is unnecessary here.
    corpus = load_corpus(eval_dir)
    id_by_point = {serving.point_id(doc_id): doc_id for doc_id in corpus}

    embed_ms: list[float] = []
    search_ms: list[float] = []
    ann: dict[str, list[str]] = {}
    query_vectors: dict[str, list[float]] = {}

    order = judged[:warmup] + judged            # warm-up pass, then the measured pass
    for position, query_id in enumerate(order):
        measured = position >= warmup
        t0 = time.perf_counter()
        vector = (await embedder.embed_queries([queries[query_id]]))[0]
        t1 = time.perf_counter()
        hits = store.query(report.collection, vector, top_k)
        t2 = time.perf_counter()
        if not measured:
            continue
        embed_ms.append((t1 - t0) * 1000)
        search_ms.append((t2 - t1) * 1000)
        query_vectors[query_id] = vector
        ann[query_id] = [
            id_by_point[str(h["id"])] for h in hits if str(h["id"]) in id_by_point
        ]

    report.latency = {
        "embed": latency_stats(embed_ms),
        "search": latency_stats(search_ms),
        "end_to_end": latency_stats([e + s for e, s in zip(embed_ms, search_ms)]),
        "warmup_discarded": warmup,
        "batch_size": 1,
        "top_k": top_k,
    }

    # ── 정확도: ANN 실측 vs 같은 벡터의 brute-force 천장 ──────────────────────
    recall_ks = tuple(sorted({1, 3, 5, 10, top_k}))
    ann_metrics = mean_metrics(per_query_metrics(ann, qrels, recall_ks))

    stored = store.scroll_vectors(report.collection)
    doc_ids = [id_by_point[pid] for pid, _ in stored if pid in id_by_point]
    doc_vectors = [vec for pid, vec in stored if pid in id_by_point]
    exact_metrics: dict[str, float] = {}
    if doc_vectors and query_vectors:
        exact = exact_rankings(
            doc_ids, doc_vectors, list(query_vectors), list(query_vectors.values()), top_k
        )
        exact_metrics = mean_metrics(per_query_metrics(exact, qrels, recall_ks))

    report.accuracy = {
        "split": resolved_split,
        "n_queries": len(ann),
        "ann": ann_metrics,
        "exact": exact_metrics,
        # exact − ann: what the index's approximation costs, separated from the model.
        "ann_loss": {
            k: round(exact_metrics[k] - ann_metrics[k], 6)
            for k in ann_metrics if k in exact_metrics
        },
    }

    # ── 자원 발자국 ───────────────────────────────────────────────────────────
    info = store.collection_info(report.collection) or {}
    points = info.get("points") or 0
    vector_bytes = points * settings.embed_dim * 4
    report.footprint = {
        "model_disk_bytes": _directory_bytes(settings.active_model),
        "points": points,
        "vector_bytes": vector_bytes,
        # The number that decides storage at production scale, not at rehearsal scale.
        "vector_bytes_per_1m_docs": settings.embed_dim * 4 * 1_000_000,
        "query_peak_bytes": None,
    }
    return report


def append_bench_run(record: dict, path: str = BENCH_RUNS_FILE) -> dict:
    """Append one bench record. Separate from runs/evals.jsonl on purpose: these rows
    are only comparable within a hardware fingerprint, and mixing them into the
    accuracy registry would invite averaging across machines."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
    return record


def load_bench_runs(path: str = BENCH_RUNS_FILE) -> list[dict]:
    file = Path(path)
    if not file.is_file():
        return []
    return [json.loads(line) for line in file.read_text(encoding="utf-8").splitlines() if line.strip()]
