"""``rag-bench`` — measure what serving one model actually costs.

    uv run rag-bench --model outputs/embedding-intranet-mnrl-e6 --label qwen-ft
    uv run rag-bench --model outputs/nemotron-intranet-mnrl-e6  --label nemo-ft
    uv run rag-bench --list                      # 기록된 벤치 (하드웨어별로 묶어서)

Indexes the eval set's corpus into a versioned Qdrant collection, then serves every
judged query one at a time and records latency percentiles, ANN-vs-exact accuracy,
and resource footprint. Needs a running Qdrant (``make qdrant``).

Accuracy transfers between machines; **latency and memory do not** — each record
carries a hardware fingerprint and ``--list`` groups by it rather than averaging
across it. Run this on the hardware you will actually serve from.
"""
from __future__ import annotations

import argparse
import asyncio

from rag import lab
from rag.config import Settings
from rag.embeddings.factory import build_embedder
from rag.evaluation.beir import eval_dir_from_env
from rag.servingbench import (
    DEFAULT_TOP_K,
    DEFAULT_WARMUP,
    append_bench_run,
    load_bench_runs,
    run_bench,
)
from rag.vectorstore.qdrant import QdrantStore


def _mb(value: int | None) -> str:
    return "—" if not value else f"{value / 1e6:,.0f} MB"


def _print_report(report, label: str) -> None:
    print(f"\n=== {label} ===")
    print(f"model    : {report.model}")
    print(f"profile  : {report.model_profile}   dim: {report.embed_dim}")
    print(f"hardware : {report.hardware.get('device')} · {report.hardware.get('gpu', '—')} "
          f"· torch {report.hardware.get('torch')} · max_seq={report.hardware.get('max_seq_length')}")
    print(f"collection: {report.collection}")

    idx = report.indexing
    print(f"\n[색인] {idx['docs']}문서 {idx['seconds']}s "
          f"({idx['docs_per_sec']} docs/s)  peak={_mb(idx['peak_bytes'])} "
          f"[{idx['peak_source'] or 'n/a'}]")

    lat = report.latency
    print(f"\n[지연시간] 쿼리 1건씩, 워밍업 {lat['warmup_discarded']}건 제외, top_k={lat['top_k']}")
    print(f"{'':10s} {'p50':>8s} {'p95':>8s} {'p99':>8s} {'mean':>8s}")
    for phase in ("embed", "search", "end_to_end"):
        s = lat.get(phase) or {}
        if s:
            print(f"{phase:10s} {s['p50_ms']:8.1f} {s['p95_ms']:8.1f} "
                  f"{s['p99_ms']:8.1f} {s['mean_ms']:8.1f}")

    acc = report.accuracy
    print(f"\n[정확도] {acc['n_queries']}쿼리 · split={acc['split']}")
    print(f"{'metric':12s} {'ANN(실서빙)':>12s} {'exact(천장)':>12s} {'ANN 손실':>10s}")
    for key in sorted(acc["ann"]):
        ann = acc["ann"][key]
        exact = acc["exact"].get(key)
        loss = acc["ann_loss"].get(key)
        print(f"{key:12s} {ann:12.4f} "
              f"{'—' if exact is None else f'{exact:12.4f}'} "
              f"{'—' if loss is None else f'{loss:10.4f}'}")

    fp = report.footprint
    print(f"\n[자원] 모델 디스크={_mb(fp['model_disk_bytes'])} · "
          f"벡터 {fp['points']}건={_mb(fp['vector_bytes'])} · "
          f"1M 문서 환산={fp['vector_bytes_per_1m_docs'] / 1e9:.1f} GB")


def _print_list() -> None:
    runs = load_bench_runs()
    if not runs:
        print("기록된 벤치가 없습니다 — 먼저 `uv run rag-bench --model ...`을 실행하세요")
        return
    groups: dict[str, list[dict]] = {}
    for r in runs:
        groups.setdefault(r.get("hardware_key", "?"), []).append(r)
    for key, rows in groups.items():
        hw = rows[0].get("hardware", {})
        print(f"\n── hardware {key}: {hw.get('device')} · {hw.get('gpu', '—')} "
              f"· torch {hw.get('torch')} ──")
        print(f"{'label':22s} {'dim':>5s} {'ndcg@10':>8s} {'e2e p50':>9s} {'e2e p95':>9s} "
              f"{'peak':>10s} {'idx docs/s':>11s}")
        for r in rows:
            e2e = (r.get("latency") or {}).get("end_to_end") or {}
            acc = (r.get("accuracy") or {}).get("ann") or {}
            idx = r.get("indexing") or {}
            peak = idx.get("peak_bytes")
            print(f"{r['label'][:22]:22s} {r.get('embed_dim', '—'):>5} "
                  f"{acc.get('ndcg@10', float('nan')):8.4f} "
                  f"{e2e.get('p50_ms', float('nan')):9.1f} {e2e.get('p95_ms', float('nan')):9.1f} "
                  f"{'—' if not peak else f'{peak / 1e6:,.0f}MB':>10s} "
                  f"{idx.get('docs_per_sec', '—'):>11}")
    print("\n하드웨어 그룹이 다르면 지연시간·메모리는 비교하지 마세요 (정확도만 이전됩니다).")


async def _run(args) -> None:
    eval_dir = args.eval_dir or eval_dir_from_env()
    base = Settings.from_env()
    dim = lab.infer_dim("sentence-transformers", args.model, base.ollama_url, args.truncate_dim)
    settings = lab.build_eval_settings(
        "sentence-transformers", args.model, dim, base.ollama_url, args.truncate_dim
    )

    device, max_seq = "", None
    async with build_embedder(settings) as embedder:
        model = getattr(embedder, "_model", None)          # ST backend exposes the model
        if model is not None:
            device = str(getattr(model, "device", "") or "")
            max_seq = getattr(model, "max_seq_length", None)
        with QdrantStore(settings.qdrant_url) as store:
            if not store.ping():
                raise SystemExit(
                    f"Qdrant({settings.qdrant_url})에 연결할 수 없습니다 — `make qdrant`로 띄우세요"
                )
            report = await run_bench(
                settings, embedder, store, eval_dir,
                split=args.split, top_k=args.top_k, warmup=args.warmup,
                device=device, max_seq_length=max_seq, recreate=not args.reuse_index,
                progress=lambda done, total: print(f"\r  색인 {done}/{total}", end="", flush=True),
            )

    label = args.label or args.model
    if args.truncate_dim and "@" not in label:
        label = f"{label}@{args.truncate_dim}"
    _print_report(report, label)
    record = append_bench_run(report.as_record(label, eval_dir, args.split, args.note))
    print(f"\n기록됨: runs/bench.jsonl ({record['id']}, hardware {record['hardware_key']})")


def main() -> None:
    parser = argparse.ArgumentParser(description="Qdrant 서빙 경로 성능·자원 벤치마크")
    parser.add_argument("--model", help="모델 경로 또는 HF 이름")
    parser.add_argument("--label", default="", help="기록에 남길 이름")
    parser.add_argument("--eval-dir", default="", help="평가셋 디렉터리 (기본: EVAL_DIR)")
    parser.add_argument("--split", default="dev")
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--warmup", type=int, default=DEFAULT_WARMUP,
                        help="측정에서 제외할 초기 쿼리 수")
    parser.add_argument("--truncate-dim", type=int, default=None,
                        help="Matryoshka 절단 차원 (저장비용 맞춘 비교용)")
    parser.add_argument("--reuse-index", action="store_true",
                        help="기존 컬렉션 재사용 (색인 비용은 측정되지 않음)")
    parser.add_argument("--note", default="")
    parser.add_argument("--list", action="store_true", help="기록된 벤치 보기")
    args = parser.parse_args()

    if args.list:
        _print_list()
        return
    if not args.model:
        parser.error("--model 이 필요합니다 (또는 --list)")
    asyncio.run(_run(args))
