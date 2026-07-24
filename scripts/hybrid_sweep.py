"""Hybrid (BM25 + dense) RRF fusion sweep on the intranet dev set — report §3.7.

Ranks the corpus with a dense model AND with BM25, fuses them with weighted RRF over
a grid of dense-weights α, and reports fused vs each component alone (per slice), with
a paired permutation test of the tuned winner against dense-alone.

    uv run python scripts/hybrid_sweep.py [MODEL] [PROFILE] [SPLIT]

Defaults: the fine-tuned intranet model, qwen3 profile, dev split. Pass the base model
to see that fusion cannot substitute for fine-tuning (base+hybrid stays near base).
"""
from __future__ import annotations

import asyncio
import os
import sys

os.environ.setdefault("EMBEDDER", "sentence-transformers")

from rag.config import Settings
from rag.diff import paired_permutation_test
from rag.embeddings import build_embedder
from rag.evaluation.beir import load_corpus, load_qrels, load_queries, load_query_slices
from rag.evaluation.bm25 import rank_eval_corpus
from rag.evaluation.hybrid import fuse_dense_lexical
from rag.evaluation.metrics import mean_metrics, per_query_metrics, slice_means
from rag.evaluation.retrieval import rank_corpus

ED = "data-intranet/eval"
DEPTH = 100
RECALL_KS = (1, 3, 5, 10, 50)


async def main() -> None:
    model = sys.argv[1] if len(sys.argv) > 1 else "outputs/embedding-intranet-mnrl-e6"
    profile = sys.argv[2] if len(sys.argv) > 2 else "qwen3"
    split = sys.argv[3] if len(sys.argv) > 3 else "dev"
    os.environ["ST_MODEL"] = model
    os.environ["MODEL_PROFILE"] = profile

    corpus = load_corpus(ED)
    queries = load_queries(ED)
    qrels = load_qrels(ED, split)
    slmap = load_query_slices(ED)
    judged = {q: queries[q] for q in queries if qrels.get(q)}

    async with build_embedder(Settings.from_env()) as emb:
        dense = await rank_corpus(emb, corpus, judged, top_n=DEPTH)
    lexical = rank_eval_corpus(corpus, judged, top_k=DEPTH)

    grid = [round(a / 10, 1) for a in range(11)]
    scored = {}
    for a in grid:
        fused = fuse_dense_lexical(dense, lexical, alpha=a, depth=DEPTH)
        pq = per_query_metrics(fused, qrels, RECALL_KS)
        scored[a] = (pq, mean_metrics(pq), slice_means(pq, slmap))
    best_a = max(grid, key=lambda a: scored[a][1]["ndcg@10"])

    def jargon(sl):
        return sl.get("jargon", {}).get("metrics", {})

    print(f"MODEL={model} PROFILE={profile} SPLIT={split} ({len(judged)} q, depth={DEPTH})\n")
    print("α sweep (nDCG@10 / jargon recall@5):")
    for a in grid:
        _, mean, sl = scored[a]
        mark = "  <- tuned best" if a == best_a else ""
        print(f"  α={a:.1f}  nDCG@10={mean['ndcg@10']:.4f}  jargon r@5={jargon(sl).get('recall@5', 0):.4f}{mark}")

    dense_pq, dense_mean, _ = scored[1.0]
    _, bm25_mean, _ = scored[0.0]
    best_pq, best_mean, best_sl = scored[best_a]
    print("\ncomponents vs fused:")
    print(f"  BM25 only (α=0)   nDCG@10={bm25_mean['ndcg@10']:.4f}")
    print(f"  dense only (α=1)  nDCG@10={dense_mean['ndcg@10']:.4f}")
    print(f"  fused (α={best_a})    nDCG@10={best_mean['ndcg@10']:.4f}  jargon nDCG={jargon(best_sl).get('ndcg@10', 0):.4f}")

    common = [q for q in best_pq if q in dense_pq]
    deltas = [best_pq[q]["ndcg@10"] - dense_pq[q]["ndcg@10"] for q in common]
    wins = sum(d > 0 for d in deltas)
    losses = sum(d < 0 for d in deltas)
    print(
        f"\nfused vs dense-alone (nDCG@10): Δ={sum(deltas) / len(deltas):+.4f}  "
        f"{wins}W {losses}L {len(deltas) - wins - losses}T  p={paired_permutation_test(deltas):.4f}"
    )


if __name__ == "__main__":
    asyncio.run(main())
