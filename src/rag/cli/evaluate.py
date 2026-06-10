"""`rag-eval` — measure retrieval quality of the configured embedder on a BEIR-format
eval set (rag.evaluation.retrieval).

Env: EVAL_DIR (the eval dataset dir) plus the embedder settings (EMBEDDER / ST_MODEL /
EMBED_MODEL / ...). Compare models by re-running with EMBEDDER changed — same corpus,
same metrics. See docs/evaluation.md.
"""
from __future__ import annotations

import asyncio

from rag.config import Settings
from rag.evaluation.beir import eval_dir_from_env
from rag.evaluation.retrieval import evaluate


def main() -> None:
    settings = Settings.from_env()
    eval_dir = eval_dir_from_env()

    report = asyncio.run(evaluate(settings, eval_dir))
    print(f"[eval] embedder={settings.embedder} model={settings.active_model} dir={eval_dir}")
    if not report.metrics:
        print("  (no judged queries found — check qrels/test.tsv)")
        return
    print(f"  n = {len(report.per_query)} judged queries")
    for key, value in report.metrics.items():
        lo, hi = report.ci95[key]
        print(f"  {key} = {value:.4f}   (95% CI {lo:.4f}–{hi:.4f})")
