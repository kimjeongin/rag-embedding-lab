"""Zero-shot public-model survey on the intranet dev set — report §3.9.

Runs each candidate through the SAME eval path as the in-house models (records into the
run registry so fingerprints match and Compare/leaderboard pick them up), with the
per-model input profile. Answers "is there an off-the-shelf model we could just use?"
— on the jargon slice, no. Fine-tuning is the only thing that moves it.

    uv run python scripts/model_survey.py [substring-filter]

Weights are pulled from the HF hub on first use; large ones may need a stable network
(this lab machine is download-constrained — run on the GPU server for the full set).
Profiles: plain = no prefix (bge family), nemotron3 = query:/passage: (e5 family).
"""
from __future__ import annotations

import asyncio
import os
import sys

os.environ["EMBEDDER"] = "sentence-transformers"
os.environ["EVAL_DIR"] = "data-intranet/eval"

from rag.evalflow import run_eval_flow

CANDIDATES = [
    ("intfloat/multilingual-e5-base", "nemotron3", "e5-base-base"),
    ("intfloat/multilingual-e5-large", "nemotron3", "e5-large-base"),
    ("BAAI/bge-m3", "plain", "bge-m3-base"),
    ("nlpai-lab/KURE-v1", "plain", "kure-v1-base"),
]


async def main() -> None:
    only = sys.argv[1] if len(sys.argv) > 1 else None
    for model, profile, label in CANDIDATES:
        if only and only not in model:
            continue
        os.environ["ST_MODEL"] = model
        os.environ["MODEL_PROFILE"] = profile
        try:
            r = await run_eval_flow(
                "sentence-transformers", model, label=label,
                eval_dir="data-intranet/eval", split="dev",
            )
            m = r["metrics"]
            sl = r["run"].get("slices", {})
            jargon = sl.get("jargon", {}).get("metrics", {}).get("ndcg@10", 0)
            print(
                f"[ok] {label:16} dim={r['embed_dim']:5}  nDCG@10={m['ndcg@10']:.4f}  "
                f"recall@50={m['recall@50']:.4f}  jargon nDCG={jargon:.4f}",
                flush=True,
            )
        except Exception as e:  # noqa: BLE001 — survey continues past a model that won't load
            print(f"[ERR] {label}: {type(e).__name__}: {str(e)[:120]}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
