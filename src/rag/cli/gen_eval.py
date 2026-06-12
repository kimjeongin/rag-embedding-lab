"""`rag-gen-eval` — write a SAMPLE BEIR-format eval set to EVAL_DIR (data/eval).

Sample data only: real gold docs + many tech-register distractors so the corpus is a
real "haystack". Replace it with your in-house data in the same layout for a real
measurement — see docs/evaluation.md.

Env: EVAL_DIR (output dir, default data/eval), N_DISTRACTORS (cap the haystack size).
"""
from __future__ import annotations

import os

from rag.datagen.eval_corpus import generate, split_qrels
from rag.evaluation.beir import DEV_SPLIT, FINAL_SPLIT, eval_dir_from_env, write_beir_dataset, write_qrels


def main() -> None:
    eval_dir = eval_dir_from_env()
    raw = os.getenv("N_DISTRACTORS")
    n_distractors = int(raw) if raw else None

    corpus, queries, qrels = generate(n_distractors=n_distractors)
    # dev = tuning split (sweeps/comparisons), final = held-out one-shot confirmation
    # for the chosen winner — selection on one set + confirmation on another is what
    # keeps "best of 20 runs" from overfitting the eval set itself.
    dev_rows, final_rows = split_qrels(qrels)
    write_beir_dataset(eval_dir, corpus, queries, dev_rows, split=DEV_SPLIT)
    write_qrels(eval_dir, final_rows, FINAL_SPLIT)
    print(
        f"[gen-eval] wrote {eval_dir}: {len(corpus)} corpus docs, {len(queries)} queries, "
        f"qrels dev={len(dev_rows)} final={len(final_rows)}"
    )
