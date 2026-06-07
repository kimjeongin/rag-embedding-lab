"""`rag-gen-eval` — write a SAMPLE BEIR-format eval set to EVAL_DIR (data/eval).

Sample data only: real gold docs + many tech-register distractors so the corpus is a
real "haystack". Replace it with your in-house data in the same layout for a real
measurement — see docs/evaluation.md.

Env: EVAL_DIR (output dir, default data/eval), N_DISTRACTORS (cap the haystack size).
"""
from __future__ import annotations

import os

from rag.datagen.eval_corpus import generate
from rag.evaluation.beir import eval_dir_from_env, write_beir_dataset


def main() -> None:
    eval_dir = eval_dir_from_env()
    raw = os.getenv("N_DISTRACTORS")
    n_distractors = int(raw) if raw else None

    corpus, queries, qrels = generate(n_distractors=n_distractors)
    write_beir_dataset(eval_dir, corpus, queries, qrels)
    print(
        f"[gen-eval] wrote {eval_dir}: "
        f"{len(corpus)} corpus docs, {len(queries)} queries, {len(qrels)} qrels"
    )
