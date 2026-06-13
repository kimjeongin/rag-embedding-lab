"""`rag-gen-synthetic` — write an LLM-generated dataset (rag.datagen.synthetic).

Env: CORPUS_FILE, GEN_MODEL, N_QUERIES, HARD_NEGATIVES, ROUND_TRIP_K (0 = no
consistency filter), NEG_MARGIN (false-negative guard for mining), plus the
embedder settings (EMBEDDER/...) used for the filter/mining similarity pass and
TRAIN_FILE/TRAIN_EVAL_FILE outputs.
"""
from __future__ import annotations

import asyncio
import os

from rag.config import Settings
from rag.datagen.synthetic import generate
from rag.dataset import dataset_paths, write_jsonl


def main() -> None:
    settings = Settings.from_env()
    train_file, eval_file = dataset_paths()
    corpus_file = os.getenv("CORPUS_FILE", "data/corpus.jsonl")
    gen_model = os.getenv("GEN_MODEL", "qwen3.5:2b")
    n_queries = int(os.getenv("N_QUERIES", "3"))
    hard_negatives = int(os.getenv("HARD_NEGATIVES", "1"))
    round_trip_k = int(os.getenv("ROUND_TRIP_K", "1"))
    neg_margin = float(os.getenv("NEG_MARGIN", "0.05"))

    print(f"[synthetic] corpus={corpus_file} gen_model={gen_model} "
          f"n_queries={n_queries} hard_negatives={hard_negatives} "
          f"round_trip_k={round_trip_k} neg_margin={neg_margin} embedder={settings.embedder}")
    train, test = asyncio.run(
        generate(corpus_file, gen_model, n_queries, hard_negatives, settings,
                 round_trip_k=round_trip_k, neg_margin=neg_margin)
    )
    write_jsonl(train_file, train)
    write_jsonl(eval_file, test)
    print(f"[synthetic] wrote {train_file} ({len(train)}) and {eval_file} ({len(test)})")
