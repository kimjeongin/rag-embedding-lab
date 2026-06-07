"""`rag-gen-synthetic` — write an LLM-generated dataset (rag.datagen.synthetic).

Env: CORPUS_FILE, GEN_MODEL, N_QUERIES, HARD_NEGATIVES, plus the embedder settings
(EMBEDDER/...) used for hard-negative mining and TRAIN_FILE/TRAIN_EVAL_FILE outputs.
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
    gen_model = os.getenv("GEN_MODEL", "qwen3:4b")
    n_queries = int(os.getenv("N_QUERIES", "3"))
    hard_negatives = int(os.getenv("HARD_NEGATIVES", "1"))

    print(f"[synthetic] corpus={corpus_file} gen_model={gen_model} "
          f"n_queries={n_queries} hard_negatives={hard_negatives} embedder={settings.embedder}")
    train, test = asyncio.run(
        generate(corpus_file, gen_model, n_queries, hard_negatives, settings)
    )
    write_jsonl(train_file, train)
    write_jsonl(eval_file, test)
    print(f"[synthetic] wrote {train_file} ({len(train)}) and {eval_file} ({len(test)})")
