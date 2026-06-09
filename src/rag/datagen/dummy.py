"""A small, coherent toy dataset for smoke-testing the training pipeline.

Built from the shared ``topics`` set (so the doc text isn't duplicated) using the
``train_queries`` side only — the eval set draws its (disjoint) ``eval_queries``, so a
toy fine-tune is never measured on the strings it trained on. Deterministic (seeded);
stdlib only. The CLI entrypoint (``rag-gen-data``) writes the result.
"""
from __future__ import annotations

import random

from rag.datagen.topics import TOPICS


def generate_dataset(test_fraction: float = 0.25, seed: int = 13):
    """Return (train, test) lists of {query, positive} records, deterministically split."""
    pairs = [
        {"query": q, "positive": {"title": t.title, "content": t.content}}
        for t in TOPICS
        for q in t.train_queries
    ]
    random.Random(seed).shuffle(pairs)
    n_test = max(1, round(len(pairs) * test_fraction))
    return pairs[n_test:], pairs[:n_test]
