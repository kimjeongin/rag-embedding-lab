"""Offline data generation — produces fine-tuning datasets (rag.dataset format).

  - ``topics``      — the shared 16-topic set (train vs eval queries split, no leakage).
  - ``dummy``       — a seeded toy split from ``topics``, stdlib only (no model needed).
  - ``synthetic``   — an LLM writes queries from a corpus + hard-negative mining.
  - ``eval_corpus`` — a sample BEIR eval set from ``topics`` + many distractors.
"""
