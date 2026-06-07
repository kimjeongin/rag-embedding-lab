"""Offline data generation — produces fine-tuning datasets (rag.dataset format).

  - ``dummy``     — a seeded toy split, stdlib only (no model needed).
  - ``synthetic`` — an LLM writes queries from a corpus + hard-negative mining.
"""
