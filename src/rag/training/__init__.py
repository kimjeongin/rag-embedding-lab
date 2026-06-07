"""Embedding fine-tuning (the training step itself).

  - ``config`` — TrainingConfig (model, hyperparameters, dataset paths), env-driven.
  - ``data``   — JSONL → trainer inputs, formatted via rag.core.formatting (parity).
  - ``model``  — device pick + load the base model as a SentenceTransformer.
  - ``train``  — the contrastive fine-tuning loop.

Data generation lives in ``rag.datagen`` and evaluation in ``rag.evaluation``; the
console entrypoints are in ``rag.cli``. Heavy ML deps (torch, sentence-transformers)
are in the optional ``training`` group (``uv sync --group training``) and imported
lazily, so the serving side never loads them.
"""
