"""Qdrant dense-retrieval RAG.

Two concerns live under this package, kept separate by design:

  - ``rag.api``       — the retrieval *serving* app (FastAPI).
  - ``rag.training``  — embedding *training* (fine-tuning) code.

They share ``rag.core`` (notably the embedding input formatting) so that what the
model is trained on matches what we send at query time (train/inference parity).

This top-level package is intentionally import-light: importing ``rag`` (or
``rag.core``) must not pull in the web framework or any training stack.
"""

__version__ = "0.2.0"
