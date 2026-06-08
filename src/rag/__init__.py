"""RAG embedding lab — fine-tune an embedding model and measure retrieval quality.

Concerns under this package:

  - ``rag.datagen`` / ``rag.training`` / ``rag.evaluation`` — the offline pipeline
    (generate data → train → evaluate).
  - ``rag.embeddings`` — the ``Embedder`` adapters (Ollama / sentence-transformers).
  - ``rag.api``        — the lab HTTP API (FastAPI) that the React UI drives.

They share ``rag.core`` (notably the embedding-input formatting) so that what the model
is trained on matches what we send at query time (train/inference parity).

Importing ``rag`` (or ``rag.core``) is intentionally light: it must not pull in the web
framework or any training stack.
"""

__version__ = "0.3.0"
