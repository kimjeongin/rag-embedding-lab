"""The innermost layer: entities, the embedder port, and the embedding-input rules.

Nothing here imports FastAPI, httpx, or any ML stack — and it does not read
configuration. Everything else depends inward on this package:

  - ``entities``   — typed values that cross boundaries (``Document``).
  - ``ports``      — the abstraction the lab depends on (``Embedder``).
  - ``errors``     — domain exceptions.
  - ``formatting`` — the asymmetric embedding-input rules (shared by serving + training).
"""
