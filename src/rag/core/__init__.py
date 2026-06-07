"""The innermost layer: entities, ports, and pure rules.

Nothing here imports FastAPI, qdrant_client, httpx, or any ML stack — and it does not
read configuration. Everything else depends inward on this package:

  - ``entities``   — typed values that cross boundaries (Document, Hit, Site, ...).
  - ``ports``      — abstractions the use cases depend on (Embedder, VectorStore).
  - ``errors``     — domain exceptions.
  - ``formatting`` — the asymmetric embedding-input rules (shared with training).
  - ``ranking``    — pure post-processing (diversity cap, site grouping/scoring).
  - ``urls``       — derive url/domain/path metadata from a page URL.
"""
