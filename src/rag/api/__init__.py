"""HTTP layer (FastAPI) for the embedding lab. The ``rag-serve`` launcher lives in
``rag.cli.serve``.

Exposes the **lab API** — ``/api/*``: generate data → train → evaluate → compare
(training streams over SSE) — and serves the built ``frontend/dist`` at ``/`` so the
React UI and the API share one origin. No vector store: evaluation ranks in-memory.

  - ``app``     — application factory + ASGI ``app`` (uvicorn target ``rag.api.app:app``).
  - ``deps``    — request-scoped dependencies (settings).
  - ``errors``  — map domain errors → HTTP status codes.
  - ``schemas`` — pydantic request/response models (``schemas.lab``).
  - ``routes``  — the lab routers live under ``routes/lab/``.
"""
