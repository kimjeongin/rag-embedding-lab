"""HTTP layer (FastAPI). The ``rag-serve`` launcher lives in ``rag.cli.serve``.

Two surfaces from one app/port:
  - **serving** — ``/health``, ``/documents``, ``/search`` (needs Qdrant + an embedder).
  - **lab** — ``/api/*``: generate data → train → evaluate → compare (needs neither;
    training streams over SSE). The React UI drives this; ``rag-serve`` also serves the
    built ``frontend/dist`` so API + UI share one origin.

Qdrant is optional: if it's down the app still starts, the lab API works, and the
serving routes return 503 (see ``app``).

  - ``app``     — application factory + ASGI ``app`` (uvicorn target ``rag.api.app:app``).
  - ``deps``    — request-scoped dependencies (use cases, optional store, settings).
  - ``errors``  — map domain errors → HTTP status codes.
  - ``schemas`` — pydantic request/response models, split per resource (+ ``lab``).
  - ``routes``  — one router per resource; the lab routers live under ``routes/lab/``.
"""
