"""HTTP serving layer (FastAPI). The ``rag-serve`` launcher lives in ``rag.cli.serve``.

  - ``app``     — application factory + ASGI ``app`` (uvicorn target ``rag.api.app:app``).
  - ``deps``    — request-scoped dependencies (use cases, store, settings).
  - ``errors``  — map domain errors → HTTP status codes.
  - ``schemas`` — pydantic request/response models, split per resource.
  - ``routes``  — one router per resource (health, documents, search).
"""
