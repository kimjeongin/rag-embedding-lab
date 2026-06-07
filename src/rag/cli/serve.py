"""`rag-serve` — launch the HTTP API server.

Separated from the app definition (rag.api.app) so "how to launch the process"
(host/port/reload) is distinct from "what the app is". For dev with autoreload you
can instead run uvicorn directly: ``uv run uvicorn rag.api.app:app --reload``.
"""
from __future__ import annotations

import os

import uvicorn


def run() -> None:
    """Launch uvicorn, reading host/port/reload from the environment."""
    uvicorn.run(
        "rag.api.app:app",
        host=os.getenv("RAG_HOST", "127.0.0.1"),
        port=int(os.getenv("RAG_PORT", "8000")),
        reload=os.getenv("RAG_RELOAD", "").lower() in {"1", "true", "yes"},
    )
