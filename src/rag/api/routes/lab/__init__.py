"""Lab API — the ``/api/*`` surface the React front-end drives.

One router per screen-concern (status / models / data / runs / evaluate / train),
combined under the ``/api`` prefix and mounted by the app factory. These routes need no
vector store: they read/write files and run the offline pipeline, so they work even when
Qdrant is down (the serving routes — /documents, /search — 503 in that case). Training
streams its progress over Server-Sent Events (see ``train``).
"""
from __future__ import annotations

from fastapi import APIRouter

from rag.api.routes.lab import data, evaluate, models, runs, status, train

router = APIRouter(prefix="/api")
router.include_router(status.router)
router.include_router(models.router)
router.include_router(data.router)
router.include_router(runs.router)
router.include_router(evaluate.router)
router.include_router(train.router)
