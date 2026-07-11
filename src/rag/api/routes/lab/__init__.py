"""Lab API — the ``/api/*`` surface the React front-end drives.

One router per screen-concern (status / models / data / runs / evaluate / jobs /
search), combined under the ``/api`` prefix and mounted by the app factory. The lab
routes run the offline pipeline (read/write files + in-memory eval — no vector store
involved); ``search`` is the serving path and is the one surface that talks to Qdrant.
Training runs as server-owned background jobs (see ``jobs``) — the browser polls, so
closing the tab never kills a run.
"""
from __future__ import annotations

from fastapi import APIRouter

from rag.api.routes.lab import data, evaluate, jobs, models, runs, search, status

router = APIRouter(prefix="/api")
router.include_router(status.router)
router.include_router(models.router)
router.include_router(data.router)
router.include_router(runs.router)
router.include_router(evaluate.router)
router.include_router(jobs.router)
router.include_router(search.router)
