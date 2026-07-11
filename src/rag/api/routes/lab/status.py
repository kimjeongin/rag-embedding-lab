"""GET /api/status — one call that powers the front-end's header + banners.

Is Ollama up (and which models)? What device would training use? Which eval set is
bound, is it the bundled sample, how big is it? Is the training stack installed? How
many runs have been recorded? All read-only and cheap, so it runs in a threadpool
(sync ``def``) to keep the event loop free of the blocking Ollama probe + file IO.
"""
from __future__ import annotations

from fastapi import APIRouter, Depends

from rag import lab, modelstore
from rag import runs as registry
from rag.api import jobs
from rag.api.deps import get_settings
from rag.api.schemas.lab import (
    EmbedInfo,
    EvalInfo,
    OllamaStatus,
    StatusResponse,
)
from rag.config import Settings
from rag.evaluation.beir import eval_dir_from_env

router = APIRouter()


@router.get("/status", response_model=StatusResponse)
def status(settings: Settings = Depends(get_settings)) -> StatusResponse:
    reachable, models = lab.ollama_status(settings.ollama_url)
    overview = lab.eval_overview(eval_dir_from_env())
    best = registry.best_per_metric(fingerprint=overview["fingerprint"])
    return StatusResponse(
        ollama=OllamaStatus(reachable=reachable, models=models),
        device=lab.device_status(),
        settings=EmbedInfo(
            embedder=settings.embedder,
            model=settings.active_model,
            embed_dim=settings.embed_dim,
        ),
        eval=EvalInfo(**overview),
        training_ready=lab.training_ready(),
        runs=len(registry.load_runs()),
        best_ndcg=best.get("ndcg@10"),
        active_job=jobs.active_job_id(),
        handed_off=modelstore.handed_off_model(),
    )
