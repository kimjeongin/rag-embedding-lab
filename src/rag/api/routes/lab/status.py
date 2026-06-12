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
from rag.evaluation.beir import available_splits, eval_dir_from_env, eval_set_fingerprint, resolve_split

router = APIRouter()


@router.get("/status", response_model=StatusResponse)
def status(settings: Settings = Depends(get_settings)) -> StatusResponse:
    reachable, models = lab.ollama_status(settings.ollama_url)
    eval_dir = eval_dir_from_env()
    fingerprint = eval_set_fingerprint(eval_dir, resolve_split(eval_dir))
    best = registry.best_per_metric(fingerprint=fingerprint)
    return StatusResponse(
        ollama=OllamaStatus(reachable=reachable, models=models),
        device=lab.device_status(),
        settings=EmbedInfo(
            embedder=settings.embedder,
            model=settings.active_model,
            embed_dim=settings.embed_dim,
        ),
        eval=EvalInfo(
            dir=eval_dir,
            is_sample=lab.is_sample_eval(eval_dir),
            corpus=lab.count_lines(f"{eval_dir}/corpus.jsonl"),
            queries=lab.count_lines(f"{eval_dir}/queries.jsonl"),
            fingerprint=fingerprint,
            splits=available_splits(eval_dir),
        ),
        training_ready=lab.training_ready(),
        runs=len(registry.load_runs()),
        best_ndcg=best.get("ndcg@10"),
        active_job=jobs.active_job_id(),
        handed_off=modelstore.handed_off_model(),
    )
