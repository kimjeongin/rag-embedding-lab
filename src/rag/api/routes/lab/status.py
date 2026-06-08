"""GET /api/status — one call that powers the front-end's header + banners.

Is Ollama up (and which models)? What device would training use? Which eval set is
bound, is it the bundled sample, how big is it? Is the training stack installed? How
many runs have been recorded? All read-only and cheap, so it runs in a threadpool
(sync ``def``) to keep the event loop free of the blocking Ollama probe + file IO.
"""
from __future__ import annotations

from fastapi import APIRouter, Depends

from rag import lab
from rag import runs as registry
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
    eval_dir = eval_dir_from_env()
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
        ),
        training_ready=lab.training_ready(),
        runs=len(registry.load_runs()),
    )
