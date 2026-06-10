"""POST /api/eval — measure one model and record the run.

Auto-detects the embedding dimension from the model (no manual field to get wrong),
ranks the eval corpus for every query, computes recall@k / MRR@10 / nDCG@10 with a
bootstrap 95% CI, appends the result (plus the eval set's content fingerprint and the
per-query scores) to the registry, and returns the metrics alongside the prior best
on the SAME eval set (so the UI can render an honest Δ). Upstream embedding failures
surface as 502; an eval set with no judged queries is a 422.
"""
from __future__ import annotations

from fastapi import APIRouter, HTTPException

from rag import lab
from rag import runs as registry
from rag.api.schemas.lab import EvalRequest, EvalResponse, RunRecord
from rag.config import Settings
from rag.evaluation.beir import eval_dir_from_env, eval_set_fingerprint
from rag.evaluation.retrieval import evaluate

router = APIRouter()


@router.post("/eval", response_model=EvalResponse)
async def run_eval(req: EvalRequest) -> EvalResponse:
    eval_dir = (req.eval_dir or "").strip() or eval_dir_from_env()
    ollama_url = req.ollama_url or Settings.from_env().ollama_url
    try:
        dim = lab.infer_dim(req.embedder, req.model, ollama_url)
        settings = lab.build_eval_settings(req.embedder, req.model, dim, ollama_url)
        # Δ is only meaningful against runs on the SAME eval-set contents (a different
        # haystack isn't comparable, and regenerating reuses the dir path), so scope
        # the prior best to the set's content fingerprint.
        fingerprint = eval_set_fingerprint(eval_dir)
        prior_best = registry.best_per_metric(fingerprint=fingerprint)
        report = await evaluate(settings, eval_dir)
    except Exception as exc:  # noqa: BLE001 — surface model/embedding failures as 502
        raise HTTPException(status_code=502, detail=f"{type(exc).__name__}: {exc}") from exc

    if not report.metrics:
        raise HTTPException(status_code=422, detail="판정된 쿼리가 없습니다 — qrels/<split>.tsv를 확인하세요")

    record = registry.append_run(
        req.label, req.embedder, settings.active_model, eval_dir, report.metrics,
        eval_fingerprint=fingerprint,
        n_queries=len(report.per_query),
        ci95=report.ci95,
        per_query=report.per_query,
    )
    return EvalResponse(
        model=settings.active_model,
        embed_dim=dim,
        metrics=report.metrics,
        n_queries=len(report.per_query),
        ci95={k: list(v) for k, v in report.ci95.items()},
        run=RunRecord(**record),
        prior_best=prior_best,
    )
