"""One evaluation, end to end — shared by the HTTP route and the job runner.

Resolve the model → rank the eval set → score → append to the registry → return the
result with the prior best for an honest Δ. Framework-free (no fastapi): the route
translates the exceptions here into HTTP codes, and the background job runner calls
it directly after each training run (auto-eval).
"""
from __future__ import annotations

from rag import lab
from rag import runs as registry
from rag.config import Settings
from rag.evaluation.beir import (
    eval_dir_from_env,
    eval_set_fingerprint,
    load_query_slices,
    resolve_split,
)
from rag.evaluation.metrics import slice_means
from rag.evaluation.retrieval import evaluate
from rag.modelprofile import resolve_profile


class NoJudgedQueries(ValueError):
    """The eval set produced zero scorable queries (empty/missing qrels)."""


async def run_eval_flow(
    embedder: str,
    model: str,
    *,
    label: str = "",
    eval_dir: str | None = None,
    split: str = "dev",
    ollama_url: str | None = None,
    note: str | None = None,
    truncate_dim: int | None = None,
) -> dict:
    """Evaluate one model on one split and record the run.

    Returns {model, embed_dim, metrics, n_queries, ci95, run, prior_best, split}.
    Raises NoJudgedQueries when nothing is scorable; embedding/model failures
    propagate as-is (the caller decides how to surface them). ``truncate_dim``
    measures the model's Matryoshka prefix (e.g. 256-d) — recorded as a distinct
    run ("…@256") so the dim→quality curve shows up in Compare.
    """
    eval_dir = (eval_dir or "").strip() or eval_dir_from_env()
    ollama_url = ollama_url or Settings.from_env().ollama_url

    dim = lab.infer_dim(embedder, model, ollama_url, truncate_dim)
    settings = lab.build_eval_settings(embedder, model, dim, ollama_url, truncate_dim)
    if truncate_dim and label and "@" not in label:
        label = f"{label}@{truncate_dim}"

    # Δ is only meaningful against runs on the SAME eval-set contents AND split (a
    # different haystack — or the held-out final split — isn't comparable), so scope
    # the prior best to the split's content fingerprint.
    resolved = resolve_split(eval_dir, split)
    fingerprint = eval_set_fingerprint(eval_dir, resolved)
    prior_best = registry.best_per_metric(fingerprint=fingerprint)

    report = await evaluate(settings, eval_dir, split=split)
    if not report.metrics:
        raise NoJudgedQueries("판정된 쿼리가 없습니다 — qrels/<split>.tsv를 확인하세요")

    # 슬라이스별 평균 — 평가셋이 쿼리에 slice 태그를 달아둔 경우(예: standard/jargon)
    # 전체 평균이 가리는 슬라이스 붕괴를 런 기록 자체에 남긴다.
    slices = slice_means(report.per_query, load_query_slices(eval_dir))

    record = registry.append_run(
        label, embedder, settings.active_model, eval_dir, report.metrics,
        eval_fingerprint=fingerprint,
        n_queries=len(report.per_query),
        ci95=report.ci95,
        per_query=report.per_query,
        rankings=report.rankings,
        split=report.split,
        note=note,
        slices=slices,
        model_profile=resolve_profile(settings.active_model, settings.model_profile).name,
        embed_dim=dim,
    )
    return {
        "model": settings.active_model,
        "embed_dim": dim,
        "metrics": report.metrics,
        "n_queries": len(report.per_query),
        "ci95": {k: list(v) for k, v in report.ci95.items()},
        "run": record,
        "prior_best": prior_best,
        "split": report.split,
    }
