"""/api/runs — the eval-run registry behind the Overview leaderboard + Compare table.

List is newest-first and carries the best-per-metric map (so the UI can highlight the
winner and show Δ-vs-best) plus the metric display order. ``best`` and the returned
``current_fingerprint`` are scoped to the eval set currently bound to the process —
runs measured on other (or regenerated) sets aren't comparable, and the UI uses the
fingerprint to keep them out of rankings. ``/runs/diff`` is the paired run-vs-run
comparison (win/loss + permutation p-value + what was retrieved). Delete removes one
run by id. Thin wrappers over ``rag.runs`` / ``rag.diff`` (stdlib stores).
"""
from __future__ import annotations

from fastapi import APIRouter, HTTPException

from rag import runs as registry
from rag.api.schemas.lab import (
    Bm25Request,
    DeleteRunResponse,
    DiffResponse,
    ImportTrecRequest,
    ImportTrecResponse,
    RunRecord,
    RunsResponse,
)
from rag.diff import DEFAULT_METRIC, candidate_union, compare_runs
from rag.evaluation.beir import (
    FINAL_SPLIT,
    eval_dir_from_env,
    eval_set_fingerprint,
    load_corpus,
    load_qrels,
    load_queries,
    load_query_slices,
    resolve_split,
)
from rag.evaluation.bm25 import rank_eval_corpus
from rag.evaluation.metrics import bootstrap_ci, mean_metrics, per_query_metrics, slice_means
from rag.evaluation.retrieval import eval_top_k, metric_recall_ks
from rag.evaluation.trec import parse_trec_run

router = APIRouter()


@router.get("/runs", response_model=RunsResponse)
def list_runs() -> RunsResponse:
    eval_dir = eval_dir_from_env()
    current = eval_set_fingerprint(eval_dir, resolve_split(eval_dir))
    final = eval_set_fingerprint(eval_dir, FINAL_SPLIT)  # None until the set has one
    return RunsResponse(
        runs=[RunRecord(**r) for r in registry.load_runs()],
        best=registry.best_per_metric(fingerprint=current),
        current_fingerprint=current,
        final_fingerprint=final,
        metric_keys=list(registry.METRIC_KEYS),
    )


@router.get("/runs/diff", response_model=DiffResponse)
def diff_runs(a: str, b: str, metric: str = DEFAULT_METRIC) -> DiffResponse:
    """Paired comparison of two runs (B relative to A) — win/loss per query, a
    sign-flip permutation p-value, per-topic slices, and (when the live eval set
    still matches the runs' fingerprint) the query texts + what each model retrieved."""
    run_a, run_b = registry.get_run(a), registry.get_run(b)
    if run_a is None or run_b is None:
        raise HTTPException(status_code=404, detail="해당 런을 찾을 수 없습니다")

    # Query texts / slice tags only join while the eval set on disk still has the
    # exact contents these runs were measured on.
    texts_available = False
    eval_dir = run_a.get("eval_dir") or eval_dir_from_env()
    split = run_a.get("split") or resolve_split(eval_dir)
    if eval_set_fingerprint(eval_dir, split) == run_a.get("eval_fingerprint"):
        texts_available = True

    try:
        result = compare_runs(
            run_a, run_b, metric,
            slice_map=load_query_slices(eval_dir) if texts_available else None,
        )
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc

    union = None
    if texts_available:
        query_texts = load_queries(eval_dir)
        corpus = load_corpus(eval_dir)
        qrels = load_qrels(eval_dir, split)
        rankings_a = run_a.get("rankings") or {}
        rankings_b = run_b.get("rankings") or {}
        # 후보군 상보성 — A∪B의 top-10 후보가 정답을 얼마나 더 덮는지 (하이브리드 관점)
        union = candidate_union(run_a, run_b, qrels, slice_map=load_query_slices(eval_dir))

        def _docs(ranked: list[str], relevant: set[str]) -> list[dict]:
            return [
                {
                    "id": doc_id,
                    "title": (corpus.get(doc_id) or {}).get("title") or doc_id,
                    "relevant": doc_id in relevant,
                }
                for doc_id in ranked[:5]
            ]

        for query in result["queries"]:
            query_id = query["query_id"]
            relevant = set(qrels.get(query_id, {}))
            query["text"] = query_texts.get(query_id)
            query["retrieved_a"] = _docs(rankings_a.get(query_id) or [], relevant)
            query["retrieved_b"] = _docs(rankings_b.get(query_id) or [], relevant)

    return DiffResponse(
        a=RunRecord(**run_a),
        b=RunRecord(**run_b),
        texts_available=texts_available,
        union=union,
        **result,
    )


@router.post("/runs/bm25", response_model=ImportTrecResponse)
def register_bm25(req: Bm25Request) -> ImportTrecResponse:
    """Score the built-in BM25 baseline (문자 bigram, from scratch) on the current
    eval set and record it as a normal run. dense 런과 diff하면 상보성(후보군 합집합
    recall)이 함께 계산된다 — "dense가 BM25 위에 실제로 보태는 정답"이 랩의 지표가
    프로덕션 하이브리드와 방향이 맞는지 알려준다."""
    eval_dir = eval_dir_from_env()
    split = resolve_split(eval_dir, req.split)
    corpus = load_corpus(eval_dir)
    queries = load_queries(eval_dir)
    qrels = load_qrels(eval_dir, split)
    if not qrels:
        raise HTTPException(status_code=422, detail="판정된 쿼리가 없습니다 — qrels를 확인하세요")

    top_k = eval_top_k()
    rankings = rank_eval_corpus(corpus, {q: t for q, t in queries.items() if q in qrels}, top_k)
    per_query = per_query_metrics(rankings, qrels, metric_recall_ks(top_k))
    label = (req.label or "").strip() or "bm25"
    record = registry.append_run(
        label, "bm25", "bm25(char-bigram)", eval_dir, mean_metrics(per_query),
        eval_fingerprint=eval_set_fingerprint(eval_dir, split),
        n_queries=len(per_query),
        ci95=bootstrap_ci(per_query),
        per_query=per_query,
        rankings={q: ranked[:10] for q, ranked in rankings.items()},
        split=split,
        note=req.note or None,
        slices=slice_means(per_query, load_query_slices(eval_dir)),
    )
    return ImportTrecResponse(
        run=RunRecord(**record),
        metrics=record["metrics"],
        n_queries=len(per_query),
        errors=[],
        message=f"BM25 베이스라인 등록 완료 — dense 런과 diff하면 상보성이 보입니다 (n={len(per_query)})",
    )


@router.post("/runs/import-trec", response_model=ImportTrecResponse)
def import_trec(req: ImportTrecRequest) -> ImportTrecResponse:
    """Score an EXTERNAL retriever's TREC run (e.g. production BM25) against the
    current qrels and record it as a normal registry run — after that, the paired
    diff view answers "what does dense add that BM25 misses?" for free."""
    rankings, errors = parse_trec_run(req.content)
    if not rankings:
        raise HTTPException(status_code=400, detail="; ".join(errors) or "읽을 수 있는 랭킹이 없습니다")

    eval_dir = eval_dir_from_env()
    split = resolve_split(eval_dir)
    qrels = load_qrels(eval_dir, split)
    per_query = per_query_metrics(rankings, qrels, metric_recall_ks(eval_top_k()))
    if not per_query:
        raise HTTPException(
            status_code=422,
            detail="qrels와 겹치는 query-id가 없습니다 — run 파일의 query-id가 평가셋의 id와 같아야 합니다",
        )

    label = req.label.strip() or "external-run"
    record = registry.append_run(
        label, "external", label, eval_dir, mean_metrics(per_query),
        eval_fingerprint=eval_set_fingerprint(eval_dir, split),
        n_queries=len(per_query),
        ci95=bootstrap_ci(per_query),
        per_query=per_query,
        rankings={q: ranked[:10] for q, ranked in rankings.items()},
        split=split,
        slices=slice_means(per_query, load_query_slices(eval_dir)),
    )
    return ImportTrecResponse(
        run=RunRecord(**record),
        metrics=record["metrics"],
        n_queries=len(per_query),
        errors=errors,
        message=f"외부 런 등록 완료 — 실험 탭에서 dense 런과 diff하면 보완성이 보입니다 (n={len(per_query)})",
    )


@router.delete("/runs/{run_id}", response_model=DeleteRunResponse)
def delete_run(run_id: str) -> DeleteRunResponse:
    remaining = registry.delete_run(run_id)
    return DeleteRunResponse(deleted=run_id, remaining=remaining)
