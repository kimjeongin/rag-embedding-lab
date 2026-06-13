"""`rag-gen-eval` — write a BEIR-format eval set to EVAL_DIR (data/eval).

Two sources (EVAL_SOURCE):

  - ``sample`` (default): the bundled toy gold docs + tech-register distractors —
    placeholder data so the harness runs end-to-end without a corpus.
  - ``corpus``: the real corpus (CORPUS_FILE) as the haystack + the held-out test
    split (TRAIN_EVAL_FILE) as queries/qrels. Run after ``rag-crawl`` +
    ``rag-gen-synthetic``; the whole site becomes the distractor set.

Env: EVAL_DIR (output dir, default data/eval), EVAL_SOURCE (sample|corpus),
N_DISTRACTORS (sample: cap the haystack), CORPUS_FILE + TRAIN_EVAL_FILE (corpus).
"""
from __future__ import annotations

import os

from rag.datagen.eval_corpus import generate, split_qrels
from rag.evaluation.beir import (
    DEV_SPLIT,
    FINAL_SPLIT,
    eval_dir_from_env,
    prune_qrels_splits,
    write_beir_dataset,
    write_qrels,
)


def _from_corpus() -> tuple[list[dict], list[dict], list[tuple[str, str, int]]]:
    from rag.datagen.eval_from_corpus import load_and_build
    from rag.dataset import dataset_paths

    corpus_file = os.getenv("CORPUS_FILE", "data/corpus.jsonl")
    _, test_file = dataset_paths()
    try:
        corpus, queries, qrels, skipped = load_and_build(corpus_file, test_file)
    except ValueError as exc:
        raise SystemExit(f"[gen-eval] {exc}") from exc
    if skipped:
        print(f"[gen-eval] 주의: corpus와 매칭되지 않은 test pair {skipped}건 건너뜀 — 데이터를 재생성하면 사라집니다")
    return corpus, queries, qrels


def main() -> None:
    eval_dir = eval_dir_from_env()
    source = os.getenv("EVAL_SOURCE", "sample")

    if source == "corpus":
        corpus, queries, qrels = _from_corpus()
    else:
        raw = os.getenv("N_DISTRACTORS")
        corpus, queries, qrels = generate(n_distractors=int(raw) if raw else None)

    # dev = tuning split (sweeps/comparisons), final = held-out one-shot confirmation
    # for the chosen winner — selection on one set + confirmation on another is what
    # keeps "best of 20 runs" from overfitting the eval set itself.
    dev_rows, final_rows = split_qrels(qrels)
    write_beir_dataset(eval_dir, corpus, queries, dev_rows, split=DEV_SPLIT)
    write_qrels(eval_dir, final_rows, FINAL_SPLIT)
    stale = prune_qrels_splits(eval_dir, keep=(DEV_SPLIT, FINAL_SPLIT))
    if stale:
        print(f"[gen-eval] 이전 세대의 qrels split 제거: {', '.join(stale)} (corpus가 바뀌어 더는 유효하지 않음)")
    print(
        f"[gen-eval] wrote {eval_dir} (source={source}): {len(corpus)} corpus docs, "
        f"{len(queries)} queries, qrels dev={len(dev_rows)} final={len(final_rows)}"
    )
