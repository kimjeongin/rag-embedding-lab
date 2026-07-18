"""`rag-gen-intranet` — 가상 인트라넷 카탈로그 데이터셋(실데이터 리허설용) 생성.

data-intranet/ 아래에 랩이 그대로 먹을 수 있는 전체 레이아웃을 쓴다:

    corpus.jsonl   페이지 카탈로그 (url·description·content·agent_prompt·metadata)
    train.jsonl    학습쌍 — 클릭로그 시뮬레이션 (은어 쿼리 포함)
    test.jsonl     평가용 held-out 쌍 (slice 태그: standard | jargon)
    eval/          BEIR 평가셋 (queries.jsonl에 slice 태그, qrels dev/final 분리)

백엔드를 이 데이터셋으로 돌리려면:

    CORPUS_FILE=data-intranet/corpus.jsonl TRAIN_FILE=data-intranet/train.jsonl \
    TRAIN_EVAL_FILE=data-intranet/test.jsonl EVAL_DIR=data-intranet/eval uv run rag-serve

Env: INTRANET_DIR (출력 위치, 기본 data-intranet), INTRANET_SEED (기본 20260718).
"""
from __future__ import annotations

import os
from collections import Counter
from pathlib import Path

from rag.dataset import write_jsonl
from rag.datagen.eval_corpus import split_qrels
from rag.datagen.eval_from_corpus import build
from rag.datagen.intranet import generate
from rag.evaluation.beir import prune_qrels_splits, write_beir_dataset, write_qrels


def main() -> None:
    out = os.getenv("INTRANET_DIR", "data-intranet")
    seed = int(os.getenv("INTRANET_SEED", "20260718"))
    pages, train, eval_pairs = generate(seed)

    write_jsonl(str(Path(out) / "corpus.jsonl"), pages)
    write_jsonl(str(Path(out) / "train.jsonl"), train)
    write_jsonl(str(Path(out) / "test.jsonl"), eval_pairs)

    corpus, queries, qrels, skipped = build(pages, eval_pairs)
    if skipped:
        raise SystemExit(f"평가쌍 {skipped}건이 corpus와 매칭되지 않았습니다 (생성기 버그)")
    for query in queries:  # q-test-{j}의 j = eval_pairs 인덱스 — slice 태그를 되붙인다
        query["slice"] = eval_pairs[int(query["_id"].rsplit("-", 1)[1])]["slice"]

    eval_dir = str(Path(out) / "eval")
    dev_rows, final_rows = split_qrels(qrels)
    write_beir_dataset(eval_dir, corpus, queries, dev_rows, split="dev")
    write_qrels(eval_dir, final_rows, split="final")
    prune_qrels_splits(eval_dir, keep=("dev", "final"))

    slices = Counter(q["slice"] for q in queries)
    print(f"[gen-intranet] {out}/")
    print(f"  corpus  {len(pages)} pages ({len({p['system'] for p in pages}) - 1} systems + policy)")
    print(f"  train   {len(train)} pairs "
          f"(jargon {sum(1 for r in train if r['slice'] == 'jargon')})")
    print(f"  eval    {len(queries)} queries "
          f"(standard {slices['standard']} / jargon {slices['jargon']}) "
          f"→ dev {len({q for q, _, _ in dev_rows})} / final {len({q for q, _, _ in final_rows})}")
