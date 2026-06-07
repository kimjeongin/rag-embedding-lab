"""BEIR-format evaluation data — corpus + queries + qrels (load and write).

Layout (the de-facto standard from the BEIR benchmark, github.com/beir-cellar/beir):

    <EVAL_DIR>/
      corpus.jsonl      one doc per line:    {"_id": "d1", "title": "...", "text": "..."}
      queries.jsonl     one query per line:  {"_id": "q1", "text": "..."}
      qrels/test.tsv    TSV: a header row, then  query-id <TAB> corpus-id <TAB> score

A `score > 0` marks a (query, doc) pair as relevant (graded scores are allowed). Any
BEIR dataset — or in-house data exported to this layout — drops straight in: point
`EVAL_DIR` at the folder. This module does IO only; scoring lives in `metrics` and the
embed-and-rank step in `retrieval`. See docs/evaluation.md for the full contract.
"""
from __future__ import annotations

import os
from collections.abc import Iterable
from pathlib import Path

from rag.dataset import load_jsonl, write_jsonl

DEFAULT_EVAL_DIR = "data/eval"
DEFAULT_SPLIT = "test"


def eval_dir_from_env() -> str:
    """The eval dataset directory (EVAL_DIR), defaulting to data/eval."""
    return os.getenv("EVAL_DIR", DEFAULT_EVAL_DIR)


def load_corpus(eval_dir: str) -> dict[str, dict[str, str | None]]:
    """{doc_id: {"title": str | None, "text": str}} from corpus.jsonl."""
    corpus: dict[str, dict[str, str | None]] = {}
    for rec in load_jsonl(str(Path(eval_dir) / "corpus.jsonl")):
        corpus[str(rec["_id"])] = {"title": rec.get("title") or None, "text": rec["text"]}
    return corpus


def load_queries(eval_dir: str) -> dict[str, str]:
    """{query_id: text} from queries.jsonl."""
    return {
        str(rec["_id"]): rec["text"]
        for rec in load_jsonl(str(Path(eval_dir) / "queries.jsonl"))
    }


def load_qrels(eval_dir: str, split: str = DEFAULT_SPLIT) -> dict[str, dict[str, float]]:
    """{query_id: {doc_id: gain}} from qrels/<split>.tsv.

    The header row is skipped (its score column isn't numeric); only pairs with
    score > 0 are kept.
    """
    qrels: dict[str, dict[str, float]] = {}
    path = Path(eval_dir) / "qrels" / f"{split}.tsv"
    with open(path, encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) != 3:
                continue
            query_id, doc_id, raw_score = parts
            try:
                score = float(raw_score)
            except ValueError:
                continue  # the header row: "query-id\tcorpus-id\tscore"
            if score > 0:
                qrels.setdefault(query_id, {})[doc_id] = score
    return qrels


def write_beir_dataset(
    eval_dir: str,
    corpus: Iterable[dict],
    queries: Iterable[dict],
    qrels_rows: Iterable[tuple[str, str, int]],
    split: str = DEFAULT_SPLIT,
) -> None:
    """Write corpus.jsonl, queries.jsonl and qrels/<split>.tsv in BEIR layout."""
    base = Path(eval_dir)
    write_jsonl(str(base / "corpus.jsonl"), corpus)
    write_jsonl(str(base / "queries.jsonl"), queries)

    qrels_path = base / "qrels" / f"{split}.tsv"
    qrels_path.parent.mkdir(parents=True, exist_ok=True)
    with qrels_path.open("w", encoding="utf-8") as f:
        f.write("query-id\tcorpus-id\tscore\n")
        for query_id, doc_id, score in qrels_rows:
            f.write(f"{query_id}\t{doc_id}\t{score}\n")
