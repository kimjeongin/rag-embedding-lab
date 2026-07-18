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

import hashlib
import os
from collections.abc import Iterable
from pathlib import Path

from rag.dataset import load_jsonl, write_jsonl

DEFAULT_EVAL_DIR = "data/eval"
DEFAULT_SPLIT = "test"   # legacy single-split layout (pre dev/final separation)
DEV_SPLIT = "dev"        # tuning split — sweeps and day-to-day comparisons run here
FINAL_SPLIT = "final"    # held-out confirmation split — touched once, for the winner


def eval_dir_from_env() -> str:
    """The eval dataset directory (EVAL_DIR), defaulting to data/eval."""
    return os.getenv("EVAL_DIR", DEFAULT_EVAL_DIR)


def resolve_split(eval_dir: str, split: str = DEV_SPLIT) -> str:
    """Map a logical split to the qrels file that actually exists.

    Picking the best of N sweep runs on one query set overfits that set — so queries
    are split: "dev" for selection, "final" for a one-shot confirmation of the winner.
    "dev" falls back to the legacy single "test" split (sets generated before the
    separation). "final" never falls back — silently confirming on the tuning set
    would defeat its purpose.
    """
    if split == FINAL_SPLIT:
        if not (Path(eval_dir) / "qrels" / f"{FINAL_SPLIT}.tsv").exists():
            raise FileNotFoundError(
                "qrels/final.tsv가 없습니다 — 데이터 탭에서 평가셋을 재생성하면 dev/final로 분리됩니다"
            )
        return FINAL_SPLIT
    if (Path(eval_dir) / "qrels" / f"{DEV_SPLIT}.tsv").exists():
        return DEV_SPLIT
    return DEFAULT_SPLIT


def available_splits(eval_dir: str) -> list[str]:
    """Which qrels splits exist in this eval dir (display/UI helper)."""
    qrels_dir = Path(eval_dir) / "qrels"
    if not qrels_dir.exists():
        return []
    return sorted(p.stem for p in qrels_dir.glob("*.tsv"))


def eval_set_fingerprint(eval_dir: str, split: str = DEFAULT_SPLIT) -> str | None:
    """A short content hash identifying this eval set's exact corpus+queries+qrels.

    Scores are only comparable between runs measured on the SAME set, and the dir
    path can't tell sets apart once files are regenerated in place (e.g. with a
    different distractor count) — the contents can. None if the set is incomplete.
    """
    h = hashlib.sha256()
    for rel in ("corpus.jsonl", "queries.jsonl", f"qrels/{split}.tsv"):
        path = Path(eval_dir) / rel
        if not path.exists():
            return None
        h.update(path.read_bytes())
        h.update(b"\x00")  # file boundary, so concatenation ambiguity can't collide
    return h.hexdigest()[:12]


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


def load_query_slices(eval_dir: str) -> dict[str, str]:
    """{query_id: slice} from queries.jsonl's optional ``slice`` key ({} if untagged).

    A slice tags WHERE a query's difficulty comes from (e.g. "standard" vs "jargon" —
    사내 은어로만 답을 찾을 수 있는 쿼리). Slice-wise means expose gaps that a whole-set
    average hides: base 모델이 표준 쿼리 0.99 / 은어 쿼리 0.15여도 평균은 0.56으로 보인다.
    """
    return {
        str(rec["_id"]): str(rec["slice"])
        for rec in load_jsonl(str(Path(eval_dir) / "queries.jsonl"))
        if rec.get("slice")
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


def write_qrels(eval_dir: str, qrels_rows: Iterable[tuple[str, str, int]], split: str) -> None:
    """Write qrels/<split>.tsv (overwrites; header row included)."""
    qrels_path = Path(eval_dir) / "qrels" / f"{split}.tsv"
    qrels_path.parent.mkdir(parents=True, exist_ok=True)
    with qrels_path.open("w", encoding="utf-8") as f:
        f.write("query-id\tcorpus-id\tscore\n")
        for query_id, doc_id, score in qrels_rows:
            f.write(f"{query_id}\t{doc_id}\t{score}\n")


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
    write_qrels(eval_dir, qrels_rows, split)


def prune_qrels_splits(eval_dir: str, keep: Iterable[str]) -> list[str]:
    """Delete qrels/<split>.tsv files NOT in `keep`; returns what was removed.

    A regenerated eval set replaces corpus/queries in place — a split file from the
    previous generation would keep pointing at doc ids that no longer exist, and the
    UI would offer it as if it were evaluable. Generators call this after writing
    their splits so the directory is coherent.
    """
    qrels_dir = Path(eval_dir) / "qrels"
    if not qrels_dir.exists():
        return []
    keep_set = set(keep)
    removed = []
    for path in qrels_dir.glob("*.tsv"):
        if path.stem not in keep_set:
            path.unlink()
            removed.append(path.stem)
    return sorted(removed)
