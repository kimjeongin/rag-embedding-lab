"""The fine-tuning **training-pair** dataset — its format, IO, and default paths.

A dataset is a JSONL file; each line is one record:

    {"query": str,
     "positive": {"title": str | None, "content": str},
     "negatives": [{"title": str | None, "content": str}]}   # optional

`datagen` writes these and `training` reads them. (Evaluation is separate — it uses a
BEIR-format corpus/queries/qrels set; see `rag.evaluation.beir`.) Keeping this format,
its IO, and the default paths in one neutral module avoids a backwards dependency
between those packages (e.g. the data generator shouldn't import training's config).
Pure stdlib.
"""
from __future__ import annotations

import json
import os
from collections.abc import Iterable, Iterator
from pathlib import Path

DEFAULT_TRAIN_FILE = "data/train.jsonl"
DEFAULT_EVAL_FILE = "data/test.jsonl"


def dataset_paths() -> tuple[str, str]:
    """(train_file, eval_file) from env (TRAIN_FILE / TRAIN_EVAL_FILE), with defaults."""
    return (
        os.getenv("TRAIN_FILE", DEFAULT_TRAIN_FILE),
        os.getenv("TRAIN_EVAL_FILE", DEFAULT_EVAL_FILE),
    )


def load_jsonl(path: str) -> Iterator[dict]:
    """Yield records from a JSONL file, skipping blank lines."""
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_jsonl(path: str, records: Iterable[dict]) -> None:
    """Write records to a JSONL file (creating parent dirs)."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
