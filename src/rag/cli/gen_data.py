"""`rag-gen-data` — write a toy train/test dataset (rag.datagen.dummy)."""
from __future__ import annotations

from rag.datagen.dummy import generate_dataset
from rag.dataset import dataset_paths, write_jsonl


def main() -> None:
    train_file, eval_file = dataset_paths()
    train, test = generate_dataset()
    write_jsonl(train_file, train)
    write_jsonl(eval_file, test)
    print(f"wrote {train_file} ({len(train)} pairs) and {eval_file} ({len(test)} pairs)")
