"""Turn the JSONL dataset (rag.dataset format) into the inputs the trainer needs —
with train/inference parity.

The one thing that must be right: a training/eval example's query and document text
must be built the same way as at inference. So both go through ``rag.core.formatting``
(the module the serving path uses): the query gets the instruction prefix, the
document gets its title prepended.

`datasets` is imported lazily so this module stays importable without the training
stack.
"""
from __future__ import annotations

from rag.core.formatting import format_document, format_query
from rag.dataset import load_jsonl


def negative_count(rows: list[dict], cap: int | None) -> int:
    """How many hard-negative columns the dataset can carry: the LARGEST count every
    record can supply (records must be uniform — a columnar dataset has no ragged
    rows), capped by the loss's appetite (TripletLoss digests exactly one; the
    MNRL/GIST family takes them all). 0 when any record has none.
    """
    if not rows:
        return 0
    common = min(len(r.get("negatives") or []) for r in rows)
    return min(common, cap) if cap is not None else common


def to_training_dataset(path: str, instruction: str, max_negatives: int | None = None):
    """Build a 🤗 ``datasets.Dataset`` for the contrastive losses.

    Columns are (anchor, positive) — anchor = instruction-prefixed query, positive =
    title-prepended document, identical to what we embed at serving time. Mined hard
    negatives become additional columns (negative, negative_2, …): every column after
    the first two is treated as a negative by the MNRL/CachedMNRL/GIST family, so ALL
    mined negatives sharpen the contrast — not just the first. ``max_negatives``
    caps the columns for losses with a fixed arity (TripletLoss → 1).
    """
    from datasets import Dataset

    rows = list(load_jsonl(path))
    data = {
        "anchor": [format_query(r["query"], instruction) for r in rows],
        "positive": [format_document(r["positive"].get("title"), r["positive"]["content"]) for r in rows],
    }
    for k in range(negative_count(rows, max_negatives)):
        column = "negative" if k == 0 else f"negative_{k + 1}"
        data[column] = [
            format_document(r["negatives"][k].get("title"), r["negatives"][k]["content"]) for r in rows
        ]
    return Dataset.from_dict(data)


def to_ir_eval(path: str, instruction: str) -> tuple[dict, dict, dict]:
    """Build (queries, corpus, relevant_docs) for sentence-transformers'
    InformationRetrievalEvaluator used DURING training.

    Documents are deduped so multiple query phrasings can point at the same doc.
    Text is formatted exactly as in training/serving.
    """
    queries: dict[str, str] = {}
    corpus: dict[str, str] = {}
    relevant: dict[str, set[str]] = {}
    doc_ids: dict[tuple[str | None, str], str] = {}

    for i, record in enumerate(load_jsonl(path)):
        pos = record["positive"]
        key = (pos.get("title"), pos["content"])
        if key not in doc_ids:
            did = f"d{len(doc_ids)}"
            doc_ids[key] = did
            corpus[did] = format_document(pos.get("title"), pos["content"])
        qid = f"q{i}"
        queries[qid] = format_query(record["query"], instruction)
        relevant[qid] = {doc_ids[key]}

    return queries, corpus, relevant
