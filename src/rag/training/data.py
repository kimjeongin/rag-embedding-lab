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


def to_training_dataset(path: str, instruction: str):
    """Build a 🤗 ``datasets.Dataset`` for MultipleNegativesRankingLoss.

    Columns are (anchor, positive) — anchor = instruction-prefixed query, positive =
    title-prepended document, identical to what we embed at serving time. If every
    record carries a hard negative, a (anchor, positive, negative) triplet is built
    instead, which trains against that explicit negative on top of in-batch ones.
    """
    from datasets import Dataset

    rows = list(load_jsonl(path))
    data = {
        "anchor": [format_query(r["query"], instruction) for r in rows],
        "positive": [format_document(r["positive"].get("title"), r["positive"]["content"]) for r in rows],
    }
    if rows and all(r.get("negatives") for r in rows):
        data["negative"] = [
            format_document(r["negatives"][0].get("title"), r["negatives"][0]["content"]) for r in rows
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
