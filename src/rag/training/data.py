"""Turn the JSONL dataset (rag.dataset format) into the inputs the trainer needs —
with train/inference parity.

The one thing that must be right: a training/eval example's query and document text
must be built the same way as at inference. So both go through ``rag.core.formatting``
(the module the serving path uses), with the SAME `ModelProfile` — the base model's
format, resolved once by the caller and passed in here.

`datasets` is imported lazily so this module stays importable without the training
stack.
"""
from __future__ import annotations

import random

from rag.core.formatting import DEFAULT_PROFILE, ModelProfile, format_document, format_query
from rag.dataset import load_jsonl


def negative_count(rows: list[dict], cap: int | None) -> int:
    """How many hard-negative columns the dataset should carry: the LARGEST count any
    record supplies, capped by the loss's appetite (TripletLoss digests exactly one;
    the MNRL/GIST family takes them all). Records with fewer are padded up to this
    target by ``pad_negatives`` — a columnar dataset has no ragged rows, and the old
    take-the-minimum rule meant ONE pair with zero mined negatives (the false-negative
    guard legitimately produces those) silently discarded every mined negative in the
    dataset. 0 when no record has any.
    """
    if not rows:
        return 0
    most = max(len(r.get("negatives") or []) for r in rows)
    return min(most, cap) if cap is not None else most


def _doc_key(doc: dict) -> tuple[str | None, str]:
    return (doc.get("title"), doc["content"])


def pad_negatives(rows: list[dict], target: int, seed: int = 13) -> int:
    """Bring every record up to ``target`` negatives by borrowing docs from the other
    records (their mined negatives first, then their positives — for THIS query those
    are ordinary random negatives, the same grade MNRL already uses in-batch). A
    record's own positive is never borrowed. Mutates ``rows``; returns how many
    records needed padding. Deterministic given ``seed``.
    """
    pool: list[dict] = []
    seen: set[tuple[str | None, str]] = set()
    for row in rows:
        for doc in (row.get("negatives") or []) + [row["positive"]]:
            if _doc_key(doc) not in seen:
                seen.add(_doc_key(doc))
                pool.append(doc)

    rng = random.Random(seed)
    padded = 0
    for row in rows:
        negatives = list(row.get("negatives") or [])
        if len(negatives) >= target:
            continue
        padded += 1
        own = {_doc_key(doc) for doc in negatives} | {_doc_key(row["positive"])}
        candidates = [doc for doc in pool if _doc_key(doc) not in own]
        rng.shuffle(candidates)
        negatives.extend(candidates[: target - len(negatives)])
        # tiny-corpus fallback: nothing left to borrow → repeat what the record has
        if negatives:
            base = list(negatives)
            while len(negatives) < target:
                negatives.append(base[(len(negatives) - len(base)) % len(base)])
        row["negatives"] = negatives
    return padded


def to_training_dataset(
    path: str,
    instruction: str,
    max_negatives: int | None = None,
    profile: ModelProfile = DEFAULT_PROFILE,
):
    """Build a 🤗 ``datasets.Dataset`` for the contrastive losses.

    Columns are (anchor, positive) — anchor = instruction-prefixed query, positive =
    title-prepended document, identical to what we embed at serving time. Mined hard
    negatives become additional columns (negative, negative_2, …): every column after
    the first two is treated as a negative by the MNRL/CachedMNRL/GIST family, so ALL
    mined negatives sharpen the contrast — not just the first. Records with fewer
    negatives than the target are padded from the other records' docs (columns can't
    be ragged; see ``pad_negatives``). ``max_negatives`` caps the columns for losses
    with a fixed arity (TripletLoss → 1).
    """
    from datasets import Dataset

    rows = list(load_jsonl(path))
    target = negative_count(rows, max_negatives)
    if target:
        padded = pad_negatives(rows, target)
        if padded:
            print(f"[data] {padded}/{len(rows)} records had fewer than {target} hard "
                  f"negatives — padded with docs borrowed from other records")
    data = {
        "anchor": [format_query(r["query"], instruction, profile) for r in rows],
        "positive": [
            format_document(r["positive"].get("title"), r["positive"]["content"], profile)
            for r in rows
        ],
    }
    for k in range(target):
        column = "negative" if k == 0 else f"negative_{k + 1}"
        data[column] = [
            format_document(r["negatives"][k].get("title"), r["negatives"][k]["content"], profile)
            for r in rows
        ]
    return Dataset.from_dict(data)


def to_ir_eval(
    path: str,
    instruction: str,
    distractor_file: str | None = None,
    profile: ModelProfile = DEFAULT_PROFILE,
) -> tuple[dict, dict, dict]:
    """Build (queries, corpus, relevant_docs) for sentence-transformers'
    InformationRetrievalEvaluator used DURING training.

    Documents are deduped so multiple query phrasings can point at the same doc.
    Text is formatted exactly as in training/serving.

    ``distractor_file`` (typically the TRAIN split) adds its positives to the corpus
    as distractors — nobody's relevant doc, just haystack. Without them the val
    corpus is only the handful of held-out docs, nDCG@10 saturates near 1.0, and the
    early-stopping signal is noise; with them validation ranks against the same kind
    of haystack the real evaluation (and production) uses.
    """
    queries: dict[str, str] = {}
    corpus: dict[str, str] = {}
    relevant: dict[str, set[str]] = {}
    doc_ids: dict[tuple[str | None, str], str] = {}

    def _add_doc(pos: dict) -> str:
        key = (pos.get("title"), pos["content"])
        if key not in doc_ids:
            doc_ids[key] = f"d{len(doc_ids)}"
            corpus[doc_ids[key]] = format_document(pos.get("title"), pos["content"], profile)
        return doc_ids[key]

    for i, record in enumerate(load_jsonl(path)):
        did = _add_doc(record["positive"])
        qid = f"q{i}"
        queries[qid] = format_query(record["query"], instruction, profile)
        relevant[qid] = {did}

    if distractor_file:
        for record in load_jsonl(distractor_file):
            _add_doc(record["positive"])

    return queries, corpus, relevant
