"""Bring real data in — search/click logs → training pairs and qrels.

For an internal-site search, the most valuable supervision already exists: the query
log and the click log. A (query, clicked_site) pair is exactly an MNRL training pair
AND exactly a relevance judgment — noisy individually, decisive in volume. This
module turns pasted/uploaded records into the lab's two formats:

    training pair:  {"query": ..., "positive": {"title": ..., "content": ...}}
    qrels row:      (query_id, doc_id, 1)  + the query into queries.jsonl

Accepted input (auto-detected): JSONL records or CSV. Records reference a corpus doc
by ``doc_id`` (looked up in the eval corpus) or carry their own ``title``/``content``
(training pairs only — qrels must point at a corpus doc). Pure transforms — the API
route does the file IO.
"""
from __future__ import annotations

import csv
import io
import json

_CSV_COLUMNS = ("query", "doc_id", "title", "content")


def parse_records(text: str) -> tuple[list[dict], list[str]]:
    """(records, errors) from pasted JSONL or CSV (auto-detected).

    CSV with a header row maps by column name; headerless 2-column CSV is taken as
    (query, doc_id) — the shape of a click log export.
    """
    text = text.strip()
    if not text:
        return [], ["내용이 비어 있습니다"]
    if text.lstrip().startswith("{"):
        return _parse_jsonl(text)
    return _parse_csv(text)


def _parse_jsonl(text: str) -> tuple[list[dict], list[str]]:
    records, errors = [], []
    for lineno, line in enumerate(text.splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            errors.append(f"{lineno}행: JSON이 아닙니다")
            continue
        if isinstance(record, dict):
            records.append(record)
        else:
            errors.append(f"{lineno}행: 객체가 아닙니다")
    return records, errors


def _parse_csv(text: str) -> tuple[list[dict], list[str]]:
    rows = list(csv.reader(io.StringIO(text)))
    if not rows:
        return [], ["내용이 비어 있습니다"]
    header = [c.strip().lower() for c in rows[0]]
    if "query" in header:
        body, columns = rows[1:], header
    else:
        body, columns = rows, list(_CSV_COLUMNS[: len(rows[0])])
    records, errors = [], []
    for lineno, row in enumerate(body, start=1):
        if not any(cell.strip() for cell in row):
            continue
        record = {col: cell.strip() for col, cell in zip(columns, row) if col in _CSV_COLUMNS}
        if record.get("query"):
            records.append(record)
        else:
            errors.append(f"{lineno}행: query 열이 비어 있습니다")
    return records, errors


def to_train_pairs(records: list[dict], corpus: dict[str, dict]) -> tuple[list[dict], list[str]]:
    """Records → training pairs. doc_id pairs resolve against the corpus; records may
    instead carry their own title/content."""
    pairs, skipped = [], []
    for i, record in enumerate(records, start=1):
        query = str(record.get("query") or "").strip()
        if not query:
            skipped.append(f"{i}번: query 없음")
            continue
        doc_id = str(record.get("doc_id") or "").strip()
        if doc_id:
            doc = corpus.get(doc_id)
            if doc is None:
                skipped.append(f"{i}번: corpus에 없는 doc_id '{doc_id}'")
                continue
            pairs.append({"query": query, "positive": {"title": doc.get("title"), "content": doc.get("text") or ""}})
        elif str(record.get("content") or "").strip():
            pairs.append({
                "query": query,
                "positive": {"title": (record.get("title") or None), "content": str(record["content"]).strip()},
            })
        else:
            skipped.append(f"{i}번: doc_id 또는 content가 필요합니다")
    return pairs, skipped


def dedupe_pairs(existing: list[dict], new: list[dict]) -> list[dict]:
    """New pairs not already present (same query + same positive content)."""
    def key(pair: dict) -> tuple[str, str]:
        positive = pair.get("positive") or {}
        return (str(pair.get("query", "")).strip().lower(), str(positive.get("content", "")).strip())

    seen = {key(p) for p in existing}
    fresh = []
    for pair in new:
        k = key(pair)
        if k not in seen:
            seen.add(k)
            fresh.append(pair)
    return fresh


def to_qrels(
    records: list[dict],
    corpus: dict[str, dict],
    taken_query_ids: set[str],
) -> tuple[list[dict], list[tuple[str, str, int]], list[str]]:
    """Records → (new queries.jsonl entries, qrels rows, skipped).

    Only doc_id records can become judgments (a qrel must point at a corpus doc).
    The same query text appearing with several docs becomes ONE query with several
    relevant docs. New ids are q-user-N (disjoint from the generated q-{topic}-{i}).
    """
    new_queries: list[dict] = []
    rows: list[tuple[str, str, int]] = []
    skipped: list[str] = []
    id_by_text: dict[str, str] = {}
    counter = 0

    def next_id() -> str:
        nonlocal counter
        while True:
            counter += 1
            candidate = f"q-user-{counter}"
            if candidate not in taken_query_ids:
                taken_query_ids.add(candidate)
                return candidate

    for i, record in enumerate(records, start=1):
        query = str(record.get("query") or "").strip()
        doc_id = str(record.get("doc_id") or "").strip()
        if not query:
            skipped.append(f"{i}번: query 없음")
            continue
        if not doc_id:
            skipped.append(f"{i}번: qrels에는 doc_id가 필요합니다")
            continue
        if doc_id not in corpus:
            skipped.append(f"{i}번: corpus에 없는 doc_id '{doc_id}'")
            continue
        text_key = query.lower()
        if text_key not in id_by_text:
            query_id = next_id()
            id_by_text[text_key] = query_id
            new_queries.append({"_id": query_id, "text": query})
        rows.append((id_by_text[text_key], doc_id, 1))
    return new_queries, rows, skipped
