# Serving Parity — running the fine-tuned model in your own stack

> **One line:** a domain fine-tune only helps in production if your serving stack embeds
> text **the exact same way** the lab trained it. A formatting / inference mismatch
> between the lab and your serving (e.g. Elasticsearch + BM25 + reranker) is the **#1
> reason a fine-tune "wins in the lab but flops in prod."** Check this *before* you trust
> any production number.

This lab trains and measures the **dense embedder** in isolation (see
[evaluation.md](evaluation.md)). When you take the resulting model and serve it yourself —
indexing documents into Elasticsearch, embedding queries at search time — **your pipeline
becomes the third place that must agree with `train` and `eval`.** If it embeds text
differently, the fine-tuned vectors live in a different space than the queries you send at
serve time, and the gains evaporate (or invert).

---

## The formatting contract

Retrieval embedding models are **asymmetric**: queries and documents are embedded
*differently*, and each model family wants its own template. The lab defines these as
`ModelProfile`s in exactly one place —
[`src/rag/core/formatting.py`](../src/rag/core/formatting.py) — and **training, evaluation,
and inference all call it**, so they can't drift apart. Your serving stack is outside this
repo, so it's on you to match it.

**Match the profile of the model you deploy**, not just the table's first row:

| Profile | Query | Document |
|---------|-------|----------|
| **`qwen3`** (Qwen3-Embedding) | `Instruct: {instruction}\nQuery: {query}` | `{title}\n\n{content}` (or `{content}` if no title) |
| **`nemotron3`** (Nemotron-3-Embed) | `query: {query}` | `passage: {title}\n\n{content}` |
| **`plain`** | `{query}` | `{title}\n\n{content}` |

A fine-tuned model records its profile in `train_meta.json`, so
`rag.modelprofile.resolve_profile()` gives you the right answer for a model directory.
Serving a `nemotron3` model with `qwen3` prompts raises nothing — it just scores worse.

- `{instruction}` is `QUERY_INSTRUCTION` (env-configurable), default:
  `Given a web search query, retrieve relevant passages that answer the query`
- Identifiers (url / domain / path) are **excluded** from the embedded text — they are
  filter/group metadata, not semantic content.

The exact strings your ES pipeline must reproduce, byte-for-byte:

```python
from rag.core.formatting import format_query, format_document, DEFAULT_QUERY_INSTRUCTION

format_query("how do I reset my password", DEFAULT_QUERY_INSTRUCTION)
# 'Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery: how do I reset my password'

format_document("Password reset", "Open Settings → Security and choose Reset password.")
# 'Password reset\n\nOpen Settings → Security and choose Reset password.'

format_document(None, "Open Settings → Security and choose Reset password.")
# 'Open Settings → Security and choose Reset password.'
```

---

## Why a mismatch silently kills your gains

There are **two** distinct failures, and the second is the sneaky one:

1. **Your base model is already underperforming today.** If your ES *query* embedding omits
   the `Instruct: …` prefix, even the un-fine-tuned 0.6B is leaving recall on the table —
   Qwen3-Embedding expects that prefix on the query side. You may be measuring the wrong
   baseline.

2. **Train/serve mismatch makes the fine-tune look like a regression.** If you train *with*
   the prefix (the lab does) but serve *without* it, the fine-tuned query vectors and the
   indexed document vectors no longer sit in the space the model was tuned for. Result:
   **lab recall@k goes up, production stays flat or drops** — and you wrongly conclude the
   fine-tune didn't work.

This is why a lab win is **necessary but not sufficient**: it only transfers if serving is
parity-correct.

---

## Four things that must match (not just the prefix)

The query prefix is the most common miss, but parity is about the *whole* embedding path:

1. **Query template** — `Instruct: {instruction}\nQuery: {q}`, with the **same instruction
   string** and the same whitespace/newlines.
2. **Document composition** — `{title}\n\n{content}`, **no** instruction prefix. Don't embed
   title and body as separate fields, and don't join them with a space or a different
   separator.
3. **Inference method / pooling** — Qwen3-Embedding uses **last-token pooling** (it's a
   causal-LM-based embedder), not mean pooling. Embed through the *same* path the lab uses
   (sentence-transformers, or the model's official inference), **not** a hand-rolled
   mean-pool over hidden states. A wrong pooling silently produces plausible-but-wrong
   vectors.
4. **Normalization & similarity** — L2-normalize the vectors and score with **cosine** (or
   dot product on normalized vectors), matching the lab. If ES is configured for raw dot
   product on un-normalized vectors, that's a mismatch.

> The reranker (your Qwen reranker) is a **separate model with its own input format** —
> don't reuse the embedder's templates for it. Parity here is only about the **embedder**.

---

## How to verify parity (do this, don't assume)

**Step 1 — string check.** Print what the lab produces and diff it against the exact string
your ES indexer / query builder sends to the embedding model. They must be identical,
including the `\nQuery:` newline and the `\n\n` document join.

**Step 2 — vector check (the real proof).** Embed the *same* text through (a) the lab's path
and (b) your ES embedding path, then compare. On identical input the cosine similarity
should be **≈ 1.0** (≳ 0.999). Anything materially lower means a hidden mismatch —
whitespace, pooling, or normalization.

```python
# (a) lab path
import numpy as np
from rag.config import Settings
from rag.core.entities import Document
from rag.embeddings import build_embedder

async def lab_vec(text_is_query: bool, text: str):
    async with build_embedder(Settings.from_env()) as emb:
        v = (await emb.embed_queries([text]))[0] if text_is_query \
            else (await emb.embed_documents([Document(content=text)]))[0]
    return np.asarray(v, "float32")

# (b) your ES path: fetch the vector ES actually stores/sends for the same text,
#     then: cos = (a·b) / (||a|| ||b||)   →   expect ≈ 1.0
```

If cosine is ~1.0 for both a sample query and a sample document, your serving is
parity-correct and the lab numbers will transfer.

---

## When you swap the fine-tuned model into Elasticsearch

- **Re-index everything.** A fine-tuned model produces a **new vector space** — its vectors
  are *not* comparable to the base model's. Re-embed and re-index the **entire** corpus;
  never mix base-model and fine-tuned vectors in one index.
- **Dimension is unchanged** (1024 for the 0.6B), so your ES mapping stays the same — only
  the vectors change.
- **Freeze `QUERY_INSTRUCTION`.** Keep the instruction string identical across training,
  eval, and serving. If you ever change it, you must re-embed the corpus and re-evaluate —
  it's part of the contract.

---

## Checklist

- [ ] ES **query** embedding uses `Instruct: {instruction}\nQuery: {q}` with the same instruction string.
- [ ] ES **document** embedding uses `{title}\n\n{content}` (no instruction, `\n\n` join).
- [ ] Same **inference path / pooling** as the lab (last-token, via sentence-transformers or the official method).
- [ ] Vectors **L2-normalized**, similarity is **cosine**.
- [ ] **Full re-index** performed after swapping models.
- [ ] **Vector parity check** passes (cosine ≈ 1.0 on identical text, query *and* document).

---

## Where this fits

- The embedder is your **dense first-stage retriever**; in a hybrid + rerank stack its job
  is **recall into the candidate pool** the reranker reorders. Measure it with
  `recall@k` at your rerank candidate depth — see [evaluation.md](evaluation.md).
- Parity here (serving) + parity in the lab (train ≈ eval, via `core.formatting`) together
  mean the model behaves the same in all four places: **train, eval, serve-index,
  serve-query.**
