# Retrieval Evaluation

How `rag-eval` measures an embedding model's **retrieval quality**, the data format it
expects, and the assumptions baked into the experiment — so you can drop in your own
in-house data and run it unchanged.

> **The one-line model:** evaluation is *needle in a haystack*. Given a query, does the
> embedder rank the **gold doc** (needle) near the top of the **whole corpus**
> (haystack)? The haystack must be big and realistic or the numbers saturate at 1.0 and
> tell you nothing.

---

## TL;DR

```bash
uv run rag-gen-eval            # write the SAMPLE eval set to data/eval (replace with real data)
uv run rag-eval                # measure the configured embedder over it

# compare two models on the same corpus, SAME backend — the deltas are the experiment result:
EMBEDDER=sentence-transformers ST_MODEL=Qwen/Qwen3-Embedding-0.6B uv run rag-eval  # base
EMBEDDER=sentence-transformers ST_MODEL=outputs/embedding-ft      uv run rag-eval  # fine-tuned
```

---

## The data contract (BEIR layout)

`rag-eval` reads a directory (`EVAL_DIR`, default `data/eval`) in the **BEIR** on-disk
format — the de-facto standard from the [BEIR benchmark](https://github.com/beir-cellar/beir).
Any BEIR dataset, or your in-house data exported to this layout, works as-is.

```
data/eval/
├── corpus.jsonl      the haystack — every searchable doc (gold + distractors)
├── queries.jsonl     the user queries
└── qrels/
    └── test.tsv      relevance judgments — which doc answers which query
```

### `corpus.jsonl` — one JSON object per line

```json
{"_id": "gold-3", "title": "PostgreSQL full text search", "text": "Full text search uses tsvector and tsquery to ..."}
{"_id": "distractor-12", "title": "Apache Kafka — scaling", "text": "Scaling Apache Kafka horizontally means ..."}
```

| field | required | notes |
|-------|----------|-------|
| `_id` | ✅ | unique string id. Referenced by qrels. |
| `text` | ✅ | the document body (embedded). |
| `title` | optional | prepended to `text` for embedding (see [parity](#3-formatting-parity-train--serve--eval)). |

> This includes **every** doc you search over — the handful of answer docs **and** all
> the distractors. Distractors are the rest of your collection; they're what make the
> task hard.

### `queries.jsonl` — one JSON object per line

```json
{"_id": "q-3-0", "text": "search text columns in postgres"}
```

### `qrels/test.tsv` — tab-separated, with a header row

```
query-id	corpus-id	score
q-3-0	gold-3	1
q-3-1	gold-3	1
```

- A row says "`corpus-id` is relevant to `query-id` with this relevance grade".
- `score > 0` ⇒ relevant. **Only relevant pairs need a row** — the thousands of
  (query, distractor) non-matches are implied, you don't list them.
- A query may have **multiple** relevant docs (several rows). Graded scores (1, 2, 3 …)
  are honoured by nDCG; binary `1` is the common case.
- `test` is the split name; `rag-eval` reads `qrels/test.tsv` by default.

---

## How the experiment works (assumptions)

What `rag-eval` actually does, and the assumptions you're signing up for:

### 1. It measures the *retriever*, not an LLM answer
The "answer" is a **document id in the corpus**, not generated text. We score whether
the right *passage* is retrieved — that's what an embedding model controls. (Generation
quality is a separate, downstream concern.)

### 2. The corpus must be a real haystack
The metric only discriminates if finding the gold doc is genuinely hard:

| corpus | what happens |
|--------|--------------|
| only the gold docs (≈10s of docs) | every model scores ≈1.0 — **saturated, useless** |
| gold + thousands of distractors | a stronger model ranks the gold higher — **discriminative** |

This is the single most important thing to get right. Two levers raise difficulty:
**size** (more distractors) and **adjacency** (distractors on topics close to the
queries). Your in-house collection gives you both for free — use the real docs as the
haystack.

### 3. Formatting parity (train ≈ serve ≈ eval)
Text is embedded with the **same** `rag.core.formatting` rules used in serving and
training, so the numbers reflect production behaviour:
- **Query** → `Instruct: {QUERY_INSTRUCTION}\nQuery: {text}`
- **Document** → `{title}\n\n{text}` (title prepended; ids/urls excluded)

> ⚠️ **This parity must extend to *your own* serving stack** (Elasticsearch, hybrid +
> rerank, …). If your production pipeline embeds text differently than the lab, these
> numbers won't transfer to prod — the most common way a fine-tune disappoints. See
> **[serving-parity.md](serving-parity.md)**.

### 4. Cosine over normalized embeddings, in-memory
Every doc and query is embedded, L2-normalized, and scored by dot product (= cosine).
The ranking runs **in-process with numpy**, and with the Ollama backend no torch is
loaded. Only the top-`N` results per query are kept (N = max cutoff = 10), so memory stays
bounded on large corpora.

### 5. Which model is measured = whatever you configure
`rag-eval` embeds with the backend selected by env (`EMBEDDER`, `ST_MODEL`,
`EMBED_MODEL`, …). To compare models, **re-run with different env** against the same
`EVAL_DIR`. Each run embeds independently, so models with different dimensions are still
comparable (cosine is computed within each run).

### 6. Queries without judgments are skipped
If a query id has no `qrels` row, it can't be scored and is dropped from the averages.

---

## Why dense-only is the right measurement (even with a hybrid + rerank stack)

In production you may search with **BM25 + dense + a reranker**. So why does this lab score
the **dense embedder alone**? Because that's the only component you're changing.

### Variable isolation
A fine-tune changes exactly one thing — the dense embedder. To measure *its* effect you
hold everything else constant and measure that one variable. Dense-only retrieval over the
corpus does exactly that. Measure end-to-end (BM25 + dense + rerank) instead and:

- the BM25 leg and the reranker **mask or mix in** the embedder's change — you can't
  *attribute* a delta to the embedder;
- a small embedder gain is often **absorbed by the reranker**, so it never shows up in the
  end-to-end number.

So dense-only isn't a shortcut or a limitation — it's the **correct experimental design**
for this change.

### Two evals, two questions
| Question | How to measure | Use |
|----------|----------------|-----|
| *Did the embedder get better?* | this lab's dense-only eval | fast, attributable — **iteration / tuning** |
| *Did end-to-end search get better?* | your hybrid + rerank **A/B** in production | slow — **final ship decision** |

While fine-tuning, the first is the one to watch; the second is the last gate before you
ship. A lab gain is **necessary** (no lab gain → don't expect an end-to-end gain) but **not
sufficient** — confirm with the A/B.

### What makes it *meaningful* (3 conditions)
The method is correct; meaningfulness depends on what you feed it:
1. **A real domain eval set** — real queries + their relevant docs + your **real corpus as
   the haystack**, not the toy sample. (Skip this and you're "measuring nothing, correctly.")
2. **`recall@k` at your reranker's candidate depth** — with a reranker downstream, the
   embedder's job is *recall* (get the gold doc into the top-k the reranker then reorders),
   not final precision. Watch `recall@50/100`, not `nDCG@10`.
3. **Formatting parity with serving** — the lab's input formatting must match your ES
   index/query pipeline or the numbers won't transfer (see [serving-parity.md](serving-parity.md)).

**Bottom line:** dense-only isolation is the textbook way to answer "did the embedder
improve?" — exactly the question you're asking while fine-tuning. Fill in a real domain
eval set and measure `recall@(rerank depth)`, and the number is trustworthy; then confirm
the production win with an end-to-end A/B.

## Metrics

All averaged over the judged queries; definitions follow the BEIR / `trec_eval`
conventions, so they're comparable to published numbers.

| metric | question it answers | formula |
|--------|--------------------|---------|
| **recall@k** (k = 1, 3, 5, 10) | are the relevant docs in the top-k? | `|relevant ∩ top-k| / |relevant|` |
| **MRR@10** | how high is the *first* relevant doc? | mean of `1 / rank_of_first_relevant` (0 if none in top-10) |
| **nDCG@10** | are relevant docs ranked *high* (graded)? | `DCG@10 / IDCG@10`, gain = qrels score, discount = `1/log2(rank+1)` |

`recall@1` is the strictest ("gold ranked #1"); `nDCG@10` is BEIR's headline number.

---

## Bringing your in-house data

1. **Export three files** into a directory, in the layout above:
   - `corpus.jsonl` — **all** your documents (this is the haystack; include everything,
     not just the answers).
   - `queries.jsonl` — your evaluation queries.
   - `qrels/test.tsv` — for each query, the doc(s) that answer it (`score 1`).
   - IDs are arbitrary strings but must be **consistent** across the three files.
2. **Point `rag-eval` at it and run:**
   ```bash
   EVAL_DIR=/path/to/your/eval uv run rag-eval
   ```
3. **Tips for a trustworthy measurement:**
   - More queries ⇒ tighter estimates. Aim for **≥100** judged queries if you can.
   - Put your **real corpus** in `corpus.jsonl` as distractors — topical adjacency is
     what gives the metric teeth.
   - Label honestly: if a query genuinely has several relevant docs, list them all
     (otherwise a correct retrieval is scored as a miss — a false negative).

No code changes are needed — the format is the contract.

---

## Comparing models (the actual experiment)

The point of measuring is to compare. Run the same corpus through each model — **with
the same backend** — and read the deltas:

```bash
# baseline — the ORIGINAL checkpoint via sentence-transformers
EMBEDDER=sentence-transformers ST_MODEL=Qwen/Qwen3-Embedding-0.6B uv run rag-eval

# candidate — the fine-tuned model via sentence-transformers
EMBEDDER=sentence-transformers ST_MODEL=outputs/embedding-ft uv run rag-eval
```

Keeping the backend fixed is what makes the Δ attributable to the fine-tune: the Ollama
build (GGUF quantisation, llama.cpp pooling/truncation) and the HF fp32 path score the
*same weights* slightly differently, and on a small query set that gap can rival the
effect you're measuring. An `EMBEDDER=ollama` run is still useful — as a **serving-path
parity check** against the ST run of the same model, not as the training baseline.

A fine-tune "helped" if its recall@1 / nDCG@10 go **up** on the *same* `EVAL_DIR` — by
more than the reported 95% CI suggests noise can explain. `rag-eval` (and the API)
bootstrap-resample the per-query scores to print that interval; with ~50 queries one
query is worth ~2 points of recall@1, so treat overlapping intervals as "no evidence",
not as a win. (Run on a couple of datasets — your domain set + a standard BEIR set like
SciFact or NFCorpus — so you see both in-domain lift and whether general ability
regressed.)

> **Not the same as training's in-loop eval.** `rag-train` prints an
> `InformationRetrievalEvaluator` score *during* training as a progress signal, computed
> over the small `data/test.jsonl` pairs. That is a sanity gauge, **not** this
> measurement. The trustworthy number is `rag-eval` over a real haystack.

---

## The sample data (`rag-gen-eval`)

`rag-gen-eval` writes **placeholder** data so the harness runs out of the box:

- **16 gold docs** — real one-line tech explainers (asyncio, Postgres FTS, Docker, …),
  each with 3 queries → **48 queries**.
- **448 distractors** — 28 subjects × 16 operational aspects (`Kafka — scaling`,
  `Prometheus — backup`, …). Subjects are kept **disjoint** from the gold subjects but
  in the same tech register: plausible noise that answers no query.
- Total corpus ≈ **464 docs**, deterministically shuffled.

> **No leak with the toy training set.** The gold docs are shared with the toy training
> data (`rag-gen-data`), but the *queries* are not: the toy set learns from each topic's
> `train_queries` and the eval scores on disjoint `eval_queries` (both defined once in
> [`datagen/topics.py`](../src/rag/datagen/topics.py)). So a toy fine-tune is measured on
> phrasings it never saw — generalisation, not memorisation.

Observed with the base `qwen3-embedding:0.6b`:

```
recall@1 = 0.98   recall@3 = 1.00   mrr@10 = 0.99   nDCG@10 = 0.99
```

These are **high on purpose** — the sample distractors are easy (disjoint subjects, clean
labels), so a strong base model nearly aces them. That's the expected behaviour, and it's
why `recall@1` (the strict one) is the only metric that visibly moves. The sample proves
the **harness is correct** (it ranks over all 464 docs; `recall@1 < 1.0` means a
distractor did out-rank a gold doc once) — it is **not** a quality verdict on the model.
Real discrimination comes from your in-house corpus, where distractors are topically
adjacent. Cap the haystack while iterating with `N_DISTRACTORS=100 uv run rag-gen-eval`.

---

## The corpus mode (`EVAL_SOURCE=corpus`)

Once a real corpus exists (`rag-crawl`, or your own pages in `data/corpus.jsonl`), stop
synthesising distractors:

```bash
EVAL_SOURCE=corpus uv run rag-gen-eval
```

builds the eval set from reality instead — **every page of the site becomes the
haystack**, the held-out test split (`data/test.jsonl`) becomes the queries, and each
query's source page is its gold doc. Train-split pages sit in the corpus as natural
distractors, exactly like the production index, where every page is always retrievable.
The qrels are split into **dev** (tuning — sweeps and comparisons select here) and
**final** (one-shot confirmation of the chosen winner), same as the sample mode.

Two properties to know:

- **The test split is deliberately NOT round-trip-filtered.** `rag-gen-synthetic`
  filters the *train* split with the base embedder (a query must retrieve its own page
  into the top-k to survive); applying that filter to the eval queries would keep only
  the queries that embedder already gets right — every metric saturates at 1.0 and a
  fine-tune can only go *down*. Unfiltered eval queries carry some label noise, but the
  noise hits every compared model equally, so the **deltas stay valid**.
- **Regenerate in order.** The eval set joins test pairs back to pages by exact
  (title, content), so after a re-crawl the old pairs no longer match — re-run
  `rag-gen-synthetic`, then `rag-gen-eval` (the tools warn/refuse rather than silently
  shrinking the query set).

Observed with the base `qwen3-embedding:0.6b` on a 300-page Korean corpus (224 dev
queries): `recall@1 ≈ 0.87`, `recall@5 ≈ 0.97`, `nDCG@10 ≈ 0.93` — discriminative
headroom the sample set can't give you.

---

## Scaling notes

- **In-memory** ranking is comfortable to ~tens of thousands of docs. Each query keeps
  only its top-10, so memory is bounded by the corpus matrix (`#docs × dim × 4 bytes`).
- **Ollama** embeds the corpus in slices of 64 inputs per HTTP call (so one giant request
  can't hit the client timeout). For a very large corpus prefer the
  **sentence-transformers** backend (on-device, batched) or split the corpus.
- **Millions of docs?** Back the ranking with a vector database or ANN index instead of
  numpy — the `metrics` module is independent of how the ranking is produced, so only the
  ranking step changes.

---

## Gotchas

- **Don't build the corpus from only the answer docs.** That's the classic trap (and what
  an earlier version of this project did): the haystack collapses to ~10 docs and every
  metric pins at 1.0.
- **Multiple relevant docs:** list every relevant pair in qrels, or correct retrievals
  count as misses.
- **Formatting changes move serving and eval together** (shared `core.formatting`) — good
  for parity, but if you re-tune the template, re-run both.
