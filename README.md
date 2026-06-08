# RAG Embedding Lab

Train, evaluate, and serve **dense-retrieval embedding models** — end to end, on a laptop.

- **Train** — fine-tune an embedding model (sentence-transformers; macOS MPS / Linux CUDA / CPU).
- **Evaluate** — measure retrieval quality (recall@k / MRR / nDCG) over a BEIR-format eval set.
- **Compare** — stack runs side by side to see which model retrieves better.
- **Serve** — a FastAPI + Qdrant dense-retrieval API (`/search`) using whichever model you pick.
- **Web UI** — a Gradio app that drives the whole loop (data → train → eval → compare); no CLI needed.

Stack: **Python 3.13 + uv** · **Ollama** (`qwen3-embedding:0.6b`) · **sentence-transformers** ·
**Qdrant** · **FastAPI** · **Gradio**. Dependencies locked in `uv.lock`.

> 🚀 **Easiest start:** `uv sync --group ui --group training`, then `uv run rag-ui` → http://127.0.0.1:7860
>
> The vector store sits behind a small `VectorStore` port (this project started on
> pgvector and moved to Qdrant by writing one adapter — see [Architecture](#architecture)).

## Commands

Every runnable command is a console script (declared in `pyproject.toml`, implemented
as a thin entrypoint in [`rag/cli/`](src/rag/cli)):

| Command | What it does |
|---------|--------------|
| `uv run rag-serve` | 🟢 run the HTTP **API** + serve the built **React UI** (one port) |
| `uv run rag-gen-data` | write a toy fine-tuning dataset |
| `uv run rag-gen-synthetic` | write an LLM-generated **training** dataset (+ hard negatives) |
| `uv run rag-gen-eval` | write a sample **BEIR-format eval set** (`data/eval`) |
| `uv run rag-train` | fine-tune the embedding model |
| `uv run rag-eval` | measure retrieval quality over a BEIR-format set (recall@k / MRR / nDCG) |
| `uv run rag-ui` | **web UI** (Gradio, legacy) — superseded by the React lab in `frontend/` |

`rag-serve` (API + UI) and `rag-ui` (Gradio) are long-running servers; the rest are batch
tools that run and exit. The **React lab** (`frontend/`) is the primary UI — see [Web UI](#web-ui).

## How it works

### Asymmetric embeddings (Qwen3)
Qwen3-Embedding treats documents and queries differently:

| Side | What we embed | `OllamaEmbedder` method |
|------|---------------|------------------------|
| Document (indexing) | `"{title}\n\n{content}"` (title prepended; no instruction prefix) | `embed_documents` (input built by `format_document`) |
| Query (search) | `Instruct: {task}\nQuery: {query}` | `embed_query` |

The asymmetry is defined once in [`core/formatting.py`](src/rag/core/formatting.py)
(`format_query` / `format_document`) and applied by the
[`OllamaEmbedder`](src/rag/embeddings/ollama.py) adapter, so it can't be mixed up
— and the training code reuses the same module. The `{task}` comes from
`QUERY_INSTRUCTION`.

### Structured page metadata
The base unit is the **page** (its URL). `/documents` accepts `url` and `title`;
the server parses the URL ([`core/urls.py`](src/rag/core/urls.py)) and stores in the
point payload's `metadata`: `url`, `domain` (host only — `example.com`), `path`
(`/docs/x`), and `title`.

> **Identifiers stay out of the embedding.** Only `title` + `content` go into the
> embedding input. `url`/`domain`/`path` are filter/group metadata (used by
> `site` mode), never part of the similarity space.

### Search
The [`QdrantVectorStore`](src/rag/stores/qdrant.py) runs the one nearest-neighbour
query — cosine search over the `documents` collection; Qdrant returns each point's
cosine `score` directly as the similarity. The `SearchDocuments` use case
([`usecases/search.py`](src/rag/usecases/search.py)) embeds the query, calls the
store, then applies one of two pure post-processings:
- **`page`** — page-level top-k, optional `max_per_domain` diversity cap.
- **`site`** — pages grouped by `domain` into ranked sites (`site_score` = best
  page similarity).

### Collection setup
On startup the app calls `ensure_collection()` (retrying until Qdrant is ready):
it creates the `documents` collection if missing — `size = EMBED_DIM`,
`distance = Cosine`. No extensions, no SQL, no migrations.

> ⚠️ **Reindex needed if the embedding input format changes** (e.g. title
> prepending). Drop + recreate the collection, then re-POST every page:
> ```bash
> curl -X DELETE http://localhost:6333/collections/documents
> # restart the app (recreates the empty collection), then re-POST to /documents
> ```

## Storage model

Each page is one Qdrant **point**:

```
point {
  id:      1, 2, 3, ...          # sequential, assigned on insert
  vector:  float[1024]           # embedding of "{title}\n\n{content}" (cosine)
  payload: {
    content:  "<raw page body>",          # title is NOT in content
    metadata: { url, domain, path, title, ...extras }
  }
}
```

`url`/`domain`/`path` live in the payload (filter/group fields) but are kept out of
the embedded text — only `title` + `content` are embedded.

Qdrant is schemaless — structure lives in the point payload, so adding metadata
fields needs no migration.

---

## Quick start

### 1. Pull the embedding model (Ollama)
```bash
ollama pull qwen3-embedding:0.6b
# sanity check: should print 1024
curl -s http://localhost:11434/api/embed \
  -d '{"model":"qwen3-embedding:0.6b","input":"hello"}' \
  | python3 -c "import sys,json; print(len(json.load(sys.stdin)['embeddings'][0]))"
```

### 2. Start Qdrant
```bash
docker compose up -d
# Qdrant is up when this returns 200:
curl -s -o /dev/null -w "%{http_code}\n" http://localhost:6333/readyz
# dashboard: http://localhost:6333/dashboard
```
The app creates the `documents` collection itself on startup (retrying until
Qdrant is ready), so there's nothing else to provision.

### 3. Install dependencies (uv)
Python is pinned to **3.13** (`.python-version`); uv installs it if missing.
The project is a `src/` package (`rag`); `uv sync` installs it editable.
Dependencies are locked in `uv.lock`.
```bash
uv sync            # creates .venv (Python 3.13), installs locked deps + the rag package
# bump everything to the latest compatible set later with:  uv lock --upgrade
```

### 4. Run the server
```bash
uv run rag-serve                              # console script (host/port via RAG_HOST/RAG_PORT)
# or, with autoreload for development:
uv run uvicorn rag.api.app:app --reload
# -> http://localhost:8000  (interactive docs at /docs)
```

Configuration is via environment variables (all optional; defaults match the
docker-compose above). See [`.env.example`](.env.example).

| Variable | Default | Purpose |
|----------|---------|---------|
| `QDRANT_URL` | `http://localhost:6333` | Qdrant base URL |
| `QDRANT_COLLECTION` | `documents` | collection name |
| `EMBED_DIM` | `1024` | embedding dimension (must match the model & the collection's vector size) |
| `QUERY_INSTRUCTION` | `Given a web search query, retrieve relevant passages that answer the query` | Qwen3 query task description |
| `EMBEDDER` | `ollama` | backend: `ollama` or `sentence-transformers` (serve a fine-tuned model) |
| `OLLAMA_URL` | `http://localhost:11434` | Ollama base URL (`ollama` backend) |
| `EMBED_MODEL` | `qwen3-embedding:0.6b` | embedding model name (`ollama` backend) |
| `ST_MODEL` | `outputs/embedding-ft` | model path/name (`sentence-transformers` backend) |
| `ST_DEVICE` | `` (auto) | `cuda`/`mps`/`cpu` (`sentence-transformers` backend) |

---

## API

### `GET /health`
```bash
curl -s http://localhost:8000/health | python3 -m json.tool
```
```json
{
  "status": "ok",
  "embed_model": "qwen3-embedding:0.6b",
  "embed_dim": 1024,
  "vector_store": "connected",
  "document_count": 10
}
```

### `POST /documents`
Send `url` + `title` + `content` per page; the server derives `domain`/`path`.
```bash
curl -s http://localhost:8000/documents \
  -H 'Content-Type: application/json' \
  -d '{
    "documents": [
      {"url": "https://docs.python.org/3/library/asyncio.html", "title": "asyncio — Asynchronous I/O", "content": "asyncio is a library to write concurrent code using async/await ..."},
      {"url": "https://realpython.com/async-io-python/", "title": "Async IO in Python", "content": "Learn asynchronous programming in Python using async and await ..."},
      {"url": "https://fastapi.tiangolo.com/async/", "title": "Concurrency and async / await", "content": "FastAPI lets you write endpoints with async def ..."}
    ]
  }' | python3 -m json.tool
```
```json
{ "ids": [1, 2, 3], "count": 3 }
```
Stored metadata for the first page (domain/path auto-derived):
```json
{ "url": "https://docs.python.org/3/library/asyncio.html",
  "domain": "docs.python.org", "path": "/3/library/asyncio.html",
  "title": "asyncio — Asynchronous I/O" }
```

### `POST /search` — `mode: "page"` (default)
Page-level top-k. Optional `max_per_domain` caps results per domain for diversity
(omit / `null` = pure similarity order).
```bash
curl -s http://localhost:8000/search \
  -H 'Content-Type: application/json' \
  -d '{"query": "asynchronous programming with async and await in Python",
       "top_k": 5, "mode": "page", "max_per_domain": 1}' \
  | python3 -m json.tool
```
```json
{
  "mode": "page",
  "query": "asynchronous programming with async and await in Python",
  "top_k": 5,
  "results": [
    {"id": 4, "url": "https://realpython.com/async-io-python/", "title": "Async IO in Python", "domain": "realpython.com", "content": "Learn asynchronous ...", "similarity": 0.772},
    {"id": 1, "url": "https://docs.python.org/3/library/asyncio.html", "title": "asyncio — Asynchronous I/O", "domain": "docs.python.org", "content": "asyncio is a library ...", "similarity": 0.676},
    {"id": 8, "url": "https://fastapi.tiangolo.com/async/", "title": "Concurrency and async / await", "domain": "fastapi.tiangolo.com", "content": "FastAPI lets you ...", "similarity": 0.633}
  ]
}
```
> Without the cap, two `docs.python.org` + two `realpython.com` pages fill the
> top-5; `max_per_domain: 1` keeps one per domain and lets other sites in.

### `POST /search` — `mode: "site"`
Pull a `fetch_k` (default 50) candidate pool, group by `domain`, score each site
by its best page (`site_score`), return `top_k` sites with their pages.
```bash
curl -s http://localhost:8000/search \
  -H 'Content-Type: application/json' \
  -d '{"query": "asynchronous programming with async and await in Python",
       "top_k": 3, "mode": "site", "fetch_k": 50}' \
  | python3 -m json.tool
```
```json
{
  "mode": "site",
  "query": "asynchronous programming with async and await in Python",
  "top_k": 3,
  "fetch_k": 50,
  "results": [
    {"domain": "realpython.com", "site_score": 0.772, "pages": [
      {"url": "https://realpython.com/async-io-python/", "title": "Async IO in Python", "similarity": 0.772},
      {"url": "https://realpython.com/python-type-checking/", "title": "Python Type Checking Guide", "similarity": 0.282}
    ]},
    {"domain": "docs.python.org", "site_score": 0.676, "pages": [
      {"url": "https://docs.python.org/3/library/asyncio.html", "title": "asyncio — Asynchronous I/O", "similarity": 0.676},
      {"url": "https://docs.python.org/3/library/typing.html", "title": "typing — Support for type hints", "similarity": 0.306},
      {"url": "https://docs.python.org/3/library/dataclasses.html", "title": "dataclasses — Data Classes", "similarity": 0.255}
    ]},
    {"domain": "fastapi.tiangolo.com", "site_score": 0.633, "pages": [
      {"url": "https://fastapi.tiangolo.com/async/", "title": "Concurrency and async / await", "similarity": 0.633}
    ]}
  ]
}
```

---

## Architecture

Layered after Clean Architecture's **dependency rule**: dependencies point inward,
toward `core`. The inner layers know nothing about the web framework, the DB
driver, or the HTTP client.

```
   entrypoints (delivery):   api/  (FastAPI server)      cli/  (rag-serve, rag-train,
                             — composition root                rag-eval, rag-gen-*)
                                     │ call
        ┌─────────── usecases (application) ───────────┐   IndexDocuments · SearchDocuments
        │        depend only on core PORTS              │   (no framework / driver imports)
        └───────────────────────────────────────────────┘
                          │ depends on
        ┌──────────────────── core ─────────────────────┐  innermost, pure, dependency-light
        │  entities · ports (Embedder, VectorStore)      │  ← adapters implement these ports
        │  ranking · formatting · urls · errors          │
        └───────────────────────────────────────────────┘
                          ▲ implement ports
        ┌─────────────── adapters (driven) ─────────────┐
        │  embeddings/{OllamaEmbedder, SentenceTransformerEmbedder}   stores/QdrantVectorStore
        └───────────────────────────────────────────────┘

   offline ML (batch, run via cli/): datagen ─train pairs▶ training/   ·   evaluation/ ranks a BEIR set
                                     all reuse core.formatting (train / serve / eval parity)
```

A CI-style check enforces it: importing `core` + `usecases` pulls in **zero**
frameworks (`fastapi`/`httpx`/`qdrant_client`/…). That's also why the use-case
tests run fast with in-memory fakes — no vector DB or Ollama needed. (Swapping
pgvector → Qdrant was exactly one new adapter under `stores/` — the use cases,
API, ranking, and tests were untouched.)

```
src/rag/
├── core/             innermost — pure, no config/framework/driver imports
│   ├── entities.py   Document · EmbeddedDocument · Hit · Site (typed, frozen)
│   ├── ports.py      Embedder · VectorStore (Protocols — the boundaries)
│   ├── ranking.py    apply_max_per_domain · site_score · group_by_site (pure)
│   ├── formatting.py format_query / format_document  (asymmetry, one place)
│   ├── urls.py · errors.py
├── usecases/         application — depends only on core ports
│   ├── indexing.py   IndexDocuments  (embed → derive metadata → store)
│   └── search.py     SearchDocuments (embed → store.search → rank)
├── embeddings/       Embedder adapters (picked per Settings.embedder)
│   ├── ollama.py · sentence_transformer.py · factory.py (build_embedder)
├── stores/qdrant.py  QdrantVectorStore → implements VectorStore (SQL-free, raw client)
├── api/              HTTP delivery (FastAPI) + composition root
│   ├── app.py        create_app(): build Settings/adapters, wire use cases, lifespan
│   ├── deps.py · errors.py · schemas/ (DTOs) · routes/ (thin)
├── cli/              ALL entrypoints (thin): serve · gen_data · gen_synthetic · gen_eval · train · evaluate · webui
├── webui/            web UI delivery (Gradio): app.py (cards/tabs) · actions.py (glue) · theme.py · runs.py (registry) · jobs.py
├── datagen/          offline: dummy.py (toy) · synthetic.py (LLM + hard negatives) · eval_corpus.py (BEIR sample)
├── training/         offline: config · data · model · train (fine-tuning)
├── evaluation/       offline: beir.py (corpus/queries/qrels IO) · retrieval.py (embed + rank) · metrics.py (recall@k / MRR / nDCG)
├── dataset.py        shared training-pair JSONL IO (datagen ↔ training)
└── config.py         serving Settings (injected; from_env at the root)

tests/                pure: ranking · urls · formatting · settings · qdrant-mapping · eval-metrics · beir · runs · webui-data
                      + use cases & datagen with in-memory fakes (no DB/Ollama/torch)
docs/evaluation.md    the eval data contract + experiment assumptions (read before measuring)
docker-compose.yml · pyproject.toml · uv.lock · .python-version (3.13)
data/                 corpus.jsonl · train/test.jsonl (training pairs) · eval/ (BEIR-format eval set)
```

### Embedding fine-tuning + evaluation (offline)
The offline ML side is small packages run via `cli/` — `datagen` (make data) →
`training` (fine-tune) → `evaluation` (measure) — all sharing `core.formatting`.
Contrastive fine-tuning with **sentence-transformers**, on **macOS (MPS)**,
**Ubuntu (CUDA)**, or **CPU** (device auto-detected). Deps are isolated in an optional
group so the server stays lean:
```bash
uv sync --group training
uv run rag-gen-data        # toy training pairs → data/train.jsonl + test.jsonl
                           #   (or rag-gen-synthetic: LLM-written queries + mined hard negatives)
uv run rag-gen-eval        # sample BEIR-format eval set → data/eval/  (replace with your data)
uv run rag-eval            # baseline recall@k / MRR / nDCG over data/eval/
uv run rag-train           # fine-tune (default: Qwen/Qwen3-Embedding-0.6B) → in-loop eval + save
```
Everything reuses `rag.core.formatting`, so the model is trained, served, **and**
evaluated on exactly the same text shape (parity).

**Two distinct datasets — don't conflate them:**
- **Training pairs** — `data/train.jsonl` / `test.jsonl`: `{query, positive[, negatives]}`
  records that `rag-train` learns from (`datagen` writes them; format below).
- **Eval set** — `data/eval/` in **BEIR format** (`corpus` + `queries` + `qrels`): a big
  haystack `rag-eval` ranks over. **This is the real performance measurement** — its data
  contract and the experiment's assumptions live in
  **[`docs/evaluation.md`](docs/evaluation.md)**. Bring your in-house data in that layout
  and `rag-eval` runs unchanged.

**Measuring a fine-tune** = run `rag-eval` on each model over the same eval set, compare
the deltas (recall@1 / nDCG@10 up = it helped):
```bash
EMBEDDER=ollama                EMBED_MODEL=qwen3-embedding:0.6b uv run rag-eval
EMBEDDER=sentence-transformers ST_MODEL=outputs/embedding-ft   uv run rag-eval
```
> ⚠️ The sample eval set's distractors are deliberately *easy* (so a strong base model
> scores ~0.98) — it proves the harness, not model quality. Real discrimination comes
> from your in-house corpus. See [`docs/evaluation.md`](docs/evaluation.md).

**Serve the fine-tuned model** through the same `/search` — flip one env var (the
`SentenceTransformerEmbedder` adapter is already wired; use cases/API/Qdrant
unchanged). Use a fresh collection since a new model means a new vector space:
```bash
EMBEDDER=sentence-transformers ST_MODEL=outputs/embedding-ft \
QDRANT_COLLECTION=documents_ft uv run rag-serve
# re-POST documents (re-embedded by the fine-tuned model), then /search
```
> ⚠️ A new model is a new vector space → use a fresh collection and re-index.

**Training-pair format** (JSONL, one record per line — written by `datagen`, read by
`training`; the eval set is separate, see [`docs/evaluation.md`](docs/evaluation.md)):
```json
{"query": "...", "positive": {"title": "...", "content": "..."},
 "negatives": [{"title": "...", "content": "..."}]}
```
`negatives` is optional; when every record has one, training adds it as an explicit
(anchor, positive, negative) triplet on top of in-batch negatives. Bring your own data
in this format (point `TRAIN_FILE`/`TRAIN_EVAL_FILE` at it) or put documents in
`data/corpus.jsonl` and run `rag-gen-synthetic`. Key env: `TRAIN_BASE_MODEL`,
`TRAIN_EPOCHS`, `TRAIN_BATCH_SIZE`, `TRAIN_DEVICE`, `GEN_MODEL`, `HARD_NEGATIVES`.

## Web UI

Two delivery layers wrap the **same `rag.*` offline loop** (generate data → train →
evaluate → compare). Both are thin — no business logic of their own. Bring your in-house
eval set ([`docs/evaluation.md`](docs/evaluation.md)) and the eval/compare screens measure
real models.

### React lab (primary) — [`frontend/`](frontend)

A single-page studio: an **Overview** dashboard (champion + leaderboard + nDCG trend),
**Data** (generate/preview training pairs & the BEIR eval set), **Train** (live SSE loss
curve + before/after nDCG), **Eval** (auto-dim model scoring with Δ-vs-best), and
**Compare** (grouped metric bars + best-per-metric table). ⌘K command palette, toasts,
focus-trapped dialogs.

**Stack:** Vite · React + TypeScript · Tailwind v4 · TanStack Query (server state) ·
React Router · Radix UI + cmdk + Sonner (behaviour) · hand-drawn SVG charts (data-viz).
The design system, data layer, and screens live in
`frontend/src/{components,lib,routes}`. It talks to the **lab API** (`/api/*`); Qdrant is
**not** required for the lab — only the search server (`/search`, `/documents`) needs it.

```bash
npm install --prefix frontend          # one-time: JS deps

# dev (two terminals): API on :8800, Vite HMR on :5273 (proxies /api → :8800)
RAG_PORT=8800 uv run rag-serve
npm run dev --prefix frontend          # http://localhost:5273

# production single-port: build the SPA, then rag-serve mounts it at /
npm run build --prefix frontend
uv run rag-serve                       # http://localhost:8000  (UI + /api together)
```

#### Lab API (`/api/*`)
The React app is a thin client over these (FastAPI, [`rag/api/routes/lab/`](src/rag/api/routes/lab)):

| Endpoint | Purpose |
|----------|---------|
| `GET /api/status` | Ollama / device / eval-set / training-ready / run-count |
| `GET /api/models?embedder=` | model list + a sensible default |
| `GET /api/data/{overview,pairs,corpus}` | dataset counts + previews |
| `POST /api/data/{pairs,eval}` | (re)generate training pairs / the BEIR eval set |
| `POST /api/eval` | score a model → append to the run registry → metrics + Δ |
| `GET`·`DELETE /api/runs[/{id}]` | list (with best-per-metric) / delete a run |
| `POST /api/train` | fine-tune, streamed live over **SSE** (log / loss / metrics / done) |

Lab support code is framework-free and shared with the Gradio UI: the run registry in
[`rag/runs.py`](src/rag/runs.py), environment/model introspection in
[`rag/lab.py`](src/rag/lab.py), and training-log parsing in
[`rag/trainlog.py`](src/rag/trainlog.py).

### Gradio (legacy) — [`rag/webui/`](src/rag/webui)
The original click-through, kept for the no-Node path (`uv sync --group ui`):
```bash
uv run rag-ui                 # http://127.0.0.1:7860   (UI_HOST / UI_PORT to change)
```
Same four-step loop in a card layout. The React lab supersedes it.

## Tests
```bash
uv run pytest            # pure domain logic + use cases via in-memory fakes (no DB/Ollama)
```

## Next step: hybrid search (not implemented yet)

The ports make lexical + dense fusion drop in without reshaping anything. With
Qdrant there are two clean routes:

- **Qdrant-native** — add a **sparse vector** (BM25/SPLADE) alongside the dense one
  in each point, then let Qdrant fuse them server-side in one Query API call
  (`prefetch` + built-in **RRF**). Only `QdrantVectorStore` changes.
- **In-process** — extend `VectorStore` with `lexical_search(query, limit)` returning
  the same `Hit` entities, add a pure `reciprocal_rank_fusion(*ranked_lists)` in
  `core.ranking`, and fuse inside `SearchDocuments` *before* the page/site step.

Either way the use cases depend on the `VectorStore` port and `page`/`site` stay pure
ranking over `Hit` entities, so fusion slots in under both modes without touching the
API — and `site_score` is already isolated for swapping in e.g. a top-N average.

## License

[MIT](LICENSE) © 2026 Jeongin Kim
