# RAG Embedding Lab

Fine-tune and measure dense-retrieval **embedding models** — generate data → train →
evaluate → compare — end to end, on a laptop.

- **Generate** — toy or LLM-synthesised training pairs, plus a BEIR-format eval set.
- **Train** — fine-tune an embedding model (sentence-transformers; macOS MPS / Linux CUDA / CPU).
- **Evaluate** — recall@k / MRR / nDCG over a needle-in-haystack BEIR eval set.
- **Compare** — stack runs side by side to see which model retrieves better.

A **React web app** drives the whole loop; every step is also a plain CLI command.

Stack: **Python 3.13 + uv** · **Ollama** (`qwen3-embedding:0.6b`) · **sentence-transformers** ·
**FastAPI** · **React** (Vite + Tailwind). Dependencies locked in `uv.lock`.

> 🚀 **Quickstart:** `uv sync` · `npm install --prefix frontend && npm run build --prefix frontend`
> · `uv run rag-serve` → http://localhost:8000 (UI + API on one port).
> No vector database to stand up — evaluation ranks the corpus in-memory.

## Commands

Every runnable command is a console script (declared in `pyproject.toml`, implemented
as a thin entrypoint in [`rag/cli/`](src/rag/cli)):

| Command | What it does |
|---------|--------------|
| `uv run rag-serve` | 🟢 serve the lab **API** (`/api/*`) + the built **React UI** (one port) |
| `uv run rag-gen-data` | write a toy fine-tuning dataset |
| `uv run rag-gen-synthetic` | write an LLM-generated **training** dataset (+ hard negatives) |
| `uv run rag-gen-eval` | write a sample **BEIR-format eval set** (`data/eval`) |
| `uv run rag-train` | fine-tune the embedding model |
| `uv run rag-eval` | measure retrieval quality over a BEIR-format set (recall@k / MRR / nDCG) |

`rag-serve` (API + UI) is a long-running server; the rest are batch tools that run and
exit. The web UI lives in [`frontend/`](frontend) — see [Web UI](#web-ui). For everyday
use, a [`Makefile`](#3-run) wraps these (`make run` / `make dev` / `make help`).

## How it works

### Asymmetric embeddings (Qwen3)
Qwen3-Embedding treats documents and queries differently:

| Side | What we embed | `Embedder` method |
|------|---------------|-------------------|
| Document | `"{title}\n\n{content}"` (title prepended; no instruction prefix) | `embed_documents` (input built by `format_document`) |
| Query | `Instruct: {task}\nQuery: {query}` | `embed_queries` |

The asymmetry is defined once in [`core/formatting.py`](src/rag/core/formatting.py)
(`format_query` / `format_document`) and applied by the embedder adapters, so it can't be
mixed up — and **training, evaluation, and inference all reuse the same module** (parity).
The `{task}` comes from `QUERY_INSTRUCTION`.

> **Serving the fine-tuned model in your own stack (Elasticsearch, a hybrid + rerank
> pipeline, …)?** Your serving pipeline must embed text with this **exact same formatting**,
> or the fine-tune won't transfer (lab score up, production flat/down — the #1 prod
> disappointment). See **[docs/serving-parity.md](docs/serving-parity.md)**.

### The loop
1. **Generate** training pairs (`{query, positive[, negatives]}`) and a BEIR eval set
   (`corpus` + `queries` + `qrels`).
2. **Train** — contrastive fine-tuning (MultipleNegativesRankingLoss; in-batch + mined
   hard negatives) on macOS/Linux/CPU.
3. **Evaluate** — embed the whole eval corpus + queries with the **same** formatting, rank
   by cosine in-memory (numpy), score against the qrels.
4. **Compare** — every eval is recorded; the UI stacks runs and highlights the winner.

## Quick start

### 1. Pull the embedding model (Ollama)
```bash
ollama pull qwen3-embedding:0.6b
# sanity check: should print 1024
curl -s http://localhost:11434/api/embed \
  -d '{"model":"qwen3-embedding:0.6b","input":"hello"}' \
  | python3 -c "import sys,json; print(len(json.load(sys.stdin)['embeddings'][0]))"
```

### 2. Install dependencies (uv)
Python is pinned to **3.13** (`.python-version`); uv installs it if missing. The project is
a `src/` package (`rag`); `uv sync` installs it editable.
```bash
uv sync                    # base: API + datagen + eval (no torch)
uv sync --group training   # add the training stack (torch, sentence-transformers) for rag-train
# bump everything to the latest compatible set later with:  uv lock --upgrade
```

### 3. Run

A `Makefile` wraps every everyday task — run `make` (or `make help`) to list them:

| Command | What it does |
|---------|--------------|
| `make install` | install backend (uv + training) + frontend (npm) deps — run once |
| `make run` | **use the lab** — build the UI, serve UI + API on one port → http://localhost:8000 |
| `make dev` | **develop the UI** — API + Vite (HMR) on two ports → http://localhost:5273 (Ctrl-C stops both) |
| `make stop` | stop any running lab servers (ports 8000 / 8800 / 5273) |
| `make build` | build the React app → `frontend/dist` |
| `make clean` | remove build artifacts (dist + caches); deps untouched |
| `make test` | run the backend tests (pytest) |
| `make lint` | ruff (Python) + eslint (frontend) |

Or drive it directly:
```bash
# build the React UI once, then serve UI + API together
npm install --prefix frontend && npm run build --prefix frontend
uv run rag-serve                       # http://localhost:8000   (host/port via RAG_HOST/RAG_PORT)

# …or skip the UI and drive the offline pipeline from the CLI (see below)
```

Configuration is via environment variables (all optional). See [`.env.example`](.env.example):

| Variable | Default | Purpose |
|----------|---------|---------|
| `EMBED_DIM` | `1024` | embedding dimension (must match the model) |
| `QUERY_INSTRUCTION` | `Given a web search query, retrieve relevant passages that answer the query` | Qwen3 query task description |
| `EMBEDDER` | `ollama` | backend: `ollama` or `sentence-transformers` (a fine-tuned model) |
| `OLLAMA_URL` | `http://localhost:11434` | Ollama base URL (`ollama` backend) |
| `EMBED_MODEL` | `qwen3-embedding:0.6b` | embedding model name (`ollama` backend) |
| `ST_MODEL` | `outputs/embedding-ft` | model path/name (`sentence-transformers` backend) |
| `ST_DEVICE` | `` (auto) | `cuda`/`mps`/`cpu` (`sentence-transformers` backend) |

## Web UI

The React app ([`frontend/`](frontend)) wraps the **same `rag.*` offline loop** (generate
data → train → evaluate → compare) — a thin client with no business logic of its own.
Bring your in-house eval set ([`docs/evaluation.md`](docs/evaluation.md)) and the
eval/compare screens measure real models.

A single-page studio: an **Overview** dashboard (champion + leaderboard + nDCG trend),
**Data** (generate/preview training pairs & the BEIR eval set), **Train** (live SSE loss
curve + before/after nDCG), **Eval** (auto-dim model scoring with Δ-vs-best), and
**Compare** (grouped metric bars + best-per-metric table). ⌘K command palette, toasts,
focus-trapped dialogs.

**Stack:** Vite · React + TypeScript · Tailwind v4 · TanStack Query (server state) ·
React Router · Radix UI + cmdk + Sonner (behaviour) · hand-drawn SVG charts (data-viz).
The design system, data layer, and screens live in
`frontend/src/{components,lib,routes}`. It talks to the **lab API** (`/api/*`); there is no
vector store — evaluation ranks in-memory, so only Ollama is needed (when you run an eval).

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

Lab support code is framework-free (keeping the route handlers thin): the run registry in
[`rag/runs.py`](src/rag/runs.py), environment/model introspection in
[`rag/lab.py`](src/rag/lab.py), and training-log parsing in
[`rag/trainlog.py`](src/rag/trainlog.py).

## Fine-tuning & evaluation (offline / CLI)

The offline ML side is small packages run via `cli/` — `datagen` (make data) → `training`
(fine-tune) → `evaluation` (measure) — all sharing `core.formatting`. Contrastive
fine-tuning with **sentence-transformers**, on **macOS (MPS)**, **Ubuntu (CUDA)**, or
**CPU** (device auto-detected). The training deps are isolated in an optional group:
```bash
uv sync --group training
uv run rag-gen-data        # toy training pairs → data/train.jsonl + test.jsonl
                           #   (or rag-gen-synthetic: LLM-written queries + mined hard negatives)
uv run rag-gen-eval        # sample BEIR-format eval set → data/eval/  (replace with your data)
uv run rag-eval            # baseline recall@k / MRR / nDCG over data/eval/
uv run rag-train           # fine-tune (default: Qwen/Qwen3-Embedding-0.6B) → in-loop eval + save
```

**Two distinct datasets — don't conflate them:**
- **Training pairs** — `data/train.jsonl` / `test.jsonl`: `{query, positive[, negatives]}`
  records that `rag-train` learns from (`datagen` writes them; format below).
- **Eval set** — `data/eval/` in **BEIR format** (`corpus` + `queries` + `qrels`): a big
  haystack `rag-eval` ranks over. **This is the real performance measurement** — its data
  contract and the experiment's assumptions live in
  **[`docs/evaluation.md`](docs/evaluation.md)**. Bring your in-house data in that layout
  and `rag-eval` runs unchanged.

**Measuring a fine-tune** = run `rag-eval` on each model over the same eval set — with
the **same backend**, so the Δ is the fine-tune and not a quantisation/pooling difference
between stacks — and compare the deltas (recall@1 / nDCG@10 up = it helped). The
Eval/Compare screens do this interactively; from the CLI:
```bash
EMBEDDER=sentence-transformers ST_MODEL=Qwen/Qwen3-Embedding-0.6B uv run rag-eval  # base
EMBEDDER=sentence-transformers ST_MODEL=outputs/embedding-ft      uv run rag-eval  # fine-tuned
```
> ⚠️ The sample eval set's distractors are deliberately *easy* (so a strong base model
> scores ~0.98) — it proves the harness, not model quality. Real discrimination comes
> from your in-house corpus. See [`docs/evaluation.md`](docs/evaluation.md).

**Training-pair format** (JSONL, one record per line — written by `datagen`, read by
`training`; the eval set is separate, see [`docs/evaluation.md`](docs/evaluation.md)):
```json
{"query": "...", "positive": {"title": "...", "content": "..."},
 "negatives": [{"title": "...", "content": "..."}]}
```
`negatives` is optional; when every record has one, training adds it as an explicit
(anchor, positive, negative) triplet on top of in-batch negatives. Bring your own data in
this format (point `TRAIN_FILE`/`TRAIN_EVAL_FILE` at it) or put documents in
`data/corpus.jsonl` and run `rag-gen-synthetic`. Key env: `TRAIN_BASE_MODEL`,
`TRAIN_EPOCHS`, `TRAIN_BATCH_SIZE`, `TRAIN_DEVICE`, `TRAIN_METHOD` (`full` or `lora` —
LoRA adapters are merged into the base on save), `GEN_MODEL`, `HARD_NEGATIVES`.

## Architecture

Layered after Clean Architecture's **dependency rule**: dependencies point inward, toward
`core`. The inner layers know nothing about the web framework, the ML stack, or the HTTP
client — the offline pipeline talks to an `Embedder` **port**, and the concrete adapter
(Ollama or sentence-transformers) is chosen at the edge.

```
src/rag/
├── core/             innermost — pure, no config/framework/driver imports
│   ├── entities.py   Document  (typed, frozen)
│   ├── ports.py      Embedder  (Protocol — the boundary the lab depends on)
│   ├── formatting.py format_query / format_document  (asymmetry, one place)
│   └── errors.py
├── embeddings/       Embedder adapters: ollama.py · sentence_transformer.py · factory.py (build_embedder)
├── datagen/          topics.py (shared 16-topic set) · dummy.py (toy) · synthetic.py (LLM + hard negatives) · eval_corpus.py (BEIR sample)
├── training/         config · data · model · train  (contrastive fine-tuning)
├── evaluation/       beir.py (corpus/queries/qrels IO) · retrieval.py (embed + in-memory rank) · metrics.py
├── api/              lab HTTP API (FastAPI) + composition root
│   ├── app.py        create_app(): Settings + lab routes + serve frontend/dist
│   └── deps.py · errors.py · schemas/lab.py · routes/lab/ (status·models·data·runs·evaluate·train SSE)
├── cli/              entrypoints (thin): serve · gen_data · gen_synthetic · gen_eval · train · evaluate
├── runs.py           eval-run registry (JSONL) ·  lab.py  env/model introspection ·  trainlog.py  log parsing
├── dataset.py        shared training-pair JSONL IO (datagen ↔ training)
└── config.py         Settings (injected; from_env at the root)

frontend/             React lab (Vite + TS + Tailwind) — see frontend/README.md
tests/                pure: formatting · settings · lab · eval-metrics · beir · runs · trainlog · datagen
                      + the SSE route (a fake subprocess) — no vector DB / torch needed
docs/evaluation.md    the eval data contract + experiment assumptions (read before measuring)
data/                 corpus.jsonl · train/test.jsonl (training pairs) · eval/ (BEIR-format eval set)
```

A CI-style check enforces the rule: importing `core` + `evaluation` pulls in **zero** web
frameworks (`fastapi`/`starlette`/…). That's also why the tests run fast with an in-memory
fake `Embedder` — no Ollama, no torch, no vector DB.

## Tests
```bash
uv run pytest            # pure domain + the offline pipeline via in-memory fakes
```

## License

[MIT](LICENSE) © 2026 Jeongin Kim
