# RAG Embedding Lab

Fine-tune, measure, and **serve** dense-retrieval **embedding models** — generate data →
train → evaluate → compare → hand off → search — end to end, on a laptop.

- **Generate** — toy or LLM-synthesised training pairs, plus a BEIR-format eval set.
- **Train** — fine-tune an embedding model (sentence-transformers; macOS MPS / Linux CUDA / CPU).
- **Evaluate** — recall@k / MRR / nDCG over a needle-in-haystack BEIR eval set.
- **Compare** — per-query diffs + permutation p-values to pick a winner honestly.
- **Hand off** — package the winner (embedding contract + parity vectors) for serving.
- **Serve** — index the corpus into **Qdrant** (versioned collections, zero-downtime
  alias swap, instant rollback) and search it live from the UI.

A **React web app** drives the whole loop; every step is also a plain CLI command.

Stack: **Python 3.13 + uv** · **sentence-transformers** (the embedder everywhere: train/eval/serve) ·
**Ollama** (the LLM that writes synthetic queries; optional parity embedder) · **FastAPI** ·
**Qdrant** (serving only) · **React** (Vite + Tailwind). Dependencies locked in `uv.lock`.

> 🚀 **Quickstart:** `uv sync --group training` · `npm install --prefix frontend && npm run build
> --prefix frontend` · `uv run rag-serve` → http://localhost:8000 (UI + API on one port).
> No vector database needed to train/evaluate — evaluation ranks the corpus in-memory.
> Qdrant enters only when you serve (`make qdrant`).

## Commands

Every runnable command is a console script (declared in `pyproject.toml`, implemented
as a thin entrypoint in [`rag/cli/`](src/rag/cli)):

| Command | What it does |
|---------|--------------|
| `uv run rag-serve` | 🟢 serve the lab **API** (`/api/*`) + the built **React UI** (one port) |
| `uv run rag-crawl <url>` | crawl a public site (or one sitemap) → page-level `data/corpus.jsonl` |
| `uv run rag-gen-data` | write a toy fine-tuning dataset |
| `uv run rag-gen-synthetic` | write an LLM-generated **training** dataset (search-box queries, round-trip filtered, + margin-guarded hard negatives) |
| `uv run rag-gen-eval` | write a **BEIR-format eval set** (`data/eval`) — `EVAL_SOURCE=corpus` uses the crawled site as the haystack |
| `uv run rag-gen-intranet` | write the **가상 인트라넷 리허설 데이터셋** (`data-intranet/`) — 사내사이트 검색의 쌍둥이(페이지 url·description·agent prompt·수집 메타데이터) + 은어 양성 대조군(standard/jargon 슬라이스) |
| `uv run rag-train` | fine-tune the embedding model |
| `uv run rag-eval` | measure retrieval quality over a BEIR-format set (recall@k / MRR / nDCG) |
| `uv run rag-index` | embed `data/corpus.jsonl` into **Qdrant** (versioned collection + atomic alias swap; `--prune` drops rollback copies) |
| `uv run rag-search "질문"` | query the live Qdrant index from the CLI (serving smoke test) |

`rag-serve` (API + UI) is a long-running server; the rest are batch tools that run and
exit. The web UI lives in [`frontend/`](frontend) — see [Web UI](#web-ui). For everyday
use, a [`Makefile`](#3-run) wraps these (`make run` / `make dev` / `make help`), and the
PoC data pipeline has one-word targets: `make crawl` / `pairs` / `evalset` / `baseline` /
`train` — or `make pipeline` for the whole chain.

**가상 인트라넷 데이터셋으로 랩 전체를 돌리기** (실데이터 리허설 + 파이프라인의
격차 검출력 검증 — 은어는 corpus에 없고 학습쌍에만 있어 base는 못 맞히고 FT만
배울 수 있다):

```bash
uv run rag-gen-intranet   # data-intranet/ 생성 (corpus 195p · train 741 · eval 222q)
CORPUS_FILE=data-intranet/corpus.jsonl TRAIN_FILE=data-intranet/train.jsonl \
TRAIN_EVAL_FILE=data-intranet/test.jsonl EVAL_DIR=data-intranet/eval uv run rag-serve
```

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
1. **Data** — generate training pairs (`{query, positive[, negatives]}`) and a BEIR eval
   set (qrels split into **dev** for tuning and **final** for one-shot confirmation), or
   import real query/click logs (`POST /api/data/import`) and grow qrels with the
   built-in judging UI.
2. **Train** — server-owned **jobs** (a refresh never kills a run): single runs or
   **sweeps** — one axis × values, optionally co-varied with **LR** (2-axis, so a
   non-LR axis is judged at its own best LR), × seeds, run sequentially and
   auto-evaluated into a **seed-aggregated** live leaderboard (mean ± std, so single-run
   noise can't reorder it); opt-in **median pruning** stops clearly-losing runs early.
   Selectable loss (MNRL / cached MNRL / GIST / triplet), optional **Matryoshka**
   (truncatable vectors), backbone dropout, LoRA knobs, per-epoch validation with early
   stopping; the **best** epoch's weights are saved and the name says so (`…-mnrl-e7`,
   `-mrl` when Matryoshka); the full recipe + data fingerprints land in `train_meta.json`.
3. **Evaluate** — embed the whole eval corpus + queries with the **same** formatting,
   rank by cosine in-memory (numpy), score against the qrels. `recall@50` is the
   headline (in a hybrid+rerank pipeline the dense model is a candidate generator);
   `EVAL_TOP_K` aligns it to your production fusion depth. **But** recall@50 only
   discriminates on a *large* haystack (top-50 of a few-hundred-doc corpus is ~everything,
   so every model ties ~1.0) — at PoC scale select on **recall@5 / recall@1 / nDCG@10**.
   A Matryoshka (`-mrl`) model can be scored at a truncated dimension
   (`EMBED_TRUNCATE_DIM`, or the Eval tab's 차원 절단) to read its dim→quality curve.
4. **Compare** — every eval is recorded; pick two runs for a **paired per-query diff**
   (win/loss, sign-flip permutation p-value, topic slices, retrieved docs side by
   side). Import production BM25 as a TREC run to measure complementarity. Confirm the
   winner once on the held-out **final** split.
5. **Hand off** — the Models page packages the winner for the serving team
   (`HANDOFF.md` + `handoff.json`: embedding contract, parity sample vectors,
   reindex checklist). The lab does not deploy; production swaps the dense model
   inside its existing hybrid + rerank pipeline.
6. **Serve (optional, this repo)** — the lab can also serve the winner itself:
   sentence-transformers in-process + **Qdrant**. `make qdrant` → `make index
   SERVE_MODEL=outputs/…` → `POST /api/search`. Reindexing is idempotent and swaps a
   live alias atomically, so a model change is one command with zero downtime. See
   **[docs/serving.md](docs/serving.md)**.

## Quick start

### 1. Pull the Ollama models (for data generation)
Embedding runs in-process via sentence-transformers — Ollama's job here is the **LLM that
writes synthetic queries** in the data tab (`GEN_MODEL`). The embedding pull is only needed
if you use the optional `EMBEDDER=ollama` parity backend.
```bash
ollama pull qwen3.5:2b            # query-synthesis LLM (data tab / rag-gen-synthetic)
ollama pull qwen3-embedding:0.6b  # optional: the ollama parity embedder
```

### 2. Install dependencies (uv)
Python is pinned to **3.13** (`.python-version`); uv installs it if missing. The project is
a `src/` package (`rag`); `uv sync` installs it editable.
```bash
uv sync --group training   # API + datagen + eval + the training stack (torch, sentence-transformers)
uv sync                    # minimal alternative (no torch) — needs EMBEDDER=ollama, embedding via Ollama
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
| `make qdrant` | start a local **Qdrant** (docker, :6333) for the serving path |
| `make index` | embed the corpus into Qdrant (`SERVE_MODEL=outputs/…` picks the model) |
| `make serve-ft` | serve UI + API with the fine-tuned model (`SERVE_MODEL`) as the embedder |

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
| `EMBEDDER` | `sentence-transformers` | backend: `sentence-transformers` (default) or `ollama` (parity check) |
| `OLLAMA_URL` | `http://localhost:11434` | Ollama base URL (query-synthesis LLM; `ollama` backend) |
| `EMBED_MODEL` | `qwen3-embedding:0.6b` | embedding model name (`ollama` backend only) |
| `ST_MODEL` | `outputs/embedding-ft` | model path/name (`sentence-transformers` backend) |
| `ST_DEVICE` | `` (auto) | `cuda`/`mps`/`cpu` (`sentence-transformers` backend) |

The offline tools have their own knobs (also in `.env.example`): `CRAWL_MAX_PAGES`/`CRAWL_DELAY`/
`CRAWL_MIN_CHARS`/`CRAWL_MAX_CHARS` (rag-crawl), `GEN_MODEL`/`N_QUERIES`/`HARD_NEGATIVES`/
`ROUND_TRIP_K`/`NEG_MARGIN` (rag-gen-synthetic), `EVAL_SOURCE`/`EVAL_DIR`/`N_DISTRACTORS`
(rag-gen-eval), and the `TRAIN_*` family ([below](#fine-tuning--evaluation-offline--cli)).

## Web UI

The React app ([`frontend/`](frontend)) wraps the **same `rag.*` offline loop** (generate
data → train → evaluate → compare → hand off → serve) — a thin client with no business
logic of its own.
Bring your in-house eval set ([`docs/evaluation.md`](docs/evaluation.md)) and the
eval/compare screens measure real models.

A single-page studio, tabs in pipeline order: **개요** (champion + leaderboard + nDCG
trend), **데이터** (crawl a site into the corpus · generate/preview training pairs & the
BEIR eval set · import real query/click logs · a judging UI that grows qrels), **학습**
(single runs or sweeps, live SSE loss curve, per-epoch validation), **평가** (auto-dim
model scoring with Δ-vs-best, Matryoshka truncation, recent-runs history), **실험**
(grouped metric bars, per-query diff + permutation p-value, final confirmation), **모델**
(recipe shelf + handoff packaging — handoff auto-starts a background reindex), **검색**
(the serving console: index status with dim/model-mismatch guards, collection inventory
with instant alias rollback + prune, reindex with progress, live search with latency
split), and **소개** (a built-in textbook + report). The header always shows the world
state: the process embedder + model, Qdrant liveness, device, eval-set fingerprint,
best score, running jobs.

**Stack:** Vite · React + TypeScript · Tailwind v4 · TanStack Query (server state) ·
React Router · Radix UI + Sonner (behaviour) · hand-drawn SVG charts (data-viz).
The design system, data layer, and screens live in
`frontend/src/{components,lib,routes}`. It talks to the **lab API** (`/api/*`); training
and evaluation need no vector store (in-memory ranking) — Qdrant is used by the 검색 tab
only.

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
| `GET /api/status` | embedder / device / Ollama / Qdrant / eval-set / running jobs — the header's world state |
| `GET /api/models?embedder=` | model list + a sensible default |
| `GET /api/data/{overview,pairs,corpus}` | dataset counts + previews |
| `POST /api/data/{pairs,eval}` | (re)generate training pairs / the BEIR eval set |
| `POST /api/eval` | score a model (optional `truncate_dim`) → append to the run registry → metrics + Δ |
| `GET`·`DELETE /api/runs[/{id}]` | list (with best-per-metric) / delete a run |
| `POST`·`GET /api/jobs[/{id}]` | create/observe **server-owned** training jobs (single or sweep); poll state, `/stop`·`/skip` |
| `GET /api/models/detail` · `POST /api/models/handoff` | model shelf (recipes, sizes) · handoff package (+ auto-reindex hook) |
| `POST /api/search` · `GET /api/search/status` | search the live Qdrant index · index health (dim/model-match guards, collection family) |
| `POST /api/index[/alias\|/prune]` · `GET /api/index/status` | background reindex (409 while running) · instant alias rollback · drop rollback copies · job progress |

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

**PoC loop for a real site** (the internal-site-search stand-in — a public site becomes
the corpus; later, swap the source for the in-house site + click logs and nothing else
changes):
```bash
uv run rag-crawl https://www.korea.kr/sitemap_policy.xml   # pages → data/corpus.jsonl
GEN_MODEL=qwen3.5:4b N_QUERIES=4 HARD_NEGATIVES=4 uv run rag-gen-synthetic
                           # search-box-style queries (doc's language) → doc-level
                           # train/test split → round-trip filter (TRAIN ONLY — filtering
                           # test with the eval embedder would saturate every metric)
                           # → margin-guarded hard negatives
EVAL_SOURCE=corpus uv run rag-gen-eval   # eval set: the WHOLE site as the haystack,
                                         # held-out test queries as dev/final qrels
uv run rag-eval && uv run rag-train
```
Same thing via make — `make pipeline` runs crawl → pairs → evalset → baseline with these
defaults (override per-invocation: `make crawl CRAWL_URL=… `, `make pairs GEN_MODEL=…`).

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
ST_MODEL=Qwen/Qwen3-Embedding-0.6B uv run rag-eval  # base
ST_MODEL=outputs/embedding-ft      uv run rag-eval  # fine-tuned
```
> ⚠️ Eval discrimination is mostly a function of **haystack size**. The bundled sample
> is deliberately easy (a strong base model scores ~0.98) — it proves the harness, not
> model quality. A real crawl (`rag-crawl` + `EVAL_SOURCE=corpus`) helps, but note even a
> ~1500-doc corpus leaves **recall@50 saturated** (top-50 ≈ top 3%); recall@1/@5/nDCG@10
> are what move and what you should select on. For recall@50 itself to bite you need a
> haystack in the thousands+. See [`docs/evaluation.md`](docs/evaluation.md).

**Training-pair format** (JSONL, one record per line — written by `datagen`, read by
`training`; the eval set is separate, see [`docs/evaluation.md`](docs/evaluation.md)):
```json
{"query": "...", "positive": {"title": "...", "content": "..."},
 "negatives": [{"title": "...", "content": "..."}]}
```
`negatives` is optional; when every record has some, the MNRL/GIST losses train against
**all** of them as extra columns on top of in-batch negatives (TripletLoss takes exactly
one). Bring your own data in this format (point `TRAIN_FILE`/`TRAIN_EVAL_FILE` at it) or
put documents in `data/corpus.jsonl` and run `rag-gen-synthetic`. Key env:
`TRAIN_BASE_MODEL`, `TRAIN_EPOCHS` (a *ceiling* — early stopping ends sooner),
`TRAIN_PATIENCE` / `TRAIN_MONITOR` (`ndcg` or `loss`; the best epoch's weights are what
gets saved), `TRAIN_LOSS` (`mnrl` / `cached_mnrl` / `gist` / `triplet`),
`TRAIN_MATRYOSHKA`(+`TRAIN_MATRYOSHKA_DIMS` — truncatable vectors; memory-heavy, reduce
the dim count if it OOMs), `TRAIN_DROPOUT`, `TRAIN_SEED`, `TRAIN_NOTE`,
`TRAIN_BATCH_SIZE`, `TRAIN_DEVICE`, `TRAIN_METHOD` (`full` or `lora` — adapters are merged
into the base on save; `TRAIN_LORA_R/ALPHA/DROPOUT/TARGET` tune them), `EVAL_TOP_K`
(ranking depth — match your production fusion depth), `EMBED_TRUNCATE_DIM` (score a
Matryoshka model at a shorter prefix), `GEN_MODEL`, `N_QUERIES`, `HARD_NEGATIVES`,
`ROUND_TRIP_K`, `NEG_MARGIN`.

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
├── vectorstore/      qdrant.py — Qdrant over plain httpx REST (no client lib; every wire call visible)
├── datagen/          crawl.py (site → corpus) · topics.py (16-topic set) · dummy.py (toy) · synthetic.py (LLM queries + filters + hard negatives) · eval_corpus.py (BEIR sample) · eval_from_corpus.py (BEIR from the real corpus)
├── training/         config · data · model · train  (contrastive fine-tuning)
├── evaluation/       beir.py (corpus/queries/qrels IO) · retrieval.py (embed + in-memory rank) · metrics.py
├── serving.py        framework-free serving flow: versioned collections · atomic alias swap ·
│                     idempotent index_corpus · search · rollback (set_live) · prune (CLI + API share it)
├── api/              lab HTTP API (FastAPI) + composition root
│   ├── app.py        create_app(): Settings + lab routes + serve frontend/dist
│   ├── jobs.py       server-owned training jobs (single/sweep) · pruning.py (median pruning, pure) · hints.py (failure→fix)
│   ├── indexjob.py   the one background-reindex slot (handoff hook + POST /api/index)
│   └── deps.py · errors.py · schemas/lab.py · routes/lab/ (status·models·data·runs·jobs·evaluate·search)
├── cli/              entrypoints (thin): serve · crawl · gen_data · gen_synthetic · gen_eval · train · evaluate · index · search
├── runs.py           eval-run registry (JSONL) ·  lab.py  env/model introspection ·  trainlog.py  log parsing
├── dataset.py        shared training-pair JSONL IO (datagen ↔ training)
└── config.py         Settings (injected; from_env at the root)

frontend/             React lab (Vite + TS + Tailwind) — see frontend/README.md
tests/                pure: formatting · settings · lab · eval-metrics · beir · runs · trainlog · datagen
                      + serving/search/indexjob via in-memory fakes + the SSE route — no vector DB / torch needed
docs/evaluation.md    the eval data contract + experiment assumptions (read before measuring)
docs/serving.md       the serving path: Qdrant, versioned collections, rollback, automation
docs/serving-parity.md porting the embedding contract to another serving stack
docs/report.md        stakeholder report — why fine-tune, what was built, honest PoC numbers
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
