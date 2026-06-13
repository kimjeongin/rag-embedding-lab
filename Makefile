# Project commands — run `make` (or `make help`) to list them.
# Backend = Python (uv) · Frontend = React/Vite (npm).
.DEFAULT_GOAL := help
FRONTEND := frontend
# Vite proxies /api → API_PORT (see frontend/vite.config.ts), so keep them in sync.
API_PORT ?= 8800

# ── PoC data pipeline defaults (override per-invocation: `make crawl CRAWL_URL=…`) ──
CRAWL_URL ?= https://www.korea.kr/sitemap_policy.xml
GEN_MODEL ?= qwen3.5:4b
N_QUERIES ?= 4
HARD_NEGATIVES ?= 4

.PHONY: help install run dev build stop clean reset test lint \
        crawl pairs evalset baseline train pipeline

help:  ## List available commands
	@grep -hE '^[a-zA-Z0-9_-]+:.*## ' $(MAKEFILE_LIST) \
	  | awk 'BEGIN{FS=":.*## "}{printf "  \033[36m%-9s\033[0m %s\n", $$1, $$2}'

install:  ## Install backend (uv, incl. training stack) + frontend (npm) deps
	uv sync --group training
	npm install --prefix $(FRONTEND)

run: build  ## Use the lab: build the UI + serve UI + API on one port → http://localhost:8000
	uv run rag-serve

dev:  ## Develop the UI: API + Vite (HMR) on two ports → http://localhost:5273 · Ctrl-C stops both
	@echo "▸ API  http://localhost:$(API_PORT)      ▸ UI  http://localhost:5273"
	@trap 'kill 0' INT TERM; \
	  RAG_PORT=$(API_PORT) uv run rag-serve & \
	  npm run dev --prefix $(FRONTEND) & \
	  wait

build:  ## Build the React app → frontend/dist
	npm run build --prefix $(FRONTEND)

stop:  ## Stop any running lab servers (ports 8000 / 8800 / 5273)
	@for p in 8000 $(API_PORT) 5273; do \
	  pids=$$(lsof -ti:$$p 2>/dev/null); \
	  if [ -n "$$pids" ]; then kill $$pids 2>/dev/null && echo "  stopped :$$p ($$pids)"; fi; \
	done; \
	echo "✓ no lab servers running"

clean:  ## Remove build artifacts (frontend/dist + Python caches) — keeps deps
	rm -rf $(FRONTEND)/dist .pytest_cache .ruff_cache
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	@echo "✓ cleaned build artifacts (deps untouched — 'make install' if needed)"

reset: clean  ## Remove ALL lab results: trained models (outputs/), eval runs (runs/), and restore data/ to last commit
	rm -rf outputs runs checkpoints
	git checkout HEAD -- data/
	git clean -fdq data/
	@echo "✓ reset: models + eval runs deleted, data/ restored to last commit"

test:  ## Run the backend tests (pytest)
	uv run pytest -q

lint:  ## Lint backend (ruff) + frontend (eslint)
	uvx ruff check src tests
	npm run lint --prefix $(FRONTEND)

# ── PoC data pipeline: crawl → pairs → eval set → baseline (README "PoC loop") ──

crawl:  ## Crawl CRAWL_URL (site root or sitemap.xml) → page-level data/corpus.jsonl
	uv run rag-crawl $(CRAWL_URL)

pairs:  ## LLM search-box queries → split → round-trip filter (train) → hard negatives
	GEN_MODEL=$(GEN_MODEL) N_QUERIES=$(N_QUERIES) HARD_NEGATIVES=$(HARD_NEGATIVES) \
	  uv run rag-gen-synthetic

evalset:  ## BEIR eval set from the crawled corpus — whole site = haystack (dev/final)
	EVAL_SOURCE=corpus uv run rag-gen-eval

baseline:  ## Measure the configured embedder on data/eval (recall@K / MRR / nDCG)
	uv run rag-eval

train:  ## Fine-tune the embedding model (TRAIN_* env; early stopping; → outputs/)
	uv run rag-train

pipeline: crawl pairs evalset baseline  ## Full PoC data pipeline (needs Ollama running)
