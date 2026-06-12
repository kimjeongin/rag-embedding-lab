# Project commands — run `make` (or `make help`) to list them.
# Backend = Python (uv) · Frontend = React/Vite (npm).
.DEFAULT_GOAL := help
FRONTEND := frontend
# Vite proxies /api → API_PORT (see frontend/vite.config.ts), so keep them in sync.
API_PORT ?= 8800

.PHONY: help install run dev build stop clean reset test lint

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
