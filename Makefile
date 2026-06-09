# Project commands — run `make` (or `make help`) to list them.
# Backend = Python (uv) · Frontend = React/Vite (npm).
.DEFAULT_GOAL := help
FRONTEND := frontend
# Vite proxies /api → API_PORT (see frontend/vite.config.ts), so keep them in sync.
API_PORT ?= 8800

.PHONY: help install dev build serve test lint

help:  ## List available commands
	@grep -hE '^[a-zA-Z0-9_-]+:.*## ' $(MAKEFILE_LIST) \
	  | awk 'BEGIN{FS=":.*## "}{printf "  \033[36m%-9s\033[0m %s\n", $$1, $$2}'

install:  ## Install backend (uv, incl. training stack) + frontend (npm) deps
	uv sync --group training
	npm install --prefix $(FRONTEND)

dev:  ## Run the API + the Vite dev server together — Ctrl-C stops both
	@echo "▸ API  http://localhost:$(API_PORT)      ▸ UI  http://localhost:5273"
	@trap 'kill 0' INT TERM; \
	  RAG_PORT=$(API_PORT) uv run rag-serve & \
	  npm run dev --prefix $(FRONTEND) & \
	  wait

build:  ## Build the React app → frontend/dist
	npm run build --prefix $(FRONTEND)

serve: build  ## Production single-port: build the UI, then serve UI + API on :8000
	uv run rag-serve

test:  ## Run the backend tests (pytest)
	uv run pytest -q

lint:  ## Lint backend (ruff) + frontend (eslint)
	uvx ruff check src tests
	npm run lint --prefix $(FRONTEND)
