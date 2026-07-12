# RAG Embedding Lab — web UI

The React front-end for the embedding lab, tabs in pipeline order:
**개요 → 데이터 → 학습 → 평가 → 실험 → 모델 → 검색 → 소개**.
A thin client over the lab API (`/api/*`); see the [root README](../README.md#web-ui) for
the full picture and run instructions.

## Run

```bash
npm install                 # deps (once)

# dev: API on :8800, this on :5273 with HMR (vite proxies /api → :8800)
#   (in another terminal, from the repo root)  RAG_PORT=8800 uv run rag-serve
npm run dev                 # http://localhost:5273

npm run build               # → dist/ ; `uv run rag-serve` then serves it at / (one port)
npm run lint                # eslint
```

## Stack & layout

- **Vite + React + TypeScript + Tailwind v4** — design tokens (ink + lime `#c6f24a`) in `src/index.css`.
- **TanStack Query** — all server state (`src/lib/queries.ts`); components never call `fetch` directly.
- **React Router** — `src/routes/` (a `Layout` shell + one file per screen).
- **Radix UI + Sonner** — dialogs and toasts (the *behavioural* pieces).
- **Hand-drawn SVG charts** — bar chart / loss curve / sparkline (`src/components/charts.tsx`), to match the design exactly.

```
src/
  lib/          types.ts (wire DTOs) · api.ts · queries.ts · format.ts · nav.tsx (steps/routes/copy)
                sse.ts · sweep.ts · crawlStore.ts · syntheticStore.ts
  components/   ui/ (primitives) · charts.tsx · diagrams.tsx (About) · Sidebar · Header · Modal
                DataTable · DiffView
  routes/       Layout · Overview · Data · Train · Eval · Compare · Models · Search · About
  main.tsx      providers (Query + Router + Toaster) and the route table
```

Conventions worth knowing:

- **Server-owned jobs** — training/reindex run on the server; the UI only polls
  (`useJobs`/`useIndexStatus`, fast while running, lazy otherwise). Closing the tab never
  kills a run. Data generation streams over SSE (`lib/sse.ts`).
- **Header = world state** — the process embedder + model, Qdrant liveness, device,
  eval-set fingerprint, best score, running jobs. Anything that would silently change
  what a number means is surfaced there.
- **Guards render as UI** — different-fingerprint runs are dimmed out of the leaderboard,
  dim/model mismatches show amber pills *before* you waste an embed pass, destructive
  actions (model delete, collection prune) use a 3-second armed two-click.
