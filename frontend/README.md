# RAG Embedding Lab — web UI

The React front-end for the embedding lab: **Overview → Data → Train → Eval → Compare**.
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
- **Radix UI + cmdk + Sonner** — dialogs, the ⌘K command palette, and toasts (the *behavioural* pieces).
- **Hand-drawn SVG charts** — bar chart / loss curve / sparkline (`src/components/charts.tsx`), to match the design exactly.

```
src/
  lib/          types.ts · api.ts · queries.ts · format.ts · nav.tsx · useTrainStream.ts (SSE)
  components/   ui.tsx (primitives) · charts.tsx · Sidebar · Header · Modal · CommandPalette · DataTable
  routes/       Layout · Overview · Data · Train · Eval · Compare
  main.tsx      providers (Query + Router + Toaster) and the route table
```

Training streams over **Server-Sent Events** — `useTrainStream` POSTs to `/api/train` and
parses the `log` / `loss` / `metrics` / `done` frames off the response stream into live UI.
