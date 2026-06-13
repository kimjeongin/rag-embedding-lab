// Module-scope crawl stream state — same shape as syntheticStore: the crawl is
// network-bound (minutes), so it must survive navigating away from the Data screen.
// One run at a time; POST /api/data/crawl/stream and parse SSE frames ourselves.
import { useSyncExternalStore } from "react";

import { readSSE } from "./sse";
import type { CrawlRequest } from "./types";

export type CrawlStatus = "idle" | "running" | "done" | "error";

export interface CrawlPage {
  url: string;
  title: string | null;
}
export interface CrawlState {
  status: CrawlStatus;
  mode: string; // sitemap | bfs — how URLs were discovered
  discovered: number; // frontier size at start
  done: number; // pages kept so far
  total: number; // the page budget (max_pages)
  pages: CrawlPage[]; // kept pages, in completion order
  result: { message: string; file: string; count: number; fetched: number; skipped: number } | null;
  error: string | null;
}

const INITIAL: CrawlState = {
  status: "idle",
  mode: "",
  discovered: 0,
  done: 0,
  total: 0,
  pages: [],
  result: null,
  error: null,
};

let state: CrawlState = INITIAL;
let ctrl: AbortController | null = null;
const listeners = new Set<() => void>();

function emit(next: CrawlState) {
  state = next;
  listeners.forEach((l) => l());
}

function reduce(s: CrawlState, event: string, p: Record<string, unknown>): CrawlState {
  switch (event) {
    case "start":
      return { ...s, mode: String(p.mode ?? ""), discovered: Number(p.discovered ?? 0), total: Number(p.max_pages ?? 0) };
    case "page":
      return {
        ...s,
        done: Number(p.done ?? s.done),
        total: Number(p.total ?? s.total),
        pages: [...s.pages, { url: String(p.url ?? ""), title: (p.title as string) ?? null }],
      };
    case "done":
      return {
        ...s,
        status: "done",
        result: {
          message: String(p.message ?? "완료"),
          file: String(p.file ?? ""),
          count: Number(p.count ?? 0),
          fetched: Number(p.fetched ?? 0),
          skipped: Number(p.skipped ?? 0),
        },
      };
    case "error":
      return { ...s, status: "error", error: String(p.detail ?? "크롤 오류") };
    default:
      return s;
  }
}

export async function startCrawl(body: CrawlRequest) {
  const before = state;
  if (before.status === "running") return; // one crawl at a time
  ctrl?.abort();
  const ac = new AbortController();
  ctrl = ac;
  emit({ ...INITIAL, status: "running" });

  try {
    await readSSE("/api/data/crawl/stream", body, ac.signal, (event, payload) =>
      emit(reduce(state, event, payload)),
    );
    if (state.status === "running") emit({ ...state, status: "done" });
  } catch (e) {
    if ((e as Error).name === "AbortError") return; // user navigated/reset
    emit({ ...state, status: "error", error: (e as Error).message });
  }
}

/** Clear the stream state (dismiss a finished/errored crawl). */
export function resetCrawl() {
  ctrl?.abort();
  emit(INITIAL);
}

const subscribe = (fn: () => void) => {
  listeners.add(fn);
  return () => {
    listeners.delete(fn);
  };
};

/** The full crawl-stream state — re-renders on every event (use on the Data screen). */
export function useCrawlState(): CrawlState {
  return useSyncExternalStore(subscribe, () => state);
}
