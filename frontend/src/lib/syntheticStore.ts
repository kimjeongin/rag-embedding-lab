// Module-scope synthetic-generation stream state — mirrors trainStore. Generation is
// LLM-bound (tens of seconds), so it must not be orphaned by navigating away from the
// Data screen: the state lives here (not in component state) and components subscribe via
// useSyncExternalStore. One run at a time.
//
// EventSource is GET-only, so we POST /api/data/pairs/stream with fetch and parse the SSE
// frames off the response stream ourselves. Aborting the fetch disconnects the stream,
// which cancels the server-side generator on its next yield.
import { useSyncExternalStore } from "react";

import type { GenPairsRequest } from "./types";

export type SynthStatus = "idle" | "running" | "done" | "error";

export interface SynthDoc {
  title: string | null;
  queries: string[];
}
export interface SynthState {
  status: SynthStatus;
  total: number; // corpus docs
  done: number; // docs finished
  thinkingDisabled: boolean; // reasoning model? we turned thinking off for speed
  docs: SynthDoc[]; // finished docs, in completion order
  mining: boolean; // hard-negative mining phase
  result: { message: string; train: number; test: number } | null;
  error: string | null;
}

const INITIAL: SynthState = {
  status: "idle",
  total: 0,
  done: 0,
  thinkingDisabled: false,
  docs: [],
  mining: false,
  result: null,
  error: null,
};

let state: SynthState = INITIAL;
let ctrl: AbortController | null = null;
const listeners = new Set<() => void>();

function emit(next: SynthState) {
  state = next;
  listeners.forEach((l) => l());
}

function reduce(s: SynthState, event: string, p: Record<string, unknown>): SynthState {
  switch (event) {
    case "start":
      return { ...s, total: Number(p.docs ?? 0), thinkingDisabled: Boolean(p.thinking_disabled) };
    case "doc":
      return {
        ...s,
        done: Number(p.done ?? s.done),
        total: Number(p.total ?? s.total),
        docs: [...s.docs, { title: (p.title as string) ?? null, queries: (p.queries as string[]) ?? [] }],
      };
    case "mining":
      return { ...s, mining: true };
    case "done":
      return {
        ...s,
        status: "done",
        mining: false,
        result: {
          message: String(p.message ?? "완료"),
          train: Number((p.train as { count?: number })?.count ?? 0),
          test: Number((p.test as { count?: number })?.count ?? 0),
        },
      };
    case "error":
      return { ...s, status: "error", error: String(p.detail ?? "생성 오류") };
    default:
      return s;
  }
}

function parseFrame(frame: string): { event: string; data: string } {
  let event = "message";
  let data = "";
  for (const line of frame.split("\n")) {
    if (line.startsWith("event:")) event = line.slice(6).trim();
    else if (line.startsWith("data:")) data += line.slice(5).trim();
  }
  return { event, data };
}

export async function startSynthetic(body: GenPairsRequest) {
  // Snapshot for the guard — narrowing the module `state` here would blind TS to
  // emit() reassigning it during the stream below (same as trainStore).
  const before = state;
  if (before.status === "running") return; // one run at a time
  ctrl?.abort();
  const ac = new AbortController();
  ctrl = ac;
  emit({ ...INITIAL, status: "running" });

  try {
    const res = await fetch("/api/data/pairs/stream", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify(body),
      signal: ac.signal,
    });
    if (!res.ok || !res.body) {
      let detail = `HTTP ${res.status}`;
      try {
        const j = await res.json();
        if (j?.detail) detail = typeof j.detail === "string" ? j.detail : JSON.stringify(j.detail);
      } catch {
        // non-JSON error body — keep the status line
      }
      throw new Error(detail);
    }

    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      let sep: number;
      while ((sep = buffer.indexOf("\n\n")) >= 0) {
        const frame = buffer.slice(0, sep);
        buffer = buffer.slice(sep + 2);
        const { event, data } = parseFrame(frame);
        if (!data) continue;
        let payload: Record<string, unknown>;
        try {
          payload = JSON.parse(data) as Record<string, unknown>;
        } catch {
          continue;
        }
        emit(reduce(state, event, payload));
      }
    }
    // stream closed without an explicit done/error
    if (state.status === "running") emit({ ...state, status: "done" });
  } catch (e) {
    if ((e as Error).name === "AbortError") return; // user navigated/reset
    emit({ ...state, status: "error", error: (e as Error).message });
  }
}

/** Clear the stream state (e.g. to dismiss a finished/errored run). */
export function resetSynthetic() {
  ctrl?.abort();
  emit(INITIAL);
}

const subscribe = (fn: () => void) => {
  listeners.add(fn);
  return () => {
    listeners.delete(fn);
  };
};

/** The full synthetic-stream state — re-renders on every event (use on the Data screen). */
export function useSyntheticState(): SynthState {
  return useSyncExternalStore(subscribe, () => state);
}
