// Subscribe to POST /api/train's Server-Sent Events. EventSource is GET-only, so we
// POST with fetch and parse the SSE frames off the response stream ourselves.
import { useCallback, useRef, useState } from "react";

import type { LossPoint, TrainMetrics, TrainRequest } from "./types";

export type TrainStatus = "idle" | "running" | "done" | "error";

export interface TrainState {
  status: TrainStatus;
  log: string[];
  loss: LossPoint[];
  metrics: TrainMetrics;
  exitCode: number | null;
  error: string | null;
}

const INITIAL: TrainState = {
  status: "idle",
  log: [],
  loss: [],
  metrics: { before: null, after: null },
  exitCode: null,
  error: null,
};

function reduce(state: TrainState, event: string, payload: Record<string, unknown>): TrainState {
  switch (event) {
    case "start":
      return { ...state, log: [...state.log, `$ ${String(payload.cmd ?? "")}`] };
    case "log":
      return { ...state, log: [...state.log, String(payload.line ?? "")] };
    case "loss":
      return { ...state, loss: [...state.loss, payload as unknown as LossPoint] };
    case "metrics":
      return { ...state, metrics: payload as unknown as TrainMetrics };
    case "done":
      return { ...state, status: "done", exitCode: Number(payload.exit_code) };
    case "error":
      return { ...state, status: "error", error: String(payload.detail ?? "training error") };
    default:
      return state;
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

export function useTrainStream() {
  const [state, setState] = useState<TrainState>(INITIAL);
  const ctrl = useRef<AbortController | null>(null);

  const start = useCallback(async (body: TrainRequest) => {
    ctrl.current?.abort();
    const ac = new AbortController();
    ctrl.current = ac;
    setState({ ...INITIAL, status: "running" });

    try {
      const res = await fetch("/api/train", {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify(body),
        signal: ac.signal,
      });
      if (!res.ok || !res.body) throw new Error(`HTTP ${res.status} — 학습을 시작할 수 없습니다`);

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
          setState((s) => reduce(s, event, payload));
        }
      }
      // stream closed without an explicit done event
      setState((s) => (s.status === "running" ? { ...s, status: "done" } : s));
    } catch (e) {
      if ((e as Error).name === "AbortError") return; // user stopped — keep what we have
      setState((s) => ({ ...s, status: "error", error: (e as Error).message }));
    }
  }, []);

  const stop = useCallback(() => {
    ctrl.current?.abort();
    setState((s) => (s.status === "running" ? { ...s, status: "idle" } : s));
  }, []);

  return { ...state, start, stop };
}
