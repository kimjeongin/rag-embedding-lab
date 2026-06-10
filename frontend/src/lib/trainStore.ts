// Module-scope training-stream state. A fine-tune outlives any one screen: the Train
// route unmounting must not orphan the stream or make a running job look dead, so the
// state lives here (not in component state) and components subscribe via
// useSyncExternalStore. One run at a time — mirrors the server's 409 guard.
//
// EventSource is GET-only, so we POST /api/train with fetch and parse the SSE frames
// off the response stream ourselves. Aborting the fetch disconnects the stream, which
// the server takes as "stop": it kills the training subprocess.
import { useSyncExternalStore } from "react";

import { readSSE } from "./sse";
import type { LossPoint, TrainMetrics, TrainRequest } from "./types";

export type TrainStatus = "idle" | "running" | "done" | "stopped" | "error";

export interface TrainState {
  status: TrainStatus;
  log: string[];
  loss: LossPoint[];
  metrics: TrainMetrics;
  exitCode: number | null;
  error: string | null;
  outputDir: string | null; // where the finished model was saved (from the done event)
}

const INITIAL: TrainState = {
  status: "idle",
  log: [],
  loss: [],
  metrics: { before: null, after: null },
  exitCode: null,
  error: null,
  outputDir: null,
};

let state: TrainState = INITIAL;
let ctrl: AbortController | null = null;
const listeners = new Set<() => void>();

function emit(next: TrainState) {
  state = next;
  listeners.forEach((l) => l());
}

function reduce(s: TrainState, event: string, payload: Record<string, unknown>): TrainState {
  switch (event) {
    case "start":
      return { ...s, log: [...s.log, `$ ${String(payload.cmd ?? "")}`] };
    case "log":
      return { ...s, log: [...s.log, String(payload.line ?? "")] };
    case "loss":
      return { ...s, loss: [...s.loss, payload as unknown as LossPoint] };
    case "metrics":
      return { ...s, metrics: payload as unknown as TrainMetrics };
    case "done":
      return {
        ...s,
        status: "done",
        exitCode: Number(payload.exit_code),
        outputDir: typeof payload.output_dir === "string" ? payload.output_dir : s.outputDir,
      };
    case "error":
      return { ...s, status: "error", error: String(payload.detail ?? "training error") };
    default:
      return s;
  }
}

export async function startTraining(body: TrainRequest) {
  // Snapshot for the guard — narrowing the module `state` here would blind TS to
  // emit() reassigning it during the stream below.
  const before = state;
  if (before.status === "running") return; // the server would 409 anyway

  ctrl?.abort();
  const ac = new AbortController();
  ctrl = ac;
  emit({ ...INITIAL, status: "running" });

  try {
    await readSSE("/api/train", body, ac.signal, (event, payload) => emit(reduce(state, event, payload)));
    // stream closed without an explicit done event
    if (state.status === "running") emit({ ...state, status: "done" });
  } catch (e) {
    if ((e as Error).name === "AbortError") return; // user stopped — stop() set the state
    emit({ ...state, status: "error", error: (e as Error).message });
  }
}

/** Abort the stream — the server kills the training subprocess on disconnect. */
export function stopTraining() {
  ctrl?.abort();
  if (state.status === "running") {
    emit({
      ...state,
      status: "stopped",
      log: [...state.log, "— 사용자가 중단했습니다 (서버의 학습 프로세스 종료)"],
    });
  }
}

const subscribe = (fn: () => void) => {
  listeners.add(fn);
  return () => {
    listeners.delete(fn);
  };
};

/** The full stream state — re-renders on every event (use on the Train screen). */
export function useTrainState(): TrainState {
  return useSyncExternalStore(subscribe, () => state);
}

/** Just the status — re-renders only when it changes (cheap for Sidebar/Header). */
export function useTrainStatus(): TrainStatus {
  return useSyncExternalStore(subscribe, () => state.status);
}
