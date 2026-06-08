// Typed fetch client for the lab API. One function per endpoint; thrown ApiError
// carries the backend's `detail` so the UI can show a real message.
import type {
  CorpusResponse,
  DataOverviewResponse,
  DeleteRunResponse,
  Embedder,
  EvalRequest,
  EvalResponse,
  GenEvalRequest,
  GenEvalResponse,
  GenPairsRequest,
  GenPairsResponse,
  ModelsResponse,
  PairsResponse,
  RunsResponse,
  StatusResponse,
} from "./types";

const BASE = "/api";

export class ApiError extends Error {
  status: number;
  constructor(status: number, message: string) {
    super(message);
    this.name = "ApiError";
    this.status = status;
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let res: Response;
  try {
    res = await fetch(BASE + path, {
      headers: { "content-type": "application/json" },
      ...init,
    });
  } catch (e) {
    throw new ApiError(0, `네트워크 오류 — 서버가 실행 중인가요? (${(e as Error).message})`);
  }
  if (!res.ok) {
    let detail = res.statusText;
    try {
      const body = await res.json();
      if (body?.detail) detail = typeof body.detail === "string" ? body.detail : JSON.stringify(body.detail);
    } catch {
      // non-JSON error body — keep statusText
    }
    throw new ApiError(res.status, detail);
  }
  return (await res.json()) as T;
}

const json = (body: unknown): RequestInit => ({ method: "POST", body: JSON.stringify(body) });

export const api = {
  status: () => request<StatusResponse>("/status"),
  models: (embedder: Embedder) => request<ModelsResponse>(`/models?embedder=${encodeURIComponent(embedder)}`),

  dataOverview: () => request<DataOverviewResponse>("/data/overview"),
  pairs: (limit = 8, content = false) => request<PairsResponse>(`/data/pairs?limit=${limit}&content=${content}`),
  corpus: (limit = 8, truncate?: number) =>
    request<CorpusResponse>(`/data/corpus?limit=${limit}${truncate ? `&truncate=${truncate}` : ""}`),

  genPairs: (body: GenPairsRequest) => request<GenPairsResponse>("/data/pairs", json(body)),
  genEval: (body: GenEvalRequest) => request<GenEvalResponse>("/data/eval", json(body)),

  runEval: (body: EvalRequest) => request<EvalResponse>("/eval", json(body)),
  runs: () => request<RunsResponse>("/runs"),
  deleteRun: (id: string) => request<DeleteRunResponse>(`/runs/${encodeURIComponent(id)}`, { method: "DELETE" }),
};
