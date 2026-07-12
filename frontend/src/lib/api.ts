// Typed fetch client for the lab API. One function per endpoint; thrown ApiError
// carries the backend's `detail` so the UI can show a real message.
import type {
  CorpusResponse,
  DataOverviewResponse,
  DeleteModelResponse,
  DeleteRunResponse,
  DiffResponse,
  Embedder,
  EvalRequest,
  EvalResponse,
  GenEvalRequest,
  GenEvalResponse,
  GenPairsRequest,
  GenPairsResponse,
  HandoffResponse,
  ImportPairsRequest,
  IndexJobStatus,
  IndexRequest,
  ImportPairsResponse,
  ImportTrecRequest,
  ImportTrecResponse,
  JobCreateRequest,
  JobsListResponse,
  JobState,
  LabelCommitRequest,
  LabelCommitResponse,
  LabelSearchRequest,
  LabelSearchResponse,
  ModelsDetailResponse,
  ModelsResponse,
  PairsResponse,
  PruneResponse,
  RunsResponse,
  SearchResponse,
  SearchStatusResponse,
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
  modelsDetail: () => request<ModelsDetailResponse>("/models/detail"),
  deleteModel: (path: string) =>
    request<DeleteModelResponse>(`/models?path=${encodeURIComponent(path)}`, { method: "DELETE" }),
  handoff: (path: string) => request<HandoffResponse>("/models/handoff", json({ path })),

  dataOverview: () => request<DataOverviewResponse>("/data/overview"),
  pairs: (limit = 8, content = false) => request<PairsResponse>(`/data/pairs?limit=${limit}&content=${content}`),
  corpus: (limit = 8, truncate?: number) =>
    request<CorpusResponse>(`/data/corpus?limit=${limit}${truncate ? `&truncate=${truncate}` : ""}`),

  genPairs: (body: GenPairsRequest) => request<GenPairsResponse>("/data/pairs", json(body)),
  genEval: (body: GenEvalRequest) => request<GenEvalResponse>("/data/eval", json(body)),
  importPairs: (body: ImportPairsRequest) => request<ImportPairsResponse>("/data/import", json(body)),
  labelSearch: (body: LabelSearchRequest) => request<LabelSearchResponse>("/data/label/search", json(body)),
  labelCommit: (body: LabelCommitRequest) => request<LabelCommitResponse>("/data/label/commit", json(body)),

  runEval: (body: EvalRequest) => request<EvalResponse>("/eval", json(body)),
  runs: () => request<RunsResponse>("/runs"),
  deleteRun: (id: string) => request<DeleteRunResponse>(`/runs/${encodeURIComponent(id)}`, { method: "DELETE" }),
  diff: (a: string, b: string, metric?: string) =>
    request<DiffResponse>(
      `/runs/diff?a=${encodeURIComponent(a)}&b=${encodeURIComponent(b)}${metric ? `&metric=${encodeURIComponent(metric)}` : ""}`,
    ),
  importTrec: (body: ImportTrecRequest) => request<ImportTrecResponse>("/runs/import-trec", json(body)),

  search: (query: string, topK = 10) => request<SearchResponse>("/search", json({ query, top_k: topK })),
  searchStatus: () => request<SearchStatusResponse>("/search/status"),
  startIndex: (body: IndexRequest) => request<IndexJobStatus>("/index", json(body)),
  indexStatus: () => request<IndexJobStatus>("/index/status"),
  setAlias: (collection: string) => request<SearchStatusResponse>("/index/alias", json({ collection })),
  pruneCollections: () => request<PruneResponse>("/index/prune", { method: "POST" }),

  jobs: () => request<JobsListResponse>("/jobs"),
  job: (id: string) => request<JobState>(`/jobs/${encodeURIComponent(id)}`),
  createJob: (body: JobCreateRequest) => request<JobState>("/jobs", json(body)),
  stopJob: (id: string) => request<JobState>(`/jobs/${encodeURIComponent(id)}/stop`, { method: "POST" }),
  skipRun: (id: string) => request<JobState>(`/jobs/${encodeURIComponent(id)}/skip`, { method: "POST" }),
};
