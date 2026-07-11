// TanStack Query hooks — the single source of server state. Components read these and
// never call `api` directly, so caching, refetching and invalidation live in one place.
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { toast } from "sonner";

import { api } from "./api";
import type {
  Embedder,
  EvalRequest,
  GenEvalRequest,
  GenPairsRequest,
  ImportPairsRequest,
  ImportTrecRequest,
  JobCreateRequest,
  LabelCommitRequest,
} from "./types";

const fail = (e: unknown) => toast.error((e as Error).message);

export const keys = {
  status: ["status"] as const,
  models: (e: Embedder) => ["models", e] as const,
  modelsDetail: ["models", "detail"] as const,
  dataOverview: ["data", "overview"] as const,
  pairs: ["data", "pairs"] as const,
  corpus: ["data", "corpus"] as const,
  runs: ["runs"] as const,
  diff: (a: string, b: string, metric?: string) => ["diff", a, b, metric ?? ""] as const,
  jobs: ["jobs"] as const,
  job: (id: string) => ["jobs", id] as const,
  searchStatus: ["search", "status"] as const,
  indexStatus: ["index", "status"] as const,
};

// ── reads ──────────────────────────────────────────────────────────────────────
export const useStatus = () =>
  useQuery({ queryKey: keys.status, queryFn: api.status, refetchInterval: 20_000 });

export const useModels = (embedder: Embedder) =>
  useQuery({ queryKey: keys.models(embedder), queryFn: () => api.models(embedder) });

export const useModelsDetail = () =>
  useQuery({ queryKey: keys.modelsDetail, queryFn: api.modelsDetail });

export const useDataOverview = () => useQuery({ queryKey: keys.dataOverview, queryFn: api.dataOverview });

export const usePairs = () => useQuery({ queryKey: keys.pairs, queryFn: () => api.pairs(8, false) });

export const useCorpus = () => useQuery({ queryKey: keys.corpus, queryFn: () => api.corpus(8, 120) });

export const useRuns = () => useQuery({ queryKey: keys.runs, queryFn: api.runs });

export const useDiff = (a?: string, b?: string, metric?: string) =>
  useQuery({
    queryKey: keys.diff(a ?? "", b ?? "", metric),
    queryFn: () => api.diff(a!, b!, metric),
    enabled: !!a && !!b,
    staleTime: 60_000,
  });

// Jobs are server-owned — the browser only polls. Fast while something runs,
// lazy otherwise; the list and the active job share one cadence.
export const useJobs = () =>
  useQuery({
    queryKey: keys.jobs,
    queryFn: api.jobs,
    refetchInterval: (q) => (q.state.data?.active ? 3_000 : 15_000),
  });

export const useJob = (id?: string | null) =>
  useQuery({
    queryKey: keys.job(id ?? ""),
    queryFn: () => api.job(id!),
    enabled: !!id,
    refetchInterval: (q) => {
      const s = q.state.data?.status;
      return s === "running" || s === "pending" ? 2_000 : false;
    },
  });

// The serving index (Qdrant) — cheap reads; the reindex job polls fast while running.
export const useSearchStatus = () =>
  useQuery({ queryKey: keys.searchStatus, queryFn: api.searchStatus, refetchInterval: 20_000 });

export const useIndexStatus = () =>
  useQuery({
    queryKey: keys.indexStatus,
    queryFn: api.indexStatus,
    refetchInterval: (q) => (q.state.data?.status === "running" ? 2_000 : false),
  });

// ── mutations (invalidate what they change) ──────────────────────────────────────
export function useGenPairs() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (body: GenPairsRequest) => api.genPairs(body),
    onError: fail,
    onSuccess: (data) => {
      toast.success(data.message);
      qc.invalidateQueries({ queryKey: keys.dataOverview });
      qc.invalidateQueries({ queryKey: keys.pairs });
      qc.invalidateQueries({ queryKey: keys.status });
    },
  });
}

export function useGenEval() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (body: GenEvalRequest) => api.genEval(body),
    onError: fail,
    onSuccess: (data) => {
      toast.success(data.message);
      qc.invalidateQueries({ queryKey: keys.dataOverview });
      qc.invalidateQueries({ queryKey: keys.corpus });
      qc.invalidateQueries({ queryKey: keys.status });
      qc.invalidateQueries({ queryKey: keys.runs }); // fingerprint moved — stale flags change
    },
  });
}

export function useImportPairs() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (body: ImportPairsRequest) => api.importPairs(body),
    onError: fail,
    onSuccess: (data) => {
      toast.success(data.message);
      qc.invalidateQueries({ queryKey: keys.dataOverview });
      qc.invalidateQueries({ queryKey: keys.pairs });
      qc.invalidateQueries({ queryKey: keys.status });
      if (data.fingerprint_changed) qc.invalidateQueries({ queryKey: keys.runs });
    },
  });
}

export function useLabelCommit() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (body: LabelCommitRequest) => api.labelCommit(body),
    onError: fail,
    onSuccess: (data) => {
      toast.success(data.message);
      qc.invalidateQueries({ queryKey: keys.dataOverview });
      qc.invalidateQueries({ queryKey: keys.status });
      qc.invalidateQueries({ queryKey: keys.runs });
    },
  });
}

export function useRunEval() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (body: EvalRequest) => api.runEval(body),
    onError: fail,
    onSuccess: (data) => {
      toast.success(`평가 완료 — ${data.model} (dim ${data.embed_dim}, ${data.split})`);
      qc.invalidateQueries({ queryKey: keys.runs });
      qc.invalidateQueries({ queryKey: keys.status });
      qc.invalidateQueries({ queryKey: keys.modelsDetail });
    },
  });
}

export function useImportTrec() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (body: ImportTrecRequest) => api.importTrec(body),
    onError: fail,
    onSuccess: (data) => {
      toast.success(data.message);
      qc.invalidateQueries({ queryKey: keys.runs });
      qc.invalidateQueries({ queryKey: keys.status });
    },
  });
}

export function useDeleteRun() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (id: string) => api.deleteRun(id),
    onError: fail,
    onSuccess: () => {
      toast.success("실험을 삭제했습니다");
      qc.invalidateQueries({ queryKey: keys.runs });
      qc.invalidateQueries({ queryKey: keys.status });
    },
  });
}

export function useCreateJob() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (body: JobCreateRequest) => api.createJob(body),
    onError: fail,
    onSuccess: (job) => {
      toast.success(job.kind === "sweep" ? `스윕 시작 — ${job.runs.length}런` : "학습 시작");
      qc.invalidateQueries({ queryKey: keys.jobs });
      qc.invalidateQueries({ queryKey: keys.status });
    },
  });
}

export function useStopJob() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (id: string) => api.stopJob(id),
    onError: fail,
    onSuccess: (job) => {
      toast.success("중단 요청 — 현재 런을 종료합니다");
      qc.setQueryData(keys.job(job.id), job);
      qc.invalidateQueries({ queryKey: keys.jobs });
    },
  });
}

export function useSkipRun() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (id: string) => api.skipRun(id),
    onError: fail,
    onSuccess: (job) => {
      toast.success("현재 런을 건너뜁니다");
      qc.setQueryData(keys.job(job.id), job);
    },
  });
}

export function useDeleteModel() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (path: string) => api.deleteModel(path),
    onError: fail,
    onSuccess: (data) => {
      toast.success(`삭제 완료 — ${data.deleted}`);
      qc.setQueryData(keys.modelsDetail, { models: data.models, disk_total_bytes: data.disk_total_bytes });
      qc.invalidateQueries({ queryKey: keys.models("sentence-transformers") });
    },
  });
}

export function useHandoff() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (path: string) => api.handoff(path),
    onError: fail,
    onSuccess: (data) => {
      toast.success(`핸드오프 패키지 생성 — ${data.path}/HANDOFF.md`);
      if (data.indexing === "started") toast.info("서빙 인덱스 재색인 시작 — 검색 탭에서 진행률 확인");
      qc.invalidateQueries({ queryKey: keys.modelsDetail });
      qc.invalidateQueries({ queryKey: keys.status });
      qc.invalidateQueries({ queryKey: keys.indexStatus });
    },
  });
}

export function useStartIndex() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (model?: string) => api.startIndex({ model: model ?? "" }),
    onError: fail,
    onSuccess: (state) => {
      toast.success(`재색인 시작 — ${state.model}`);
      qc.setQueryData(keys.indexStatus, state);
      qc.invalidateQueries({ queryKey: keys.searchStatus });
    },
  });
}
