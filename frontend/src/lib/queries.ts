// TanStack Query hooks — the single source of server state. Components read these and
// never call `api` directly, so caching, refetching and invalidation live in one place.
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { toast } from "sonner";

import { api } from "./api";
import type { Embedder, EvalRequest, GenEvalRequest, GenPairsRequest } from "./types";

const fail = (e: unknown) => toast.error((e as Error).message);

export const keys = {
  status: ["status"] as const,
  models: (e: Embedder) => ["models", e] as const,
  dataOverview: ["data", "overview"] as const,
  pairs: ["data", "pairs"] as const,
  corpus: ["data", "corpus"] as const,
  runs: ["runs"] as const,
};

// ── reads ──────────────────────────────────────────────────────────────────────
export const useStatus = () =>
  useQuery({ queryKey: keys.status, queryFn: api.status, refetchInterval: 20_000 });

export const useModels = (embedder: Embedder) =>
  useQuery({ queryKey: keys.models(embedder), queryFn: () => api.models(embedder) });

export const useDataOverview = () => useQuery({ queryKey: keys.dataOverview, queryFn: api.dataOverview });

export const usePairs = () => useQuery({ queryKey: keys.pairs, queryFn: () => api.pairs(8, false) });

export const useCorpus = () => useQuery({ queryKey: keys.corpus, queryFn: () => api.corpus(8, 120) });

export const useRuns = () => useQuery({ queryKey: keys.runs, queryFn: api.runs });

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
    },
  });
}

export function useRunEval() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (body: EvalRequest) => api.runEval(body),
    onError: fail,
    onSuccess: (data) => {
      toast.success(`평가 완료 — ${data.model} (dim ${data.embed_dim})`);
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
