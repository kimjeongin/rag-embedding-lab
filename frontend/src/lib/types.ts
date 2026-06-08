// Wire DTOs — mirror the FastAPI lab API (src/rag/api/schemas/lab.py). Keep in sync.

export const METRICS = ["recall@1", "recall@3", "recall@5", "recall@10", "mrr@10", "ndcg@10"] as const;
export type MetricKey = (typeof METRICS)[number];
export type Metrics = Record<string, number>;

export type Embedder = "ollama" | "sentence-transformers";

export interface FileCount {
  file: string;
  count: number;
}
export interface EvalInfo {
  dir: string;
  is_sample: boolean;
  corpus: number;
  queries: number;
}

// GET /api/status
export interface OllamaStatus {
  reachable: boolean;
  models: string[];
}
export interface EmbedInfo {
  embedder: string;
  model: string;
  embed_dim: number;
}
export interface StatusResponse {
  ollama: OllamaStatus;
  device: string;
  settings: EmbedInfo;
  eval: EvalInfo;
  training_ready: boolean;
  runs: number;
}

// GET /api/models
export interface ModelsResponse {
  embedder: Embedder;
  models: string[];
  default: string;
}

// GET /api/data/*
export interface DataOverviewResponse {
  train: FileCount;
  test: FileCount;
  eval: EvalInfo;
}
export interface PairItem {
  query: string | null;
  title: string | null;
  content: string | null;
}
export interface PairsResponse {
  file: string;
  total: number;
  items: PairItem[];
}
export interface CorpusDoc {
  id: string | null;
  title: string | null;
  text: string;
}
export interface CorpusResponse {
  dir: string;
  total: number;
  items: CorpusDoc[];
}

// POST /api/data/pairs
export interface GenPairsRequest {
  method: "toy" | "synthetic";
  corpus_file?: string | null;
  gen_model?: string | null;
  n_queries?: number;
  hard_negatives?: number;
}
export interface GenPairsResponse {
  message: string;
  train: FileCount;
  test: FileCount;
  preview: PairItem[];
}

// POST /api/data/eval
export interface GenEvalRequest {
  n_distractors?: number | null;
}
export interface GenEvalResponse {
  message: string;
  dir: string;
  corpus: number;
  queries: number;
  qrels: number;
  preview: CorpusDoc[];
}

// runs registry
export interface RunRecord {
  id: string;
  created_at: string;
  label: string;
  embedder: string;
  model: string;
  eval_dir: string;
  metrics: Metrics;
}
export interface RunsResponse {
  runs: RunRecord[];
  best: Metrics;
  metric_keys: string[];
}
export interface DeleteRunResponse {
  deleted: string;
  remaining: number;
}

// POST /api/eval
export interface EvalRequest {
  embedder: Embedder;
  model: string;
  ollama_url?: string | null;
  eval_dir?: string | null;
  label?: string;
}
export interface EvalResponse {
  model: string;
  embed_dim: number;
  metrics: Metrics;
  run: RunRecord;
  prior_best: Metrics;
}

// POST /api/train (SSE) — request body; the response is an event stream (see useTrainStream).
export interface TrainRequest {
  base_model: string;
  output_dir: string;
  epochs: number;
  batch_size: number;
  learning_rate: number;
  device: string;
}

// SSE event payloads
export interface LossPoint {
  step: number;
  epoch: number;
  loss: number;
}
export interface TrainMetrics {
  before: number | null;
  after: number | null;
}
