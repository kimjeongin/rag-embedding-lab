// Wire DTOs — mirror the FastAPI lab API (src/rag/api/schemas/lab.py). Keep in sync.

export const METRICS = ["recall@1", "recall@3", "recall@5", "recall@10", "recall@50", "mrr@10", "ndcg@10"] as const;
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
  fingerprint?: string | null; // content hash of the tuning (dev) split
  splits?: string[]; // qrels splits present (dev/final/test)
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
  best_ndcg?: number | null; // best nDCG@10 on the current eval set
  active_job?: string | null; // running training job id
  handed_off?: { model: string; at: string } | null; // latest delivery marker
  qdrant_reachable: boolean; // serving vector store up? (header dot)
  indexing: boolean; // background reindex running? (header pill)
}

// GET /api/models
export interface ModelsResponse {
  embedder: Embedder;
  models: string[];
  default: string;
}

// GET /api/models/detail — the saved-model shelf
export interface ModelEvalSummary {
  run_id?: string;
  metrics: Metrics;
  n_queries?: number;
  split?: string;
  eval_fingerprint?: string | null; // which eval-set content scored this run
}
export interface ModelDetail {
  path: string;
  size_bytes: number;
  dim?: number | null;
  created_at?: string | null;
  meta?: Record<string, unknown> | null; // train_meta.json (recipe + history)
  eval_dev?: ModelEvalSummary | null;
  eval_final?: ModelEvalSummary | null;
  handed_off: boolean;
}
export interface ModelsDetailResponse {
  models: ModelDetail[];
  disk_total_bytes: number;
  current_fingerprint?: string | null; // the bound eval set's content hash
}
export interface DeleteModelResponse extends ModelsDetailResponse {
  deleted: string;
}
export interface HandoffResponse {
  path: string;
  markdown: string;
  handoff: Record<string, unknown>;
  indexing?: string | null; // 재색인 훅 결과: "started" | 사유
}

// /api/search — Qdrant 서빙 경로
export interface SearchHit {
  score: number;
  url?: string | null;
  title?: string | null;
  content: string;
}
export interface SearchResponse {
  query: string;
  collection: string;
  model: string;
  embed_ms: number; // query-embedding latency (model-bound)
  search_ms: number; // Qdrant ANN latency
  hits: SearchHit[];
}
export interface CollectionInfo {
  name: string;
  model_slug?: string | null; // which model built it (from the name)
  dim?: number | null;
  points: number;
  fingerprint?: string | null; // corpus-content hash (from the name)
  live: boolean; // alias points here right now
}
export interface SearchStatusResponse {
  reachable: boolean;
  alias: string;
  collection?: string | null;
  points: number;
  dim?: number | null;
  dim_matches?: boolean | null;
  model_matches?: boolean | null; // index model == query embedder (same-dim trap guard)
  collections: CollectionInfo[];
  embedder: string;
  model: string;
}
export interface IndexRequest {
  model?: string;
  recreate?: boolean;
}
export interface PruneResponse {
  pruned: string[];
}
export interface IndexJobStatus {
  status: "idle" | "running" | "done" | "failed";
  model?: string | null;
  done: number;
  total?: number | null;
  error?: string | null;
  summary?: Record<string, unknown> | null;
  started_at?: string | null;
  finished_at?: string | null;
}

// GET /api/data/*
export interface DataOverviewResponse {
  train: FileCount;
  test: FileCount;
  eval: EvalInfo;
  train_has_negatives?: boolean; // every train record carries hard negatives
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
  round_trip_k?: number; // consistency filter, TRAIN split only (0 = off)
  neg_margin?: number; // false-negative guard for hard-negative mining (0 = off)
}
export interface GenPairsResponse {
  message: string;
  train: FileCount;
  test: FileCount;
  preview: PairItem[];
}

// POST /api/data/eval
export interface GenEvalRequest {
  source?: "sample" | "corpus"; // corpus = the crawled site is the haystack
  n_distractors?: number | null; // sample-only
  corpus_file?: string; // corpus-only
}

// POST /api/data/crawl/stream (SSE — see crawlStore)
export interface CrawlRequest {
  url: string; // site root, or a sitemap.xml directly
  max_pages?: number;
  corpus_file?: string;
}
export interface GenEvalResponse {
  message: string;
  dir: string;
  corpus: number;
  queries: number;
  qrels: number;
  preview: CorpusDoc[];
}

// POST /api/data/import — real query/click logs
export interface ImportPairsRequest {
  content: string; // pasted JSONL or CSV
  target: "train" | "qrels" | "both";
}
export interface ImportPairsResponse {
  parsed: number;
  added_train: number;
  added_qrels: number;
  skipped: string[];
  fingerprint_changed: boolean;
  message: string;
}

// /api/data/label — judge queries to grow qrels
export interface LabelSearchRequest {
  query: string;
  embedder: Embedder;
  model: string;
}
export interface LabelDoc {
  id: string;
  title?: string | null;
  text: string;
}
export interface LabelSearchResponse {
  query: string;
  results: LabelDoc[];
}
export interface LabelCommitRequest {
  query: string;
  doc_ids: string[];
  also_train: boolean;
}
export interface LabelCommitResponse {
  query_id: string;
  added_qrels: number;
  added_train: number;
  message: string;
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
  /** Content hash of the eval set this run was measured on (null for legacy runs).
   * Scores are only comparable between runs sharing a fingerprint. */
  eval_fingerprint?: string | null;
  n_queries?: number | null;
  ci95?: Record<string, number[]> | null;
  split?: string | null; // dev (tuning) | final (one-shot confirm) | test (legacy)
  note?: string | null; // experimenter's hypothesis/memo
  /** 슬라이스별 평균 — 평가 쿼리에 slice 태그가 있을 때만 (예: standard/jargon).
   * 전체 평균이 가리는 슬라이스 붕괴를 런 자체가 들고 다닌다. */
  slices?: Record<string, { n: number; metrics: Metrics }> | null;
}
export interface RunsResponse {
  runs: RunRecord[];
  best: Metrics; // max per metric on the CURRENT eval set
  current_fingerprint?: string | null;
  final_fingerprint?: string | null;
  metric_keys: string[];
}
export interface DeleteRunResponse {
  deleted: string;
  remaining: number;
}

// GET /api/runs/diff — paired comparison (B relative to A)
export interface DiffDoc {
  id: string;
  title: string;
  relevant: boolean;
}
export interface DiffQuery {
  query_id: string;
  a: number;
  b: number;
  delta: number;
  text?: string | null;
  retrieved_a?: DiffDoc[];
  retrieved_b?: DiffDoc[];
}
export interface DiffResponse {
  a: RunRecord;
  b: RunRecord;
  metric: string;
  n: number;
  wins: number;
  losses: number;
  ties: number;
  mean_a: number;
  mean_b: number;
  delta: number;
  p_value: number;
  queries: DiffQuery[];
  by_metric: Record<string, { mean_a: number; mean_b: number; delta: number; p_value: number }>;
  slices: { topic: string; n: number; mean_a: number; mean_b: number; delta: number }[];
  texts_available: boolean;
  /** 후보군 상보성 — A/B/A∪B top-k의 recall과 한계 기여(marginal_b = union − A).
   * 랭킹이 저장된 런끼리, 평가셋이 디스크와 일치할 때만 채워진다. */
  union?: {
    k: number;
    n: number;
    recall_a: number;
    recall_b: number;
    recall_union: number;
    marginal_b: number;
    marginal_a: number;
    slices: { topic: string; n: number; recall_a: number; recall_b: number; recall_union: number; marginal_b: number }[];
  } | null;
}

// POST /api/data/import-clicklog — 세션 클릭로그를 클리닝해 학습쌍으로
export interface ImportClicklogRequest {
  content: string;
  min_dwell?: number; // 이 미만 dwell 클릭은 바운스 (기본 20초)
  transfer?: boolean; // 실패한 앞 쿼리를 세션의 최종 만족 문서로 전이 (기본 켬)
}
export interface ImportClicklogResponse {
  parsed: number;
  added_train: number;
  report: Record<string, number>; // 규칙별 카운트 (dropped_pii, bounces_ignored, positives_transferred …)
  skipped: string[];
  message: string;
}

// POST /api/runs/bm25 — 내장 BM25(문자 bigram) 베이스라인 채점·등록
export interface Bm25Request {
  label?: string;
  split?: string;
  note?: string;
}

// POST /api/runs/import-trec
export interface ImportTrecRequest {
  label: string;
  content: string;
}
export interface ImportTrecResponse {
  run: RunRecord;
  metrics: Metrics;
  n_queries: number;
  errors: string[];
  message: string;
}

// POST /api/eval
export interface EvalRequest {
  embedder: Embedder;
  model: string;
  ollama_url?: string | null;
  eval_dir?: string | null;
  label?: string;
  split?: "dev" | "final"; // final = one-shot confirmation of a chosen winner
  note?: string;
  truncate_dim?: number | null; // Matryoshka: score the truncated prefix (ST models only)
}
export interface EvalResponse {
  model: string;
  embed_dim: number;
  metrics: Metrics;
  n_queries: number;
  ci95: Record<string, number[]>; // {metric: [lo, hi]} bootstrap 95% CI
  run: RunRecord;
  prior_best: Metrics; // best before this run, same eval set only
  split: string;
}

// training config (one run inside a job)
export type TrainLoss = "mnrl" | "cached_mnrl" | "gist" | "triplet";
export interface TrainRequest {
  base_model: string;
  output_dir: string; // name prefix — auto_name appends -{loss}[-r{r}]-e{best}
  epochs: number; // a ceiling — early stopping may end sooner
  batch_size: number;
  learning_rate: number;
  device: string;
  loss?: TrainLoss;
  matryoshka?: boolean; // wrap the loss so truncated vectors (768→256→…) stay strong
  matryoshka_dims?: number[]; // empty + matryoshka on → derived from the model's dim
  dropout?: number | null; // backbone dropout; null = model defaults
  /** 데이터의 hard-negative 컬럼 상한: null = 전부, 0 = 제외(in-batch만).
   * negative 컬럼 수만큼 배치가 무거워져 OOM 시 줄이는 knob (triplet은 항상 1). */
  max_negatives?: number | null;
  early_stop_patience?: number; // 0 = off (run all epochs, save the last)
  early_stop_metric?: "ndcg" | "loss";
  auto_name?: boolean;
  seed?: number;
  note?: string; // hypothesis/memo — lands in train_meta + next to the run
  method?: "full" | "lora";
  lora_r?: number;
  lora_alpha?: number;
  lora_dropout?: number;
  lora_target?: "all-linear" | "attention";
}

// /api/jobs — server-owned training jobs (single run or sweep)
export interface LossPoint {
  step: number;
  epoch: number;
  loss: number;
}
export interface EpochPoint {
  epoch: number;
  max_epochs: number;
  eval_loss: number | null;
  ndcg: number | null;
  best_epoch: number; // best epoch so far — what early stopping will keep
  elapsed?: number; // seconds since the run started (ETA material)
}
export interface JobRunSpec {
  label: string;
  config: TrainRequest;
}
export interface JobCreateRequest {
  runs: JobRunSpec[];
  auto_eval?: boolean;
  keep_top_k?: number | null;
  prune?: boolean; // median pruning — kill runs trailing the completed median (sweep-only)
}
export type JobRunStatus =
  | "pending"
  | "running"
  | "trained"
  | "evaluated"
  | "failed"
  | "skipped"
  | "stopped"
  | "interrupted"
  | "pruned";
export interface JobRunState {
  idx: number;
  label: string;
  status: JobRunStatus;
  config: TrainRequest;
  loss: LossPoint[];
  epochs: EpochPoint[];
  result?: {
    output_dir: string;
    best_epoch?: number;
    ran?: number;
    early_stopped?: boolean;
    ndcg_before?: number | null;
    ndcg_after?: number | null;
  } | null;
  eval?: { run_id: string; metrics: Metrics; n_queries: number; split: string } | null;
  error?: string | null;
  hint?: string | null; // actionable next step for a failure
  started_at?: string | null;
  finished_at?: string | null;
  model_deleted?: boolean;
}
export interface JobState {
  id: string;
  kind: "train" | "sweep";
  status: "pending" | "running" | "done" | "stopped" | "failed" | "interrupted";
  created_at: string;
  auto_eval: boolean;
  keep_top_k?: number | null;
  current?: number | null;
  error?: string | null;
  runs: JobRunState[];
}
export interface JobSummary {
  id: string;
  kind: "train" | "sweep";
  status: string;
  created_at: string;
  n_runs: number;
  n_finished: number;
  labels: string[];
}
export interface JobsListResponse {
  jobs: JobSummary[];
  active?: string | null;
}
