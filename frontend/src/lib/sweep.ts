// Sweep planning + seed aggregation — the pure logic behind the Train tab, lifted out
// of the component so it's unit-testable and the gnarliest code in the app stops living
// inside a 900-line view. No React here.
import type { JobRunSpec, JobRunState, TrainLoss, TrainRequest } from "./types";

const MAX_RUNS = 64; // mirrors JobCreateRequest.runs max_length on the server
const LOSS_SET = new Set<TrainLoss>(["mnrl", "cached_mnrl", "gist", "triplet"]);

// One PRIMARY axis at a time — a single-variable probe, not a joint search. Because
// learning rate interacts with almost everything, a non-LR axis can be co-swept with LR
// ("LR 동반") so each setting is judged at its own best LR rather than one frozen guess —
// the nuisance-parameter caveat from Google's Tuning Playbook. ("none" = repeat the same
// config over seeds only, to measure run-to-run variance.)
export const AXES = [
  { value: "learning_rate", label: "learning rate", hint: "예: 1e-5, 2e-5, 1e-4" },
  { value: "loss", label: "loss", hint: "예: mnrl, gist, triplet" },
  { value: "dropout", label: "dropout", hint: "예: 0, 0.1, 0.2" },
  { value: "lora_r", label: "LoRA rank", hint: "예: 8, 16, 32" },
  { value: "batch_size", label: "batch size", hint: "예: 8, 16, 32" },
  { value: "none", label: "(축 없음 — 시드만 반복)", hint: "같은 설정 × N시드로 분산 측정" },
] as const;
export type Axis = (typeof AXES)[number]["value"];

const axisShort: Record<Axis, string> = {
  learning_rate: "lr",
  loss: "loss",
  dropout: "drop",
  lora_r: "r",
  batch_size: "batch",
  none: "",
};

/** Cast a raw axis value to the type that axis's config field expects. */
export function castAxisValue(axis: Axis, raw: string): number | string {
  if (axis === "loss") return raw;
  if (axis === "lora_r" || axis === "batch_size") return parseInt(raw, 10);
  return parseFloat(raw);
}

export interface SweepForm {
  mode: "single" | "sweep";
  axis: Axis;
  axisValues: string;
  coVaryLr: boolean; // cross the primary axis with LR (2-axis)
  lrValues: string;
  seeds: number;
  base: TrainRequest; // the base config every run is derived from
  trainHasNegatives?: boolean; // for the Triplet pre-flight warning
}

/** Expand the form into the explicit run list the server will execute — what the UI's
 *  preview shows is exactly what runs. Returns validation `problems` alongside. Pure:
 *  single → [base]; sweep → primary-axis values (× LR if co-varied) × seeds. */
export function planRuns(form: SweepForm): { runs: JobRunSpec[]; problems: string[] } {
  const { mode, axis, axisValues, coVaryLr, lrValues, seeds, base, trainHasNegatives } = form;
  const problems: string[] = [];
  let runs: JobRunSpec[];

  if (mode === "single" || axis === "none") {
    runs = [{ label: "", config: base }];
  } else {
    const values = axisValues.split(",").map((s) => s.trim()).filter(Boolean);
    if (values.length === 0) problems.push("축 값을 쉼표로 구분해 입력하세요");
    values.forEach((v) => {
      if (axis === "loss" && !LOSS_SET.has(v as TrainLoss)) problems.push(`loss가 아닙니다: ${v}`);
      else if (axis !== "loss" && Number.isNaN(castAxisValue(axis, v))) problems.push(`숫자가 아닙니다: ${v}`);
    });
    // optional 2nd axis = learning rate, so a non-LR axis is judged at its own best LR
    const crossLr = coVaryLr && axis !== "learning_rate";
    const lrs = crossLr ? lrValues.split(",").map((s) => s.trim()).filter(Boolean) : [];
    if (crossLr) {
      if (lrs.length === 0) problems.push("LR 동반: LR 값을 쉼표로 구분해 입력하세요");
      lrs.forEach((v) => Number.isFinite(parseFloat(v)) || problems.push(`LR이 숫자가 아닙니다: ${v}`));
    }
    runs = values.flatMap((v) => {
      const row = { label: `${axisShort[axis]}=${v}`, config: { ...base, [axis]: castAxisValue(axis, v) } as TrainRequest };
      if (!crossLr) return [row];
      return lrs.map((lrv) => ({
        label: `${row.label} · lr=${lrv}`,
        config: { ...row.config, learning_rate: parseFloat(lrv) },
      }));
    });
  }

  if (mode === "sweep" && seeds > 1) {
    runs = runs.flatMap((r) =>
      Array.from({ length: seeds }, (_, i) => ({
        label: `${r.label ? `${r.label} · ` : ""}seed=${42 + i}`,
        config: { ...r.config, seed: 42 + i },
      })),
    );
  }

  if (runs.length > MAX_RUNS) problems.push(`런이 너무 많습니다 (${runs.length} > ${MAX_RUNS})`);
  if (base.loss === "triplet" && trainHasNegatives === false) {
    problems.push("Triplet loss에는 hard negative가 필요합니다 — 데이터 탭에서 mining을 켜고 재생성하세요");
  }
  return { runs, problems };
}

// ── seed-aggregated leaderboard ────────────────────────────────────────────────
export interface LeaderRow {
  label: string;
  idx: number;
  n: number; // how many seeds folded into this row
  ndcgMean: number;
  ndcgStd: number;
  recallMean: number;
}

const mean = (xs: number[]) => xs.reduce((a, b) => a + b, 0) / xs.length;
const std = (xs: number[]) => {
  if (xs.length < 2) return 0;
  const m = mean(xs);
  return Math.sqrt(xs.reduce((a, b) => a + (b - m) ** 2, 0) / (xs.length - 1));
};
/** Drop the "· seed=NN" segment so seed-siblings share one display label. */
const stripSeed = (label: string) => label.split(" · ").filter((p) => !p.startsWith("seed=")).join(" · ");

/** Collapse runs that differ only by seed into one variance-aware row, ranked by mean
 *  nDCG@10. A single-seed sweep is just every group with n=1 (no ± shown). Ranking on
 *  the seed mean — not a lone run — is what keeps training noise from reordering it. */
export function aggregateBySeed(runs: JobRunState[]): LeaderRow[] {
  const groups = new Map<string, { label: string; idx: number; ndcg: number[]; recall: number[] }>();
  for (const r of runs) {
    if (!r.eval) continue;
    const rest = { ...r.config };
    delete rest.seed; // everything-but-seed is the grouping key
    const key = JSON.stringify(rest);
    const g = groups.get(key) ?? { label: stripSeed(r.label) || `run ${r.idx}`, idx: r.idx, ndcg: [], recall: [] };
    g.ndcg.push(r.eval.metrics["ndcg@10"] ?? 0);
    g.recall.push(r.eval.metrics["recall@50"] ?? 0);
    groups.set(key, g);
  }
  return [...groups.values()]
    .map((g) => ({
      label: g.label,
      idx: g.idx,
      n: g.ndcg.length,
      ndcgMean: mean(g.ndcg),
      ndcgStd: std(g.ndcg),
      recallMean: mean(g.recall),
    }))
    .sort((a, b) => b.ndcgMean - a.ndcgMean);
}
