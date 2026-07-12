import { useEffect, useRef, useState } from "react";
import { useNavigate } from "react-router-dom";
import { ArrowRight, ChevronRight, Play, SkipForward, Square } from "lucide-react";

import { fmt } from "../lib/format";
import { PATH } from "../lib/nav";
import { useCreateJob, useDataOverview, useJob, useJobs, useSkipRun, useStatus, useStopJob, keys } from "../lib/queries";
import { useQueryClient } from "@tanstack/react-query";
import { LossCurve } from "../components/charts";
import { Btn, ErrorNote, Field, Info, Input, Panel, Section, SectionLabel, Seg, Stat } from "../components/ui";
import { AXES, type Axis, aggregateBySeed, planRuns } from "../lib/sweep";
import type { JobRunState, JobState, TrainLoss, TrainRequest } from "../lib/types";

const LOSS_OPTIONS: { value: TrainLoss; label: string }[] = [
  { value: "mnrl", label: "MNRL" },
  { value: "cached_mnrl", label: "Cached MNRL" },
  { value: "gist", label: "GIST" },
  { value: "triplet", label: "Triplet" },
];

const RUN_STATUS: Record<string, { label: string; cls: string }> = {
  pending: { label: "대기", cls: "text-faint" },
  running: { label: "학습 중", cls: "text-signal2" },
  trained: { label: "평가 대기", cls: "text-cyan" },
  evaluated: { label: "완료", cls: "text-signal" },
  failed: { label: "실패", cls: "text-danger" },
  skipped: { label: "건너뜀", cls: "text-faint" },
  stopped: { label: "중단", cls: "text-amber" },
  interrupted: { label: "끊김", cls: "text-amber" },
  pruned: { label: "가지치기", cls: "text-faint" },
};

function etaLabel(run: JobRunState): string | null {
  const last = run.epochs[run.epochs.length - 1];
  if (!last?.elapsed || last.epoch <= 0) return null;
  const remaining = (last.max_epochs - last.epoch) * (last.elapsed / last.epoch);
  if (remaining <= 0) return null;
  const m = Math.floor(remaining / 60);
  const s = Math.round(remaining % 60);
  return `남은 시간 ~${m > 0 ? `${m}분 ` : ""}${s}초 (early stop 시 더 일찍)`;
}

/** Per-epoch validation table + the live loss curve for one run. */
function RunDetail({ run, running }: { run: JobRunState; running: boolean }) {
  const lossValues = run.loss.map((p) => p.loss);
  const lossLabel =
    lossValues.length > 1 ? `${lossValues[0].toFixed(3)} → ${lossValues[lossValues.length - 1].toFixed(3)}` : "—";
  const lastEpoch = run.epochs[run.epochs.length - 1];
  const bestSoFar = lastEpoch?.best_epoch ?? 0;
  const stale = lastEpoch ? lastEpoch.epoch - lastEpoch.best_epoch : 0;
  const patience = run.config.early_stop_patience ?? 3;
  const eta = running ? etaLabel(run) : null;

  return (
    <>
      <div className="grid gap-5 lg:grid-cols-[1.6fr_1fr]">
        <Panel className="p-5">
          <div className="mb-3 flex items-center justify-between">
            <span className="flex items-center gap-1.5 text-[13px] font-medium text-mut">
              training loss
              <Info title="loss가 들쭉날쭉한 건 정상입니다" align="left">
                스텝별 <span className="mono">contrastive loss</span>는 batch마다 in-batch negative가 달라 원래
                출렁입니다. 봐야 할 건 <b className="text-fg">추세</b> — <span className="mono">{lossLabel}</span>
                (처음→끝)이 내려가면 정상입니다.
              </Info>
            </span>
            <span className="mono text-[12px] text-signal2">{lossLabel}</span>
          </div>
          <LossCurve points={lossValues} />
          {eta && <div className="mono mt-2 text-[11.5px] text-faint">{eta}</div>}
        </Panel>
        <div className="grid grid-rows-2 gap-3">
          <Stat
            label="검증쌍 nDCG@10 · 학습 전"
            value={run.result?.ndcg_before != null ? fmt(run.result.ndcg_before) : "—"}
            tone="cyan"
            sub="held-out 학습쌍"
          />
          <Stat
            label="평가셋 nDCG@10 · 자동 평가"
            value={run.eval ? fmt(run.eval.metrics["ndcg@10"] ?? 0) : "—"}
            tag={run.eval ? "기록됨" : undefined}
            tone="signal"
            sub={run.eval ? `recall@50 ${fmt(run.eval.metrics["recall@50"] ?? 0)} · 실험 탭에 누적` : "학습 후 자동 실행"}
          />
        </div>
      </div>

      {run.epochs.length > 0 && (
        <Panel className="mt-5 p-5">
          <div className="mb-3 flex items-center justify-between">
            <span className="flex items-center gap-1.5 text-[13px] font-medium text-mut">
              epoch별 검증
              <Info title="early stopping이 보는 값" align="left">
                매 epoch 끝에 held-out 검증쌍으로 <span className="mono">val loss</span>와{" "}
                <span className="mono">nDCG@10</span>을 측정합니다. ★ = 지금까지 최고(저장될 가중치). loss는 내려가는데
                nDCG가 꺾이면 과적합 — patience만큼 개선이 없으면 자동으로 멈춥니다.
              </Info>
            </span>
            <span className="mono text-[12px] text-signal2">
              best e{bestSoFar || "—"}
              {running && patience > 0 && stale > 0 && (
                <span className="text-amber">
                  {" "}
                  · 개선 없음 {stale}/{patience}
                </span>
              )}
            </span>
          </div>
          <table className="w-full text-left">
            <thead>
              <tr className="text-[11px] uppercase tracking-wider text-faint">
                <th className="py-1 pr-4 font-medium">epoch</th>
                <th className="py-1 pr-4 font-medium">val loss</th>
                <th className="py-1 pr-4 font-medium">val nDCG@10</th>
                <th className="py-1 font-medium" />
              </tr>
            </thead>
            <tbody className="mono text-[12px]">
              {run.epochs.map((e) => {
                const isBest = e.epoch === bestSoFar;
                return (
                  <tr key={e.epoch} className={`border-t border-line/60 ${isBest ? "text-signal2" : "text-mut"}`}>
                    <td className="py-1.5 pr-4">
                      {e.epoch}/{e.max_epochs}
                    </td>
                    <td className="py-1.5 pr-4">{e.eval_loss != null ? e.eval_loss.toFixed(4) : "—"}</td>
                    <td className="py-1.5 pr-4">{e.ndcg != null ? fmt(e.ndcg) : "—"}</td>
                    <td className="py-1.5">{isBest ? "★ best" : ""}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </Panel>
      )}
    </>
  );
}

/** Live leaderboard — auto-evaluated runs ranked while the rest still train. */
function SweepLeaderboard({ job }: { job: JobState }) {
  const rows = aggregateBySeed(job.runs);
  if (rows.length < 2) return null;
  const top = rows[0].ndcgMean;
  const seeded = rows.some((r) => r.n > 1);
  return (
    <Panel className="mt-5 p-5">
      <div className="mb-3 flex items-center gap-1.5 text-[13px] font-medium text-mut">
        리더보드 (자동 평가 · dev split{seeded ? " · 시드 평균±편차" : ""})
        {seeded && (
          <Info title="시드 평균으로 줄 세웁니다" align="left">
            같은 설정을 여러 <span className="mono">seed</span>로 돌린 런은 하나로 묶어 <b className="text-fg">평균 ± 표준편차</b>로
            표시합니다. 편차만큼 겹치는 두 설정은 사실상 동률 — 단발 점수로 순위를 매기면 학습 노이즈에 속습니다.
          </Info>
        )}
      </div>
      <table className="w-full text-left">
        <tbody className="mono text-[12px]">
          {rows.map((r, i) => (
            <tr key={r.label + r.idx} className={`border-t border-line/60 ${i === 0 ? "text-signal" : "text-mut"}`}>
              <td className="w-8 py-1.5">{i + 1}</td>
              <td className="py-1.5">
                {r.label}
                {r.n > 1 && <span className="ml-1.5 text-faint">×{r.n} seed</span>}
              </td>
              <td className="py-1.5 text-right">
                nDCG@10 {fmt(r.ndcgMean)}
                {r.n > 1 && <span className="text-faint"> ±{r.ndcgStd.toFixed(3)}</span>}
              </td>
              <td className="py-1.5 pl-4 text-right">recall@50 {fmt(r.recallMean)}</td>
              <td className="py-1.5 pl-4 text-right text-faint">
                {i === 0 ? "1등" : `Δ ${(r.ndcgMean - top).toFixed(4)}`}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </Panel>
  );
}

export default function Train() {
  const nav = useNavigate();
  const qc = useQueryClient();
  const status = useStatus();
  const overview = useDataOverview();
  const jobsList = useJobs();
  const createJob = useCreateJob();
  const stopJob = useStopJob();
  const skipRun = useSkipRun();

  // ── form state ────────────────────────────────────────────────────────────────
  const [mode, setMode] = useState<"single" | "sweep">("single");
  const [base, setBase] = useState("Qwen/Qwen3-Embedding-0.6B");
  const [out, setOut] = useState("outputs/embedding-ft");
  const [epochs, setEpochs] = useState(12);
  const [batch, setBatch] = useState(16);
  const [lr, setLr] = useState("2e-5");
  const [device, setDevice] = useState("");
  const [loss, setLoss] = useState<TrainLoss>("mnrl");
  const [matryoshka, setMatryoshka] = useState(false);
  const [matryoshkaDims, setMatryoshkaDims] = useState(""); // blank = auto from model dim
  const [dropout, setDropout] = useState("");
  const [patience, setPatience] = useState(3);
  const [monitor, setMonitor] = useState<"ndcg" | "loss">("ndcg");
  const [method, setMethod] = useState<"full" | "lora">("full");
  const [loraR, setLoraR] = useState(16);
  const [loraAlpha, setLoraAlpha] = useState(32);
  const [loraDropout, setLoraDropout] = useState("0.05");
  const [loraTarget, setLoraTarget] = useState<"all-linear" | "attention">("all-linear");
  const [note, setNote] = useState("");
  const [autoEval, setAutoEval] = useState(true);
  const [showAdvanced, setShowAdvanced] = useState(false); // device · dropout · early stopping · memo
  // sweep-only
  const [axis, setAxis] = useState<Axis>("learning_rate");
  const [axisValues, setAxisValues] = useState("1e-5, 2e-5, 1e-4");
  const [coVaryLr, setCoVaryLr] = useState(false); // cross the primary axis with LR (2-axis)
  const [lrValues, setLrValues] = useState("1e-5, 2e-5, 1e-4");
  const [seeds, setSeeds] = useState(1);
  const [keepTopK, setKeepTopK] = useState(0); // 0 = keep everything
  const [prune, setPrune] = useState(false); // median pruning — kill clearly-losing runs early

  const [confirmStop, setConfirmStop] = useState(false);
  useEffect(() => {
    if (!confirmStop) return;
    const t = setTimeout(() => setConfirmStop(false), 3000);
    return () => clearTimeout(t);
  }, [confirmStop]);

  // ── which job is on screen ────────────────────────────────────────────────────
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const viewId = selectedId ?? jobsList.data?.active ?? jobsList.data?.jobs[0]?.id ?? null;
  const job = useJob(viewId).data;
  const running = job?.status === "running" || job?.status === "pending";

  // refresh downstream caches once when the job reaches a terminal state
  const prevStatus = useRef<string | undefined>(undefined);
  useEffect(() => {
    const s = job?.status;
    if (prevStatus.current === "running" && s && s !== "running") {
      qc.invalidateQueries({ queryKey: keys.runs });
      qc.invalidateQueries({ queryKey: keys.modelsDetail });
      qc.invalidateQueries({ queryKey: keys.status });
      qc.invalidateQueries({ queryKey: keys.jobs });
    }
    prevStatus.current = s;
  }, [job?.status, qc]);

  // which run's detail is open (default: the one training now, else the last with data).
  // Plain derivation — a find over a handful of runs; the React Compiler memoizes it.
  const [detailIdx, setDetailIdx] = useState<number | null>(null);
  const detailRun = !job
    ? null
    : detailIdx != null
      ? (job.runs.find((r) => r.idx === detailIdx) ?? null)
      : job.current != null
        ? (job.runs.find((r) => r.idx === job.current) ?? null)
        : ([...job.runs].reverse().find((r) => r.loss.length > 0 || r.epochs.length > 0) ?? job.runs[0] ?? null);

  const ready = status.data?.training_ready ?? true;

  const changeMethod = (m: "full" | "lora") => {
    setMethod(m);
    if (m === "lora" && out.trim() === "outputs/embedding-ft") setOut("outputs/embedding-ft-lora");
    if (m === "full" && out.trim() === "outputs/embedding-ft-lora") setOut("outputs/embedding-ft");
    // LoRA convention: ~5–10× the full-FT learning rate — only if the user hasn't touched it
    if (m === "lora" && lr.trim() === "2e-5") setLr("1e-4");
    if (m === "full" && lr.trim() === "1e-4") setLr("2e-5");
  };

  const baseConfig = (): TrainRequest => {
    const parsedDropout = parseFloat(dropout);
    return {
      base_model: base.trim(),
      output_dir: out.trim(),
      epochs,
      batch_size: batch,
      learning_rate: parseFloat(lr) || (method === "lora" ? 1e-4 : 2e-5),
      device: device.trim(),
      loss,
      matryoshka,
      matryoshka_dims: matryoshka
        ? matryoshkaDims.split(",").map((s) => parseInt(s.trim(), 10)).filter((n) => Number.isFinite(n) && n > 0)
        : [],
      dropout: Number.isFinite(parsedDropout) ? parsedDropout : null,
      early_stop_patience: patience,
      early_stop_metric: monitor,
      auto_name: true,
      seed: 42,
      note: note.trim(),
      method,
      lora_r: loraR,
      lora_alpha: loraAlpha,
      lora_dropout: parseFloat(loraDropout) || 0,
      lora_target: loraTarget,
    };
  };

  // Expand the sweep into the explicit run list (the preview = exactly what runs).
  // planRuns is a pure function (lib/sweep) — the React Compiler memoizes this call, so
  // there's no hand-maintained dependency array to drift.
  const { runs: plannedRuns, problems } = planRuns({
    mode,
    axis,
    axisValues,
    coVaryLr,
    lrValues,
    seeds,
    base: baseConfig(),
    trainHasNegatives: overview.data?.train_has_negatives,
  });

  const submit = () =>
    createJob.mutate(
      {
        runs: plannedRuns,
        auto_eval: autoEval,
        keep_top_k: mode === "sweep" && keepTopK > 0 ? keepTopK : null,
        prune: mode === "sweep" && prune,
      },
      { onSuccess: (j) => setSelectedId(j.id) },
    );

  const namePreview = `${out.trim() || "…"}-${loss}${matryoshka ? "-mrl" : ""}${method === "lora" ? `-r${loraR}` : ""}-eN`;
  const finishedRun =
    job?.status === "done" && job.kind === "train" && job.runs[0]?.result ? job.runs[0] : null;
  const bestSweepRun =
    job?.status === "done" && job.kind === "sweep"
      ? [...job.runs]
          .filter((r) => r.eval)
          .sort((x, y) => (y.eval!.metrics["ndcg@10"] ?? 0) - (x.eval!.metrics["ndcg@10"] ?? 0))[0] ?? null
      : null;

  return (
    <div className="space-y-9">
      <Section>
        <SectionLabel hint="잡은 서버가 소유 — 탭을 닫거나 새로고침해도 계속 돕니다">학습 설정</SectionLabel>
        <Panel className="p-5">
          {!ready && (
            <div className="mb-4 rounded-xl border border-amber/30 bg-amber/10 px-4 py-3 text-[12.5px] text-amber">
              ⚠️ 학습 라이브러리(torch 등)가 설치되어 있지 않습니다 —{" "}
              <code className="mono">uv sync --group training</code> 후 사용하세요.
            </div>
          )}

          <div className="mb-4 flex flex-wrap items-center gap-2.5">
            <Seg
              options={[
                { value: "single", label: "단일 학습" },
                { value: "sweep", label: "스윕 (여러 설정 비교)" },
              ]}
              value={mode}
              onChange={setMode}
            />
            <Info title="스윕이란" align="left">
              아래 설정을 베이스로 <b className="text-fg">한 축(하이퍼파라미터 하나)</b>을 바꿔가며 여러 런을 순차 실행하고,
              끝날 때마다 <b className="text-fg">자동 평가</b>해 리더보드에 쌓습니다. 단일 변수 탐침이라 결과는 “고정한 다른
              값 아래에서의” 비교 — LR은 거의 모든 것과 상호작용하니 비-LR 축은 <b className="text-fg">LR 동반(2축)</b>으로, 학습
              노이즈는 <b className="text-fg">시드 반복</b>으로 다루세요. 다 돌면 <span className="mono">비교</span> 탭에서
              diff·유의성으로 확정합니다.
            </Info>
          </div>

          <div className="grid gap-4 sm:grid-cols-2">
            <Field label="base 모델" hint="HuggingFace 또는 outputs/ 경로 (이어서 학습)">
              <Input value={base} onChange={(e) => setBase(e.target.value)} disabled={running} />
            </Field>
            <Field label="저장 이름" hint="N = best epoch">
              <Input value={out} onChange={(e) => setOut(e.target.value)} disabled={running} />
              <div className="mono mt-1 text-[11px] text-faint">저장 시 자동 이름: {namePreview}</div>
            </Field>
          </div>

          <div className="mt-4">
            <Field label="학습 방법">
              <div className="flex flex-wrap items-center gap-2.5">
                <Seg
                  options={[
                    { value: "full", label: "전체 (full)" },
                    { value: "lora", label: "LoRA" },
                  ]}
                  value={method}
                  onChange={changeMethod}
                />
                <Info title="full vs LoRA" align="left">
                  <b className="text-fg">전체(full)</b> = 모든 parameter 학습(천장이 약간 높지만 무거움).{" "}
                  <b className="text-fg">LoRA</b> = 작은 adapter만 학습(메모리·속도 유리, 과적합 내성↑)하고 저장 시
                  base에 <b className="text-fg">병합</b> — 결과물은 똑같이 일반 모델입니다.
                </Info>
              </div>
            </Field>
            {method === "lora" && (
              <div className="mt-3 grid grid-cols-2 gap-3 rounded-xl border border-line bg-ink-925/60 p-3.5 sm:max-w-md">
                <Field label="LoRA rank (r)" hint="클수록 표현력↑·무거움">
                  <Input type="number" min={1} value={loraR} onChange={(e) => setLoraR(+e.target.value)} className="mono" disabled={running} />
                </Field>
                <Field label="LoRA alpha" hint="스케일 (보통 2×r)">
                  <Input type="number" min={1} value={loraAlpha} onChange={(e) => setLoraAlpha(+e.target.value)} className="mono" disabled={running} />
                </Field>
                <Field label="LoRA dropout" hint="보통 0~0.1">
                  <Input value={loraDropout} onChange={(e) => setLoraDropout(e.target.value)} className="mono" disabled={running} />
                </Field>
                <Field label="target" hint="adapter 붙일 위치">
                  <div className="flex items-center gap-2">
                    <Seg
                      options={[
                        { value: "attention", label: "attention만" },
                        { value: "all-linear", label: "모든 linear" },
                      ]}
                      value={loraTarget}
                      onChange={setLoraTarget}
                    />
                    <Info title="LoRA 튜닝 관례" align="left">
                      <b className="text-fg">alpha는 r의 1~2배로 고정</b>하고 r과 LR을 스윕하는 게 관행 — alpha와 LR은
                      효과가 겹칩니다. <b className="text-fg">target</b>은 효과가 가장 큰 선택지: attention만(가볍고
                      보수적) vs 모든 linear(표현력↑). LoRA는 LR을 full보다 <b className="text-fg">5~10배 높게</b>.
                    </Info>
                  </div>
                </Field>
              </div>
            )}
          </div>

          <div className="mt-4">
            <Field label="loss function">
              <div className="flex flex-wrap items-center gap-2.5">
                <Seg options={LOSS_OPTIONS} value={loss} onChange={setLoss} />
                <Info title="어떤 loss를 쓸까" align="left">
                  <b className="text-fg">MNRL</b> = in-batch negative InfoNCE — 검색 임베딩의 표준(기본).{" "}
                  <b className="text-fg">Cached MNRL</b> = 메모리 제약 없이 batch(=negative 수)를 키울 때.{" "}
                  <b className="text-fg">GIST</b> = guide 모델이 가짜 negative를 걸러낸 개선판.{" "}
                  <b className="text-fg">Triplet</b> = 명시 hard negative 삼중항(데이터에 negatives 필요).
                </Info>
              </div>
            </Field>
            <label className="mt-3 flex cursor-pointer items-center gap-2 text-[12.5px] text-mut">
              <input
                type="checkbox"
                checked={matryoshka}
                onChange={(e) => setMatryoshka(e.target.checked)}
                disabled={running}
                className="h-4 w-4 accent-[#c6f24a]"
              />
              Matryoshka — 차원 절단 학습
              <Info title="Matryoshka (차원을 잘라 써도 견디는 벡터)" align="left">
                같은 loss를 여러 prefix 길이(예: 1024·512·256·128·64)에서 동시에 학습해, <b className="text-fg">앞부분만
                잘라 써도</b> 순위 품질이 유지되는 벡터를 만듭니다. 프로덕션이 저장·ANN 비용을 줄이려 짧은 벡터를 쓸 때
                유리한 dense 부품 — wrapper라 어떤 loss와도 합쳐지고, 모델 이름에 <span className="mono">-mrl</span>이 붙습니다.
                여러 차원의 backward 그래프를 동시에 들고 있어 <b className="text-fg">메모리를 크게 씁니다</b> —
                batch·LoRA로도 잘 안 줄어듭니다. OOM이면 차원 수를 줄이거나(예: 256·64) 더 작은 base 모델 / VRAM이 큰
                GPU를 쓰세요.
              </Info>
            </label>
            {matryoshka && (
              <div className="mt-2 sm:max-w-md">
                <Field label="절단 차원" hint="비우면 모델 차원에서 자동 (d·d/2·d/4…64)">
                  <Input
                    value={matryoshkaDims}
                    onChange={(e) => setMatryoshkaDims(e.target.value)}
                    placeholder="예: 512, 256, 128, 64 (비우면 자동)"
                    className="mono"
                    disabled={running}
                  />
                </Field>
                <p className="mt-1.5 text-[11px] leading-relaxed text-amber/90">
                  ⚠ 차원마다 backward 그래프를 들고 있어 메모리를 많이 씁니다 — OOM이면 batch·LoRA보다 차원 수를 줄이세요.
                </p>
              </div>
            )}
          </div>

          <div className="mt-4 grid grid-cols-2 gap-4 sm:grid-cols-3">
            <Field label="epochs (최대)" hint="early stop이 일찍 멈춤">
              <Input type="number" min={1} value={epochs} onChange={(e) => setEpochs(+e.target.value)} className="mono" disabled={running} />
            </Field>
            <Field label="batch size" hint="MNRL엔 negative 수">
              <Input type="number" min={1} value={batch} onChange={(e) => setBatch(+e.target.value)} className="mono" disabled={running} />
            </Field>
            <Field label="learning rate">
              <Input value={lr} onChange={(e) => setLr(e.target.value)} className="mono" disabled={running} />
            </Field>
          </div>

          {/* Advanced: sensible defaults (device=auto, dropout=model, patience=3, monitor=ndcg) —
              hidden so the 80% path isn't a wall; a model dev opens it when needed. */}
          <div className="mt-4">
            <button
              type="button"
              onClick={() => setShowAdvanced((v) => !v)}
              className="flex items-center gap-1.5 text-[12.5px] font-medium text-mut transition-colors hover:text-fg"
            >
              <ChevronRight size={14} className={`transition-transform ${showAdvanced ? "rotate-90" : ""}`} />
              고급 설정 <span className="font-normal text-faint">— device · dropout · early stopping · 메모</span>
            </button>
            {showAdvanced && (
              <>
                <div className="mt-3 grid grid-cols-2 gap-4 sm:grid-cols-4">
                  <Field label="device" hint="빈칸=auto">
                    <Input value={device} onChange={(e) => setDevice(e.target.value)} placeholder="auto" className="mono" disabled={running} />
                  </Field>
                  <Field label="dropout" hint="빈칸=모델 기본값">
                    <div className="flex items-center gap-2">
                      <Input value={dropout} onChange={(e) => setDropout(e.target.value)} placeholder="예: 0.1" className="mono" disabled={running} />
                      <Info title="dropout (과적합 억제)" align="left">
                        아키텍처마다 적용 키가 달라(BERT: <span className="mono">hidden_dropout_prob</span>, Qwen:{" "}
                        <span className="mono">attention_dropout</span>) 실제 적용 키는 학습 로그 첫 줄에 표시됩니다. 작은
                        데이터일수록 효과적 — 보통 0~0.3 비교.
                      </Info>
                    </div>
                  </Field>
                  <Field label="patience" hint="0 = early stop 끔">
                    <div className="flex items-center gap-2">
                      <Input type="number" min={0} value={patience} onChange={(e) => setPatience(+e.target.value)} className="mono" disabled={running} />
                      <Info title="early stopping" align="left">
                        epochs를 크게 잡아두고 매 epoch 검증을 봅니다. <b className="text-fg">개선 없는 epoch이 patience번
                        연속</b>되면 멈추고 <b className="text-fg">가장 좋았던 epoch</b>이 저장됩니다 — 모델 이름의{" "}
                        <span className="mono">-eN</span>이 그 epoch.
                      </Info>
                    </div>
                  </Field>
                  <Field label="중단 기준" hint="무엇이 '개선'인가">
                    <div className="flex flex-wrap items-center gap-2">
                      <Seg
                        options={[
                          { value: "ndcg", label: "val nDCG@10" },
                          { value: "loss", label: "val loss" },
                        ]}
                        value={monitor}
                        onChange={setMonitor}
                      />
                      <Info title="nDCG vs loss" align="left">
                        <b className="text-fg">val nDCG@10</b> = 실제 검색 성능(기본 추천). <b className="text-fg">val loss</b>
                        와 갈리면(loss↓ nDCG↓) 과적합 신호입니다.
                      </Info>
                    </div>
                  </Field>
                </div>
                <div className="mt-4 sm:max-w-md">
                  <Field label="가설 메모" hint="실험 탭에 함께 표시">
                    <Input value={note} onChange={(e) => setNote(e.target.value)} placeholder="예: lr 올리면 더 좋을까?" disabled={running} />
                  </Field>
                </div>
              </>
            )}
          </div>

          {mode === "sweep" && (
            <div className="mt-4 rounded-xl border border-line bg-ink-925/60 p-3.5">
              <div className="grid gap-4 sm:grid-cols-4">
                <Field label="바꿀 축" hint="한 번에 한 축">
                  <select
                    value={axis}
                    onChange={(e) => setAxis(e.target.value as Axis)}
                    disabled={running}
                    className="w-full rounded-xl border border-line bg-ink-925 px-3.5 py-2.5 text-sm text-fg outline-none focus:border-signal/50"
                  >
                    {AXES.map((a) => (
                      <option key={a.value} value={a.value}>
                        {a.label}
                      </option>
                    ))}
                  </select>
                </Field>
                <Field label="값 목록" hint={AXES.find((a) => a.value === axis)?.hint}>
                  <Input value={axisValues} onChange={(e) => setAxisValues(e.target.value)} className="mono" disabled={running || axis === "none"} />
                </Field>
                <Field label="시드 반복" hint="×N — 분산 측정">
                  <Input type="number" min={1} max={10} value={seeds} onChange={(e) => setSeeds(+e.target.value)} className="mono" disabled={running} />
                </Field>
                <Field label="모델 보관 top-k" hint="0 = 전부 보관">
                  <div className="flex items-center gap-2">
                    <Input type="number" min={0} value={keepTopK} onChange={(e) => setKeepTopK(+e.target.value)} className="mono" disabled={running} />
                    <Info title="디스크 관리" align="left">
                      런당 모델이 약 <b className="text-fg">1.1GB</b>입니다. top-k를 정하면 스윕이 끝난 뒤{" "}
                      <b className="text-fg">하위 런의 모델 폴더를 삭제</b>합니다 — 평가 기록(점수)은 남고 가중치만
                      지워져요.
                    </Info>
                  </div>
                </Field>
              </div>
              <div className="mt-3 flex flex-wrap items-center gap-x-5 gap-y-2.5 border-t border-line/60 pt-3">
                <label
                  className={`flex items-center gap-2 text-[12.5px] ${
                    axis === "learning_rate" || axis === "none" ? "cursor-not-allowed text-faint" : "cursor-pointer text-mut"
                  }`}
                >
                  <input
                    type="checkbox"
                    checked={coVaryLr && axis !== "learning_rate" && axis !== "none"}
                    onChange={(e) => setCoVaryLr(e.target.checked)}
                    disabled={running || axis === "learning_rate" || axis === "none"}
                    className="h-4 w-4 accent-[#c6f24a]"
                  />
                  LR 동반 (2축)
                  <Info title="왜 LR을 함께 변주하나" align="left">
                    learning rate는 거의 모든 설정과 상호작용합니다. <b className="text-fg">batch·rank·loss를 고정 LR 하나에서만</b>{" "}
                    비교하면 “그 축”이 아니라 “누구의 LR이 우연히 맞았나”로 순위가 갈릴 수 있어요. 각 값을 자기 최적 LR에서
                    재보려면 LR을 2번째 축으로 함께 돌립니다 (런 수 = 값 × LR × 시드).
                  </Info>
                  {(axis === "learning_rate" || axis === "none") && (
                    <span className="text-[11px] text-faint">
                      {axis === "learning_rate" ? "(축이 이미 LR)" : "(먼저 축을 고르세요)"}
                    </span>
                  )}
                </label>
                {coVaryLr && axis !== "learning_rate" && axis !== "none" && (
                  <div className="flex items-center gap-2">
                    <span className="text-[12px] text-faint">LR 값</span>
                    <Input value={lrValues} onChange={(e) => setLrValues(e.target.value)} className="mono w-44" disabled={running} />
                  </div>
                )}
                <label className="flex cursor-pointer items-center gap-2 text-[12.5px] text-mut">
                  <input
                    type="checkbox"
                    checked={prune}
                    onChange={(e) => setPrune(e.target.checked)}
                    disabled={running}
                    className="h-4 w-4 accent-[#c6f24a]"
                  />
                  median pruning
                  <Info title="분명히 지는 런 조기 종료" align="left">
                    학습 중 매 epoch, 진행 중인 런의 최고 검증 지표(모니터링하는 <b className="text-fg">nDCG 또는 loss</b>)가{" "}
                    <b className="text-fg">이미 끝난 런들의 중앙값</b>에 못 미치면 멈추고 다음으로 넘어갑니다(Optuna의
                    MedianPruner). 비싼 순차 스윕에서 나쁜 후보에 시간을 안 써요 — 완료된 런이 3개 이상 쌓인 뒤 2 epoch부터
                    판단합니다.
                  </Info>
                </label>
              </div>
              <div className="mono mt-3 border-t border-line/60 pt-3 text-[11.5px] text-mut">
                실행 목록 ({plannedRuns.length}런 · 예상 디스크 ~{(plannedRuns.length * 1.1).toFixed(1)}GB
                {keepTopK > 0 ? ` → 보관 ${Math.min(keepTopK, plannedRuns.length) * 1.1}GB` : ""}):{" "}
                {plannedRuns.map((r) => r.label || "base").join(" · ") || "—"}
              </div>
            </div>
          )}

          {problems.length > 0 && (
            <div className="mt-4 rounded-xl border border-amber/30 bg-amber/10 px-4 py-3 text-[12.5px] text-amber">
              {problems.map((p, i) => (
                <div key={i}>⚠️ {p}</div>
              ))}
            </div>
          )}

          <div className="mt-5 flex flex-wrap items-center gap-3">
            {running ? (
              <>
                {confirmStop ? (
                  <Btn
                    variant="ghost"
                    icon={<Square size={14} />}
                    className="border-danger/40 text-danger"
                    onClick={() => {
                      setConfirmStop(false);
                      if (viewId) stopJob.mutate(viewId);
                    }}
                  >
                    정말 중단할까요? — 남은 런은 건너뜁니다
                  </Btn>
                ) : (
                  <Btn variant="ghost" icon={<Square size={14} />} onClick={() => setConfirmStop(true)}>
                    중단
                  </Btn>
                )}
                {job && job.kind === "sweep" && (
                  <Btn variant="ghost" icon={<SkipForward size={14} />} onClick={() => viewId && skipRun.mutate(viewId)}>
                    현재 런 건너뛰기
                  </Btn>
                )}
                <span className="text-[12px] text-faint">탭을 닫아도 서버에서 계속 진행됩니다</span>
              </>
            ) : (
              <>
                <Btn icon={<Play size={15} />} onClick={submit} disabled={createJob.isPending || problems.length > 0}>
                  {mode === "sweep" ? `스윕 시작 (${plannedRuns.length}런)` : "학습 시작"}
                </Btn>
                <label className="flex cursor-pointer items-center gap-2 text-[12.5px] text-mut">
                  <input
                    type="checkbox"
                    checked={autoEval}
                    onChange={(e) => setAutoEval(e.target.checked)}
                    className="h-4 w-4 accent-[#c6f24a]"
                  />
                  학습 후 자동 평가
                  <Info title="자동 평가" align="left">
                    런이 끝나는 즉시 평가셋(dev split)으로 측정해 <span className="mono">비교</span> 탭에 기록합니다 —
                    클릭 한 번이 줄고, 스윕은 라이브 리더보드가 됩니다.
                  </Info>
                </label>
              </>
            )}
          </div>
        </Panel>
      </Section>

      {finishedRun?.result && (
        <Section>
          <Panel className="flex flex-wrap items-center justify-between gap-3 border-signal/25 bg-signal/[0.06] p-5">
            <div>
              <div className="text-[14px] font-semibold text-fg">
                학습 완료
                {finishedRun.result.best_epoch != null && finishedRun.result.ran != null && (
                  <span className="text-mut">
                    {" "}
                    — best epoch {finishedRun.result.best_epoch} / {finishedRun.result.ran} epochs
                    {finishedRun.result.early_stopped ? " (early stopped)" : ""}
                  </span>
                )}
              </div>
              <div className="mono mt-1 text-[12px] text-mut">
                새 모델: {finishedRun.result.output_dir}
                {finishedRun.eval ? " — 자동 평가 완료, 실험 탭에 기록됐어요" : " — 평가셋으로 실측해야 진짜 점수를 알 수 있어요"}
              </div>
            </div>
            {finishedRun.eval ? (
              <Btn icon={<ArrowRight size={15} />} onClick={() => nav(PATH.compare)}>
                실험에서 보기
              </Btn>
            ) : (
              <Btn
                icon={<ArrowRight size={15} />}
                onClick={() =>
                  nav(PATH.eval, { state: { backend: "sentence-transformers", model: finishedRun.result!.output_dir } })
                }
              >
                이 모델 평가하기
              </Btn>
            )}
          </Panel>
        </Section>
      )}

      {bestSweepRun?.eval && bestSweepRun.result && (
        <Section>
          <Panel className="flex flex-wrap items-center justify-between gap-3 border-signal/25 bg-signal/[0.06] p-5">
            <div>
              <div className="text-[14px] font-semibold text-fg">
                스윕 완료 — 1등: {bestSweepRun.label || `run ${bestSweepRun.idx}`}{" "}
                <span className="mono text-mut">nDCG@10 {fmt(bestSweepRun.eval.metrics["ndcg@10"] ?? 0)}</span>
              </div>
              <div className="mono mt-1 text-[12px] text-mut">
                {bestSweepRun.result.output_dir} — 실험 탭에서 diff·유의성으로 확인한 뒤 최종 확정하세요
              </div>
            </div>
            <Btn icon={<ArrowRight size={15} />} onClick={() => nav(PATH.compare)}>
              실험에서 고르기
            </Btn>
          </Panel>
        </Section>
      )}

      {job && (
        <Section delay={70}>
          <SectionLabel
            hint={
              running
                ? `실행 중 · ${job.kind === "sweep" ? `런 ${(job.current ?? 0) + 1}/${job.runs.length}` : "단일 학습"}`
                : `상태: ${job.status}`
            }
          >
            <span className="inline-flex items-center gap-2.5">
              진행 상황
              {(jobsList.data?.jobs.length ?? 0) > 1 && (
                <select
                  value={viewId ?? ""}
                  onChange={(e) => {
                    setSelectedId(e.target.value);
                    setDetailIdx(null);
                  }}
                  className="rounded-lg border border-line bg-ink-925 px-2 py-1 text-[11px] font-normal normal-case tracking-normal text-mut outline-none"
                >
                  {jobsList.data?.jobs.map((j) => (
                    <option key={j.id} value={j.id}>
                      {j.created_at.slice(5, 16).replace("T", " ")} · {j.kind === "sweep" ? `스윕 ${j.n_runs}런` : "단일"} · {j.status}
                    </option>
                  ))}
                </select>
              )}
            </span>
          </SectionLabel>

          {job.kind === "sweep" && (
            <Panel className="mb-5 overflow-hidden">
              <table className="w-full text-left text-[12.5px]">
                <thead>
                  <tr className="border-b border-line bg-ink-880/60 text-[11px] uppercase tracking-wider text-faint">
                    <th className="px-4 py-2.5 font-medium">#</th>
                    <th className="px-3 py-2.5 font-medium">run</th>
                    <th className="px-3 py-2.5 font-medium">상태</th>
                    <th className="mono px-3 py-2.5 text-right font-medium normal-case">best epoch</th>
                    <th className="mono px-3 py-2.5 text-right font-medium normal-case">nDCG@10</th>
                    <th className="mono px-3 py-2.5 text-right font-medium normal-case">recall@50</th>
                  </tr>
                </thead>
                <tbody>
                  {job.runs.map((r) => {
                    const st = RUN_STATUS[r.status] ?? { label: r.status, cls: "text-mut" };
                    const isCurrent = job.current === r.idx;
                    return (
                      <tr
                        key={r.idx}
                        onClick={() => setDetailIdx(r.idx)}
                        className={`cursor-pointer border-b border-line/60 last:border-0 hover:bg-ink-880/40 ${
                          detailRun?.idx === r.idx ? "bg-ink-880/50" : ""
                        }`}
                      >
                        <td className="mono px-4 py-2.5 text-faint">{r.idx + 1}</td>
                        <td className="px-3 py-2.5">
                          <span className="font-medium text-fg">{r.label || "base"}</span>
                          {r.model_deleted && <span className="ml-2 text-[10.5px] text-faint">(모델 정리됨)</span>}
                          {r.error && <div className="mt-0.5 text-[11px] text-danger">{r.error}</div>}
                          {r.hint && <div className="mt-0.5 text-[11px] text-amber">💡 {r.hint}</div>}
                        </td>
                        <td className={`px-3 py-2.5 ${st.cls}`}>
                          {st.label}
                          {isCurrent && running && <span className="ml-1 animate-pulse">●</span>}
                        </td>
                        <td className="mono px-3 py-2.5 text-right text-mut">
                          {r.result?.best_epoch != null ? `e${r.result.best_epoch}/${r.result.ran}` : "—"}
                        </td>
                        <td className="mono px-3 py-2.5 text-right text-mut">
                          {r.eval ? fmt(r.eval.metrics["ndcg@10"] ?? 0) : "—"}
                        </td>
                        <td className="mono px-3 py-2.5 text-right text-mut">
                          {r.eval ? fmt(r.eval.metrics["recall@50"] ?? 0) : "—"}
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </Panel>
          )}

          {job.kind === "train" && job.runs[0]?.error && (
            <div className="mb-5">
              <ErrorNote>
                {job.runs[0].error}
                {job.runs[0].hint && <div className="mt-1.5 text-amber">💡 {job.runs[0].hint}</div>}
              </ErrorNote>
            </div>
          )}

          {detailRun && <RunDetail run={detailRun} running={running && job.current === detailRun.idx} />}
          {job.kind === "sweep" && <SweepLeaderboard job={job} />}
        </Section>
      )}
    </div>
  );
}
