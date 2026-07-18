import { useEffect, useMemo, useState } from "react";
import { useNavigate } from "react-router-dom";
import { BarChart3, Check, ClipboardCopy, GitCompareArrows, Play, Sigma, Star, Trash2, Upload } from "lucide-react";
import { toast } from "sonner";

import { RUN_COLORS, runColor } from "../lib/colors";
import { fmt, short } from "../lib/format";
import { PATH } from "../lib/nav";
import { useDeleteRun, useImportTrec, useModelsDetail, useRegisterBm25, useRunEval, useRuns } from "../lib/queries";
import { METRICS, type Embedder, type ModelDetail, type RunRecord } from "../lib/types";
import { BarChart } from "../components/charts";
import { DiffView } from "../components/DiffView";
import { Modal } from "../components/Modal";
import { Btn, ErrorNote, Field, Info, Input, Loading, Panel, Section, SectionLabel, Tag } from "../components/ui";

/** Two-click delete: first click arms ("삭제?") for 3s, second click commits — an
 * eval run is expensive to reproduce, one stray click must not erase it. */
function DeleteRunBtn({ id }: { id: string }) {
  const del = useDeleteRun();
  const [armed, setArmed] = useState(false);
  useEffect(() => {
    if (!armed) return;
    const t = setTimeout(() => setArmed(false), 3000);
    return () => clearTimeout(t);
  }, [armed]);

  if (armed) {
    return (
      <button
        onClick={() => del.mutate(id)}
        disabled={del.isPending}
        className="mono rounded-md bg-danger/15 px-2 py-1 text-[11px] font-semibold text-danger transition-colors hover:bg-danger/25 disabled:opacity-40"
      >
        삭제?
      </button>
    );
  }
  return (
    <button onClick={() => setArmed(true)} aria-label="실험 삭제" title="삭제" className="text-faint transition-colors hover:text-danger">
      <Trash2 size={15} />
    </button>
  );
}

/** One-shot confirmation on the held-out final split — selection happened on dev,
 * the winner gets ONE measurement here before handoff. */
function FinalConfirmBtn({ run, hasFinal }: { run: RunRecord; hasFinal: boolean }) {
  const evalRun = useRunEval();
  const [armed, setArmed] = useState(false);
  useEffect(() => {
    if (!armed) return;
    const t = setTimeout(() => setArmed(false), 3000);
    return () => clearTimeout(t);
  }, [armed]);

  if (run.embedder === "external") return null;
  if (!armed) {
    return (
      <button
        onClick={() => setArmed(true)}
        title={hasFinal ? "held-out final split로 1회 확정 측정" : "평가셋을 재생성해야 final split이 생깁니다"}
        className="mono rounded-md border border-line2 px-2 py-1 text-[10.5px] text-mut transition-colors hover:border-signal/40 hover:text-signal"
      >
        최종 확정
      </button>
    );
  }
  return (
    <button
      onClick={() => {
        setArmed(false);
        evalRun.mutate({
          embedder: run.embedder as Embedder,
          model: run.model,
          label: `${run.label} · final`,
          split: "final",
        });
      }}
      disabled={evalRun.isPending}
      className="mono rounded-md bg-signal/15 px-2 py-1 text-[10.5px] font-semibold text-signal disabled:opacity-40"
    >
      {evalRun.isPending ? "측정 중…" : "final로 측정?"}
    </button>
  );
}

function TrecImportModal({ open, onClose }: { open: boolean; onClose: () => void }) {
  const importTrec = useImportTrec();
  const [label, setLabel] = useState("BM25 (production)");
  const [content, setContent] = useState("");
  return (
    <Modal open={open} onClose={onClose} title="외부 랭킹 가져오기 (TREC run)">
      <p className="mb-4 text-[13px] leading-relaxed text-mut">
        프로덕션 BM25(또는 다른 검색기)가 평가 쿼리들에 대해 내놓은 랭킹을 표준 TREC run 형식으로 붙여넣으면, 같은
        qrels로 채점해 일반 런처럼 등록합니다. 그다음 dense 런과 <b className="text-fg">diff</b>하면 "dense가 BM25가
        놓치는 걸 얼마나 건지는가"(보완성)가 쿼리 단위로 보여요.
      </p>
      <div className="mb-3 max-w-xs">
        <Field label="라벨">
          <Input value={label} onChange={(e) => setLabel(e.target.value)} />
        </Field>
      </div>
      <Field label="run 파일 내용" hint="query-id Q0 doc-id rank score tag">
        <textarea
          value={content}
          onChange={(e) => setContent(e.target.value)}
          rows={8}
          placeholder={"q-0-0 Q0 gold-3 1 12.8 bm25\nq-0-0 Q0 distractor-41 2 11.2 bm25\n…"}
          className="mono w-full rounded-xl border border-line bg-ink-925 px-3.5 py-2.5 text-[12px] text-fg outline-none placeholder:text-faint focus:border-signal/50"
        />
      </Field>
      <div className="mt-4">
        <Btn
          icon={<Upload size={14} />}
          disabled={!content.trim() || importTrec.isPending}
          onClick={() => importTrec.mutate({ label, content }, { onSuccess: onClose })}
        >
          {importTrec.isPending ? "채점 중…" : "가져와서 채점"}
        </Btn>
      </div>
    </Modal>
  );
}

function recipeOf(meta: ModelDetail["meta"]): string | null {
  if (!meta) return null;
  const m = meta as Record<string, unknown>;
  const parts: string[] = [];
  if (m.loss) parts.push(String(m.loss));
  if (m.method) parts.push(m.method === "lora" ? `lora r${m.lora_r}` : "full");
  if (m.learning_rate != null) parts.push(`lr ${m.learning_rate}`);
  if (m.saved_epoch != null) parts.push(`e${m.saved_epoch}/${m.epochs_ran}`);
  if (m.dropout != null) parts.push(`drop ${m.dropout}`);
  if (m.seed != null && m.seed !== 42) parts.push(`seed ${m.seed}`);
  return parts.join(" · ") || null;
}

const PIN_KEY = "rag.pinnedRuns";

export default function Compare() {
  const nav = useNavigate();
  const { data, isLoading, error } = useRuns();
  const details = useModelsDetail();
  const [sel, setSel] = useState<string[]>([]);
  const [trecOpen, setTrecOpen] = useState(false);
  const bm25 = useRegisterBm25();
  const [pinned, setPinned] = useState<string[]>(() => {
    try {
      return JSON.parse(localStorage.getItem(PIN_KEY) ?? "[]");
    } catch {
      return [];
    }
  });

  const metaByModel = useMemo(() => {
    const map = new Map<string, ModelDetail>();
    for (const m of details.data?.models ?? []) map.set(m.path, m);
    return map;
  }, [details.data]);

  if (isLoading) return <Loading label="실험 결과를 불러오는 중…" />;
  if (error) return <ErrorNote>{(error as Error).message}</ErrorNote>;

  const allRuns = data?.runs ?? [];
  const best = data?.best ?? {};
  const current = data?.current_fingerprint ?? null;
  const hasFinal = !!data?.final_fingerprint;
  const metricKeys = data?.metric_keys?.length ? data.metric_keys : [...METRICS];
  const onCurrent = (r: RunRecord) => !current || r.eval_fingerprint === current;

  const togglePin = (id: string) => {
    const next = pinned.includes(id) ? pinned.filter((p) => p !== id) : [...pinned, id];
    setPinned(next);
    localStorage.setItem(PIN_KEY, JSON.stringify(next));
  };
  const toggleSel = (id: string) =>
    setSel((s) => (s.includes(id) ? s.filter((x) => x !== id) : [...s.slice(-1), id]));

  // pinned rows float to the top (baselines stay visible while the list grows)
  const runs = [...allRuns].sort((a, b) => Number(pinned.includes(b.id)) - Number(pinned.includes(a.id)));

  // Bars from different eval sets would be a meaningless comparison — the chart
  // shows only dev runs measured on the current set; the table keeps everything.
  const chartRuns = allRuns.filter((r) => onCurrent(r) && r.split !== "final").slice(0, 8);
  const chartIdx = new Map(chartRuns.map((r, i) => [r.id, i]));
  const chartMetrics = metricKeys.filter((m) => chartRuns.some((r) => r.metrics[m] != null));
  const staleCount = allRuns.filter((r) => !onCurrent(r)).length;

  const selRuns = sel
    .map((id) => allRuns.find((r) => r.id === id))
    .filter((r): r is RunRecord => !!r);

  const exportMd = () => {
    const head = `| run | recipe | ${metricKeys.join(" | ")} |`;
    const sep = `|---|---|${metricKeys.map(() => "---:").join("|")}|`;
    const rows = allRuns.map((r) => {
      const recipe = recipeOf(metaByModel.get(r.model)?.meta) ?? r.model;
      const cells = metricKeys.map((m) => (r.metrics[m] != null ? fmt(r.metrics[m]) : "—"));
      return `| ${r.label}${r.split === "final" ? " **(final)**" : ""} | ${recipe} | ${cells.join(" | ")} |`;
    });
    navigator.clipboard.writeText([head, sep, ...rows].join("\n"));
    toast.success("표를 markdown으로 복사했어요 — 그대로 공유하세요");
  };

  if (allRuns.length === 0) {
    return (
      <Section>
        <Panel className="p-10 text-center">
          <BarChart3 size={26} className="mx-auto text-faint" />
          <h2 className="mt-4 text-[18px] font-semibold text-fg">비교할 실험이 없어요</h2>
          <p className="mt-2 text-[13px] text-mut">학습(자동 평가)하거나 모델을 평가하면 결과가 여기에 누적됩니다.</p>
          <div className="mt-5 flex justify-center">
            <Btn onClick={() => nav(PATH.train)} icon={<Play size={15} />}>
              학습 시작
            </Btn>
          </div>
        </Panel>
      </Section>
    );
  }

  return (
    <div className="space-y-9">
      <Section>
        <SectionLabel hint={`현재 평가셋 dev runs · y축 데이터 범위로 확대`}>
          <span className="inline-flex items-center gap-1.5">
            지표 비교
            <Info title="지표 읽는 법" align="left">
              <b className="text-fg">recall@50</b> 후보 깊이 안에 정답이 들었나 — 뒤에 리랭커가 있는 프로덕션에선 이게
              dense의 본업입니다 · <b className="text-fg">nDCG@10</b> 상위 정렬 품질(BEIR 대표) ·{" "}
              <b className="text-fg">MRR@10</b> 첫 정답 순위. 점수는 <b className="text-fg">같은 평가셋끼리만</b> 비교
              가능합니다.
            </Info>
          </span>
        </SectionLabel>
        <Panel className="p-5">
          {chartRuns.length > 0 ? (
            <>
              <div className="mb-3 flex flex-wrap items-center gap-x-5 gap-y-2">
                {chartRuns.map((r, i) => (
                  <div key={r.id} className="flex items-center gap-2">
                    <span className="h-2.5 w-2.5 rounded-[3px]" style={{ background: runColor(i) }} />
                    <span className="text-[13px] font-medium text-fg">{r.label}</span>
                    <span className="mono text-[11px] text-faint">{short(r.model)}</span>
                  </div>
                ))}
              </div>
              <BarChart runs={chartRuns} metrics={chartMetrics} colors={RUN_COLORS} />
            </>
          ) : (
            <div className="grid h-40 place-items-center text-[13px] text-mut">
              현재 평가셋에서 측정한 실험이 없어 차트를 표시하지 않습니다.
            </div>
          )}
        </Panel>
      </Section>

      <Section delay={70}>
        <SectionLabel
          hint={`체크 2개 → 쿼리별 diff · ★ 고정 · 초록 = 현재 평가셋 1등${staleCount > 0 ? ` · 흐린 행 ${staleCount}개 = 다른 평가셋` : ""}`}
        >
          <span className="inline-flex items-center gap-2.5">
            실험 테이블
            <span className="inline-flex gap-1.5 normal-case tracking-normal">
              <Btn
                variant="subtle"
                className="px-2 py-1 text-[11.5px]"
                icon={<Sigma size={13} />}
                disabled={bm25.isPending}
                onClick={() => bm25.mutate({ note: "내장 BM25(문자 bigram) — dense와의 상보성 기준선" })}
                title="내장 BM25(문자 bigram)를 현재 평가셋에 채점해 런으로 등록 — dense 런과 diff하면 상보성이 보입니다"
              >
                {bm25.isPending ? "BM25 채점 중…" : "BM25 베이스라인"}
              </Btn>
              <Btn variant="subtle" className="px-2 py-1 text-[11.5px]" icon={<Upload size={13} />} onClick={() => setTrecOpen(true)}>
                외부 랭킹(BM25)
              </Btn>
              <Btn variant="subtle" className="px-2 py-1 text-[11.5px]" icon={<ClipboardCopy size={13} />} onClick={exportMd}>
                markdown 복사
              </Btn>
            </span>
          </span>
        </SectionLabel>
        <Panel className="overflow-x-auto">
          <table className="w-full text-left text-[13px]">
            <thead>
              <tr className="border-b border-line bg-ink-880/60 text-[11px] uppercase tracking-wider text-faint">
                <th className="px-3 py-3" />
                <th className="px-2 py-3 font-medium">run</th>
                {metricKeys.map((m) => (
                  <th key={m} className="mono px-3 py-3 text-right font-medium normal-case">
                    {m}
                  </th>
                ))}
                <th className="px-3 py-3" />
              </tr>
            </thead>
            <tbody>
              {runs.map((r) => {
                const isCurrent = onCurrent(r);
                const ci = chartIdx.get(r.id);
                const detail = metaByModel.get(r.model);
                const recipe = recipeOf(detail?.meta);
                const isPinned = pinned.includes(r.id);
                return (
                  <tr
                    key={r.id}
                    className={`border-b border-line/60 last:border-0 hover:bg-ink-880/40 ${isCurrent ? "" : "opacity-55"}`}
                  >
                    <td className="px-3 py-3">
                      <div className="flex items-center gap-2">
                        <input
                          type="checkbox"
                          checked={sel.includes(r.id)}
                          onChange={() => toggleSel(r.id)}
                          title="diff 비교 대상으로 선택 (2개)"
                          className="h-3.5 w-3.5 accent-[#c6f24a]"
                        />
                        <button onClick={() => togglePin(r.id)} title="기준선으로 고정" className={isPinned ? "text-signal" : "text-faint hover:text-mut"}>
                          <Star size={13} fill={isPinned ? "currentColor" : "none"} />
                        </button>
                      </div>
                    </td>
                    <td className="px-2 py-3">
                      <div className="flex items-center gap-2.5">
                        <span
                          className="h-2.5 w-2.5 shrink-0 rounded-[3px]"
                          style={{ background: ci != null ? runColor(ci) : "var(--color-ink-700)" }}
                        />
                        <div className="min-w-0">
                          <div className="flex flex-wrap items-center gap-1.5 font-medium text-fg">
                            <span className="truncate">{r.label}</span>
                            {r.split === "final" && <Tag tone="signal">final ✓</Tag>}
                            {r.embedder === "external" && <Tag tone="cyan">외부</Tag>}
                            {!isCurrent && <Tag>다른 평가셋</Tag>}
                          </div>
                          <div className="mono truncate text-[11px] text-faint">
                            {recipe ?? short(r.model)}
                            {r.n_queries != null && ` · n=${r.n_queries}`}
                          </div>
                          {r.note && <div className="truncate text-[11px] italic text-faint">“{r.note}”</div>}
                        </div>
                      </div>
                    </td>
                    {metricKeys.map((m) => {
                      const v = r.metrics[m];
                      const isBest =
                        isCurrent && r.split !== "final" && v != null && v > 0 && Math.abs(v - (best[m] ?? 0)) < 1e-9;
                      return (
                        <td
                          key={m}
                          className={`mono px-3 py-3 text-right ${isBest ? "rounded bg-signal/12 font-semibold text-signal" : "text-mut"}`}
                        >
                          {v != null ? fmt(v) : "—"}
                        </td>
                      );
                    })}
                    <td className="px-3 py-3">
                      <div className="flex items-center justify-end gap-2">
                        {isCurrent && r.split !== "final" && <FinalConfirmBtn run={r} hasFinal={hasFinal} />}
                        <DeleteRunBtn id={r.id} />
                      </div>
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </Panel>
        {!hasFinal && (
          <p className="mt-2 text-[11.5px] text-faint">
            ※ 최종 확정(final split)을 쓰려면 데이터 탭에서 평가셋을 재생성하세요 — dev(선택용)/final(확정용)로
            분리됩니다. 여러 런을 돌려 dev에서 1등을 골랐다면, 발표 전 final로 한 번만 확인하는 게 선택 편향을
            막는 표준 절차예요.
          </p>
        )}
      </Section>

      {selRuns.length === 2 && (
        <Section delay={100}>
          <SectionLabel hint={`A = ${selRuns[0].label} (기준) · B = ${selRuns[1].label} (후보)`}>
            <span className="inline-flex items-center gap-1.5">
              <GitCompareArrows size={14} className="text-signal" /> 쿼리별 diff
            </span>
          </SectionLabel>
          <DiffView a={selRuns[0]} b={selRuns[1]} />
        </Section>
      )}
      {selRuns.length === 1 && (
        <p className="flex items-center gap-1.5 text-[12.5px] text-faint">
          <Check size={13} /> 하나 더 선택하면 쿼리별 diff가 열립니다 — 어떤 쿼리가 좋아지고 나빠졌는지, 그 차이가
          유의미한지(p값)까지.
        </p>
      )}

      <TrecImportModal open={trecOpen} onClose={() => setTrecOpen(false)} />
    </div>
  );
}
