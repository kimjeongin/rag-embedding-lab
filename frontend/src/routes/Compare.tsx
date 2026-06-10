import { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import { BarChart3, Play, Trash2 } from "lucide-react";

import { RUN_COLORS, runColor } from "../lib/colors";
import { fmt, short } from "../lib/format";
import { PATH } from "../lib/nav";
import { useDeleteRun, useRuns } from "../lib/queries";
import { METRICS, type RunRecord } from "../lib/types";
import { BarChart } from "../components/charts";
import { Btn, ErrorNote, Loading, Panel, Section, SectionLabel, Tag } from "../components/ui";

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
    <button
      onClick={() => setArmed(true)}
      aria-label="실험 삭제"
      title="삭제"
      className="text-faint transition-colors hover:text-danger"
    >
      <Trash2 size={15} />
    </button>
  );
}

export default function Compare() {
  const nav = useNavigate();
  const { data, isLoading, error } = useRuns();

  if (isLoading) return <Loading label="실험 결과를 불러오는 중…" />;
  if (error) return <ErrorNote>{(error as Error).message}</ErrorNote>;

  const runs = data?.runs ?? [];
  const best = data?.best ?? {}; // server-scoped to the current eval set
  const current = data?.current_fingerprint ?? null;
  const onCurrent = (r: RunRecord) => !current || r.eval_fingerprint === current;

  // Bars from different eval sets would be a meaningless comparison — the chart
  // shows only runs measured on the current set; the table below keeps everything.
  const chartRuns = runs.filter(onCurrent);
  const chartIdx = new Map(chartRuns.map((r, i) => [r.id, i]));
  const staleCount = runs.length - chartRuns.length;

  if (runs.length === 0) {
    return (
      <Section>
        <Panel className="p-10 text-center">
          <BarChart3 size={26} className="mx-auto text-faint" />
          <h2 className="mt-4 text-[18px] font-semibold text-fg">비교할 실험이 없어요</h2>
          <p className="mt-2 text-[13px] text-mut">모델을 평가하면 결과가 여기에 누적되어 나란히 비교됩니다.</p>
          <div className="mt-5 flex justify-center">
            <Btn onClick={() => nav(PATH.eval)} icon={<Play size={15} />}>
              평가 실행
            </Btn>
          </div>
        </Panel>
      </Section>
    );
  }

  return (
    <div className="space-y-9">
      <Section>
        <SectionLabel hint={`현재 평가셋 ${chartRuns.length} runs · y축 데이터 범위로 확대`}>지표 비교</SectionLabel>
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
              <BarChart runs={chartRuns} metrics={METRICS} colors={RUN_COLORS} />
            </>
          ) : (
            <div className="grid h-40 place-items-center text-[13px] text-mut">
              현재 평가셋에서 측정한 실험이 없어 차트를 표시하지 않습니다 — 점수는 같은 평가셋끼리만 비교할 수 있어요.
            </div>
          )}
        </Panel>
      </Section>

      <Section delay={70}>
        <SectionLabel
          hint={`초록 = 현재 평가셋 1등 · 🗑 행 삭제${staleCount > 0 ? ` · 흐린 행 ${staleCount}개 = 다른 평가셋` : ""}`}
        >
          결과 표
        </SectionLabel>
        <Panel className="overflow-hidden">
          <table className="w-full text-left text-[13px]">
            <thead>
              <tr className="border-b border-line bg-ink-880/60 text-[11px] uppercase tracking-wider text-faint">
                <th className="px-4 py-3 font-medium">run</th>
                {METRICS.map((m) => (
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
                return (
                  <tr
                    key={r.id}
                    className={`border-b border-line/60 last:border-0 hover:bg-ink-880/40 ${isCurrent ? "" : "opacity-55"}`}
                  >
                    <td className="px-4 py-3">
                      <div className="flex items-center gap-2.5">
                        <span
                          className="h-2.5 w-2.5 rounded-[3px]"
                          style={{ background: ci != null ? runColor(ci) : "var(--color-ink-700)" }}
                        />
                        <div>
                          <div className="flex items-center gap-2 font-medium text-fg">
                            {r.label}
                            {!isCurrent && <Tag>다른 평가셋</Tag>}
                          </div>
                          <div className="mono text-[11px] text-faint">{short(r.model)}</div>
                        </div>
                      </div>
                    </td>
                    {METRICS.map((m) => {
                      const v = r.metrics[m] ?? 0;
                      const isBest = isCurrent && v > 0 && Math.abs(v - (best[m] ?? 0)) < 1e-9;
                      return (
                        <td
                          key={m}
                          className={`mono px-3 py-3 text-right ${isBest ? "rounded bg-signal/12 font-semibold text-signal" : "text-mut"}`}
                        >
                          {fmt(v)}
                        </td>
                      );
                    })}
                    <td className="px-3 py-3 text-right">
                      <DeleteRunBtn id={r.id} />
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </Panel>
      </Section>
    </div>
  );
}
