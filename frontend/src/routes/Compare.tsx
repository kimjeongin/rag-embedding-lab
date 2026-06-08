import { useNavigate } from "react-router-dom";
import { BarChart3, Play, Trash2 } from "lucide-react";

import { fmt, short } from "../lib/format";
import { PATH, RUN_COLORS, runColor } from "../lib/nav";
import { useDeleteRun, useRuns } from "../lib/queries";
import { METRICS } from "../lib/types";
import { BarChart } from "../components/charts";
import { Btn, ErrorNote, Loading, Panel, Section, SectionLabel } from "../components/ui";

export default function Compare() {
  const nav = useNavigate();
  const { data, isLoading, error } = useRuns();
  const del = useDeleteRun();

  if (isLoading) return <Loading label="실험 결과를 불러오는 중…" />;
  if (error) return <ErrorNote>{(error as Error).message}</ErrorNote>;

  const runs = data?.runs ?? [];
  const best = data?.best ?? {};

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
        <SectionLabel hint={`${runs.length} runs · y축 데이터 범위로 확대`}>지표 비교</SectionLabel>
        <Panel className="p-5">
          <div className="mb-3 flex flex-wrap items-center gap-x-5 gap-y-2">
            {runs.map((r, i) => (
              <div key={r.id} className="flex items-center gap-2">
                <span className="h-2.5 w-2.5 rounded-[3px]" style={{ background: runColor(i) }} />
                <span className="text-[13px] font-medium text-fg">{r.label}</span>
                <span className="mono text-[11px] text-faint">{short(r.model)}</span>
              </div>
            ))}
          </div>
          <BarChart runs={runs} metrics={METRICS} colors={RUN_COLORS} />
        </Panel>
      </Section>

      <Section delay={70}>
        <SectionLabel hint="초록 = 지표별 1등 · 🗑 행 삭제">결과 표</SectionLabel>
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
              {runs.map((r, i) => (
                <tr key={r.id} className="border-b border-line/60 last:border-0 hover:bg-ink-880/40">
                  <td className="px-4 py-3">
                    <div className="flex items-center gap-2.5">
                      <span className="h-2.5 w-2.5 rounded-[3px]" style={{ background: runColor(i) }} />
                      <div>
                        <div className="font-medium text-fg">{r.label}</div>
                        <div className="mono text-[11px] text-faint">{short(r.model)}</div>
                      </div>
                    </div>
                  </td>
                  {METRICS.map((m) => {
                    const v = r.metrics[m] ?? 0;
                    const isBest = v > 0 && Math.abs(v - (best[m] ?? 0)) < 1e-9;
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
                    <button
                      onClick={() => del.mutate(r.id)}
                      disabled={del.isPending}
                      title="삭제"
                      className="text-faint transition-colors hover:text-danger disabled:opacity-40"
                    >
                      <Trash2 size={15} />
                    </button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </Panel>
      </Section>
    </div>
  );
}
