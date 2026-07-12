import { useNavigate } from "react-router-dom";
import { BarChart3, Database, FlaskConical, Gauge, Play, Search, Trophy } from "lucide-react";

import { delta, fmt, short, when } from "../lib/format";
import { PATH } from "../lib/nav";
import { useRuns } from "../lib/queries";
import type { RunRecord } from "../lib/types";
import { Sparkline } from "../components/charts";
import { ActionCard, Btn, ErrorNote, Info, Loading, Metric, Panel, Section, SectionLabel, Tag } from "../components/ui";

const ndcg = (r: RunRecord) => r.metrics["ndcg@10"] ?? 0;

function QuickStart({ go }: { go: (p: string) => void }) {
  return (
    <Section delay={120}>
      <SectionLabel>빠른 시작</SectionLabel>
      <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
        <ActionCard icon={<Database size={18} />} title="데이터 생성" sub="학습 · 평가 데이터 만들기" onClick={() => go(PATH.data)} />
        <ActionCard icon={<FlaskConical size={18} />} title="모델 학습" sub="base를 fine-tune" onClick={() => go(PATH.train)} />
        <ActionCard icon={<Gauge size={18} />} title="성능 평가" sub="recall · nDCG 측정" onClick={() => go(PATH.eval)} />
        <ActionCard icon={<Search size={18} />} title="실검색 (서빙)" sub="Qdrant 색인 · 검색 테스트" onClick={() => go(PATH.search)} />
      </div>
    </Section>
  );
}

function EmptyHero({
  title,
  body,
  go,
}: {
  title: string;
  body: string;
  go: (p: string) => void;
}) {
  return (
    <Section>
      <Panel className="relative overflow-hidden p-10 text-center">
        <div className="absolute -right-12 -top-14 h-44 w-44 rounded-full bg-signal/10 blur-3xl" />
        <div className="relative mx-auto max-w-md">
          <Trophy size={28} className="mx-auto text-signal" />
          <h2 className="mt-4 text-[20px] font-semibold text-fg">{title}</h2>
          <p className="mt-2 text-[13.5px] leading-relaxed text-mut">{body}</p>
          <div className="mt-6 flex justify-center gap-2.5">
            <Btn onClick={() => go(PATH.eval)} icon={<Play size={15} />}>
              평가 실행
            </Btn>
            <Btn variant="ghost" onClick={() => go(PATH.compare)} icon={<BarChart3 size={15} />}>
              모든 실험 보기
            </Btn>
          </div>
        </div>
      </Panel>
    </Section>
  );
}

export default function Overview() {
  const nav = useNavigate();
  const go = (p: string) => nav(p);
  const { data, isLoading, error } = useRuns();

  if (isLoading) return <Loading label="실험 결과를 불러오는 중…" />;
  if (error) return <ErrorNote>{(error as Error).message}</ErrorNote>;

  const all = data?.runs ?? [];
  const current = data?.current_fingerprint ?? null;
  // Scores are only comparable on the same eval-set contents — the leaderboard ranks
  // only runs measured on the eval set bound right now. Older sets live in 실험.
  const runs = current ? all.filter((r) => r.eval_fingerprint === current) : all;
  const stale = all.length - runs.length;

  if (all.length === 0) {
    return (
      <div className="space-y-9">
        <Section>
          <Panel className="relative overflow-hidden p-10 text-center">
            <div className="absolute -right-12 -top-14 h-44 w-44 rounded-full bg-signal/10 blur-3xl" />
            <div className="relative mx-auto max-w-md">
              <Trophy size={28} className="mx-auto text-signal" />
              <h2 className="mt-4 text-[20px] font-semibold text-fg">아직 평가한 모델이 없어요</h2>
              <p className="mt-2 text-[13.5px] leading-relaxed text-mut">
                모델을 평가하면 여기 리더보드에 점수가 쌓이고, 최고 모델이 한눈에 보입니다.
              </p>
              <div className="mt-6 flex justify-center gap-2.5">
                <Btn onClick={() => go(PATH.eval)} icon={<Play size={15} />}>
                  첫 평가 실행
                </Btn>
                <Btn variant="ghost" onClick={() => go(PATH.data)} icon={<Database size={15} />}>
                  데이터 준비
                </Btn>
              </div>
            </div>
          </Panel>
        </Section>
        <QuickStart go={go} />
      </div>
    );
  }

  if (runs.length === 0) {
    return (
      <div className="space-y-9">
        <EmptyHero
          title="현재 평가셋에서 측정한 실험이 없어요"
          body={`기존 실험 ${stale}개는 다른(또는 재생성 이전의) 평가셋에서 측정되어 점수를 직접 비교할 수 없습니다. 지금 평가셋에서 다시 측정하면 리더보드가 채워집니다.`}
          go={go}
        />
        <QuickStart go={go} />
      </div>
    );
  }

  const sorted = [...runs].sort((a, b) => ndcg(b) - ndcg(a));
  const champion = sorted[0];
  const base = runs.find((r) => r.label === "base") ?? runs[runs.length - 1];
  const baseN = ndcg(base);
  const dN = ndcg(champion) - baseN;
  const trend = [...runs].reverse().map(ndcg);

  return (
    <div className="space-y-9">
      <Section>
        <div className="grid gap-5 lg:grid-cols-[1.55fr_1fr]">
          <Panel className="relative overflow-hidden p-6">
            <div className="absolute -right-12 -top-14 h-44 w-44 rounded-full bg-signal/10 blur-3xl" />
            <div className="relative">
              <div className="flex items-center gap-1.5 text-[11.5px] font-medium uppercase tracking-[0.14em] text-faint">
                <Trophy size={13} className="text-signal" /> 현재 최고 모델
              </div>
              <div className="mt-2 flex items-baseline gap-3">
                <h2 className="text-[26px] font-semibold tracking-tight text-fg">{champion.label}</h2>
                <span className="mono text-[13px] text-faint">{short(champion.model)}</span>
                <Tag tone={champion.embedder === "ollama" ? "cyan" : "signal"}>{champion.embedder}</Tag>
              </div>
              <div className="mt-6 grid grid-cols-3 gap-5">
                <Metric label="nDCG@10" value={ndcg(champion)} delta={dN} big />
                <Metric
                  label="recall@1"
                  value={champion.metrics["recall@1"] ?? 0}
                  delta={(champion.metrics["recall@1"] ?? 0) - (base.metrics["recall@1"] ?? 0)}
                  big
                />
                <Metric
                  label="MRR@10"
                  value={champion.metrics["mrr@10"] ?? 0}
                  delta={(champion.metrics["mrr@10"] ?? 0) - (base.metrics["mrr@10"] ?? 0)}
                  big
                />
              </div>
              <div className="mono mt-2 text-[10.5px] text-faint">Δ는 {base.label} 대비</div>
              <div className="mt-5 flex gap-2.5">
                <Btn onClick={() => go(PATH.compare)} icon={<BarChart3 size={15} />}>
                  실험 비교
                </Btn>
                <Btn variant="ghost" onClick={() => go(PATH.eval)} icon={<Play size={15} />}>
                  새 평가
                </Btn>
              </div>
            </div>
          </Panel>

          <Panel className="flex flex-col p-6">
            <div className="text-[11.5px] font-medium uppercase tracking-[0.14em] text-faint">nDCG@10 추이</div>
            <div className="mono mt-3 text-[33px] font-semibold leading-none text-fg">{fmt(ndcg(champion))}</div>
            <div className="mono mt-2 text-[12.5px] text-signal2">
              {dN === 0 ? "—" : `${delta(dN)} vs ${base.label}`}
            </div>
            <div className="mt-auto pt-6">
              {trend.length > 1 ? (
                <Sparkline data={trend} className="h-16 w-full" />
              ) : (
                <div className="h-16" />
              )}
              <div className="mono mt-1.5 flex justify-between text-[10.5px] text-faint">
                <span>{runs[runs.length - 1].label}</span>
                <span>{runs.length} runs</span>
              </div>
            </div>
          </Panel>
        </div>
      </Section>

      <Section delay={70}>
        <SectionLabel
          hint={`nDCG@10 순 · Δ는 ${base.label} 대비 · 현재 평가셋${stale > 0 ? ` (다른 평가셋 ${stale}개 제외)` : ""}`}
        >
          <span className="inline-flex items-center gap-1.5">
            리더보드
            <Info title="리더보드 읽는 법" align="left">
              <b className="text-fg">nDCG@10</b> 높은 순 정렬. <b className="text-fg">Δ</b>는 기준 모델(
              <span className="mono">{base.label}</span>) 대비 변화입니다. 점수는 <b className="text-fg">같은 평가셋</b>
              끼리만 비교 가능해서, 다른 평가셋의 run은 리더보드에서 제외됩니다.
            </Info>
          </span>
        </SectionLabel>
        <Panel className="overflow-hidden">
          <table className="w-full text-left text-[13px]">
            <thead>
              <tr className="border-b border-line bg-ink-880/60 text-[11px] uppercase tracking-wider text-faint">
                <th className="px-4 py-3 font-medium">#</th>
                <th className="px-3 py-3 font-medium">model</th>
                <th className="mono px-3 py-3 text-right font-medium normal-case">recall@1</th>
                <th className="mono px-3 py-3 text-right font-medium normal-case">mrr@10</th>
                <th className="mono px-3 py-3 text-right font-medium normal-case">ndcg@10</th>
                <th className="mono px-3 py-3 text-right font-medium normal-case">Δ vs {base.label}</th>
                <th className="mono px-4 py-3 text-right font-medium normal-case">when</th>
              </tr>
            </thead>
            <tbody>
              {sorted.map((r, i) => {
                const d = ndcg(r) - baseN;
                const top = i === 0;
                return (
                  <tr
                    key={r.id}
                    className={`border-b border-line/60 last:border-0 hover:bg-ink-880/40 ${top ? "bg-signal/[0.05]" : ""}`}
                  >
                    <td className="px-4 py-3">
                      <span
                        className={`mono grid h-6 w-6 place-items-center rounded-md text-[11px] font-semibold ${
                          top ? "bg-signal text-ink-950" : "bg-ink-800 text-mut"
                        }`}
                      >
                        {i + 1}
                      </span>
                    </td>
                    <td className="px-3 py-3">
                      <div className="font-medium text-fg">{r.label}</div>
                      <div className="mono text-[11px] text-faint">{short(r.model)}</div>
                    </td>
                    <td className="mono px-3 py-3 text-right text-mut">{fmt(r.metrics["recall@1"])}</td>
                    <td className="mono px-3 py-3 text-right text-mut">{fmt(r.metrics["mrr@10"])}</td>
                    <td className={`mono px-3 py-3 text-right font-semibold ${top ? "text-signal" : "text-fg"}`}>
                      {fmt(ndcg(r))}
                    </td>
                    <td
                      className={`mono px-3 py-3 text-right ${d > 0 ? "text-signal2" : d < 0 ? "text-danger" : "text-faint"}`}
                    >
                      {d === 0 ? "—" : delta(d)}
                    </td>
                    <td className="mono px-4 py-3 text-right text-faint">{when(r.created_at)}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </Panel>
      </Section>

      <QuickStart go={go} />
    </div>
  );
}
