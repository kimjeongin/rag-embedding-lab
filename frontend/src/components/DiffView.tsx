// Paired run-vs-run diff — the screen where a model choice actually gets made.
// Aggregate scores start arguments; "32승 9패 7무, p=0.003 + 어떤 쿼리가 왜 졌나"가 끝냅니다.
import { useState } from "react";
import { ChevronDown, ChevronRight } from "lucide-react";

import { fmt } from "../lib/format";
import { useDiff } from "../lib/queries";
import type { DiffDoc, DiffQuery, RunRecord } from "../lib/types";
import { ErrorNote, Info, Loading, Panel, Tag } from "./ui";

function RetrievedList({ docs, title }: { docs: DiffDoc[]; title: string }) {
  return (
    <div>
      <div className="mb-1 text-[10.5px] uppercase tracking-wider text-faint">{title}</div>
      <ol className="space-y-0.5">
        {docs.length === 0 && <li className="text-[11.5px] text-faint">기록 없음 (예전 런)</li>}
        {docs.map((d, i) => (
          <li key={d.id} className={`text-[11.5px] ${d.relevant ? "font-medium text-signal" : "text-mut"}`}>
            <span className="mono text-faint">{i + 1}.</span> {d.title}
            {d.relevant && " ✓"}
          </li>
        ))}
      </ol>
    </div>
  );
}

function QueryRow({ q, open, onToggle }: { q: DiffQuery; open: boolean; onToggle: () => void }) {
  const tone = q.delta > 0 ? "text-signal" : q.delta < 0 ? "text-danger" : "text-faint";
  const expandable = (q.retrieved_a?.length ?? 0) > 0 || (q.retrieved_b?.length ?? 0) > 0;
  return (
    <>
      <tr
        onClick={expandable ? onToggle : undefined}
        className={`border-t border-line/60 ${expandable ? "cursor-pointer hover:bg-ink-880/40" : ""}`}
      >
        <td className="w-6 py-2 pl-1 text-faint">
          {expandable && (open ? <ChevronDown size={13} /> : <ChevronRight size={13} />)}
        </td>
        <td className="max-w-[340px] truncate py-2 pr-3 text-[12.5px] text-fg" title={q.text ?? q.query_id}>
          {q.text ?? <span className="mono text-mut">{q.query_id}</span>}
        </td>
        <td className="mono py-2 pr-3 text-right text-[12px] text-mut">{fmt(q.a)}</td>
        <td className="mono py-2 pr-3 text-right text-[12px] text-mut">{fmt(q.b)}</td>
        <td className={`mono py-2 text-right text-[12px] font-medium ${tone}`}>
          {q.delta > 0 ? "+" : ""}
          {q.delta.toFixed(4)}
        </td>
      </tr>
      {open && (
        <tr className="border-t border-line/40 bg-ink-925/50">
          <td />
          <td colSpan={4} className="px-2 py-3">
            <div className="grid gap-4 sm:grid-cols-2">
              <RetrievedList docs={q.retrieved_a ?? []} title="A가 검색한 것 (top 5)" />
              <RetrievedList docs={q.retrieved_b ?? []} title="B가 검색한 것 (top 5)" />
            </div>
          </td>
        </tr>
      )}
    </>
  );
}

export function DiffView({ a, b }: { a: RunRecord; b: RunRecord }) {
  const [metric, setMetric] = useState("ndcg@10");
  const [openId, setOpenId] = useState<string | null>(null);
  const { data, isLoading, error } = useDiff(a.id, b.id, metric);

  if (isLoading) return <Loading label="쿼리별로 비교하는 중…" />;
  if (error) return <ErrorNote>{(error as Error).message}</ErrorNote>;
  if (!data) return null;

  const significant = data.p_value < 0.05;
  const verdict =
    data.delta === 0
      ? "두 런이 동률입니다"
      : data.delta > 0
        ? `B (${b.label})가 우세`
        : `A (${a.label})가 우세`;
  const regressions = data.queries.filter((q) => q.delta < 0);
  const improvements = data.queries.filter((q) => q.delta > 0).reverse(); // biggest gains first

  return (
    <Panel className="p-5">
      <div className="mb-4 flex flex-wrap items-center justify-between gap-3">
        <div>
          <div className="text-[14px] font-semibold text-fg">
            {verdict}
            <span className={`mono ml-2 text-[12.5px] ${significant ? "text-signal" : "text-amber"}`}>
              {data.wins}승 {data.losses}패 {data.ties}무 · Δ {data.delta > 0 ? "+" : ""}
              {data.delta.toFixed(4)} · p={data.p_value.toFixed(3)}
            </span>
          </div>
          <div className="mt-1 text-[12px] text-mut">
            {significant
              ? "p < 0.05 — 우연으로 보기 어려운 차이입니다."
              : "p ≥ 0.05 — 이 차이는 우연일 수 있습니다. 쿼리 수를 늘리거나 시드 반복으로 더 확인하세요."}
            <Info title="paired 검정이란" align="left">
              같은 쿼리를 두 모델이 풀었으니 쿼리별 점수 차(Δ)를 보고, "차이가 없다면 Δ의 부호는 동전던지기"라는
              가정으로 부호를 1만 번 뒤집어 <b className="text-fg">지금 평균 Δ보다 극단적인 경우의 비율</b>을 셉니다 —
              그게 p값. CI 겹침보다 민감한, 작은 평가셋에 맞는 검정입니다.
            </Info>
          </div>
        </div>
        <select
          value={metric}
          onChange={(e) => setMetric(e.target.value)}
          className="mono rounded-lg border border-line bg-ink-925 px-2.5 py-1.5 text-[12px] text-mut outline-none"
        >
          {Object.keys(data.by_metric).map((m) => (
            <option key={m} value={m}>
              {m}
            </option>
          ))}
        </select>
      </div>

      <div className="mb-4 grid gap-2 sm:grid-cols-3 lg:grid-cols-4">
        {Object.entries(data.by_metric).map(([m, s]) => (
          <div key={m} className="rounded-lg border border-line bg-ink-925/60 px-3 py-2">
            <div className="mono text-[10.5px] text-faint">{m}</div>
            <div className="mono mt-0.5 text-[12px] text-mut">
              {fmt(s.mean_a)} → {fmt(s.mean_b)}{" "}
              <span className={s.delta > 0 ? "text-signal" : s.delta < 0 ? "text-danger" : "text-faint"}>
                ({s.delta > 0 ? "+" : ""}
                {s.delta.toFixed(4)})
              </span>
              {s.p_value < 0.05 && <span className="ml-1 text-signal2">*</span>}
            </div>
          </div>
        ))}
      </div>

      {data.slices.length > 0 && (
        <div className="mb-4">
          <div className="mb-1.5 flex items-center gap-1.5 text-[12px] font-medium text-mut">
            슬라이스별 Δ
            <Info title="평균이 숨기는 것" align="left">
              전체 평균이 올라도 특정 슬라이스(쿼리 태그 또는 topic)만 조용히 무너질 수 있습니다 — 음수 Δ
              슬라이스는 그 쿼리들을 위 표에서 직접 확인하세요.
            </Info>
          </div>
          <div className="flex flex-wrap gap-1.5">
            {data.slices.map((s) => (
              <Tag key={s.topic} tone={s.delta < 0 ? "mut" : "signal"}>
                <span className={s.delta < 0 ? "text-danger" : ""}>
                  {s.topic} (n={s.n}) {s.delta > 0 ? "+" : ""}
                  {s.delta.toFixed(3)}
                </span>
              </Tag>
            ))}
          </div>
        </div>
      )}

      {data.union && (
        <div className="mb-4">
          <div className="mb-1.5 flex items-center gap-1.5 text-[12px] font-medium text-mut">
            후보군 상보성 (top-{data.union.k} 합집합)
            <Info title="하이브리드 관점의 가치" align="left">
              프로덕션은 여러 랭커의 후보를 합쳐 리랭커에 넘깁니다. 그래서 B의 실제 가치는 단독 점수가 아니라{" "}
              <b className="text-fg">A가 놓친 정답을 후보군에 보태는 양</b> — recall(A∪B) − recall(A) — 입니다. A에
              BM25 런을 두면 "dense가 어휘 일치 위에 실제로 보태는 정답"이 나와요.
            </Info>
          </div>
          <div className="grid gap-2 sm:grid-cols-2 lg:grid-cols-4">
            {[
              ["A 단독", data.union.recall_a],
              ["B 단독", data.union.recall_b],
              ["A∪B", data.union.recall_union],
            ].map(([labelText, v]) => (
              <div key={labelText as string} className="rounded-lg border border-line bg-ink-925/60 px-3 py-2">
                <div className="mono text-[10.5px] text-faint">recall@{data.union!.k} · {labelText}</div>
                <div className="mono mt-0.5 text-[12px] text-mut">{fmt(v as number)}</div>
              </div>
            ))}
            <div className="rounded-lg border border-line bg-ink-925/60 px-3 py-2">
              <div className="mono text-[10.5px] text-faint">B의 한계 기여</div>
              <div className={`mono mt-0.5 text-[12px] ${data.union.marginal_b > 0 ? "text-signal" : "text-faint"}`}>
                {data.union.marginal_b > 0 ? "+" : ""}
                {data.union.marginal_b.toFixed(4)}
              </div>
            </div>
          </div>
          {data.union.slices.length > 0 && (
            <div className="mt-2 flex flex-wrap gap-1.5">
              {data.union.slices.map((s) => (
                <Tag key={s.topic} tone={s.marginal_b > 0 ? "signal" : "mut"}>
                  {s.topic} (n={s.n}) 한계 기여 {s.marginal_b > 0 ? "+" : ""}
                  {s.marginal_b.toFixed(3)}
                </Tag>
              ))}
            </div>
          )}
        </div>
      )}

      {!data.texts_available && (
        <div className="mb-3 rounded-lg border border-amber/25 bg-amber/8 px-3 py-2 text-[12px] text-amber">
          평가셋이 그 후 변경되어 쿼리 원문/검색결과는 표시할 수 없습니다 (점수 비교는 유효).
        </div>
      )}

      <div className="max-h-96 overflow-auto rounded-lg border border-line/60">
        <table className="w-full text-left">
          <thead className="sticky top-0 bg-ink-900">
            <tr className="text-[10.5px] uppercase tracking-wider text-faint">
              <th className="py-2 pl-1" />
              <th className="py-2 pr-3 font-medium">
                쿼리 — 나빠진 것 먼저 ({regressions.length}개 하락 · {improvements.length}개 상승)
              </th>
              <th className="mono py-2 pr-3 text-right font-medium normal-case">A</th>
              <th className="mono py-2 pr-3 text-right font-medium normal-case">B</th>
              <th className="mono py-2 text-right font-medium normal-case">Δ</th>
            </tr>
          </thead>
          <tbody>
            {data.queries.map((q) => (
              <QueryRow
                key={q.query_id}
                q={q}
                open={openId === q.query_id}
                onToggle={() => setOpenId(openId === q.query_id ? null : q.query_id)}
              />
            ))}
          </tbody>
        </table>
      </div>
    </Panel>
  );
}
