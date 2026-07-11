// 검색 — 학습한 모델이 Qdrant 인덱스에서 실제로 검색하는 서빙 플레이그라운드.
// 위: 인덱스 상태(alias → 컬렉션, 문서 수, dim 일치) + 재색인(백그라운드 잡, 진행률 폴링).
// 아래: 실검색 — 쿼리는 서버 프로세스의 임베더(EMBEDDER/ST_MODEL)로 임베딩되므로
// 인덱스를 만든 모델과 같아야 의미가 있다 (dim 불일치는 서버가 503으로 막는다).
import { useEffect, useState } from "react";
import { useMutation, useQueryClient } from "@tanstack/react-query";
import { ExternalLink, RefreshCw, Search as SearchIcon } from "lucide-react";

import { api } from "../lib/api";
import { fmt } from "../lib/format";
import { keys, useIndexStatus, useModels, useSearchStatus, useStartIndex } from "../lib/queries";
import type { SearchResponse } from "../lib/types";
import { Btn, ErrorNote, Field, Info, Input, Loading, Panel, Pill, Section, SectionLabel, Tag } from "../components/ui";

export default function Search() {
  return (
    <div className="space-y-6">
      <IndexPanel />
      <SearchPanel />
    </div>
  );
}

// ── 인덱스 상태 + 재색인 ───────────────────────────────────────────────────────
function IndexPanel() {
  const qc = useQueryClient();
  const status = useSearchStatus();
  const job = useIndexStatus();
  const models = useModels("sentence-transformers");
  const start = useStartIndex();
  const [model, setModel] = useState("");

  const running = job.data?.status === "running";
  // 재색인이 끝나면 인덱스 상태(컬렉션/문서 수)도 새로고침
  useEffect(() => {
    if (job.data?.status === "done" || job.data?.status === "failed")
      qc.invalidateQueries({ queryKey: keys.searchStatus });
  }, [job.data?.status, qc]);

  const s = status.data;
  return (
    <Section>
      <SectionLabel
        hint={
          s?.collection ? (
            <span className="mono">
              {s.alias} → {s.collection}
            </span>
          ) : undefined
        }
      >
        서빙 인덱스
        <Info title="컬렉션 버저닝" align="left" className="ml-2">
          모델·차원·코퍼스 내용마다 별도 컬렉션이 만들어지고, 검색은 항상 alias(<span className="mono">-live</span>)만
          봅니다. 재색인은 새 컬렉션을 다 만든 뒤 alias를 원자적으로 옮기므로 무중단입니다.
        </Info>
      </SectionLabel>
      <Panel className="p-5">
        {status.isLoading ? (
          <Loading />
        ) : !s?.reachable ? (
          <ErrorNote>
            Qdrant에 연결할 수 없습니다 — <span className="mono">make qdrant</span>로 로컬 인스턴스를 띄우세요.
          </ErrorNote>
        ) : (
          <div className="space-y-4">
            <div className="flex flex-wrap items-center gap-2">
              <Pill tone="signal">Qdrant 연결됨</Pill>
              {s.collection ? (
                <>
                  <Pill>{s.points.toLocaleString()} 문서</Pill>
                  <Pill>dim {s.dim}</Pill>
                  {s.dim_matches === false && <Pill tone="amber">임베더 dim({s.model}) 불일치 — 재색인 필요</Pill>}
                </>
              ) : (
                <Pill tone="amber">색인 없음 — 아래에서 모델을 골라 재색인하세요</Pill>
              )}
              <Tag>쿼리 임베더: {s.model}</Tag>
            </div>

            <div className="flex flex-wrap items-end gap-3">
              <div className="min-w-64 flex-1">
                <Field label="재색인할 모델" hint="data/corpus.jsonl 전체를 다시 임베딩합니다">
                  <select
                    value={model}
                    onChange={(e) => setModel(e.target.value)}
                    disabled={running}
                    className="w-full rounded-xl border border-line bg-ink-925 px-3.5 py-2.5 text-sm text-fg outline-none focus:border-signal/50"
                  >
                    <option value="">서버 기본 모델 (ST_MODEL)</option>
                    {(models.data?.models ?? []).map((m) => (
                      <option key={m} value={m}>
                        {m}
                      </option>
                    ))}
                  </select>
                </Field>
              </div>
              <Btn
                variant="ghost"
                icon={<RefreshCw size={15} className={running ? "animate-spin" : undefined} />}
                disabled={running || start.isPending}
                onClick={() => start.mutate(model || undefined)}
              >
                {running ? "재색인 중…" : "재색인"}
              </Btn>
            </div>

            {running && (
              <JobProgress done={job.data?.done ?? 0} total={job.data?.total ?? null} model={job.data?.model} />
            )}
            {job.data?.status === "failed" && <ErrorNote>재색인 실패 — {job.data.error}</ErrorNote>}
          </div>
        )}
      </Panel>
    </Section>
  );
}

function JobProgress({ done, total, model }: { done: number; total: number | null; model?: string | null }) {
  const pct = total ? Math.round((done / total) * 100) : 0;
  return (
    <div>
      <div className="mb-1.5 flex items-baseline justify-between text-[12px] text-mut">
        <span className="mono">{model}</span>
        <span className="mono">
          {done.toLocaleString()}{total ? ` / ${total.toLocaleString()}` : ""} ({pct}%)
        </span>
      </div>
      <div className="h-1.5 overflow-hidden rounded-full bg-ink-800">
        <div className="h-full rounded-full bg-signal transition-all duration-500" style={{ width: `${pct}%` }} />
      </div>
    </div>
  );
}

// ── 실검색 ─────────────────────────────────────────────────────────────────────
function SearchPanel() {
  const [query, setQuery] = useState("");
  const search = useMutation({ mutationFn: (q: string) => api.search(q, 10) });

  const submit = () => {
    const q = query.trim();
    if (q) search.mutate(q);
  };

  return (
    <Section delay={60}>
      <SectionLabel hint="쿼리 1건만 임베딩하므로 수백 ms 수준 (첫 검색은 모델 로드로 수 초)">실검색</SectionLabel>
      <Panel className="p-5">
        <div className="flex gap-3">
          <Input
            placeholder="검색어를 입력하세요 — 예: 청년 주거 지원 정책"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            onKeyDown={(e) => e.key === "Enter" && submit()}
          />
          <Btn icon={<SearchIcon size={15} />} onClick={submit} disabled={search.isPending || !query.trim()}>
            검색
          </Btn>
        </div>

        <div className="mt-4">
          {search.isPending && <Loading label="검색 중…" />}
          {search.isError && <ErrorNote>{(search.error as Error).message}</ErrorNote>}
          {search.isSuccess && <Results data={search.data} />}
        </div>
      </Panel>
    </Section>
  );
}

function Results({ data }: { data: SearchResponse }) {
  if (data.hits.length === 0)
    return <div className="py-8 text-center text-[13px] text-faint">결과가 없습니다.</div>;
  return (
    <div>
      <div className="mb-3 flex flex-wrap items-center gap-2 text-[12px] text-faint">
        <Tag tone="cyan">{data.model}</Tag>
        <span className="mono">{data.collection}</span>
      </div>
      <ol className="space-y-3">
        {data.hits.map((hit, i) => (
          <li key={`${hit.url ?? i}`} className="rounded-xl border border-line bg-ink-925/60 p-4">
            <div className="flex items-baseline gap-3">
              <span className="mono text-[12px] text-faint">{i + 1}</span>
              <span className="mono text-[12px] text-signal2">{fmt(hit.score)}</span>
              <span className="min-w-0 flex-1 truncate text-[14px] font-medium text-fg">
                {hit.title || "(제목 없음)"}
              </span>
              {hit.url && (
                <a
                  href={hit.url}
                  target="_blank"
                  rel="noreferrer"
                  className="inline-flex shrink-0 items-center gap-1 text-[12px] text-cyan hover:underline"
                >
                  원문 <ExternalLink size={12} />
                </a>
              )}
            </div>
            <p className="mt-2 line-clamp-2 text-[13px] leading-relaxed text-mut">{hit.content}</p>
          </li>
        ))}
      </ol>
    </div>
  );
}
