// 검색 — 학습한 모델이 Qdrant 인덱스에서 실제로 검색하는 서빙 플레이그라운드.
// 위: 인덱스 상태(alias → 컬렉션, 문서 수, dim·모델 일치) + 재색인(백그라운드 잡, 진행률
// 폴링) + 컬렉션 인벤토리(라이브 전환 = 즉시 롤백, 정리 = 롤백 사본 삭제).
// 아래: 실검색 — 쿼리는 서버 프로세스의 임베더(EMBEDDER/ST_MODEL)로 임베딩되므로
// 인덱스를 만든 모델과 같아야 의미가 있다 (dim 불일치는 서버가 503으로 막고,
// 같은 dim의 다른 모델은 model_matches 경고가 잡는다).
import { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { useMutation, useQueryClient } from "@tanstack/react-query";
import { ClipboardCheck, ExternalLink, RefreshCw, Search as SearchIcon, Trash2, Undo2 } from "lucide-react";

import { api } from "../lib/api";
import { PATH } from "../lib/nav";
import { cx, fmt } from "../lib/format";
import {
  keys,
  useIndexStatus,
  useModels,
  usePruneCollections,
  useSearchStatus,
  useSetAlias,
  useStartIndex,
} from "../lib/queries";
import type { CollectionInfo, SearchResponse } from "../lib/types";
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
                  {/* dim 가드가 못 잡는 함정: 같은 차원, 다른 모델 → 검색은 되지만 순위가 무의미 */}
                  {s.dim_matches !== false && s.model_matches === false && (
                    <Pill
                      tone="amber"
                      title="인덱스를 만든 모델과 쿼리를 임베딩하는 모델이 다릅니다. 차원이 같아 검색은 동작하지만 두 벡터는 다른 좌표계라 순위가 무의미합니다 — 현재 모델로 재색인하거나, 인덱스 모델의 컬렉션으로 전환하세요."
                    >
                      ⚠ 인덱스 모델 ≠ 쿼리 임베더 — 순위 무의미, 재색인 필요
                    </Pill>
                  )}
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

            {/* 사전 예방 — 몇 분짜리 임베딩을 돌리기 전에, 결과가 쓸모없을 조합임을 알린다 */}
            {!running && !!model && model !== s.model && (
              <p className="text-[12px] leading-relaxed text-amber">
                ⚠ 이 모델로 색인해도 쿼리는 여전히 서버 임베더(<span className="mono">{s.model}</span>)가
                임베딩하므로 순위가 무의미해집니다. 이 모델을 서빙하려면 핸드오프(모델 탭)하거나 서버의{" "}
                <span className="mono">ST_MODEL</span>을 바꾸세요.
              </p>
            )}
            {running && (
              <JobProgress done={job.data?.done ?? 0} total={job.data?.total ?? null} model={job.data?.model} />
            )}
            {job.data?.status === "failed" && <ErrorNote>재색인 실패 — {job.data.error}</ErrorNote>}

            {s.collections.length > 0 && <Collections items={s.collections} running={running} />}
          </div>
        )}
      </Panel>
    </Section>
  );
}

// ── 컬렉션 인벤토리 — 라이브 전환(즉시 롤백) · 정리(롤백 사본 삭제) ────────────────
function Collections({ items, running }: { items: CollectionInfo[]; running: boolean }) {
  const setAlias = useSetAlias();
  const prune = usePruneCollections();
  const stale = items.filter((c) => !c.live).length;

  return (
    <div className="border-t border-line/60 pt-4">
      <div className="mb-2.5 flex items-center justify-between">
        <div className="flex items-center gap-1.5 text-[12px] font-medium text-mut">
          컬렉션 {items.length}개
          <Info title="버전 컬렉션과 롤백" align="left">
            재색인은 (모델·차원·코퍼스 지문)마다 새 컬렉션을 만들고 alias만 옮깁니다. 이전 컬렉션은
            그대로 남아 있으므로 <b className="text-fg">라이브 전환</b>은 재임베딩 없이 즉시 롤백이
            됩니다. 새 인덱스가 좋다고 확인되면 <b className="text-fg">정리</b>로 라이브가 아닌
            컬렉션을 삭제해 디스크를 회수하세요.
          </Info>
        </div>
        {stale > 0 && <PruneBtn count={stale} disabled={running || prune.isPending} onPrune={() => prune.mutate()} />}
      </div>
      <div className="overflow-x-auto rounded-xl border border-line">
        <table className="w-full text-left text-[12px]">
          <thead>
            <tr className="border-b border-line bg-ink-880/60 text-[10.5px] uppercase tracking-wider text-faint">
              <th className="px-3 py-2 font-medium">모델</th>
              <th className="px-3 py-2 font-medium">dim</th>
              <th className="px-3 py-2 font-medium">문서</th>
              <th className="px-3 py-2 font-medium">코퍼스 지문</th>
              <th className="px-3 py-2 font-medium" />
            </tr>
          </thead>
          <tbody>
            {items.map((c, i) => (
              <tr key={c.name} className={cx(i < items.length - 1 && "border-b border-line/60", c.live && "bg-signal/[0.04]")}>
                <td className="max-w-72 px-3 py-2">
                  <span className="mono block truncate text-fg" title={c.name}>
                    {c.model_slug ?? c.name}
                  </span>
                </td>
                <td className="mono px-3 py-2 text-mut">{c.dim ?? "—"}</td>
                <td className="mono px-3 py-2 text-mut">{c.points.toLocaleString()}</td>
                <td className="mono px-3 py-2 text-faint">{c.fingerprint ?? "—"}</td>
                <td className="px-3 py-2 text-right">
                  {c.live ? (
                    <Tag tone="signal">LIVE</Tag>
                  ) : (
                    <button
                      onClick={() => setAlias.mutate(c.name)}
                      disabled={running || setAlias.isPending}
                      title="alias를 이 컬렉션으로 전환 — 재임베딩 없이 즉시 적용"
                      className="inline-flex items-center gap-1 rounded-md px-1.5 py-0.5 text-[11px] text-faint transition-colors hover:bg-ink-800 hover:text-fg disabled:opacity-40"
                    >
                      <Undo2 size={12} /> 라이브 전환
                    </button>
                  )}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

/** 정리(prune) — 파괴적이라 2클릭(armed) 패턴: 한 번 누르면 3초간 확인 상태. */
function PruneBtn({ count, disabled, onPrune }: { count: number; disabled: boolean; onPrune: () => void }) {
  const [armed, setArmed] = useState(false);
  useEffect(() => {
    if (!armed) return;
    const t = setTimeout(() => setArmed(false), 3000);
    return () => clearTimeout(t);
  }, [armed]);
  if (armed) {
    return (
      <button
        onClick={() => {
          onPrune();
          setArmed(false);
        }}
        disabled={disabled}
        className="mono rounded-md bg-danger/15 px-2 py-1 text-[11px] font-semibold text-danger hover:bg-danger/25 disabled:opacity-40"
      >
        롤백 사본 {count}개 삭제?
      </button>
    );
  }
  return (
    <button
      onClick={() => setArmed(true)}
      disabled={disabled}
      title="라이브가 아닌 컬렉션 삭제 — 롤백이 불가능해집니다"
      className="inline-flex items-center gap-1 text-[11.5px] text-faint transition-colors hover:text-danger disabled:opacity-40"
    >
      <Trash2 size={13} /> 정리
    </button>
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
  const [topK, setTopK] = useState(10);
  const search = useMutation({ mutationFn: (q: string) => api.search(q, topK) });

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
          <select
            value={topK}
            onChange={(e) => setTopK(Number(e.target.value))}
            title="가져올 후보 수 — 프로덕션 recall 깊이(50)로도 확인해 보세요"
            className="mono shrink-0 rounded-xl border border-line bg-ink-925 px-2.5 py-2.5 text-[12.5px] text-mut outline-none focus:border-signal/50"
          >
            {[5, 10, 20, 50].map((k) => (
              <option key={k} value={k}>
                top {k}
              </option>
            ))}
          </select>
          <Btn icon={<SearchIcon size={15} />} className="shrink-0" onClick={submit} disabled={search.isPending || !query.trim()}>
            검색
          </Btn>
        </div>

        <div className="mt-4">
          {search.isPending && <Loading label="검색 중…" />}
          {search.isError && <ErrorNote>{(search.error as Error).message}</ErrorNote>}
          {search.isSuccess && <Results data={search.data} query={search.variables ?? ""} />}
        </div>
      </Panel>
    </Section>
  );
}

function Results({ data, query }: { data: SearchResponse; query: string }) {
  if (data.hits.length === 0)
    return <div className="py-8 text-center text-[13px] text-faint">결과가 없습니다.</div>;
  return (
    <div>
      <div className="mb-3 flex flex-wrap items-center gap-2 text-[12px] text-faint">
        <Tag tone="cyan">{data.model}</Tag>
        <span className="mono">{data.collection}</span>
        {/* 결과가 이상하다 → 그 자리에서 판정해 평가셋(qrels)으로 만드는 지름길 */}
        <Link
          to={`${PATH.data}?label=${encodeURIComponent(query)}`}
          title="이 쿼리를 데이터 탭의 라벨링으로 가져가 정답을 판정합니다 — 판정하면 평가셋(qrels)이 됩니다"
          className="inline-flex items-center gap-1 text-cyan hover:underline"
        >
          <ClipboardCheck size={12} /> 이 쿼리 판정 → 평가셋
        </Link>
        <span
          className="mono ml-auto"
          title="임베딩 = 쿼리를 벡터로 (모델 추론) · 검색 = Qdrant ANN 조회 — 느리면 어느 쪽이 병목인지 여기서 갈립니다"
        >
          임베딩 {data.embed_ms}ms · 검색 {data.search_ms}ms
        </span>
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
