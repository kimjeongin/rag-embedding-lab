import { useEffect, useRef, useState } from "react";
import { useNavigate, useSearchParams } from "react-router-dom";
import { useQuery, useQueryClient } from "@tanstack/react-query";
import { toast } from "sonner";
import { Database, FileText, FlaskConical, Gauge, Globe, Sparkles } from "lucide-react";

import { api } from "../lib/api";
import { short } from "../lib/format";
import { PATH } from "../lib/nav";
import {
  keys,
  useCorpus,
  useDataOverview,
  useGenEval,
  useGenPairs,
  useImportClicklog,
  useImportPairs,
  useLabelCommit,
  useModels,
  usePairs,
  useStatus,
} from "../lib/queries";
import { startCrawl, useCrawlState } from "../lib/crawlStore";
import { startSynthetic, useSyntheticState } from "../lib/syntheticStore";
import { DataTable } from "../components/DataTable";
import { Modal } from "../components/Modal";
import { Btn, ErrorNote, Field, Info, Input, Loading, Panel, Section, SectionLabel, Seg, Stat, Tag } from "../components/ui";
import { useMutation } from "@tanstack/react-query";
import { Search, Upload } from "lucide-react";
import type { Embedder, LabelDoc } from "../lib/types";

type Method = "toy" | "synthetic";

/** Crawl a public site into data/corpus.jsonl — the PoC stand-in for the internal
 * site. Page = retrieval unit; the synthetic generator reads this file next. */
function CrawlPanel({ corpusFile, onCorpusFile }: { corpusFile: string; onCorpusFile: (v: string) => void }) {
  const crawl = useCrawlState();
  const [url, setUrl] = useState("https://www.korea.kr/sitemap_policy.xml");
  const [maxPages, setMaxPages] = useState(300);
  const running = crawl.status === "running";

  return (
    <Panel className="p-5">
      <div className="mb-4 flex items-center gap-2">
        <Globe size={16} className="text-signal" />
        <h3 className="text-[15px] font-semibold text-fg">코퍼스 수집 — 사이트 크롤</h3>
        <Info title="사내사이트의 공개 프록시" align="left">
          사이트 루트나 <span className="mono">sitemap.xml</span>을 주면 <b className="text-fg">페이지 단위</b>로
          본문을 추출해 corpus를 만듭니다 (robots.txt 준수 · 순차 요청). 페이지 텍스트는{" "}
          <b className="text-fg">title + 본문 앞부분</b>(FirstP) — 웹 페이지는 핵심이 앞에 옵니다. 이 corpus가
          학습쌍 생성과 corpus 평가셋의 입력이 됩니다.
        </Info>
      </div>
      <div className="grid gap-3 sm:grid-cols-[1fr_auto_auto]">
        <Field label="시작 URL" hint="사이트 루트 또는 sitemap.xml">
          <Input value={url} onChange={(e) => setUrl(e.target.value)} className="mono" placeholder="https://…/sitemap.xml" />
        </Field>
        <Field label="최대 페이지">
          <Input type="number" value={maxPages} onChange={(e) => setMaxPages(+e.target.value)} className="mono w-24" />
        </Field>
        <Field label="저장 파일" hint="학습쌍 생성의 corpus 파일과 동일">
          <Input value={corpusFile} onChange={(e) => onCorpusFile(e.target.value)} className="mono w-44" />
        </Field>
      </div>
      <div className="mt-4">
        <Btn icon={<Globe size={15} />} disabled={!url.trim() || running} onClick={() => startCrawl({ url: url.trim(), max_pages: maxPages, corpus_file: corpusFile })}>
          {running ? `크롤 중… ${crawl.done}/${crawl.total || "?"}` : "크롤 시작"}
        </Btn>
      </div>

      {crawl.status !== "idle" && (
        <div className="mt-4 rounded-xl border border-line bg-ink-925/60 p-4">
          <div className="mb-2 flex items-center justify-between text-[12px]">
            <span className="font-medium text-mut">
              {crawl.status === "error"
                ? "오류 발생"
                : crawl.status === "done"
                  ? `완료 — ${crawl.result?.count ?? 0} 페이지 (fetch ${crawl.result?.fetched ?? 0} · skip ${crawl.result?.skipped ?? 0})`
                  : `수집 중 · ${crawl.done}/${crawl.total || "?"} 페이지`}
            </span>
            {crawl.mode && <Tag tone="cyan">{crawl.mode === "sitemap" ? "sitemap 발견" : "링크 따라가기(BFS)"} · {crawl.discovered} URL</Tag>}
          </div>
          <div className="h-1.5 w-full overflow-hidden rounded-full bg-ink-700">
            <div
              className={`h-full rounded-full transition-all ${crawl.status === "error" ? "bg-danger" : "bg-signal"}`}
              style={{ width: `${crawl.status === "done" ? 100 : crawl.total ? Math.round((crawl.done / crawl.total) * 100) : 8}%` }}
            />
          </div>
          {crawl.pages.length > 0 && crawl.status === "running" && (
            <div className="mono mt-3 max-h-32 overflow-auto rounded-lg border border-line bg-ink-950 p-3 text-[11px] leading-relaxed">
              {crawl.pages.slice(-6).map((p, i) => (
                <div key={i} className="truncate text-mut">▸ {p.title || p.url}</div>
              ))}
            </div>
          )}
          {crawl.status === "error" && crawl.error && (
            <div className="mt-3">
              <ErrorNote>{crawl.error}</ErrorNote>
            </div>
          )}
        </div>
      )}
    </Panel>
  );
}

/** Paste a query/click log → training pairs and/or qrels. The single highest-value
 * data source for an internal-site search: (쿼리, 클릭한 사이트) = 학습쌍이자 정답 판정. */
function ImportPanel() {
  const importPairs = useImportPairs();
  const importClicklog = useImportClicklog();
  const [mode, setMode] = useState<"pairs" | "clicklog">("pairs");
  const [content, setContent] = useState("");
  const [target, setTarget] = useState<"train" | "qrels" | "both">("both");
  const result = importPairs.data;
  const logReport = importClicklog.data?.report;
  const skipped = mode === "pairs" ? (result?.skipped ?? []) : (importClicklog.data?.skipped ?? []);
  const busy = importPairs.isPending || importClicklog.isPending;
  const submit = () => {
    const done = { onSuccess: () => setContent("") };
    if (mode === "pairs") importPairs.mutate({ content, target }, done);
    else importClicklog.mutate({ content }, done);
  };

  return (
    <Panel className="p-5">
      <div className="mb-4 flex items-center gap-2">
        <Upload size={16} className="text-signal" />
        <h3 className="text-[15px] font-semibold text-fg">실데이터 가져오기</h3>
        {mode === "pairs" ? (
          <Info title="실로그가 최고의 데이터입니다" align="left">
            사내 검색의 <b className="text-fg">쿼리 로그·클릭 로그</b>가 이미 최고의 학습/평가 데이터입니다. (쿼리,
            클릭한 문서) 한 줄이 <b className="text-fg">MNRL 학습쌍</b>이자 <b className="text-fg">정답 판정(qrels)</b> —
            개별 클릭은 노이즈여도 양으로 이깁니다. 형식: CSV <span className="mono">query,doc_id</span>(헤더 생략
            가능) 또는 JSONL <span className="mono">{'{"query": …, "doc_id": …}'}</span>. doc_id 대신{" "}
            <span className="mono">title/content</span>를 주면 학습쌍으로만 들어갑니다.
          </Info>
        ) : (
          <Info title="세션 로그는 클릭보다 많이 안다" align="left">
            세션 단위 원본 로그를 붙여넣으면 <b className="text-fg">노이즈 규칙</b>을 거쳐 학습쌍이 됩니다: 짧은
            dwell 클릭은 바운스로 무시, <b className="text-fg">재검색 세션의 실패한 앞 쿼리는 최종 만족 문서에
            연결</b>(현행 엔진이 못 푸는 사내 표현의 supervision이 여기서 나옵니다), 만족 클릭 위에서 스킵된 문서는
            hard negative, PII 쿼리는 통째로 드롭. 형식: JSONL 한 줄 = 검색 1회{" "}
            <span className="mono">{'{"session", "query", "results": [doc_id…], "clicks": [{"doc_id","rank","dwell_sec"}…]}'}</span>
          </Info>
        )}
      </div>
      <div className="mb-3">
        <Seg
          options={[
            { value: "pairs", label: "쌍 (query,doc)" },
            { value: "clicklog", label: "클릭로그 (세션)" },
          ]}
          value={mode}
          onChange={setMode}
        />
      </div>
      <textarea
        value={content}
        onChange={(e) => setContent(e.target.value)}
        rows={5}
        placeholder={
          mode === "pairs"
            ? "vpn 안됨,site-vpn-guide\n연차 신청,site-hr-portal\n…  (또는 JSONL)"
            : '{"session": "s-1", "query": "돌핀 정산 안돼", "results": ["page-3", …], "clicks": []}\n{"session": "s-1", "query": "머니핀 정산", "results": […], "clicks": [{"doc_id": "page-17", "rank": 1, "dwell_sec": 95}]}'
        }
        className="mono w-full rounded-xl border border-line bg-ink-925 px-3.5 py-2.5 text-[12px] text-fg outline-none placeholder:text-faint focus:border-signal/50"
      />
      <div className="mt-3 flex flex-wrap items-center gap-2.5">
        {mode === "pairs" ? (
          <Seg
            options={[
              { value: "both", label: "학습쌍 + qrels" },
              { value: "train", label: "학습쌍만" },
              { value: "qrels", label: "qrels만" },
            ]}
            value={target}
            onChange={setTarget}
          />
        ) : (
          <span className="text-[11.5px] text-faint">클리닝을 거쳐 학습쌍으로만 들어갑니다 (판정은 판정 루프에서)</span>
        )}
        <Btn icon={<Upload size={14} />} disabled={!content.trim() || busy} onClick={submit}>
          {busy ? "가져오는 중…" : mode === "pairs" ? "가져오기" : "클리닝해서 가져오기"}
        </Btn>
      </div>
      {mode === "clicklog" && logReport && (
        <div className="mt-3 flex flex-wrap gap-1.5">
          <Tag tone="signal">직접 쌍 {logReport.positives_direct}</Tag>
          <Tag tone="signal">재검색 전이 {logReport.positives_transferred}</Tag>
          <Tag>hard negative {logReport.hard_negatives}</Tag>
          <Tag tone="mut">바운스 무시 {logReport.bounces_ignored}</Tag>
          <Tag tone="mut">PII 드롭 {logReport.dropped_pii}</Tag>
          <Tag tone="mut">이탈 세션 {logReport.abandoned_sessions}</Tag>
        </div>
      )}
      {skipped.length > 0 && (
        <div className="mono mt-3 max-h-24 overflow-auto rounded-lg border border-amber/25 bg-amber/8 p-2.5 text-[11px] text-amber">
          {skipped.map((s, i) => (
            <div key={i}>· {s}</div>
          ))}
        </div>
      )}
      {mode === "pairs" && result?.fingerprint_changed && (
        <p className="mt-2 text-[11.5px] text-amber">
          평가셋 내용이 바뀌었습니다 — 이전 런들은 "다른 평가셋"으로 표시되며 새 런과 비교되지 않습니다 (의도된 동작).
        </p>
      )}
    </Panel>
  );
}

/** The judging loop: type a real query → see what the current model retrieves →
 * click what's actually relevant → it becomes qrels (+ a training pair). */
function LabelPanel() {
  // 검색 탭의 "이 쿼리 판정 → 평가셋" 링크가 ?label=<쿼리>로 들어온다 — 프리필 + 자동 검색
  const [params, setParams] = useSearchParams();
  const prefill = params.get("label") ?? "";
  const anchor = useRef<HTMLDivElement>(null);
  const [backend, setBackend] = useState<Embedder>("sentence-transformers");
  const models = useModels(backend);
  const [override, setOverride] = useState("");
  const model = override || models.data?.default || "";
  const [query, setQuery] = useState(prefill);
  const [picked, setPicked] = useState<Set<string>>(new Set());
  const [alsoTrain, setAlsoTrain] = useState(true);
  const commit = useLabelCommit();
  const search = useMutation({
    mutationFn: () => api.labelSearch({ query: query.trim(), embedder: backend, model }),
    onError: (e) => toast.error((e as Error).message),
    onSuccess: () => setPicked(new Set()),
  });
  const results: LabelDoc[] = search.data?.results ?? [];

  useEffect(() => {
    if (prefill) anchor.current?.scrollIntoView({ behavior: "smooth", block: "center" });
  }, [prefill]);
  // 자동 검색은 타이머로 지연 발사 — StrictMode가 마운트 이펙트를 2회 돌리는 창 안에서
  // mutate하면 관찰이 끊겨 pending이 고착된다(클린업으로 타이머만 취소되고 재예약됨)
  useEffect(() => {
    if (!prefill || !model || !search.isIdle) return;
    const timer = setTimeout(() => search.mutate(), 50);
    return () => clearTimeout(timer);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [prefill, model, search.isIdle]);

  const togglePick = (id: string) =>
    setPicked((s) => {
      const next = new Set(s);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });

  return (
    <Panel className="p-5">
      <div ref={anchor} className="mb-4 flex items-center gap-2">
        <Search size={16} className="text-cyan" />
        <h3 className="text-[15px] font-semibold text-fg">라벨링 — 정답 판정으로 평가셋 키우기</h3>
        <Info title="판정 루프" align="left">
          실제 쿼리를 넣으면 <b className="text-fg">현재 모델의 top-10</b>이 나옵니다. 정답인 문서를 클릭해 저장하면{" "}
          <b className="text-fg">qrels(+학습쌍)</b>가 됩니다. 하루 10개씩만 판정해도 평가셋이 점점 "진짜"가 돼요 —
          평가셋의 신뢰가 모든 비교의 전제입니다. 판정 모델이 서빙 인덱스와 같으면(기본) 인덱스 벡터를 재사용해
          즉시 나옵니다 — 다른 모델은 첫 검색에서 corpus 전체를 임베딩해 오래 걸립니다(모델·코퍼스당 한 번).
        </Info>
      </div>
      <div className="grid items-end gap-3 sm:grid-cols-[1fr_auto]">
        <Field label="실제 사용자 쿼리" hint="짧고 거친 실쿼리일수록 좋아요">
          <Input
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            onKeyDown={(e) => e.key === "Enter" && query.trim() && model && search.mutate()}
            placeholder="예: vpn 안됨"
          />
        </Field>
        <Btn icon={<Search size={14} />} disabled={!query.trim() || !model || search.isPending} onClick={() => search.mutate()}>
          {search.isPending ? "검색 중…" : "검색"}
        </Btn>
      </div>
      <div className="mt-2 flex flex-wrap items-center gap-2">
        <Seg
          options={[
            { value: "sentence-transformers", label: "학습 모델" },
            { value: "ollama", label: "Ollama" },
          ]}
          value={backend}
          onChange={(b) => {
            setBackend(b);
            setOverride("");
          }}
        />
        <Input value={model} onChange={(e) => setOverride(e.target.value)} list="label-models" className="mono max-w-xs !w-auto flex-1 text-[12px]" placeholder="모델" />
        <datalist id="label-models">
          {(models.data?.models ?? []).map((m) => (
            <option key={m} value={m} />
          ))}
        </datalist>
      </div>

      {results.length > 0 && (
        <>
          <div className="mt-4 max-h-60 space-y-1 overflow-auto">
            {results.map((d, i) => (
              <button
                key={d.id}
                onClick={() => togglePick(d.id)}
                className={`flex w-full items-start gap-2.5 rounded-lg border px-3 py-2 text-left transition-colors ${
                  picked.has(d.id) ? "border-signal/50 bg-signal/8" : "border-line bg-ink-925/50 hover:border-line2"
                }`}
              >
                <span className="mono pt-0.5 text-[10.5px] text-faint">{i + 1}</span>
                <span className="min-w-0 flex-1">
                  <span className={`block truncate text-[12.5px] ${picked.has(d.id) ? "font-medium text-signal" : "text-fg"}`}>
                    {d.title ?? d.id}
                  </span>
                  <span className="block truncate text-[11px] text-faint">{d.text}</span>
                </span>
                {picked.has(d.id) && <span className="mono pt-0.5 text-[10.5px] text-signal">정답 ✓</span>}
              </button>
            ))}
          </div>
          <div className="mt-3 flex flex-wrap items-center gap-3">
            <Btn
              disabled={picked.size === 0 || commit.isPending}
              onClick={() =>
                commit.mutate(
                  { query: query.trim(), doc_ids: [...picked], also_train: alsoTrain },
                  {
                    onSuccess: () => {
                      setPicked(new Set());
                      setQuery("");
                      search.reset();
                      // 판정 완료 — ?label 프리필을 지워 자동 검색이 재발사되지 않게
                      if (prefill) setParams({}, { replace: true });
                    },
                  },
                )
              }
            >
              {commit.isPending ? "저장 중…" : `정답 ${picked.size}개 저장`}
            </Btn>
            <label className="flex cursor-pointer items-center gap-2 text-[12px] text-mut">
              <input type="checkbox" checked={alsoTrain} onChange={(e) => setAlsoTrain(e.target.checked)} className="h-3.5 w-3.5 accent-[#c6f24a]" />
              학습쌍에도 추가
            </label>
          </div>
        </>
      )}
    </Panel>
  );
}

export default function Data() {
  const overview = useDataOverview();
  const status = useStatus();
  const pairs = usePairs();
  const corpus = useCorpus();
  const genPairs = useGenPairs();
  const genEval = useGenEval();
  const synth = useSyntheticState();
  const qc = useQueryClient();
  const nav = useNavigate();

  const [method, setMethod] = useState<Method>("toy");
  const [corpusFile, setCorpusFile] = useState("data/corpus.jsonl");
  const [genModel, setGenModel] = useState("qwen3.5:2b");
  const [nQueries, setNQueries] = useState(5);
  const [hardNegatives, setHardNegatives] = useState(4);
  const [roundTripK, setRoundTripK] = useState(1);
  const [negMargin, setNegMargin] = useState(0.05);
  const [evalSource, setEvalSource] = useState<"sample" | "corpus">("sample");
  const [ndist, setNdist] = useState(448);
  const [modal, setModal] = useState<null | "pairs" | "corpus">(null);

  const fullPairs = useQuery({ queryKey: ["data", "pairs", "full"], queryFn: () => api.pairs(10000, false), enabled: modal === "pairs" });
  const fullCorpus = useQuery({ queryKey: ["data", "corpus", "full"], queryFn: () => api.corpus(10000), enabled: modal === "corpus" });

  // Synthetic streams its own progress; on finish, refresh the data views (the toy path
  // goes through the mutation, which already invalidates).
  useEffect(() => {
    if (synth.status === "done" && synth.result) {
      toast.success(synth.result.message);
      qc.invalidateQueries({ queryKey: keys.dataOverview });
      qc.invalidateQueries({ queryKey: keys.pairs });
      qc.invalidateQueries({ queryKey: keys.status });
    }
    if (synth.status === "error" && synth.error) toast.error(synth.error);
  }, [synth.status, synth.result, synth.error, qc]);

  const synthRunning = synth.status === "running";
  // AI 자동 생성은 Ollama의 LLM이 쿼리를 쓰므로, 죽어 있으면 누르기 전에 알린다
  const ollamaDown = status.data ? !status.data.ollama.reachable : false;
  const busy = method === "toy" ? genPairs.isPending : synthRunning;
  const busyLabel =
    method === "synthetic" && synthRunning
      ? synth.mining
        ? "오답 마이닝 중…"
        : `생성 중… ${synth.done}/${synth.total || "?"}`
      : "생성 중…";

  const runGenPairs = () => {
    if (method === "toy") {
      genPairs.mutate({ method: "toy" });
    } else {
      startSynthetic({
        method: "synthetic",
        corpus_file: corpusFile,
        gen_model: genModel,
        n_queries: nQueries,
        hard_negatives: hardNegatives,
        round_trip_k: roundTripK,
        neg_margin: negMargin,
      });
    }
  };

  const inv = overview.data;

  return (
    <div className="space-y-9">
      <Section>
        <SectionLabel hint="각 파일의 레코드 수 · 다음 단계로 흘러갑니다">
          <span className="inline-flex items-center gap-1.5">
            보유 데이터
            <Info title="두 종류 데이터" align="left">
              <b className="text-fg">학습 데이터</b>(training pairs: query↔정답 + hard negative)는 모델이{" "}
              <i>배우는</i> 것, <b className="text-fg">평가셋</b>(eval set: corpus / queries / qrels)은{" "}
              <i>채점하는</i> 것입니다. 둘의 query는 겹치면 안 됩니다 — 학습한 질문으로 평가하면 일반화가 아니라 암기를
              재게 되니까요(leakage).
            </Info>
          </span>
        </SectionLabel>
        {overview.isLoading ? (
          <Loading />
        ) : overview.error ? (
          <ErrorNote>{(overview.error as Error).message}</ErrorNote>
        ) : inv ? (
          <div className="grid grid-cols-2 gap-3 lg:grid-cols-4">
            <Stat label="학습쌍 train" value={inv.train.count} tag="→ 학습" tone="signal" sub={short(inv.train.file)} />
            <Stat label="학습쌍 test" value={inv.test.count} tag="→ 검증" tone="signal" sub={short(inv.test.file)} />
            <Stat label="평가 corpus" value={inv.eval.corpus} tag="→ 평가" tone="cyan" sub="corpus.jsonl" />
            <Stat label="평가 queries" value={inv.eval.queries} tag="→ 평가" tone="cyan" sub="queries.jsonl" />
          </div>
        ) : null}
      </Section>

      <Section delay={50}>
        <SectionLabel hint="공개 사이트 = 사내사이트의 PoC 프록시 · 여기서 만든 corpus가 아래 두 카드의 입력입니다">
          0단계 — 코퍼스
        </SectionLabel>
        <CrawlPanel corpusFile={corpusFile} onCorpusFile={setCorpusFile} />
      </Section>

      <Section delay={70}>
        <div className="grid gap-5 lg:grid-cols-2">
          {/* training pairs */}
          <Panel className="p-5">
            <div className="mb-4 flex items-center gap-2">
              <Database size={16} className="text-signal" />
              <h3 className="text-[15px] font-semibold text-fg">학습 데이터</h3>
              <Tag tone="signal">→ 학습</Tag>
            </div>
            <Seg
              options={[
                { value: "toy", label: "예제" },
                { value: "synthetic", label: "AI 자동 생성" },
              ]}
              value={method}
              onChange={setMethod}
            />
            <p className="mt-3 text-[12.5px] leading-relaxed text-mut">
              예제 = 미리 만든 (질문, 정답) 샘플 · AI 자동 = 내 corpus를 LLM이 읽고 질문을 작성.
            </p>

            {method === "synthetic" && (
              <div className="mt-4 grid grid-cols-2 gap-3 rounded-xl border border-line bg-ink-925/60 p-3.5">
                <Field label="corpus 파일" hint="title·content jsonl">
                  <Input value={corpusFile} onChange={(e) => setCorpusFile(e.target.value)} className="mono" />
                </Field>
                <Field label="생성 모델" hint="Ollama chat">
                  <Input value={genModel} onChange={(e) => setGenModel(e.target.value)} className="mono" />
                </Field>
                <Field label="문서당 질문 수">
                  <Input type="number" min={1} value={nQueries} onChange={(e) => setNQueries(+e.target.value)} className="mono" />
                </Field>
                <Field label="hard negative 수" hint="쌍마다 채굴할 유사 오답">
                  <Input type="number" min={0} max={10} value={hardNegatives} onChange={(e) => setHardNegatives(+e.target.value)} className="mono" />
                </Field>
                <Field label="라운드트립 필터 k" hint="쿼리가 자기 문서를 top-k에 못 올리면 버림 · train만 · 0=off">
                  <Input type="number" min={0} max={10} value={roundTripK} onChange={(e) => setRoundTripK(+e.target.value)} className="mono" />
                </Field>
                <Field label="오답 margin" hint="정답 점수의 (1−m)배 넘는 후보는 가짜 오답으로 제외 · 0=off">
                  <Input type="number" min={0} max={0.5} step={0.01} value={negMargin} onChange={(e) => setNegMargin(+e.target.value)} className="mono" />
                </Field>
              </div>
            )}

            {method === "synthetic" && ollamaDown && (
              <p className="mt-3 text-[12px] leading-relaxed text-amber">
                ⚠ Ollama에 연결할 수 없습니다 — AI 자동 생성은 Ollama의 LLM이 corpus를 읽고 질문을 씁니다.{" "}
                <span className="mono">ollama serve</span>로 띄운 뒤 다시 시도하세요. (예제 데이터는 Ollama 없이 됩니다)
              </p>
            )}
            <div className="mt-4 flex flex-wrap gap-2.5">
              <Btn icon={<Sparkles size={15} />} onClick={runGenPairs} disabled={busy || (method === "synthetic" && ollamaDown)}>
                {busy ? busyLabel : "학습 데이터 생성"}
              </Btn>
              <Btn variant="ghost" icon={<FileText size={15} />} onClick={() => setModal("pairs")}>
                전체 보기 {pairs.data ? `(${pairs.data.total})` : ""}
              </Btn>
            </div>

            {method === "synthetic" && synth.status !== "idle" && (
              <div className="mt-4 rounded-xl border border-line bg-ink-925/60 p-4">
                <div className="mb-2 flex items-center justify-between text-[12px]">
                  <span className="font-medium text-mut">
                    {synth.status === "error"
                      ? "오류 발생"
                      : synth.status === "done"
                        ? "완료"
                        : synth.mining
                          ? "유사 오답(hard negative) 마이닝 중…"
                          : `LLM이 쿼리 생성 중 · 문서 ${synth.done}/${synth.total || "?"}`}
                  </span>
                  {synth.thinkingDisabled && <Tag tone="cyan">추론 모드 끔 · 속도↑</Tag>}
                </div>
                <div className="h-1.5 w-full overflow-hidden rounded-full bg-ink-700">
                  <div
                    className={`h-full rounded-full transition-all ${synth.status === "error" ? "bg-danger" : "bg-signal"}`}
                    style={{
                      width: `${
                        synth.status === "done"
                          ? 100
                          : synth.total
                            ? Math.round((synth.done / synth.total) * 100)
                            : 8
                      }%`,
                    }}
                  />
                </div>
                {synth.filtered && (
                  <div className="mono mt-2 text-[11px] text-amber">
                    라운드트립 필터(train 스플릿만): {synth.filtered.kept} 통과 · {synth.filtered.dropped} 탈락 (top-{synth.filtered.k} 기준)
                  </div>
                )}
                {synth.docs.length > 0 && (
                  <div className="mono mt-3 max-h-44 overflow-auto rounded-lg border border-line bg-ink-950 p-3 text-[11px] leading-relaxed">
                    {synth.docs.slice(-6).map((d, i) => (
                      <div key={i} className="mb-1.5 last:mb-0">
                        <span className="text-faint">▸ {short(d.title) || "(제목 없음)"}</span>
                        {d.queries.map((q, j) => (
                          <div key={j} className="pl-3 text-mut">
                            · {q}
                          </div>
                        ))}
                      </div>
                    ))}
                  </div>
                )}
                {synth.status === "error" && synth.error && (
                  <div className="mt-3">
                    <ErrorNote>{synth.error}</ErrorNote>
                  </div>
                )}
              </div>
            )}

            <div className="mt-5">
              <div className="mb-2 text-[11px] uppercase tracking-wider text-faint">미리보기 · 앞 5개</div>
              {pairs.isLoading ? (
                <Loading />
              ) : (
                <DataTable
                  cols={["query", "정답 제목"]}
                  rows={(pairs.data?.items ?? []).slice(0, 5).map((p) => [p.query, p.title])}
                />
              )}
            </div>
          </Panel>

          {/* eval set */}
          <Panel className="p-5">
            <div className="mb-4 flex items-center gap-2">
              <Gauge size={16} className="text-cyan" />
              <h3 className="text-[15px] font-semibold text-fg">평가 데이터</h3>
              <Tag tone="cyan">→ 평가</Tag>
            </div>
            <Seg
              options={[
                { value: "sample", label: "샘플 (합성 distractor)" },
                { value: "corpus", label: "크롤 corpus (사이트 전체)" },
              ]}
              value={evalSource}
              onChange={setEvalSource}
            />
            {evalSource === "sample" ? (
              <div className="mt-3">
                <Field label="distractor 수" hint="많을수록 난이도 ↑ · 모델 차이가 잘 드러남">
                  <div className="flex items-center gap-3">
                    <input
                      type="range"
                      min={16}
                      max={448}
                      step={16}
                      value={ndist}
                      onChange={(e) => setNdist(+e.target.value)}
                      className="h-1.5 flex-1 cursor-pointer appearance-none rounded-full bg-ink-700 accent-signal"
                    />
                    <span className="mono w-12 text-right text-[15px] font-semibold text-fg">{ndist}</span>
                  </div>
                </Field>
              </div>
            ) : (
              <p className="mt-3 text-[12.5px] leading-relaxed text-mut">
                크롤한 <span className="mono">{corpusFile}</span>의 <b className="text-fg">사이트 전체가 건초더미</b>가
                되고, 학습에서 제외해 둔 test 스플릿 쿼리가 평가 쿼리가 됩니다 (dev/final 자동 분리). test 스플릿은
                라운드트립 필터를 <b className="text-fg">안 거친</b> 쿼리라 베이스 모델 점수가 포화되지 않습니다.
              </p>
            )}
            <div className="mt-4 flex flex-wrap gap-2.5">
              <Btn
                icon={<Sparkles size={15} />}
                onClick={() =>
                  genEval.mutate(
                    evalSource === "sample"
                      ? { source: "sample", n_distractors: ndist }
                      : { source: "corpus", corpus_file: corpusFile },
                  )
                }
                disabled={genEval.isPending}
              >
                {genEval.isPending ? "생성 중…" : "평가 데이터 생성"}
              </Btn>
              <Btn variant="ghost" icon={<FileText size={15} />} onClick={() => setModal("corpus")}>
                전체 보기 {corpus.data ? `(${corpus.data.total})` : ""}
              </Btn>
            </div>

            <div className="mt-5">
              <div className="mb-2 text-[11px] uppercase tracking-wider text-faint">corpus 미리보기 · 앞 5개</div>
              {corpus.isLoading ? (
                <Loading />
              ) : (
                <DataTable
                  cols={["_id", "title"]}
                  rows={(corpus.data?.items ?? []).slice(0, 5).map((c) => [c.id, c.title])}
                />
              )}
            </div>
          </Panel>
        </div>
      </Section>

      <Section delay={100}>
        <SectionLabel hint="합성 데이터는 시작점 — 실로그가 쌓일수록 평가가 진짜가 됩니다">
          실데이터 (검색 로그 · 판정)
        </SectionLabel>
        <div className="grid gap-5 lg:grid-cols-2">
          <ImportPanel />
          <LabelPanel />
        </div>
      </Section>

      {(inv?.train.count ?? 0) > 0 && (
        <Section delay={120}>
          <Panel className="flex flex-wrap items-center justify-between gap-3 p-5">
            <div>
              <div className="text-[14px] font-semibold text-fg">데이터 준비 완료 — 다음은 학습</div>
              <p className="mt-1 text-[12.5px] text-mut">
                학습쌍 {inv?.train.count}개{(inv?.eval.corpus ?? 0) > 0 ? ` · 평가셋 ${inv?.eval.corpus} docs` : ""} 준비됨.
                base 모델을 이 데이터로 fine-tune 해보세요.
              </p>
            </div>
            <Btn icon={<FlaskConical size={15} />} onClick={() => nav(PATH.train)}>
              학습하러 가기
            </Btn>
          </Panel>
        </Section>
      )}

      <Modal open={modal === "pairs"} onClose={() => setModal(null)} title={`학습 데이터 — 전체 ${fullPairs.data ? `(${fullPairs.data.total})` : ""}`}>
        {fullPairs.isLoading ? (
          <Loading />
        ) : (
          <DataTable cols={["query", "정답 제목"]} rows={(fullPairs.data?.items ?? []).map((p) => [p.query, p.title])} />
        )}
      </Modal>
      <Modal open={modal === "corpus"} onClose={() => setModal(null)} title={`평가 corpus — 전체 ${fullCorpus.data ? `(${fullCorpus.data.total})` : ""}`}>
        {fullCorpus.isLoading ? (
          <Loading />
        ) : (
          <DataTable cols={["_id", "title", "text"]} rows={(fullCorpus.data?.items ?? []).map((c) => [c.id, c.title, c.text])} />
        )}
      </Modal>
    </div>
  );
}
