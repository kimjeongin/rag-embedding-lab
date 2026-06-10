import { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import { useQuery, useQueryClient } from "@tanstack/react-query";
import { toast } from "sonner";
import { Database, FileText, FlaskConical, Gauge, Sparkles } from "lucide-react";

import { api } from "../lib/api";
import { short } from "../lib/format";
import { PATH } from "../lib/nav";
import { keys, useCorpus, useDataOverview, useGenEval, useGenPairs, usePairs } from "../lib/queries";
import { startSynthetic, useSyntheticState } from "../lib/syntheticStore";
import { DataTable } from "../components/DataTable";
import { Modal } from "../components/Modal";
import { Btn, ErrorNote, Field, Info, Input, Loading, Panel, Section, SectionLabel, Seg, Stat, Tag } from "../components/ui";

type Method = "toy" | "synthetic";

export default function Data() {
  const overview = useDataOverview();
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
      startSynthetic({ method: "synthetic", corpus_file: corpusFile, gen_model: genModel, n_queries: nQueries, hard_negatives: 4 });
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
                  <Input type="number" value={nQueries} onChange={(e) => setNQueries(+e.target.value)} className="mono" />
                </Field>
              </div>
            )}

            <div className="mt-4 flex flex-wrap gap-2.5">
              <Btn icon={<Sparkles size={15} />} onClick={runGenPairs} disabled={busy}>
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
            <div className="mt-4 flex flex-wrap gap-2.5">
              <Btn icon={<Sparkles size={15} />} onClick={() => genEval.mutate({ n_distractors: ndist })} disabled={genEval.isPending}>
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
