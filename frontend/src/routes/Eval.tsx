import { useState } from "react";
import { useLocation } from "react-router-dom";
import { Play } from "lucide-react";

import { useModels, useRunEval } from "../lib/queries";
import type { Embedder } from "../lib/types";
import { Btn, ErrorNote, Field, Info, Input, Metric, Panel, Section, SectionLabel, Seg } from "../components/ui";

const KPIS = ["recall@1", "recall@3", "mrr@10", "ndcg@10"] as const;

/** The Train screen's "이 모델 평가하기" CTA lands here with the model preset. */
interface EvalPreset {
  backend?: Embedder;
  model?: string;
}

export default function Eval() {
  const preset = (useLocation().state ?? {}) as EvalPreset;
  const [backend, setBackend] = useState<Embedder>(preset.backend ?? "ollama");
  const [override, setOverride] = useState(preset.model ?? ""); // user-typed model ("" = use the query's default)
  const [label, setLabel] = useState("");
  const models = useModels(backend);
  const runEval = useRunEval();

  // Effective model = the user's override, else the backend's default — derived, no effect needed.
  const model = override || models.data?.default || "";
  const changeBackend = (b: Embedder) => {
    setBackend(b);
    setOverride(""); // drop the override so the new backend's default shows
  };

  const submit = () => {
    const m = model.trim();
    if (!m) return;
    runEval.mutate({ embedder: backend, model: m, label: label.trim() });
  };

  const result = runEval.data;
  const prior = result?.prior_best ?? {};
  const hasPrior = Object.keys(prior).length > 0;

  return (
    <div className="space-y-9">
      <Section>
        <SectionLabel hint="모델이 평가셋에서 정답을 얼마나 잘 검색하는지">평가 실행</SectionLabel>
        <Panel className="p-5">
          <div className="grid items-end gap-4 lg:grid-cols-[1fr_1.4fr_auto]">
            <Field label="백엔드">
              <Seg
                options={[
                  { value: "ollama", label: "Ollama" },
                  { value: "sentence-transformers", label: "학습 모델" },
                ]}
                value={backend}
                onChange={changeBackend}
              />
            </Field>
            <Field label="모델" hint="차원은 자동 감지">
              <Input
                value={model}
                onChange={(e) => setOverride(e.target.value)}
                list="model-options"
                placeholder={models.isLoading ? "모델 목록 불러오는 중…" : "모델 선택 또는 입력"}
                className="mono"
              />
              <datalist id="model-options">
                {(models.data?.models ?? []).map((m) => (
                  <option key={m} value={m} />
                ))}
              </datalist>
            </Field>
            <Btn icon={<Play size={15} />} className="h-[42px]" onClick={submit} disabled={runEval.isPending || !model.trim()}>
              {runEval.isPending ? "평가 중…" : "평가 실행"}
            </Btn>
          </div>
          <div className="mt-4 max-w-xs">
            <Field label="라벨" hint="비우면 모델명 사용">
              <Input value={label} onChange={(e) => setLabel(e.target.value)} placeholder="예: base, ft·3ep" />
            </Field>
          </div>
        </Panel>
      </Section>

      {runEval.isError && (
        <Section>
          <ErrorNote>{(runEval.error as Error).message}</ErrorNote>
        </Section>
      )}

      {result && (
        <Section delay={70}>
          <SectionLabel
            hint={`${hasPrior ? "▲▼ 같은 평가셋 기존 best 대비" : "이 평가셋의 첫 기록 — 다음부터 Δ 표시"} · 쿼리 ${result.n_queries}개`}
          >
            결과 · {result.model} <span className="mono text-faint">(dim {result.embed_dim})</span>{" "}
            <Info title="지표 · Δ 읽는 법" align="left">
              <b className="text-fg">recall@k</b> 정답이 상위 k에 들 확률 · <b className="text-fg">MRR@10</b> 첫 정답의
              순위 · <b className="text-fg">nDCG@10</b> 상위 정렬 품질. ▲▼는 <b className="text-fg">같은 평가셋</b>의 기존
              best 대비입니다. 뒤에 reranker가 있으면 <span className="mono">recall@(후보 깊이)</span>를 더 중시하세요.
            </Info>
          </SectionLabel>
          <div className="grid grid-cols-2 gap-3 lg:grid-cols-4">
            {KPIS.map((k) => {
              const v = result.metrics[k] ?? 0;
              const p = prior[k];
              const ci = result.ci95?.[k];
              return (
                <div key={k} className="rounded-xl border border-line bg-ink-880/60 p-4">
                  <Metric
                    label={k}
                    value={v}
                    delta={hasPrior && p != null ? v - p : undefined}
                    sub={ci ? `95% CI ${ci[0].toFixed(3)}–${ci[1].toFixed(3)}` : undefined}
                  />
                </div>
              );
            })}
          </div>
        </Section>
      )}
    </div>
  );
}
