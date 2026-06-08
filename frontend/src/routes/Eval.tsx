import { useState } from "react";
import { Play } from "lucide-react";

import { useModels, useRunEval } from "../lib/queries";
import type { Embedder } from "../lib/types";
import { Btn, ErrorNote, Field, Input, Metric, Panel, Section, SectionLabel, Seg } from "../components/ui";

const KPIS = ["recall@1", "recall@3", "mrr@10", "ndcg@10"] as const;

export default function Eval() {
  const [backend, setBackend] = useState<Embedder>("ollama");
  const [override, setOverride] = useState(""); // user-typed model ("" = use the query's default)
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
          <SectionLabel hint={hasPrior ? "▲▼ 기존 best 대비" : "첫 평가 — 다음부터 Δ 표시"}>
            결과 · {result.model} <span className="mono text-faint">(dim {result.embed_dim})</span>
          </SectionLabel>
          <div className="grid grid-cols-2 gap-3 lg:grid-cols-4">
            {KPIS.map((k) => {
              const v = result.metrics[k] ?? 0;
              const p = prior[k];
              return (
                <div key={k} className="rounded-xl border border-line bg-ink-880/60 p-4">
                  <Metric label={k} value={v} delta={hasPrior && p != null ? v - p : undefined} />
                </div>
              );
            })}
          </div>
        </Section>
      )}
    </div>
  );
}
