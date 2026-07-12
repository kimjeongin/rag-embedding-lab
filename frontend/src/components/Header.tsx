import { useLocation } from "react-router-dom";
import { useQueryClient } from "@tanstack/react-query";
import { Cpu, RotateCw } from "lucide-react";

import { META, stepFromPath } from "../lib/nav";
import { useStatus } from "../lib/queries";
import { Pill } from "./ui";

/** Green/amber liveness dot. */
const Dot = ({ ok }: { ok: boolean }) => (
  <span
    className={
      ok
        ? "h-1.5 w-1.5 rounded-full bg-signal shadow-[0_0_6px] shadow-signal"
        : "h-1.5 w-1.5 rounded-full bg-amber"
    }
  />
);

/** The context bar — the lab's "현재 세계 상태"를 모든 화면에서 보이게.
 * 무엇이 쿼리를 임베딩하는지(임베더), 벡터 스토어는 살아있는지(Qdrant), 어떤 평가셋에서
 * 비교 중인지, 현재 최고 점수, 도는 잡(학습·재색인), 납품된 모델. */
function StatusPills() {
  const { data } = useStatus();
  if (!data) return null;
  const device = data.training_ready ? data.device : "cpu";
  const hasFinal = data.eval.splits?.includes("final");
  const st = data.settings.embedder === "sentence-transformers";
  const modelShort = data.settings.model.split("/").pop();
  return (
    <div className="hidden items-center gap-2 lg:flex">
      {/* 임베더 — 이 프로세스가 쿼리·평가를 임베딩하는 주체 (하드코딩 아님) */}
      <Pill tone={st || data.ollama.reachable ? "signal" : "amber"} title={data.settings.model}>
        <Dot ok={st || data.ollama.reachable} />
        {st ? "ST" : "Ollama"} <span className="mono text-fg">{modelShort}</span>
      </Pill>
      {/* 데이터 탭의 LLM 합성이 Ollama를 쓰므로, ST 서빙 중이라도 죽어 있으면 알림 */}
      {st && !data.ollama.reachable && (
        <Pill tone="amber" title="LLM 데이터 합성(데이터 탭)이 Ollama를 씁니다">
          <Dot ok={false} /> Ollama ↓
        </Pill>
      )}
      <Pill tone={data.qdrant_reachable ? "mut" : "amber"} title="서빙 벡터 스토어">
        <Dot ok={data.qdrant_reachable} /> Qdrant
      </Pill>
      <Pill tone="mut">
        <Cpu size={12} /> <span className="mono text-fg">{device}</span>
      </Pill>
      <Pill tone={data.eval.is_sample ? "amber" : "mut"}>
        eval{" "}
        <span className="mono">
          {data.eval.fingerprint ? `#${data.eval.fingerprint.slice(0, 6)}` : data.eval.dir}
          {` · ${data.eval.queries}q`}
        </span>
        {hasFinal && <span className="mono text-signal2">+final</span>}
        {data.eval.is_sample && " · 샘플"}
      </Pill>
      {data.best_ndcg != null && (
        <Pill tone="mut">
          best <span className="mono text-fg">{data.best_ndcg.toFixed(4)}</span>
        </Pill>
      )}
      {data.active_job && (
        <Pill tone="signal">
          <span className="h-1.5 w-1.5 animate-pulse rounded-full bg-signal" />
          학습 중
        </Pill>
      )}
      {data.indexing && (
        <Pill tone="cyan">
          <span className="h-1.5 w-1.5 animate-pulse rounded-full bg-cyan" />
          재색인 중
        </Pill>
      )}
      {data.handed_off && (
        <Pill tone="cyan">
          납품 <span className="mono">{data.handed_off.model.split("/").pop()}</span>
        </Pill>
      )}
    </div>
  );
}

export default function Header() {
  const step = stepFromPath(useLocation().pathname);
  const qc = useQueryClient();
  return (
    <header className="sticky top-0 z-20 flex items-center justify-between gap-4 border-b border-line bg-ink-950/70 px-8 py-3.5 backdrop-blur-xl">
      <div>
        <h1 className="text-[19px] font-semibold tracking-tight text-fg">{META[step].title}</h1>
        <p className="text-[12.5px] text-mut">{META[step].sub}</p>
      </div>
      <div className="flex items-center gap-2.5">
        <StatusPills />
        <button
          onClick={() => qc.invalidateQueries()}
          title="새로고침"
          className="grid h-9 w-9 place-items-center rounded-xl border border-line2 text-mut transition-colors hover:bg-ink-800 hover:text-fg"
        >
          <RotateCw size={15} />
        </button>
      </div>
    </header>
  );
}
