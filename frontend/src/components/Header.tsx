import { useLocation } from "react-router-dom";
import { useQueryClient } from "@tanstack/react-query";
import { Cpu, RotateCw } from "lucide-react";

import { META, stepFromPath } from "../lib/nav";
import { useStatus } from "../lib/queries";
import { Pill } from "./ui";

function StatusPills() {
  const { data } = useStatus();
  if (!data) return null;
  const device = data.training_ready ? data.device : "cpu";
  return (
    <div className="hidden items-center gap-2 lg:flex">
      <Pill tone={data.ollama.reachable ? "signal" : "amber"}>
        <span
          className={
            data.ollama.reachable
              ? "h-1.5 w-1.5 rounded-full bg-signal shadow-[0_0_6px] shadow-signal"
              : "h-1.5 w-1.5 rounded-full bg-amber"
          }
        />
        Ollama
      </Pill>
      <Pill tone="mut">
        <Cpu size={12} /> device <span className="mono text-fg">{device}</span>
      </Pill>
      <Pill tone={data.eval.is_sample ? "amber" : "mut"}>
        eval <span className="mono">{data.eval.dir}</span>
        {data.eval.is_sample && " · 샘플"}
      </Pill>
      <Pill tone="mut">
        runs <span className="mono text-fg">{data.runs}</span>
      </Pill>
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
