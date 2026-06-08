import type { ReactNode } from "react";

import { cx } from "../../lib/format";
import { Tag } from "./badge";

/** A single big-number KPI card with a soft glow. */
export function Stat({
  label,
  value,
  tag,
  tone = "signal",
  sub,
}: {
  label: string;
  value: ReactNode;
  tag?: string;
  tone?: "signal" | "cyan";
  sub?: string;
}) {
  return (
    <div className="group relative overflow-hidden rounded-xl border border-line bg-ink-880/60 p-4">
      <div
        className={cx(
          "absolute -right-6 -top-8 h-20 w-20 rounded-full blur-2xl transition-opacity",
          tone === "signal" ? "bg-signal/15" : "bg-cyan/15",
        )}
      />
      <div className="relative">
        <div className="text-[12px] text-mut">{label}</div>
        <div className="mono mt-1 text-[30px] font-semibold leading-none text-fg">{value}</div>
        <div className="mt-2.5 flex items-center justify-between">
          {tag && <Tag tone={tone}>{tag}</Tag>}
          {sub && <span className="mono text-[10.5px] text-faint">{sub}</span>}
        </div>
      </div>
    </div>
  );
}

/** A 4-decimal metric with an optional ▲/▼ delta (green up / red down). */
export function Metric({
  label,
  value,
  delta,
  big,
}: {
  label: string;
  value: number;
  delta?: number;
  big?: boolean;
}) {
  return (
    <div>
      <div className="text-[12px] text-mut">{label}</div>
      <div className={cx("mono font-semibold leading-none text-fg", big ? "mt-2 text-[33px]" : "mt-1 text-[23px]")}>
        {value.toFixed(4)}
      </div>
      {delta !== undefined && (
        <div className={cx("mono mt-1.5 text-[12px]", delta >= 0 ? "text-signal2" : "text-danger")}>
          {delta >= 0 ? "▲ +" : "▼ "}
          {Math.abs(delta).toFixed(4)}
        </div>
      )}
    </div>
  );
}

/** Icon + title + sub quick-action card (the Overview shortcuts). */
export function ActionCard({
  icon,
  title,
  sub,
  onClick,
}: {
  icon: ReactNode;
  title: string;
  sub: string;
  onClick?: () => void;
}) {
  return (
    <button
      onClick={onClick}
      className="group flex items-center gap-3.5 rounded-2xl border border-line bg-ink-900/60 p-4 text-left transition-all hover:-translate-y-0.5 hover:border-signal/30 hover:bg-ink-880"
    >
      <span className="grid h-10 w-10 shrink-0 place-items-center rounded-xl border border-line2 bg-ink-800 text-signal transition-colors group-hover:bg-signal group-hover:text-ink-950">
        {icon}
      </span>
      <span className="min-w-0 flex-1">
        <span className="block text-[14px] font-medium text-fg">{title}</span>
        <span className="block text-[12px] text-mut">{sub}</span>
      </span>
    </button>
  );
}
