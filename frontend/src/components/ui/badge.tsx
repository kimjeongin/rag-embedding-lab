import type { ReactNode } from "react";

import { cx } from "../../lib/format";

/** Rounded status pill (header chips). */
export function Pill({ children, tone = "mut" }: { children: ReactNode; tone?: "signal" | "cyan" | "amber" | "mut" }) {
  const map = {
    signal: "text-signal border-signal/30 bg-signal/10",
    cyan: "text-cyan border-cyan/30 bg-cyan/10",
    amber: "text-amber border-amber/30 bg-amber/10",
    mut: "text-mut border-line2 bg-ink-800/50",
  }[tone];
  return (
    <span
      className={cx(
        "inline-flex items-center gap-1.5 rounded-full border px-2.5 py-1 text-[12px] font-medium leading-none",
        map,
      )}
    >
      {children}
    </span>
  );
}

/** Small monospace label chip (e.g. "→ 학습", a backend name). */
export function Tag({ children, tone = "mut" }: { children: ReactNode; tone?: "signal" | "cyan" | "mut" }) {
  const map = {
    signal: "text-signal2 bg-signal/8",
    cyan: "text-cyan bg-cyan/8",
    mut: "text-faint bg-ink-800/60",
  }[tone];
  return <span className={cx("mono rounded-md px-1.5 py-0.5 text-[11px] font-medium", map)}>{children}</span>;
}

/** Keyboard-key hint (e.g. ⌘K). */
export function Kbd({ children }: { children: ReactNode }) {
  return (
    <kbd className="mono rounded-md border border-line2 bg-ink-800 px-1.5 py-px text-[10.5px] text-mut">{children}</kbd>
  );
}
