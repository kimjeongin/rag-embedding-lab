import type { ReactNode } from "react";

import { cx } from "../../lib/format";

/** Bordered, faintly-frosted surface — the base card for every section. */
export function Panel({ className, children }: { className?: string; children: ReactNode }) {
  return (
    <div className={cx("rounded-2xl border border-line bg-ink-900/60 backdrop-blur-[2px]", className)}>{children}</div>
  );
}

/** Small uppercase section heading with an optional right-aligned hint. */
export function SectionLabel({ children, hint }: { children: ReactNode; hint?: ReactNode }) {
  return (
    <div className="mb-3 flex items-baseline justify-between gap-3">
      <h2 className="text-[13px] font-semibold uppercase tracking-[0.14em] text-mut">{children}</h2>
      {hint && <span className="text-xs text-faint">{hint}</span>}
    </div>
  );
}

/** Fade-and-rise section wrapper; stagger siblings with `delay`. */
export function Section({ children, delay = 0 }: { children: ReactNode; delay?: number }) {
  return (
    <section className="rise" style={{ animationDelay: `${delay}ms` }}>
      {children}
    </section>
  );
}
