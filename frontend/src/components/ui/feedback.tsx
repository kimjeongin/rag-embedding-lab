import { useState, type ReactNode } from "react";

import { cx } from "../../lib/format";

/** A small `?` affordance that reveals an explanation on hover, or pins it on click.
 * Dependency-free (no Radix). `align` keeps the popover on-screen next to edge labels. */
export function Info({
  children,
  title,
  align = "center",
  className,
}: {
  children: ReactNode;
  title?: string;
  align?: "left" | "center" | "right";
  className?: string;
}) {
  const [pinned, setPinned] = useState(false);
  const pos = align === "left" ? "left-0" : align === "right" ? "right-0" : "left-1/2 -translate-x-1/2";
  return (
    <span className={cx("group relative inline-flex align-middle", className)}>
      <button
        type="button"
        aria-label="설명 보기"
        onClick={(e) => {
          e.stopPropagation();
          setPinned((p) => !p);
        }}
        onBlur={() => setPinned(false)}
        className={cx(
          "grid h-[15px] w-[15px] cursor-help place-items-center rounded-full border text-[10px] font-semibold leading-none transition-colors",
          pinned
            ? "border-signal/50 bg-signal/15 text-signal"
            : "border-line2 text-faint hover:border-white/30 hover:text-mut",
        )}
      >
        ?
      </button>
      <span
        role="tooltip"
        className={cx(
          // reset any inherited uppercase/tracking from table headers etc.
          "absolute top-full z-50 mt-2 w-64 rounded-xl border border-line bg-ink-900 p-3 text-left text-[12px] font-normal normal-case leading-relaxed tracking-normal text-mut shadow-[0_20px_50px_-15px_rgba(0,0,0,0.85)] transition-opacity duration-150",
          pos,
          pinned ? "opacity-100" : "pointer-events-none opacity-0 group-hover:opacity-100",
        )}
      >
        {title && <span className="mb-1 block text-[12px] font-semibold text-fg">{title}</span>}
        {children}
      </span>
    </span>
  );
}

/** Inline loading row for a query in flight. */
export function Loading({ label = "불러오는 중…" }: { label?: string }) {
  return (
    <div className="flex items-center justify-center gap-2.5 py-12 text-[13px] text-faint">
      <span className="h-3.5 w-3.5 animate-spin rounded-full border-2 border-line2 border-t-signal" />
      {label}
    </div>
  );
}

/** Error banner carrying the API's detail message. */
export function ErrorNote({ children }: { children: ReactNode }) {
  return (
    <div className="rounded-xl border border-danger/30 bg-danger/10 px-4 py-3 text-[13px] text-danger">{children}</div>
  );
}
