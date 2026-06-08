import type { ReactNode } from "react";

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
