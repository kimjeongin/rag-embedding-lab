import type { ReactNode } from "react";

import { cx } from "../lib/format";

/** A compact, read-only table for previews and modals. */
export function DataTable({ cols, rows }: { cols: string[]; rows: (ReactNode | null)[][] }) {
  return (
    <div className="overflow-hidden rounded-xl border border-line">
      <table className="w-full text-left text-[13px]">
        <thead>
          <tr className="border-b border-line bg-ink-880/60 text-[11px] uppercase tracking-wider text-faint">
            {cols.map((c) => (
              <th key={c} className="px-3.5 py-2.5 font-medium">
                {c}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((r, i) => (
            <tr key={i} className="border-b border-line/60 last:border-0 hover:bg-ink-880/40">
              {r.map((cell, j) => (
                <td key={j} className={cx("px-3.5 py-2.5 align-top", j === 0 ? "text-fg" : "text-mut")}>
                  {cell ?? <span className="text-faint">—</span>}
                </td>
              ))}
            </tr>
          ))}
          {rows.length === 0 && (
            <tr>
              <td colSpan={cols.length} className="px-3.5 py-6 text-center text-[12.5px] text-faint">
                데이터가 없습니다 — 먼저 생성하세요.
              </td>
            </tr>
          )}
        </tbody>
      </table>
    </div>
  );
}
