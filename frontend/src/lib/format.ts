// Tiny presentation helpers shared across views.
import { clsx, type ClassValue } from "clsx";
import { twMerge } from "tailwind-merge";

/** Merge class names with Tailwind-aware conflict resolution (later wins). */
export const cx = (...c: ClassValue[]) => twMerge(clsx(c));

/** Fixed-decimal metric, or an em-dash for missing values. */
export const fmt = (n: number | null | undefined, d = 4) => (n == null ? "—" : n.toFixed(d));

/** Signed percentage, e.g. +1.23% / -0.45%. */
export const pct = (n: number) => `${n >= 0 ? "+" : ""}${(n * 100).toFixed(2)}%`;

/** The last path segment of a model id (outputs/embedding-ft → embedding-ft). */
export const short = (model: string | null | undefined) => (model ? model.split("/").pop()! : "");

/** "06-08 07:13" from an ISO timestamp (created_at) — compact, local-ish. */
export const when = (iso: string | null | undefined) => (iso ? iso.slice(5, 16).replace("T", " ") : "");
