// Tiny presentation helpers shared across views.
import { clsx, type ClassValue } from "clsx";
import { twMerge } from "tailwind-merge";

/** Merge class names with Tailwind-aware conflict resolution (later wins). */
export const cx = (...c: ClassValue[]) => twMerge(clsx(c));

/** Fixed-decimal metric, or an em-dash for missing values. */
export const fmt = (n: number | null | undefined, d = 4) => (n == null ? "—" : n.toFixed(d));

/** Signed absolute delta of a 0–1 metric, e.g. +0.0345 / −0.0345 — same scale as the
 * scores themselves (a "%" would read as a relative improvement, which it isn't). */
export const delta = (n: number, d = 4) => `${n >= 0 ? "+" : "−"}${Math.abs(n).toFixed(d)}`;

/** The last path segment of a model id (outputs/embedding-ft → embedding-ft). */
export const short = (model: string | null | undefined) => (model ? model.split("/").pop()! : "");

/** "06-08 07:13" from an ISO timestamp (created_at) — compact, local-ish. */
export const when = (iso: string | null | undefined) => (iso ? iso.slice(5, 16).replace("T", " ") : "");
