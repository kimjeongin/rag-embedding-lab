import type { ReactNode } from "react";

import { cx } from "../../lib/format";

/** Primary/ghost/subtle button with an optional leading icon. */
export function Btn({
  children,
  onClick,
  variant = "primary",
  className,
  icon,
  disabled,
}: {
  children: ReactNode;
  onClick?: () => void;
  variant?: "primary" | "ghost" | "subtle";
  className?: string;
  icon?: ReactNode;
  disabled?: boolean;
}) {
  const styles = {
    primary:
      "bg-signal text-ink-950 font-semibold hover:bg-[#d6ff62] shadow-[0_0_0_1px_rgba(198,242,74,0.3),0_8px_24px_-8px_rgba(198,242,74,0.5)]",
    ghost: "border border-line2 text-fg hover:bg-ink-800/60 hover:border-white/20",
    subtle: "text-mut hover:text-fg hover:bg-ink-800/60",
  }[variant];
  return (
    <button
      onClick={onClick}
      disabled={disabled}
      className={cx(
        "inline-flex items-center justify-center gap-2 rounded-xl px-4 py-2.5 text-sm transition-all duration-150 active:scale-[0.98] disabled:opacity-40",
        styles,
        className,
      )}
    >
      {icon}
      {children}
    </button>
  );
}

/** Segmented single-choice toggle (string-typed value). */
export function Seg<T extends string>({
  options,
  value,
  onChange,
}: {
  options: { value: T; label: string }[];
  value: T;
  onChange: (v: T) => void;
}) {
  return (
    <div className="inline-flex rounded-xl border border-line bg-ink-880/60 p-1">
      {options.map((o) => (
        <button
          key={o.value}
          onClick={() => onChange(o.value)}
          className={cx(
            "rounded-lg px-3.5 py-2 text-sm transition-all duration-150",
            value === o.value ? "bg-ink-700 text-fg shadow-sm" : "text-mut hover:text-fg",
          )}
        >
          {o.label}
        </button>
      ))}
    </div>
  );
}

/** Labelled form row with an optional hint. */
export function Field({ label, children, hint }: { label: string; children: ReactNode; hint?: string }) {
  return (
    <label className="block">
      <div className="mb-1.5 flex items-center gap-2">
        <span className="text-[12.5px] font-medium text-mut">{label}</span>
        {hint && <span className="text-[11px] text-faint">{hint}</span>}
      </div>
      {children}
    </label>
  );
}

/** Themed text input (forwards every native input prop). */
export function Input(props: React.InputHTMLAttributes<HTMLInputElement>) {
  return (
    <input
      {...props}
      className={cx(
        "w-full rounded-xl border border-line bg-ink-925 px-3.5 py-2.5 text-sm text-fg outline-none transition-colors placeholder:text-faint focus:border-signal/50 focus:ring-2 focus:ring-signal/15",
        props.className,
      )}
    />
  );
}
