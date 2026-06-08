import type { ReactNode } from "react";
import { cx } from "../lib/format";

export function Panel({ className, children }: { className?: string; children: ReactNode }) {
  return (
    <div className={cx("rounded-2xl border border-line bg-ink-900/60 backdrop-blur-[2px]", className)}>{children}</div>
  );
}

export function SectionLabel({ children, hint }: { children: ReactNode; hint?: ReactNode }) {
  return (
    <div className="mb-3 flex items-baseline justify-between gap-3">
      <h2 className="text-[13px] font-semibold uppercase tracking-[0.14em] text-mut">{children}</h2>
      {hint && <span className="text-xs text-faint">{hint}</span>}
    </div>
  );
}

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

export function Tag({ children, tone = "mut" }: { children: ReactNode; tone?: "signal" | "cyan" | "mut" }) {
  const map = {
    signal: "text-signal2 bg-signal/8",
    cyan: "text-cyan bg-cyan/8",
    mut: "text-faint bg-ink-800/60",
  }[tone];
  return (
    <span className={cx("mono rounded-md px-1.5 py-0.5 text-[11px] font-medium", map)}>{children}</span>
  );
}

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

export function Sparkline({ data, color = "#c6f24a", className }: { data: number[]; color?: string; className?: string }) {
  const W = 120;
  const H = 40;
  const lo = Math.min(...data);
  const hi = Math.max(...data);
  const rng = hi - lo || 1;
  const yy = (v: number) => H - ((v - lo) / rng) * (H - 8) - 4;
  const pts = data.map((v, i) => `${(i / (data.length - 1)) * W},${yy(v)}`).join(" ");
  return (
    <svg viewBox={`0 0 ${W} ${H}`} className={className} preserveAspectRatio="none">
      <polyline points={pts} fill="none" stroke={color} strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
      <circle cx={W} cy={yy(data[data.length - 1])} r="2.6" fill={color} />
    </svg>
  );
}

export function Kbd({ children }: { children: ReactNode }) {
  return (
    <kbd className="mono rounded-md border border-line2 bg-ink-800 px-1.5 py-px text-[10.5px] text-mut">{children}</kbd>
  );
}

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

/** Fade-and-rise section wrapper; stagger with `delay`. */
export function Section({ children, delay = 0 }: { children: ReactNode; delay?: number }) {
  return (
    <section className="rise" style={{ animationDelay: `${delay}ms` }}>
      {children}
    </section>
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
