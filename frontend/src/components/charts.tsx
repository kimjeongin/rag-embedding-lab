// Domain data-viz — hand-drawn SVG so they match the design exactly (custom, not shadcn).
import type { RunRecord } from "../lib/types";

/** Grouped (side-by-side) bars of all metrics; y-axis zoomed to the data so small gaps show. */
export function BarChart({
  runs,
  metrics,
  colors,
}: {
  runs: RunRecord[];
  metrics: readonly string[];
  colors: string[];
}) {
  const W = 820;
  const H = 300;
  const mL = 42;
  const mR = 14;
  const mT = 14;
  const mB = 38;
  const pw = W - mL - mR;
  const ph = H - mT - mB;

  const values = runs.flatMap((r) => metrics.map((m) => r.metrics[m] ?? 0)).filter((v) => v > 0);
  const lo = values.length ? Math.max(0, Math.min(...values) - 0.02) : 0;
  const hi = 1.0;
  const y = (v: number) => mT + ph - ((v - lo) / (hi - lo)) * ph;
  const groupW = pw / metrics.length;
  const bw = Math.min(24, (groupW * 0.66) / Math.max(runs.length, 1));
  const grid = Array.from({ length: 5 }, (_, i) => lo + (i * (hi - lo)) / 4);

  return (
    <svg viewBox={`0 0 ${W} ${H}`} className="w-full" style={{ fontFamily: "Geist Mono, monospace" }}>
      <defs>
        {colors.map((c, i) => (
          <linearGradient id={`bg${i}`} key={i} x1="0" y1="0" x2="0" y2="1">
            <stop offset="0" stopColor={c} stopOpacity="0.95" />
            <stop offset="1" stopColor={c} stopOpacity="0.5" />
          </linearGradient>
        ))}
      </defs>

      {grid.map((gv, i) => (
        <g key={i}>
          <line x1={mL} x2={W - mR} y1={y(gv)} y2={y(gv)} stroke="rgba(255,255,255,0.055)" />
          <text x={mL - 9} y={y(gv) + 3.5} textAnchor="end" fontSize="10" fill="#5b646e">
            {gv.toFixed(2)}
          </text>
        </g>
      ))}

      {metrics.map((m, gi) => {
        const gx = mL + gi * groupW;
        const inner = bw * runs.length;
        const start = gx + (groupW - inner) / 2;
        return (
          <g key={m}>
            {runs.map((r, ri) => {
              const v = r.metrics[m] ?? 0;
              const yy = y(v);
              const bh = Math.max(1.5, mT + ph - yy);
              return (
                <rect key={r.id} x={start + ri * bw + 1} y={yy} width={bw - 2} height={bh} rx="3" fill={`url(#bg${ri % colors.length})`}>
                  <title>{`${r.label} · ${m} = ${v.toFixed(4)}`}</title>
                </rect>
              );
            })}
            <text x={gx + groupW / 2} y={H - 13} textAnchor="middle" fontSize="10.5" fill="#98a1ab">
              {m}
            </text>
          </g>
        );
      })}
    </svg>
  );
}

/** Training loss over steps. Empty until the stream produces points. */
export function LossCurve({ points }: { points: number[] }) {
  if (points.length < 2) {
    return (
      <div className="grid h-[150px] place-items-center rounded-lg border border-dashed border-line text-[12px] text-faint">
        학습을 시작하면 loss 곡선이 실시간으로 그려집니다
      </div>
    );
  }
  const W = 760;
  const H = 150;
  const max = Math.max(...points) * 1.05 || 1;
  const path = points.map((p, i) => `${(i / (points.length - 1)) * W},${H - (p / max) * H}`).join(" ");
  return (
    <svg viewBox={`0 0 ${W} ${H}`} className="w-full" preserveAspectRatio="none">
      <defs>
        <linearGradient id="loss" x1="0" y1="0" x2="0" y2="1">
          <stop offset="0" stopColor="#c6f24a" stopOpacity="0.25" />
          <stop offset="1" stopColor="#c6f24a" stopOpacity="0" />
        </linearGradient>
      </defs>
      <polygon points={`0,${H} ${path} ${W},${H}`} fill="url(#loss)" />
      <polyline points={path} fill="none" stroke="#c6f24a" strokeWidth="2" strokeLinejoin="round" />
    </svg>
  );
}
