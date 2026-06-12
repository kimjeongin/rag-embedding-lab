// Teaching diagrams for the About page — hand-drawn SVG (same approach as charts.tsx)
// so the "textbook" reads in the app's own visual language. Static, no data deps.

const INK = "#111419";
const LINE = "rgba(255,255,255,0.16)";
const GRID = "rgba(255,255,255,0.045)";
const FG = "#e9ecef";
const MUT = "#98a1ab";
const FAINT = "#5b646e";
const SIGNAL = "#c6f24a";
const CYAN = "#5ad1d6";
const AMBER = "#f3b14a";
const DANGER = "#f1685e";
const MONO = "Geist Mono, monospace";

function Box({
  x,
  y,
  w,
  h,
  title,
  sub,
  tone = "ink",
}: {
  x: number;
  y: number;
  w: number;
  h: number;
  title: string;
  sub?: string;
  tone?: "ink" | "signal" | "cyan";
}) {
  const stroke = tone === "signal" ? SIGNAL : tone === "cyan" ? "rgba(90,209,214,0.55)" : LINE;
  const fill = tone === "signal" ? "rgba(198,242,74,0.07)" : tone === "cyan" ? "rgba(90,209,214,0.06)" : INK;
  const titleFill = tone === "signal" ? SIGNAL : tone === "cyan" ? CYAN : FG;
  return (
    <g>
      <rect x={x} y={y} width={w} height={h} rx={10} fill={fill} stroke={stroke} strokeWidth={tone === "signal" ? 1.5 : 1} />
      <text
        x={x + w / 2}
        y={y + (sub ? h / 2 - 3 : h / 2 + 4)}
        textAnchor="middle"
        fontSize={11.5}
        fontWeight={600}
        fill={titleFill}
      >
        {title}
      </text>
      {sub && (
        <text x={x + w / 2} y={y + h / 2 + 14} textAnchor="middle" fontSize={9.5} fill={FAINT}>
          {sub}
        </text>
      )}
    </g>
  );
}

/** Where the lab sits: the production pipeline with the dense slot highlighted. */
export function PipelineDiagram() {
  return (
    <svg viewBox="0 0 860 196" className="w-full">
      <defs>
        <marker id="pl-arr" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
          <path d="M0,0 L8,4 L0,8 z" fill={MUT} />
        </marker>
      </defs>
      <text x={16} y={28} fontSize={10.5} fill={FAINT}>
        프로덕션 검색 파이프라인 (이미 운영 중)
      </text>

      <Box x={16} y={62} w={120} h={46} title="사용자 쿼리" sub="“vpn이 안 돼요”" />
      <path d="M136 85 H166 V40 H190" fill="none" stroke={MUT} strokeWidth={1.2} markerEnd="url(#pl-arr)" />
      <path d="M136 85 H166 V130 H190" fill="none" stroke={MUT} strokeWidth={1.2} markerEnd="url(#pl-arr)" />

      <Box x={196} y={16} w={200} h={48} title="BM25 — 키워드 검색" sub="같은 단어가 있어야 찾음" />
      <Box x={196} y={106} w={200} h={48} title="Dense 임베딩 — 의미 검색" sub="단어가 달라도 의미로 찾음" tone="signal" />
      <text x={296} y={174} textAnchor="middle" fontSize={11} fontWeight={700} fill={SIGNAL}>
        ★ 이 랩이 만드는 부품
      </text>

      <path d="M396 40 H424 V85 H440" fill="none" stroke={MUT} strokeWidth={1.2} markerEnd="url(#pl-arr)" />
      <path d="M396 130 H424 V85 H440" fill="none" stroke={MUT} strokeWidth={1.2} markerEnd="url(#pl-arr)" />
      <Box x={446} y={62} w={130} h={46} title="하이브리드 융합" sub="두 후보 목록을 합침" />

      <line x1={576} y1={85} x2={600} y2={85} stroke={MUT} strokeWidth={1.2} markerEnd="url(#pl-arr)" />
      <Box x={606} y={62} w={120} h={46} title="리랭커" sub="상위 후보 재정렬" />

      <line x1={726} y1={85} x2={750} y2={85} stroke={MUT} strokeWidth={1.2} markerEnd="url(#pl-arr)" />
      <Box x={756} y={62} w={92} h={46} title="결과" sub="상위 N개" tone="cyan" />
    </svg>
  );
}

/** The embedding space: a query, its positive pulled in, negatives pushed out. */
export function SpaceDiagram() {
  const q = { x: 350, y: 128 };
  const dots: [number, number][] = [
    [96, 58], [180, 210], [252, 44], [470, 226], [640, 52], [764, 96],
    [700, 196], [120, 136], [560, 34], [806, 170], [430, 40], [620, 150],
  ];
  return (
    <svg viewBox="0 0 860 252" className="w-full">
      <defs>
        <marker id="sp-pull" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
          <path d="M0,0 L8,4 L0,8 z" fill={CYAN} />
        </marker>
        <marker id="sp-push" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
          <path d="M0,0 L8,4 L0,8 z" fill={DANGER} />
        </marker>
      </defs>

      {[63, 126, 189].map((y) => (
        <line key={y} x1={0} x2={860} y1={y} y2={y} stroke={GRID} />
      ))}
      {[143, 286, 429, 572, 715].map((x) => (
        <line key={x} x1={x} x2={x} y1={0} y2={252} stroke={GRID} />
      ))}

      <text x={16} y={22} fontSize={10.5} fill={FAINT}>
        임베딩 공간 — 1024차원을 2차원으로 그린 그림. 학습은 이 점들의 위치를 조금씩 옮기는 일입니다.
      </text>

      {dots.map(([x, y], i) => (
        <circle key={i} cx={x} cy={y} r={3} fill={FAINT} opacity={0.5} />
      ))}

      <circle cx={q.x} cy={q.y} r={110} fill="none" stroke="rgba(198,242,74,0.22)" strokeDasharray="4 6" />
      <text x={q.x + 80} y={q.y - 86} fontSize={10} fill={FAINT}>
        가까움 = 관련 있음
      </text>

      <circle cx={q.x} cy={q.y} r={9} fill="rgba(198,242,74,0.18)" stroke={SIGNAL} strokeWidth={1.5} />
      <circle cx={q.x} cy={q.y} r={3.5} fill={SIGNAL} />
      <text x={q.x} y={q.y + 28} textAnchor="middle" fontSize={11.5} fontWeight={600} fill={FG}>
        쿼리 「vpn이 안 돼요」
      </text>

      <circle cx={520} cy={86} r={6} fill={CYAN} />
      <text x={534} y={82} fontSize={11} fill={CYAN}>
        VPN 접속 가이드 (정답)
      </text>
      <line x1={504} y1={92} x2={386} y2={116} stroke={CYAN} strokeWidth={1.5} strokeDasharray="5 4" markerEnd="url(#sp-pull)" />
      <text x={464} y={94} textAnchor="middle" fontSize={10} fill={CYAN}>
        가깝게 당김
      </text>

      <circle cx={433} cy={196} r={5.5} fill={MUT} />
      <text x={447} y={200} fontSize={11} fill={MUT}>
        구내식당 메뉴 (오답)
      </text>
      <line x1={445} y1={204} x2={516} y2={232} stroke={DANGER} strokeWidth={1.3} strokeDasharray="5 4" markerEnd="url(#sp-push)" />
      <text x={545} y={246} textAnchor="middle" fontSize={10} fill={DANGER}>
        멀게 밀어냄
      </text>

      <circle cx={236} cy={178} r={5.5} fill={MUT} />
      <text x={224} y={174} textAnchor="end" fontSize={11} fill={MUT}>
        프린터 드라이버 (오답)
      </text>
      <line x1={222} y1={186} x2={152} y2={212} stroke={DANGER} strokeWidth={1.3} strokeDasharray="5 4" markerEnd="url(#sp-push)" />
    </svg>
  );
}

/** MNRL's free negatives: the in-batch similarity matrix. */
export function MnrlMatrix() {
  const cell = 54;
  const ox = 116;
  const oy = 66;
  const sub = ["₁", "₂", "₃", "₄"];
  return (
    <svg viewBox="0 0 430 336" className="w-full max-w-[430px]">
      <text x={16} y={24} fontSize={11} fill={MUT}>
        batch = 4일 때: 학습쌍 (q₁,d₁) … (q₄,d₄)
      </text>

      {sub.map((s, c) => (
        <text key={c} x={ox + c * cell + cell / 2} y={oy - 12} textAnchor="middle" fontSize={12} fill={FG} fontFamily={MONO}>
          d{s}
        </text>
      ))}
      {sub.map((s, r) => (
        <text key={r} x={ox - 14} y={oy + r * cell + cell / 2 + 4} textAnchor="end" fontSize={12} fill={FG} fontFamily={MONO}>
          q{s}
        </text>
      ))}

      {sub.map((_, r) =>
        sub.map((_2, c) => {
          const diag = r === c;
          return (
            <g key={`${r}-${c}`}>
              <rect
                x={ox + c * cell + 2}
                y={oy + r * cell + 2}
                width={cell - 4}
                height={cell - 4}
                rx={8}
                fill={diag ? "rgba(198,242,74,0.16)" : INK}
                stroke={diag ? SIGNAL : "rgba(255,255,255,0.08)"}
                strokeWidth={diag ? 1.3 : 1}
              />
              <text
                x={ox + c * cell + cell / 2}
                y={oy + r * cell + cell / 2 + 4}
                textAnchor="middle"
                fontSize={10.5}
                fontWeight={diag ? 700 : 400}
                fill={diag ? SIGNAL : FAINT}
              >
                {diag ? "정답" : "오답"}
              </text>
            </g>
          );
        }),
      )}

      <rect x={ox - 1} y={oy + cell - 1} width={cell * 4 + 2} height={cell + 2} rx={10} fill="none" stroke="rgba(90,209,214,0.5)" strokeDasharray="4 4" />
      <text x={ox + cell * 4 + 10} y={oy + cell + cell / 2 + 4} fontSize={10} fill={CYAN}>
        q₂의 시험지
      </text>

      <text x={16} y={oy + cell * 4 + 30} fontSize={10.5} fill={MUT}>
        q₂의 정답은 d₂ 하나 — 같은 batch의 d₁·d₃·d₄는 자동으로 오답.
      </text>
      <text x={16} y={oy + cell * 4 + 46} fontSize={10.5} fill={MUT}>
        라벨링 없이 batch 크기만큼 선택지가 생깁니다 (16이면 1 vs 15).
      </text>
    </svg>
  );
}

/** Early stopping: validation curve, best-epoch star, patience, auto-stop. */
export function EarlyStopDiagram() {
  const W = 860;
  const H = 256;
  const mL = 52;
  const mR = 18;
  const mT = 32;
  const mB = 46;
  const pw = W - mL - mR;
  const ph = H - mT - mB;
  const vals = [0.62, 0.74, 0.81, 0.86, 0.895, 0.91, 0.918, 0.912, 0.909, 0.905];
  const X = (e: number) => mL + ((e - 1) / 11) * pw;
  const Y = (v: number) => mT + ph - ((v - 0.58) / (0.96 - 0.58)) * ph;
  const pts = vals.map((v, i) => `${X(i + 1)},${Y(v)}`).join(" ");
  const stopX = X(10) + 16;
  return (
    <svg viewBox={`0 0 ${W} ${H}`} className="w-full">
      {[0.6, 0.7, 0.8, 0.9].map((g) => (
        <g key={g}>
          <line x1={mL} x2={W - mR} y1={Y(g)} y2={Y(g)} stroke={GRID} />
          <text x={mL - 8} y={Y(g) + 3.5} textAnchor="end" fontSize={9.5} fill={FAINT} fontFamily={MONO}>
            {g.toFixed(2)}
          </text>
        </g>
      ))}
      {Array.from({ length: 12 }, (_, i) => i + 1).map((e) => (
        <text
          key={e}
          x={X(e)}
          y={mT + ph + 18}
          textAnchor="middle"
          fontSize={9.5}
          fill={e > 10 ? "rgba(91,100,110,0.45)" : FAINT}
          fontFamily={MONO}
        >
          {e}
        </text>
      ))}
      <text x={mL} y={16} fontSize={10.5} fill={FAINT}>
        검증 점수 (val nDCG@10) — 매 epoch 끝마다 측정
      </text>
      <text x={W - mR} y={H - 6} textAnchor="end" fontSize={10.5} fill={FAINT}>
        epoch — 천장(12)까지 다 돌 필요가 없습니다
      </text>

      <line x1={X(10)} y1={Y(0.905)} x2={X(12)} y2={Y(0.9)} stroke={FAINT} strokeDasharray="3 5" opacity={0.6} />
      <text x={X(11)} y={Y(0.9) + 20} textAnchor="middle" fontSize={9.5} fill={FAINT}>
        돌지 않음 (시간 절약)
      </text>

      <polyline points={pts} fill="none" stroke={SIGNAL} strokeWidth={2} strokeLinejoin="round" />
      {vals.map((v, i) => {
        const e = i + 1;
        const noImp = e >= 8;
        return (
          <circle
            key={e}
            cx={X(e)}
            cy={Y(v)}
            r={e === 7 ? 5 : 3.4}
            fill={e === 7 ? SIGNAL : "#0d0f13"}
            stroke={noImp ? AMBER : SIGNAL}
            strokeWidth={1.4}
          />
        );
      })}
      <circle cx={X(7)} cy={Y(0.918)} r={10} fill="none" stroke="rgba(198,242,74,0.35)" />
      <text x={X(7)} y={Y(0.918) - 18} textAnchor="middle" fontSize={11} fontWeight={700} fill={SIGNAL}>
        ★ 최고 기록 — 이 시점의 가중치가 저장됨 (이름의 -e7)
      </text>

      {[8, 9, 10].map((e, i) => (
        <text key={e} x={X(e)} y={Y(vals[e - 1]) + 22} textAnchor="middle" fontSize={9.5} fill={AMBER} fontFamily={MONO}>
          {i + 1}
        </text>
      ))}
      <text x={X(9)} y={Y(vals[8]) + 40} textAnchor="middle" fontSize={10} fill={AMBER}>
        개선 없음 ×3 (= patience)
      </text>

      <line x1={stopX} y1={mT} x2={stopX} y2={mT + ph} stroke={DANGER} strokeDasharray="4 4" opacity={0.8} />
      <text x={stopX + 6} y={mT + 14} fontSize={10.5} fontWeight={600} fill={DANGER}>
        자동 중단
      </text>
    </svg>
  );
}

/** LoRA: frozen W, trainable low-rank bypass, merge-on-save. */
export function LoraDiagram() {
  return (
    <svg viewBox="0 0 860 238" className="w-full">
      <defs>
        <marker id="lr-arr" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
          <path d="M0,0 L8,4 L0,8 z" fill={MUT} />
        </marker>
        <marker id="lr-arr-s" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
          <path d="M0,0 L8,4 L0,8 z" fill={SIGNAL} />
        </marker>
      </defs>

      <text x={18} y={95} fontSize={11.5} fill={FG}>
        입력
      </text>
      <line x1={48} y1={91} x2={82} y2={91} stroke={MUT} strokeWidth={1.2} markerEnd="url(#lr-arr)" />

      <Box x={90} y={58} w={200} h={66} title="W — 원본 가중치 (6억 개)" sub="❄ 동결 — 학습하지 않음" />

      <path d="M66 91 V178 H82" fill="none" stroke="rgba(198,242,74,0.6)" strokeWidth={1.3} markerEnd="url(#lr-arr-s)" />
      <Box x={90} y={156} w={86} h={44} title="A" sub="d×r로 압축" tone="signal" />
      <line x1={176} y1={178} x2={196} y2={178} stroke="rgba(198,242,74,0.6)" strokeWidth={1.3} markerEnd="url(#lr-arr-s)" />
      <Box x={202} y={156} w={86} h={44} title="B" sub="r×d로 복원" tone="signal" />
      <path d="M288 178 H330 V104" fill="none" stroke="rgba(198,242,74,0.6)" strokeWidth={1.3} markerEnd="url(#lr-arr-s)" />
      <text x={338} y={152} fontSize={10} fill={SIGNAL}>
        × α/r (반영 강도)
      </text>

      <line x1={290} y1={91} x2={315} y2={91} stroke={MUT} strokeWidth={1.2} markerEnd="url(#lr-arr)" />
      <circle cx={330} cy={91} r={11} fill={INK} stroke={MUT} />
      <text x={330} y={95.5} textAnchor="middle" fontSize={14} fill={FG}>
        +
      </text>
      <line x1={341} y1={91} x2={388} y2={91} stroke={MUT} strokeWidth={1.2} markerEnd="url(#lr-arr)" />
      <text x={394} y={95} fontSize={11.5} fill={FG}>
        출력
      </text>

      <line x1={428} y1={91} x2={462} y2={91} stroke={MUT} strokeWidth={1.2} strokeDasharray="4 4" markerEnd="url(#lr-arr)" />
      <text x={445} y={78} textAnchor="middle" fontSize={10} fill={MUT}>
        저장할 때
      </text>

      <rect x={470} y={30} width={372} height={150} rx={14} fill="rgba(90,209,214,0.05)" stroke="rgba(90,209,214,0.35)" />
      <text x={656} y={62} textAnchor="middle" fontSize={12} fontWeight={700} fill={CYAN}>
        저장 시 병합 (merge)
      </text>
      <text x={656} y={94} textAnchor="middle" fontSize={13} fill={FG} fontFamily={MONO}>
        W′ = W + (α/r)·B·A
      </text>
      <text x={656} y={126} textAnchor="middle" fontSize={10.5} fill={MUT}>
        어댑터가 본체에 합쳐져 보통의 모델 파일이 됩니다
      </text>
      <text x={656} y={146} textAnchor="middle" fontSize={10.5} fill={MUT}>
        → 서빙은 full로 학습한 모델과 똑같이 취급
      </text>

      <text x={90} y={228} fontSize={10.5} fill={SIGNAL}>
        학습되는 건 A·B(초록)뿐 — 전체 파라미터의 1% 미만
      </text>
    </svg>
  );
}
