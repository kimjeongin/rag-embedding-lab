// Report — the project's standing report to management, updated continuously until the
// project completes. Mirrors docs/report.md but built for the boss to read on screen:
// the 6-section frame (배경·목표·벤치마킹·추진·성과·향후) plus a "journey" timeline that
// keeps the PROCESS and the WHY behind each choice, not just the final state.
// Content lives in the arrays at the top so updating the report = editing data.
import type { ReactNode } from "react";
import { useNavigate } from "react-router-dom";
import { ArrowRight, BookOpen, CheckCircle2, FlaskConical, Target } from "lucide-react";

import { PATH } from "../lib/nav";
import { Btn, Panel, Section, SectionLabel, Stat, Tag } from "../components/ui";

/* ── local building blocks ─────────────────────────────────────────────── */

const M = ({ children }: { children: ReactNode }) => <span className="mono text-[11.5px] text-fg">{children}</span>;

/** Tinted insight callout (reused for "왜 이렇게 했나" rationale blocks). */
function Note({ tone = "signal", title, children }: { tone?: "signal" | "cyan" | "amber" | "danger"; title?: ReactNode; children: ReactNode }) {
  const map = {
    signal: "border-signal/30 bg-signal/[0.06]",
    cyan: "border-cyan/30 bg-cyan/[0.06]",
    amber: "border-amber/30 bg-amber/[0.07]",
    danger: "border-danger/30 bg-danger/[0.06]",
  }[tone];
  return (
    <div className={`rounded-xl border ${map} px-4 py-3`}>
      {title && <div className="mb-1 text-[12.5px] font-semibold text-fg">{title}</div>}
      <div className="text-[12.5px] leading-relaxed text-mut">{children}</div>
    </div>
  );
}

/** Bordered table; first column emphasized. `emph` marks rows to highlight. */
function Tbl({ head, rows, emph = [] }: { head: string[]; rows: ReactNode[][]; emph?: number[] }) {
  return (
    <div className="overflow-x-auto rounded-xl border border-line">
      <table className="w-full text-left text-[12.5px]">
        <thead>
          <tr className="border-b border-line bg-ink-880/60 text-[11px] uppercase tracking-wider text-faint">
            {head.map((h) => (
              <th key={h} className="whitespace-nowrap px-3.5 py-2.5 font-medium">{h}</th>
            ))}
          </tr>
        </thead>
        <tbody className="text-mut">
          {rows.map((r, i) => (
            <tr key={i} className={`${i < rows.length - 1 ? "border-b border-line/60" : ""} ${emph.includes(i) ? "bg-signal/[0.05]" : ""}`}>
              {r.map((c, j) => (
                <td key={j} className={`px-3.5 py-2.5 align-top leading-relaxed ${j === 0 ? "whitespace-nowrap font-medium text-fg" : "mono"}`}>{c}</td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

/** Anchor-able section: TOC chips jump here. */
function Topic({ id, n, title, children }: { id: string; n: string; title: string; children: ReactNode }) {
  return (
    <Section>
      <div id={id} className="scroll-mt-24">
        <SectionLabel hint={<span className="mono text-faint">{n}</span>}>{title}</SectionLabel>
        <div className="space-y-3.5">{children}</div>
      </div>
    </Section>
  );
}

/** One node of the journey timeline: date · what · WHY · result. */
function TimelineNode({ date, phase, title, did, why, result, last }: {
  date: string; phase: string; title: string; did: ReactNode; why: ReactNode; result?: ReactNode; last?: boolean;
}) {
  return (
    <div className="relative pl-7">
      <span className="absolute left-0 top-1 grid h-4 w-4 place-items-center">
        <span className="h-2.5 w-2.5 rounded-full border-2 border-signal bg-ink-950" />
      </span>
      {!last && <span className="absolute left-[7px] top-5 h-[calc(100%-0.5rem)] w-px bg-line2" />}
      <div className="pb-6">
        <div className="mb-1 flex flex-wrap items-center gap-2">
          <Tag tone="signal">{phase}</Tag>
          <span className="mono text-[11px] text-faint">{date}</span>
        </div>
        <div className="text-[13.5px] font-semibold text-fg">{title}</div>
        <div className="mt-1 text-[12.5px] leading-relaxed text-mut">{did}</div>
        <div className="mt-2 rounded-lg border-l-2 border-amber/50 bg-amber/[0.05] px-3 py-2 text-[12px] leading-relaxed text-mut">
          <b className="text-amber">왜 이렇게 했나 — </b>{why}
        </div>
        {result && <div className="mt-2 text-[12.5px] leading-relaxed text-fg"><b className="text-signal2">결과 · </b>{result}</div>}
      </div>
    </div>
  );
}

/* ── content ───────────────────────────────────────────────────────────── */

const TOC: [string, string][] = [
  ["background", "1 · 추진배경"],
  ["goals", "2 · 과제목표"],
  ["bench", "3 · 관련연구 벤치마킹"],
  ["journey", "4 · 과제추진 (경과)"],
  ["value", "5 · 과제성과/기대효과"],
  ["future", "6 · 향후계획"],
  ["trust", "부록 · 신뢰 장치"],
];

const HEADLINE = [
  { label: "은어 검색 품질 (nDCG@10)", value: "0.15→0.77", tag: "p = 1e-4", sub: "양성 대조군" },
  { label: "하이브리드 한계 기여", value: "+3→+33%p", tag: "BM25 위 dense 기여", sub: "은어 +64%p" },
  { label: "차기 모델 후보 비교", value: "2종", tag: "Qwen · Nemotron", sub: "정확도·비용 동시" },
  { label: "자동화 테스트", value: "202건", tag: "전 구간 회귀 방어", sub: "" },
] as const;

/* ── page ──────────────────────────────────────────────────────────────── */

export default function Report() {
  const nav = useNavigate();
  return (
    <div className="mx-auto max-w-4xl space-y-8 pb-20">
      {/* header */}
      <Section>
        <Panel className="p-6">
          <div className="flex flex-wrap items-start justify-between gap-4">
            <div>
              <div className="mb-2 flex items-center gap-2">
                <Tag tone="signal">진행 중 · 계속 갱신</Tag>
                <span className="mono text-[11px] text-faint">최종 갱신 2026-07-24</span>
              </div>
              <h1 className="text-[22px] font-semibold tracking-tight text-fg">
                임베딩 모델 파인튜닝으로 사내 사이트 검색 품질 개선
              </h1>
              <p className="mt-1.5 max-w-2xl text-[13px] leading-relaxed text-mut">
                공개 임베딩 모델이 모르는 <b className="text-fg">사내 언어(은어·약어)</b>를 파인튜닝으로 메워
                검색 품질을 올리고, 신규 공개 모델과 비교해 <b className="text-fg">차기 모델을 선정</b>하는 과제.
                아래는 추진 배경부터 경과·성과·향후 계획까지의 보고이며, 프로젝트 종료까지 계속 갱신됩니다.
              </p>
            </div>
            <Btn onClick={() => nav(PATH.about)}>
              <BookOpen size={15} /> 개념 참고 페이지
            </Btn>
          </div>
          <div className="mt-5 grid grid-cols-2 gap-3 sm:grid-cols-4">
            {HEADLINE.map((h) => (
              <Stat key={h.label} label={h.label} value={h.value} tag={h.tag} sub={h.sub} />
            ))}
          </div>
        </Panel>
      </Section>

      {/* TOC */}
      <Section>
        <div className="flex flex-wrap gap-2">
          {TOC.map(([id, label]) => (
            <a key={id} href={`#${id}`} className="rounded-full border border-line bg-ink-880/50 px-3 py-1 text-[12px] text-mut transition-colors hover:border-signal/40 hover:text-fg">
              {label}
            </a>
          ))}
        </div>
      </Section>

      {/* 1 · 배경 */}
      <Topic id="background" n="01" title="추진배경">
        <p className="text-[13px] leading-relaxed text-mut">
          사내 검색 프로덕션은 <b className="text-fg">BM25(키워드) + Dense 임베딩(의미) + 리랭커</b>의 3단
          하이브리드로 운영됩니다. 이 중 dense 부품은 공개 임베딩 모델을 그대로 쓰는데, 공개 모델은 일반
          한국어는 알아도 <b className="text-fg">우리 조직의 언어를 모릅니다</b> — 사내 시스템 이름·약어·신조어,
          "그 업무를 우리가 부르는 말"과 실제 문서를 연결하지 못합니다.
        </p>
        <Note tone="cyan" title="이 랩의 역할 = dense 부품 하나만 더 좋게 교체">
          검색 시스템을 새로 만드는 게 아니라, 돌아가는 파이프라인의 부품 하나만 우리 데이터로 파인튜닝해
          바꿉니다 — 리스크·배포 범위가 작음. 임베더의 임무는 "정답을 1등으로"가 아니라 <b className="text-fg">
          "정답을 상위 후보(K=50)에 넣기"</b>이고, 순서는 뒤의 리랭커가 정리합니다. 그래서 <M>recall@K</M>가
          검색 품질의 천장이자 주 지표입니다.
        </Note>
      </Topic>

      {/* 2 · 목표 */}
      <Topic id="goals" n="02" title="과제목표">
        <div className="grid gap-3 sm:grid-cols-2">
          <Panel className="p-4">
            <div className="mb-1.5 flex items-center gap-2 text-fg"><Target size={15} className="text-signal" /><b className="text-[13px]">주 목표 ①</b></div>
            <div className="text-[12.5px] leading-relaxed text-mut">파인튜닝으로 <b className="text-fg">사내 검색 품질이 실제로 올랐는지</b> 통계적으로 검증.</div>
          </Panel>
          <Panel className="p-4">
            <div className="mb-1.5 flex items-center gap-2 text-fg"><Target size={15} className="text-signal" /><b className="text-[13px]">주 목표 ②</b></div>
            <div className="text-[12.5px] leading-relaxed text-mut">신규 공개 모델(Nemotron)과 비교해 <b className="text-fg">차기 모델을 선정</b>. 서비스 환경상 비용차가 작아 <b className="text-fg">정확도가 결정 기준</b>(비용은 측정·관측).</div>
          </Panel>
        </div>
        <Note tone="signal" title="보조 성과 경로 — 하이브리드 자체 개선">
          최종 목표가 검색 품질이므로, 모델 교체 외에 <b className="text-fg">BM25+dense 융합 자체를 튜닝</b>해
          올려도 성과로 인정됩니다. 랩은 이미 BM25 상보성(dense의 한계 기여)을 측정하고 있어 이 경로의
          출발점이 마련돼 있습니다.
        </Note>
      </Topic>

      {/* 3 · 벤치마킹 */}
      <Topic id="bench" n="03" title="관련연구 벤치마킹 조사">
        <p className="text-[13px] leading-relaxed text-mut">
          차기 모델 후보로 <b className="text-fg">Qwen3-Embedding-0.6B</b>(현행)와 NVIDIA가 최근 공개한
          <b className="text-fg"> Nemotron-3-Embed-1B</b>(RTEB 1위 계열)를 놓고, 두 모델을 같은 레시피로
          학습·비교했습니다.
        </p>
        <Tbl
          head={["항목", "Qwen3-0.6B", "Nemotron-3-Embed-1B"]}
          rows={[
            ["파라미터", "0.6B", "1.14B"],
            ["임베딩 차원", "1024", "2048"],
            ["구조", "28층 × 1024 (깊음)", "16층 × 2048 (넓음)"],
            ["쿼리 포맷", "Instruct 지시문", "query: 접두사"],
            ["풀링", "last-token", "평균"],
          ]}
        />
        <p className="text-[12.5px] leading-relaxed text-mut">
          채택 방법론: 대조학습 손실 4종(<M>MNRL</M>/Cached/GIST/Triplet), <M>Matryoshka</M>(차원 절단
          가능 학습), 하이브리드 상보성 측정, 튜닝 방법론(좌표하강 + LR 동반 스윕 + median pruning).
          평가는 <M>BEIR</M> 표준(recall/nDCG/MRR)을 따릅니다.
        </p>
      </Topic>

      {/* 4 · 경과 (타임라인) */}
      <Topic id="journey" n="04" title="과제추진 사항 — 경과와 선택 근거">
        <p className="text-[12.5px] leading-relaxed text-mut">
          최종 결과만이 아니라 <b className="text-fg">각 단계에서 무엇을·왜 선택했는지</b>를 남깁니다.
          핵심 서사: 파이프라인을 만들고 → 그 파이프라인이 <b className="text-fg">진짜 격차를 잡아내는지 먼저 증명</b>한 뒤
          → 실데이터 수용을 준비하고 → 모델을 공정하게 비교했습니다.
        </p>
        <Panel className="p-5">
          <TimelineNode
            date="2026-06"
            phase="파이프라인"
            title="데이터→학습→평가→비교→납품→서빙 6단계 구축"
            did={<>웹 UI 하나에서 학습쌍 생성·파인튜닝·통계 비교·Qdrant 서빙까지 끝나는 구조. Qdrant 서빙은 버전 컬렉션 + alias 원자 전환으로 무중단·롤백·멱등.</>}
            why={<>부품 교체식 개선을 <b className="text-fg">반복 가능·재현 가능</b>하게 만들어야 이후 모든 실험이 자산으로 쌓임.</>}
          />
          <TimelineNode
            date="2026-06"
            phase="PoC"
            title="공공 사이트(korea.kr 1,500문서)로 end-to-end 검증"
            did={<>파이프라인 전체가 정상 동작함을 확인. 다만 파인튜닝 개선폭이 신뢰구간 안(유의하지 않음).</>}
            why={<>개선이 없는 게 <b className="text-fg">파이프라인 탓인지 데이터 탓인지 구분할 수 없는</b> 상태였음 — 이 모호함을 정직하게 남기고 다음 단계에서 해소하기로.</>}
            result={<>base가 이미 아는 공공 텍스트라 파고들 도메인 격차가 작다는 가설 → 대조군으로 검증 필요.</>}
          />
          <TimelineNode
            date="2026-07-18"
            phase="측정기 교정"
            title="양성 대조군 — 은어 격차를 설계로 주입"
            did={<>실 운영 payload 스키마를 본뜬 가상 인트라넷(27시스템·195페이지)에 <b className="text-fg">사내 은어</b>를 심되, 은어→시스템 연결은 학습쌍에만 존재하고 문서 본문엔 절대 안 나오게 설계. base는 구조적으로 못 맞히고 파인튜닝만 배울 수 있는 상황.</>}
            why={<>"격차가 있으면 이 파이프라인이 <b className="text-fg">정말 잡아내는가</b>"를 증명해야, 이후 실데이터에서 나온 어떤 숫자든 신뢰할 수 있음. 측정기부터 교정하는 것.</>}
            result={<>은어 nDCG <M>0.152 → 0.772</M>, 전체 <M>0.559 → 0.879</M> (p=0.0001, 81승1패86무). 표준 쿼리는 비회귀. <b className="text-fg">파이프라인은 격차가 있으면 잡고, 없으면 없다고 말한다</b>를 입증.</>}
          />
          <TimelineNode
            date="2026-07-20"
            phase="실로그 준비"
            title="클릭로그 노이즈 리허설 — 정제 계층"
            did={<>실사용 로그(포지션 바이어스·오클릭·재검색·PII)를 확률 모형으로 재현하고, 정답을 아는 모의 로그로 정제 규칙을 미리 만들어 채점. 두 학습셋(naive vs cleaned)으로 실제 파인튜닝해 대조.</>}
            why={<>실로그는 그대로 쓰면 안 되는 데이터 — 도착 <b className="text-fg">전에</b> 정제 규칙을 만들어 검증해두면, 데이터가 오는 순간 붙이는 것 외에 할 일이 없음.</>}
            result={<>naive 학습은 무학습과 통계적 동급(p=0.72), cleaned는 은어 recall@5 <M>0.17→0.90</M> 회복. 재검색 전이 규칙이 은어 supervision의 전부임을 확인.</>}
          />
          <TimelineNode
            date="2026-07-24"
            phase="모델 비교"
            title="모델 중립 비교 인프라 — ModelProfile + Nemotron 대조 + rag-bench"
            did={<>입력 포맷을 모델별 <M>ModelProfile</M>로 추상화(학습·평가·서빙 단일 참조)하고, Nemotron을 같은 레시피로 파인튜닝. <M>rag-bench</M>로 실제 Qdrant 경로의 지연·GPU·저장까지 측정.</>}
            why={<>한 모델에 묶이면 "차기 모델은?"에 답할 수 없음. 그리고 포맷이 틀리면 <b className="text-fg">예외 없이 점수만 조용히 무너지므로</b> 포맷을 한 곳에 강제해야 함.</>}
            result={<>정확도 차 통계적 무의미(p=0.55), 포맷 ablation로 추상화 가치 증명(0.89→0.35), 서빙 비용 수치화. 상세는 아래 §5.</>}
          />
          <TimelineNode
            date="계획"
            phase="다음"
            title="체계적 튜닝 스윕 — 각 모델 '제대로' 학습"
            did={<>단일 레시피가 아니라 LR×batch→loss→dropout→Matryoshka→hard negative→LoRA를 좌표하강으로 탐색, 각 모델을 자기 최고점까지.</>}
            why={<>"각 모델을 제대로 튜닝하고 비교했나"에 방어하려면 단일 레시피 한 방으론 부족. 설계서는 <M>docs/model-tuning-plan.md</M>에.</>}
            last
          />
        </Panel>
      </Topic>

      {/* 5 · 성과 */}
      <Topic id="value" n="05" title="과제성과/기대효과">
        <div className="grid gap-3 sm:grid-cols-2">
          {[
            ["검증된 파인튜닝 효과", "은어 슬라이스 nDCG 0.15→0.77 (p=1e-4). 파이프라인이 도메인 격차를 실제로 메움을 대조군으로 입증."],
            ["차기 모델 비교 결론", "Qwen vs Nemotron 정확도 차 무의미(p=0.55). 저장비용 맞추면(1024d) 오히려 Qwen 우세 — 현 근거로는 교체 이유 약함."],
            ["정직한 측정 인프라", "평가셋 지문 강제, dev/final 분리, paired 유의성 검정, 하드웨어 지문 — '좋아 보이는 숫자'가 아니라 '믿을 수 있는 숫자'."],
            ["실서비스 비용 가시화", "rag-bench로 지연 p50/p95/p99·GPU·저장·색인을 실제 Qdrant 경로에서 측정. 정확도 외 축까지 결정 근거화."],
          ].map(([t, d]) => (
            <Panel key={t} className="p-4">
              <div className="mb-1 flex items-center gap-2"><CheckCircle2 size={15} className="text-signal2" /><b className="text-[13px] text-fg">{t}</b></div>
              <div className="text-[12.5px] leading-relaxed text-mut">{d}</div>
            </Panel>
          ))}
        </div>

        <div className="pt-1 text-[12.5px] font-semibold text-mut">모델 비교 상세 — 파인튜닝 후 (dev 168쿼리)</div>
        <Tbl
          head={["모델", "dim", "전체", "표준", "은어", "recall@50", "Qwen 대비 p"]}
          rows={[
            ["Qwen 0.6B FT", "1024", "0.8794", "0.9915", "0.7725", "1.0000", "—"],
            ["Nemotron 1B FT", "2048", "0.8885", "0.9939", "0.7879", "1.0000", "0.55 (무의미)"],
            ["Nemotron @1024 절단", "1024", "0.8548", "0.9894", "0.7265", "1.0000", "0.14 (무의미)"],
          ]}
          emph={[0]}
        />
        <Note tone="amber" title="포맷 ablation — ModelProfile의 가치가 수치로">
          Nemotron FT 모델을 일부러 틀린(Qwen) 포맷으로 서빙하면 <M>0.889 → 0.345</M> (138패 1승, p=1e-4).
          <b className="text-fg"> 예외는 하나도 안 나고</b> 점수만 무너집니다 — 심지어 학습조차 안 한 base(0.543)보다
          낮아, 파인튜닝에 쓴 시간이 마이너스가 됩니다. 포맷을 한 곳에 강제한 이유.
        </Note>

        <div className="pt-1 text-[12.5px] font-semibold text-mut">서빙 비용 — 실제 Qdrant 경로 (M2 Pro MPS 기준, CUDA 재측정 예정)</div>
        <Tbl
          head={["항목", "Qwen 0.6B FT", "Nemotron 1B FT"]}
          rows={[
            ["응답 p50 / p95 (ms)", "45.8 / 74.5", "32.9 / 59.1"],
            ["GPU 피크", "1,554 MB", "2,753 MB"],
            ["모델 디스크", "1.2 GB", "2.3 GB"],
            ["1M 문서 벡터 저장", "4.1 GB", "8.2 GB"],
          ]}
        />
        <Note tone="cyan" title='"큰 모델 = 느림"은 단건 응답에서 틀립니다'>
          Nemotron은 파라미터가 2배인데 응답이 <b className="text-fg">28% 빠릅니다</b> — 크기가 아니라 구조 때문.
          배치=1 지연에선 순차 실행되는 <b className="text-fg">층수(28 vs 16)가 병렬화되는 폭보다 지배적</b>입니다.
          대신 메모리·저장은 정직하게 2배. 현 서비스 환경에선 이 비용차가 작아 사실상 정확도가 결정.
        </Note>
      </Topic>

      {/* 6 · 향후 */}
      <Topic id="future" n="06" title="향후계획 / 연구보완">
        <Tbl
          head={["순위", "과제", "내용"]}
          rows={[
            ["1", "실사용 로그·corpus 확보", "파인튜닝 이득의 원천. 인입 경로는 구현 완료 — 데이터 연결만 남음. 실 corpus면 지표 포화도 해소."]
              ,
            ["2", "측정 해상도 회복", "195문서는 recall 포화라 모델 변별 불가. 더 크고 어려운 평가셋으로 nDCG/recall@1의 변별력 확보 → 모델 확정의 선행조건."],
            ["3", "체계적 튜닝 스윕", "각 모델을 제대로 학습해 공정 비교. CUDA 서버에서 실행(설계서 완비)."],
            ["4", "서빙 벤치 CUDA 재측정", "지연·GPU 절대 수치 확보. 층수↔폭의 속도 관계가 GPU 커널 특성상 뒤집힐 수 있어 필요."],
            ["5", "하이브리드 융합 튜닝", "모델 교체 외의 두 번째 개선 레버. BM25+dense 융합 가중치 튜닝으로 검색 품질 직접 개선."],
            ["6", "final 확정 + A/B", "dev로 승자 선택 → final 1회 확정 → 프로덕션 하이브리드에서 A/B로 서비스 지표 최종 검증."],
          ]}
        />
      </Topic>

      {/* 부록 */}
      <Topic id="trust" n="—" title="부록 · 이 숫자를 믿어도 되는 이유">
        <Tbl
          head={["주장", "보장 장치"]}
          rows={[
            ["모든 결과는 재현 가능", "레시피·데이터 지문·epoch별 점수를 train_meta.json에 자동 기록"],
            ["비교는 항상 공정", "평가셋 지문이 다른 점수끼리는 비교 자체가 차단"],
            ["'좋은 점수 고르기' 방지", "dev(선택)/final(확정) 분리, final은 승자 1회만"],
            ["차이는 우연이 아님", "쿼리별 paired 순열 검정으로 p값 제시"],
            ["랩 점수 = 서빙 점수", "임베딩 계약 + 패리티 벡터(cosine≥0.999), 서빙에도 동일 포맷 코드 강제"],
            ["모델별 포맷 오류 차단", "입력 포맷을 ModelProfile로 한 곳에 정의, 학습·평가·서빙 단일 참조"],
            ["서빙 성능은 하드웨어별로만 비교", "벤치 기록마다 하드웨어 지문, 지문 다르면 지연·메모리 비교 차단"],
          ]}
        />
        <div className="flex flex-wrap gap-2 pt-1">
          <Btn onClick={() => nav(PATH.compare)}><FlaskConical size={15} /> 실험 탭에서 실물 비교</Btn>
          <Btn onClick={() => nav(PATH.about)}>개념 참고 페이지 <ArrowRight size={15} /></Btn>
        </div>
      </Topic>
    </div>
  );
}
