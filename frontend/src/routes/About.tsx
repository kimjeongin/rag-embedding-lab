// About — the lab's textbook *and* its pitch deck. Written so that someone who has
// never trained a model can learn the field from this page, and so that the page can
// be shown as-is to a senior/manager as the "why you can trust these numbers" report.
import { useNavigate } from "react-router-dom";
import type { ReactNode } from "react";
import {
  ArrowRight,
  BarChart3,
  Boxes,
  Database,
  FlaskConical,
  Gauge,
  HardDrive,
  Lightbulb,
  Package,
  Search,
  ShieldCheck,
} from "lucide-react";

import { PATH } from "../lib/nav";
import { Btn, Panel, Section, SectionLabel, Tag } from "../components/ui";
import { EarlyStopDiagram, LoraDiagram, MnrlMatrix, PipelineDiagram, SpaceDiagram } from "../components/diagrams";

/* ── local building blocks ─────────────────────────────────────────────── */

/** A bordered callout for a key insight, tinted by tone. */
function Note({ tone = "signal", title, children }: { tone?: "signal" | "cyan" | "amber"; title?: string; children: ReactNode }) {
  const map = {
    signal: "border-signal/30 bg-signal/[0.06]",
    cyan: "border-cyan/30 bg-cyan/[0.06]",
    amber: "border-amber/30 bg-amber/[0.07]",
  }[tone];
  return (
    <div className={`rounded-xl border ${map} px-4 py-3.5`}>
      {title && <div className="mb-1 text-[12.5px] font-semibold text-fg">{title}</div>}
      <div className="text-[12.5px] leading-relaxed text-mut">{children}</div>
    </div>
  );
}

/** "쉽게 말하면" — the analogy box that keeps beginners on board. */
function Analogy({ children }: { children: ReactNode }) {
  return (
    <div className="flex items-start gap-2.5 rounded-xl border border-dashed border-line2 bg-ink-880/40 px-4 py-3">
      <Lightbulb size={15} className="mt-0.5 shrink-0 text-amber" />
      <div className="text-[12.5px] leading-relaxed text-mut">
        <b className="text-fg">쉽게 말하면 — </b>
        {children}
      </div>
    </div>
  );
}

/** Monospace snippet block (data schemas, contracts). */
function Code({ children }: { children: ReactNode }) {
  return (
    <pre className="mono overflow-x-auto rounded-xl border border-line bg-ink-950 p-3.5 text-[11.5px] leading-relaxed text-mut">
      {children}
    </pre>
  );
}

/** Bordered table; first column emphasized. */
function Tbl({ head, rows }: { head: string[]; rows: ReactNode[][] }) {
  return (
    <div className="overflow-hidden rounded-xl border border-line">
      <table className="w-full text-left text-[12.5px]">
        <thead>
          <tr className="border-b border-line bg-ink-880/60 text-[11px] uppercase tracking-wider text-faint">
            {head.map((h) => (
              <th key={h} className="px-3.5 py-2.5 font-medium">
                {h}
              </th>
            ))}
          </tr>
        </thead>
        <tbody className="text-mut">
          {rows.map((r, i) => (
            <tr key={i} className={i < rows.length - 1 ? "border-b border-line/60" : ""}>
              {r.map((c, j) => (
                <td key={j} className={`px-3.5 py-2.5 align-top leading-relaxed ${j === 0 ? "whitespace-nowrap font-medium text-fg" : ""}`}>
                  {c}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

/** Anchor-able section: TOC chips jump here. */
function Topic({ id, title, hint, delay = 0, children }: { id: string; title: ReactNode; hint?: ReactNode; delay?: number; children: ReactNode }) {
  return (
    <Section delay={delay}>
      <div id={id} className="scroll-mt-24">
        <SectionLabel hint={hint}>{title}</SectionLabel>
        {children}
      </div>
    </Section>
  );
}

/** Inline term highlight inside muted prose. */
const M = ({ children }: { children: ReactNode }) => <span className="mono text-[11.5px] text-fg">{children}</span>;

/* ── content data ──────────────────────────────────────────────────────── */

const TOC: [string, string][] = [
  ["role", "랩의 역할"],
  ["basics", "임베딩 기초"],
  ["flow", "6단계 플로우"],
  ["training", "학습의 원리"],
  ["loss", "Loss 함수"],
  ["knobs", "하이퍼파라미터"],
  ["method", "Full vs LoRA"],
  ["eval", "평가"],
  ["compare", "비교·확정"],
  ["handoff", "납품"],
  ["serving", "서빙"],
  ["report", "보고 포인트"],
  ["glossary", "용어 사전"],
];

const FLOW = [
  {
    icon: Database,
    to: PATH.data,
    title: "데이터",
    what: "사이트를 크롤해 페이지 corpus를 만들고, (쿼리, 정답 문서) 학습쌍을 채웁니다 — 실사용 로그 가져오기 · LLM 합성(검색창 스타일 쿼리 + 라운드트립 필터 + 마진 가드 오답 채굴) · 직접 라벨링. 채점용 평가셋(corpus / queries / qrels)은 따로 준비합니다.",
    why: "모델은 데이터만큼만 배웁니다. 그리고 학습에 쓴 쿼리로 시험을 보면 암기를 실력으로 착각하므로, 두 데이터는 절대 겹치면 안 됩니다.",
  },
  {
    icon: FlaskConical,
    to: PATH.train,
    title: "학습",
    what: "loss · learning rate · dropout · 방식(full/LoRA)을 정해 base 모델을 미세 조정합니다. 여러 레시피를 줄 세우는 스윕도 같은 화면에서.",
    why: "매 epoch 검증해 가장 좋았던 시점을 저장(early stopping)하므로, “몇 epoch이 적당한가”는 사람이 고민할 필요가 없습니다.",
  },
  {
    icon: Gauge,
    to: PATH.eval,
    title: "평가",
    what: "학습에 쓰지 않은 평가셋에서 recall@50 · nDCG@10 · MRR@10을 측정합니다. 크롤 corpus가 있으면 사이트 전체가 건초더미가 되고, 모든 점수에는 평가셋의 지문(fingerprint)이 함께 박힙니다.",
    why: "“좋아진 것 같다”를 “몇 점 좋아졌다”로 바꾸는 단계 — 같은 자로 재야 비교가 성립합니다.",
  },
  {
    icon: BarChart3,
    to: PATH.compare,
    title: "실험",
    what: "런들을 나란히 놓고 쿼리별 diff와 p값으로 차이가 우연인지 판정합니다. dev 평가셋에서 고르고, 승자만 final로 1회 확정합니다.",
    why: "평균 한 줄에 속지 않기 위한 단계 — 어디서 이기고 어디서 망가졌는지까지 봐야 “채택”을 결정할 수 있습니다.",
  },
  {
    icon: HardDrive,
    to: PATH.models,
    title: "모델",
    what: "학습된 모델의 전체 레시피(train_meta.json)를 열람·정리하고, 승자를 핸드오프 패키지로 서빙팀에 납품합니다.",
    why: "모델 파일만 던지면 서빙에서 점수가 증발합니다 — 임베딩 계약서와 패리티 검증 벡터까지가 납품입니다.",
  },
  {
    icon: Search,
    to: PATH.search,
    title: "검색",
    what: "납품할 모델로 corpus 전체를 Qdrant(벡터 DB)에 색인하고, 실제 쿼리를 넣어 검색해 봅니다. 모델을 핸드오프하면 그 모델로 자동 재색인이 걸립니다.",
    why: "점수표가 아니라 실물로 확인하는 단계 — 그리고 “모델 교체 = 전면 재색인”이라는 서빙 절차를 무중단·자동으로 만드는 레퍼런스 구현입니다.",
  },
];

const LOSSES: {
  name: string;
  full: string;
  tag: string;
  tone: "signal" | "cyan" | "mut";
  how: ReactNode;
  when: string;
  caveat: string;
}[] = [
  {
    name: "MNRL",
    full: "MultipleNegativesRankingLoss",
    tag: "기본값 · 업계 표준",
    tone: "signal",
    how: (
      <>
        오답 출처 = <b className="text-fg">같은 batch</b>. batch 안 다른 쌍의 정답 문서가 내 쿼리에겐 자동으로 오답이
        됩니다(왼쪽 행렬). 매 step “N지선다에서 정답 고르기” 시험을 치르게 하는 것과 같고, 벌점은{" "}
        <M>−log P(정답)</M> — 정답을 뽑을 확률이 낮을수록 큽니다. 라벨링 비용 0으로 batch−1개의 오답을 얻습니다.
      </>
    ),
    when: "모든 실험의 출발점. 확신이 없으면 이것 — 그래서 기본값입니다.",
    caveat: "batch 안에 우연히 정답과 같은 내용의 문서가 섞이면(false negative) 멀쩡한 문서를 밀어냅니다 → GIST가 해결.",
  },
  {
    name: "Cached MNRL",
    full: "CachedMultipleNegativesRankingLoss",
    tag: "메모리 한계 돌파",
    tone: "cyan",
    how: (
      <>
        수학적으로 MNRL과 <b className="text-fg">100% 동일</b>합니다. GradCache 기법으로 batch를 잘게 쪼개 순차
        계산하므로, GPU 메모리를 더 쓰지 않고 큰 batch의 효과(= 많은 오답)를 냅니다. “batch가 클수록 좋다”와
        “메모리가 모자라다” 사이의 다리.
      </>
    ),
    when: "batch를 키우고 싶은데 OOM(메모리 부족)이 날 때. batch 64+를 노릴 때.",
    caveat: "같은 batch 기준 학습 시간이 20–30% 느려집니다. 메모리 ↔ 시간 교환.",
  },
  {
    name: "GIST",
    full: "GISTEmbedLoss",
    tag: "가짜 오답 필터",
    tone: "cyan",
    how: (
      <>
        MNRL + <b className="text-fg">가이드 모델</b>(이미 잘하는 공개 임베딩 모델)이 in-batch 오답들을 검사해 “이
        ‘오답’, 사실 정답 아닌가?” 싶은 쌍을 벌점 계산에서 <b className="text-fg">제외</b>합니다. 비슷한
        문서를 억울하게 밀어내는 일을 막아 학습 신호가 깨끗해집니다.
      </>
    ),
    when: "사내 위키처럼 비슷한 문서·중복 문서가 많은 코퍼스 (false negative가 잦은 환경).",
    caveat: "가이드 모델 추론이 추가되어 느려지고, 가이드가 우리 도메인을 너무 모르면 필터 품질도 낮아집니다.",
  },
  {
    name: "Triplet",
    full: "TripletLoss",
    tag: "hard negative 필수",
    tone: "mut",
    how: (
      <>
        오답을 <b className="text-fg">우리가 직접 지정</b>합니다 — (쿼리, 정답, hard negative) 삼중항. 벌점은{" "}
        <M>max(0, margin − s(q,d⁺) + s(q,d⁻))</M>: 정답 유사도가 오답보다 최소 margin만큼 높지 않으면 그만큼
        벌점입니다. “헷갈리는 그 둘”을 정밀하게 갈라놓는 손맛.
      </>
    ),
    when: "VPN 가이드 vs VPN 장애 공지처럼 비슷해서 틀리는 쌍을 데이터로 갖고 있을 때.",
    caveat: "hard negative가 마이닝된 데이터가 필요 — 없으면 시작 자체가 거부됩니다(데이터 탭에서 hard negative 마이닝).",
  },
];

const GLOSSARY: [string, string][] = [
  ["임베딩", "텍스트를 의미를 보존하는 숫자 좌표(벡터)로 바꾼 것"],
  ["벡터", "숫자 목록 = 고차원 공간의 한 점. 이 랩의 기본 모델은 1024차원"],
  ["cosine 유사도", "두 벡터의 방향이 같을수록 1 — ‘의미가 가깝다’의 수치화"],
  ["base 모델", "파인튜닝의 출발점인 공개 사전학습 모델 (Qwen3-Embedding-0.6B)"],
  ["파인튜닝", "사전학습 모델을 내 데이터로 추가 학습해 도메인에 맞추는 일"],
  ["대조 학습", "정답은 가깝게, 오답은 멀게 — 검색 임베딩 학습의 공통 원리"],
  ["in-batch negative", "같은 batch 안 남의 정답을 내 오답으로 쓰는 것 (공짜 오답)"],
  ["hard negative", "정답과 헷갈리게 비슷한 오답 — 모델을 정밀하게 단련시키는 재료"],
  ["false negative", "오답 취급했지만 사실 정답인 것 — GIST가 걸러내는 대상"],
  ["loss (손실)", "‘얼마나 틀렸나’의 벌점. 학습은 이 값을 줄이는 방향으로만 움직임"],
  ["learning rate", "한 step에 가중치를 고치는 정도 — 학습의 보폭"],
  ["batch", "한 step에 함께 처리하는 학습쌍 묶음 (기본 16)"],
  ["epoch", "학습 데이터 전체를 1회 다 보는 것 (1회독)"],
  ["과적합", "일반 규칙 대신 학습 데이터를 암기한 상태 — 새 데이터에서 점수 하락"],
  ["early stopping", "검증 점수가 patience번 연속 안 오르면 멈추고 최고 시점을 저장"],
  ["dropout", "학습 중 뉴런 일부를 무작위로 꺼 암기를 방해하는 정규화 기법"],
  ["LoRA", "본체는 동결, 저랭크 어댑터(B·A)만 학습 — 저장 시 본체에 병합"],
  ["r (rank)", "LoRA 어댑터의 폭 = 수정사항의 표현력 (기본 16)"],
  ["α (alpha)", "LoRA 수정사항의 반영 강도 — 관례적으로 2r (기본 32)"],
  ["recall@K", "정답이 상위 K 후보에 든 비율 — 이 랩의 주 지표 (K=50)"],
  ["nDCG@10", "상위 10 안에서 정답의 ‘위치’까지 반영한 순위 품질"],
  ["MRR@10", "첫 정답 순위의 역수 평균 — 첫 화면에 정답이 보이는가"],
  ["qrels", "쿼리 → 정답 문서 매핑, 즉 채점표"],
  ["fingerprint", "평가셋 내용의 해시 — ‘같은 시험지였다’의 증명"],
  ["p값", "관찰된 차이가 우연일 확률 — 관례상 0.05 미만이면 유의"],
  ["dev / final", "모의고사(선택용) / 수능(확정용)으로 분리한 평가셋"],
  ["핸드오프", "모델 + 임베딩 계약 + 패리티 벡터를 묶은 납품 패키지"],
  ["벡터 DB", "문서 벡터를 저장하고 ‘가장 가까운 벡터’를 빠르게 찾는 DB (이 랩은 Qdrant)"],
  ["색인(인덱싱)", "corpus 전체를 모델로 임베딩해 벡터 DB에 저장 — 모델 교체 시 전면 재실행"],
  ["컬렉션", "벡터 DB 안의 테이블 — 이 랩은 (모델·차원·corpus 지문)마다 별도 생성"],
  ["alias", "검색이 바라보는 고정 이름 — 실제 컬렉션을 원자적으로 갈아끼우는 스위치"],
  ["무중단 전환", "새 인덱스를 다 만든 뒤 alias만 바꿔치기 — 검색이 멈추는 순간이 없음"],
  ["멱등", "같은 작업을 몇 번 실행해도 결과가 같음 — 자동화에 걸어도 안전한 조건"],
];

const GUARANTEES: ReactNode[][] = [
  [
    "모든 결과는 재현 가능",
    <>
      모델마다 전체 레시피·학습 데이터 지문·epoch별 점수 기록이 <M>train_meta.json</M>으로 자동 저장 — “이 모델
      어떻게 만들었지?”가 발생하지 않음
    </>,
    "모델 탭 → 상세",
  ],
  [
    "비교는 항상 공정",
    "모든 점수에 평가셋 지문이 박히고, 지문이 다른 점수끼리는 비교 자체가 차단됨 — 다른 시험지끼리 등수를 매기는 사고 방지",
    "실험 탭",
  ],
  [
    "‘좋은 점수 고르기’ 방지",
    "dev 평가셋으로 여러 후보 중 선택하고, final 평가셋은 승자 1개에 한 번만 사용 — 선택 편향(우연히 잘 나온 모델 채택) 차단",
    "실험 탭 → 최종 확정",
  ],
  [
    "차이는 우연이 아님",
    "모델 간 점수 차이를 쿼리별 paired 순열 검정으로 검증해 p값 제시 — “0.01 올랐다”가 아니라 “우연일 확률 3%”로 보고",
    "실험 탭 → diff",
  ],
  [
    "랩 점수 = 서빙 점수",
    "납품 시 임베딩 계약(쿼리 포맷·풀링·정규화)과 패리티 검증 벡터(cosine ≥ 0.999) 동봉 — 서빙에서 점수가 증발하는 1순위 원인 차단",
    "모델 탭 → 납품",
  ],
  [
    "학습은 끊기지 않음",
    "학습 잡은 서버 소유 — 브라우저를 닫아도 진행되고, 끝나면 자동 평가까지. 스윕은 순차 실행 + median pruning으로 지는 런 조기 종료 + 상위 K개만 보관해 디스크도 자동 정리",
    "학습 탭",
  ],
  [
    "모델 교체는 무중단",
    "재색인은 새 컬렉션을 다 만든 뒤 alias를 원자적으로 전환 — 검색이 반쯤 만든 인덱스를 보는 순간이 없고, 옛 컬렉션은 롤백용으로 남음. 핸드오프하면 재색인이 자동으로 걸림",
    "검색 탭",
  ],
];

/* ── page ──────────────────────────────────────────────────────────────── */

export default function About() {
  const nav = useNavigate();

  return (
    <div className="space-y-10">
      {/* ── Hero ───────────────────────────────────────────────────────── */}
      <Section>
        <Panel className="relative overflow-hidden p-7">
          <div className="absolute -right-16 -top-16 h-52 w-52 rounded-full bg-signal/10 blur-3xl" />
          <div className="relative">
            <div className="flex items-center gap-2.5">
              <div className="grid h-9 w-9 place-items-center rounded-xl bg-signal text-ink-950 shadow-[0_0_26px_-6px_rgba(198,242,74,0.7)]">
                <Boxes size={18} strokeWidth={2.4} />
              </div>
              <h1 className="text-[24px] font-semibold tracking-tight text-fg">RAG Embedding Lab</h1>
            </div>
            <p className="mt-4 max-w-2xl text-[14px] leading-relaxed text-mut">
              사내 사이트 검색에 쓰는 <b className="text-fg">임베딩 모델</b>을 우리 데이터로{" "}
              <b className="text-fg">다시 가르치고(fine-tuning)</b>, 그 효과를 <b className="text-fg">정직하게 측정·비교</b>
              해서, 검증된 모델만 프로덕션에 <b className="text-fg">납품</b>하는 실험실입니다.
            </p>
            <p className="mt-3 max-w-2xl text-[13px] leading-relaxed text-faint">
              이 페이지는 두 가지로 쓰입니다 — ① 모델 학습을 처음 접하는 사람도 따라올 수 있는{" "}
              <b className="text-mut">기초 교과서</b>, ② 이 랩의 숫자를 왜 믿어도 되는지에 대한{" "}
              <b className="text-mut">보고 자료</b>. 위에서 아래로 읽으면 배경 지식 없이도 전체가 이어집니다.
            </p>
            <div className="mt-5 flex flex-wrap gap-2">
              {TOC.map(([id, label]) => (
                <a
                  key={id}
                  href={`#${id}`}
                  className="mono rounded-full border border-line2 bg-ink-880/50 px-3 py-1.5 text-[11.5px] text-mut transition-colors hover:border-signal/40 hover:text-fg"
                >
                  {label}
                </a>
              ))}
            </div>
          </div>
        </Panel>
      </Section>

      {/* ── 1. Role ────────────────────────────────────────────────────── */}
      <Topic id="role" title="랩의 위치와 역할" hint="검색 엔진이 아니라, 그 안의 부품 하나를 만듭니다" delay={40}>
        <Panel className="space-y-4 p-5">
          <PipelineDiagram />
          <p className="text-[13px] leading-relaxed text-mut">
            사내 검색 프로덕션은 이미 3단으로 돌아갑니다 — <b className="text-fg">BM25</b>(키워드)와{" "}
            <b className="text-fg">Dense 임베딩</b>(의미)이 각자 후보를 모으고, 융합한 뒤, <b className="text-fg">리랭커</b>
            가 최종 순서를 정합니다. 이 랩이 만드는 것은 그중 <b className="text-fg">Dense 임베딩 모델 한 부품</b>입니다.
            검색 시스템을 새로 만드는 게 아니라, 잘 돌아가는 파이프라인의 부품을 더 좋은 것으로 교체하는 프로젝트입니다.
          </p>
          <Note tone="signal" title="그래서 이 랩은 dense 단독 성능만 잽니다 (변수 격리)">
            바꾸는 변수가 임베딩 모델 하나라면, 측정도 그 변수만 격리해서 해야 개선의 원인을 모델에{" "}
            <b className="text-fg">귀속</b>시킬 수 있습니다. 처음부터 BM25·리랭커까지 섞어 재면 점수가 움직여도{" "}
            <i>무엇이</i> 좋아진 건지 알 수 없습니다. dense-only는 한계가 아니라 올바른 실험 설계입니다.
          </Note>
          <Tbl
            head={["질문", "재는 방법", "성격"]}
            rows={[
              ["임베딩 모델이 좋아졌나?", "이 랩의 dense 단독 평가", "빠르고 원인 귀속 가능 — 수십 번 반복하며 튜닝"],
              ["검색 서비스가 좋아졌나?", "프로덕션의 hybrid + rerank A/B 테스트", "느리고 비싸지만 최종 판정 — 납품 후 1회"],
            ]}
          />
        </Panel>
      </Topic>

      {/* ── 2. Basics ──────────────────────────────────────────────────── */}
      <Topic id="basics" title="배경 ① — 임베딩 검색은 어떻게 동작하나" hint="3분 기초: 벡터, cosine, 그리고 왜 파인튜닝인가" delay={60}>
        <Panel className="space-y-4 p-5">
          <p className="text-[13px] leading-relaxed text-mut">
            <b className="text-fg">임베딩(embedding)</b>은 텍스트를 숫자 목록, 즉 <b className="text-fg">벡터</b>로 바꾸는
            일입니다. 이 벡터는 고차원 공간의 좌표이고(이 랩의 기본 모델 <M>Qwen3-Embedding-0.6B</M>는 1024차원), 핵심
            성질은 하나입니다 — <b className="text-fg">의미가 비슷한 텍스트일수록 좌표가 가깝다</b>. 검색은 이 성질을 그대로
            씁니다: ① 모든 문서를 미리 벡터로 바꿔 저장(색인)하고 ② 쿼리가 오면 쿼리도 벡터로 바꾼 뒤 ③ 가장 가까운 문서
            벡터들을 찾습니다. “가깝다”는 <b className="text-fg">cosine 유사도</b>(두 벡터의 방향이 같을수록 1)로
            잽니다.
          </p>
          <SpaceDiagram />
          <p className="text-[13px] leading-relaxed text-mut">
            <b className="text-fg">그런데 왜 파인튜닝이 필요할까요?</b> 공개 임베딩 모델은 위키·웹 텍스트로 배웠기 때문에
            일반적인 한국어 의미는 잘 알지만, <b className="text-fg">우리 회사의 말은 모릅니다</b> — “지라 안
            열려요”가 이슈 트래커 포털을 가리키는지, “맥북 신청”이 어느 사이트의 메뉴인지. 파인튜닝은 우리의
            (쿼리, 정답 문서) 쌍 수천 개로 이 좌표계를 살짝 재배치해서, <b className="text-fg">사내 표현과 사내 사이트가
            가까워지도록</b> 만드는 일입니다.
          </p>
          <Analogy>
            임베딩 모델은 “의미를 지도 위 위치로 옮기는 지도 제작자”입니다. 범용 모델은 세계지도를 잘 그리지만 우리
            동네 골목은 모릅니다. 파인튜닝은 동네 주민(우리 데이터)의 안내를 받아 골목 지도를 고치는 일이고요.
          </Analogy>
          <Note tone="cyan" title="BM25와 왜 같이 쓰나">
            BM25는 같은 단어가 등장해야 찾는 대신 정확한 코드명·약어·신조어에 강하고, dense는 단어가 달라도(“연차 쓰는
            법” ↔ “휴가 신청 절차”) 의미로 찾습니다. 서로의 빈틈을 메우는 조합이라 프로덕션은 둘 다 쓰고,
            리랭커가 마무리합니다.
          </Note>
        </Panel>
      </Topic>

      {/* ── 3. Flow ────────────────────────────────────────────────────── */}
      <Topic id="flow" title="전체 플로우 — 여섯 단계의 의미" hint="사이드바의 탭 순서 그대로" delay={80}>
        <div className="space-y-3">
          {FLOW.map((s, i) => {
            const Icon = s.icon;
            return (
              <Panel key={s.title} className="flex items-start gap-4 p-5">
                <div className="flex shrink-0 items-center gap-2.5">
                  <span className="mono grid h-7 w-7 place-items-center rounded-lg bg-ink-800 text-[12px] font-semibold text-mut">
                    {i + 1}
                  </span>
                  <Icon size={17} className="text-signal" />
                </div>
                <div className="min-w-0 flex-1">
                  <div className="flex items-center gap-2.5">
                    <h3 className="text-[14.5px] font-semibold text-fg">{s.title}</h3>
                    <button
                      onClick={() => nav(s.to)}
                      className="inline-flex items-center gap-1 text-[11.5px] text-faint transition-colors hover:text-signal"
                    >
                      탭 열기 <ArrowRight size={12} />
                    </button>
                  </div>
                  <p className="mt-1.5 text-[12.5px] leading-relaxed text-mut">{s.what}</p>
                  <p className="mt-1.5 text-[12.5px] leading-relaxed text-faint">
                    <b className="text-mut">왜 이 단계가 있나 — </b>
                    {s.why}
                  </p>
                </div>
              </Panel>
            );
          })}
        </div>
        <div className="mt-4 grid gap-5 lg:grid-cols-2">
          <Panel className="p-5">
            <div className="mb-2.5 flex items-center gap-2">
              <h3 className="text-[14px] font-semibold text-fg">학습 데이터의 생김새</h3>
              <Tag tone="signal">→ 학습</Tag>
            </div>
            <p className="mb-2.5 text-[12.5px] leading-relaxed text-mut">모델이 배우는 (질문, 정답 문서) 쌍 — 한 줄에 한 레코드:</p>
            <Code>{`{"query": "비밀번호 재설정 방법",
 "positive": {"title": "...", "content": "..."},
 "negatives": [{"title": "...", "content": "..."}]}`}</Code>
            <p className="mt-2.5 text-[12px] leading-relaxed text-faint">
              negatives(헷갈리는 오답)는 선택 — 있으면 Triplet loss를 쓸 수 있고, MNRL 계열은 채굴된 오답을{" "}
              <b className="text-mut">전부</b> 추가 신호로 씁니다. 채굴 때는 정답 점수에 너무 가까운 후보(가짜 오답일
              확률 높음)를 margin으로 걸러냅니다 — 진짜 정답을 오답으로 가르치면 모델이 역주행합니다.
            </p>
          </Panel>
          <Panel className="p-5">
            <div className="mb-2.5 flex items-center gap-2">
              <h3 className="text-[14px] font-semibold text-fg">평가셋의 생김새 (BEIR 표준)</h3>
              <Tag tone="cyan">→ 평가</Tag>
            </div>
            <p className="mb-2.5 text-[12.5px] leading-relaxed text-mut">건초더미에서 바늘 찾기 — corpus 안에서 정답을 위로 올리는지 봅니다:</p>
            <Code>{`corpus.jsonl     {"_id","title","text"}    # 건초더미(전체 문서)
queries.jsonl    {"_id","text"}             # 사용자 질문
qrels/dev.tsv    query-id  corpus-id  1     # 채점표 (선택용)
qrels/final.tsv  query-id  corpus-id  1     # 채점표 (확정용)`}</Code>
            <p className="mt-2.5 text-[12px] leading-relaxed text-faint">
              corpus는 정답 + 충분히 많은 distractor로 커야 합니다 — 정답만 있으면 모든 모델이 만점이라 구분이 안 됩니다.
              크롤 corpus 모드에서는 <b className="text-mut">사이트 전체가 그대로 건초더미</b>가 되어 distractor를 합성할
              필요가 없습니다 — 프로덕션 인덱스와 같은 상황입니다.
            </p>
          </Panel>
        </div>
        <div className="mt-4">
          <Note tone="amber" title="철칙: 학습 쿼리와 평가 쿼리는 겹치면 안 됩니다 (leakage)">
            학습에서 본 쿼리로 시험을 보면 일반화가 아니라 <i>암기</i>를 재게 됩니다. 점수는 화려한데 실제 사용자 쿼리에서
            무너지는 모델이 이렇게 만들어집니다. 이 랩의 데이터 생성기는 문서 단위로 train/test를 먼저 가른 뒤, 품질
            필터(라운드트립)는 <b className="text-fg">train 쪽에만</b> 겁니다 — 평가 쿼리를 평가에 쓸 임베더로 거르면
            “그 모델이 이미 맞히는 문제”만 남아 모든 지표가 1.0으로 포화되거든요. 이 랩이 실제로 한 번 밟았던 함정입니다.
          </Note>
        </div>
      </Topic>

      {/* ── 4. Training mechanics ──────────────────────────────────────── */}
      <Topic id="training" title="배경 ② — 학습에서 실제로 일어나는 일" hint="경사하강법, epoch, 그리고 early stopping" delay={100}>
        <Panel className="space-y-4 p-5">
          <p className="text-[13px] leading-relaxed text-mut">
            학습의 1 <b className="text-fg">step</b>은 이렇습니다 — ① 학습쌍 한 묶음(<b className="text-fg">batch</b>, 기본
            16쌍)을 모델에 통과시키고 ② 지금 좌표 기준으로 벌점(<b className="text-fg">loss</b>)을 계산한 뒤 ③ 벌점이
            줄어드는 방향(gradient)으로 모델의 모든 가중치를 아주 조금씩 수정합니다. 이 “조금”의 크기가{" "}
            <b className="text-fg">learning rate</b>입니다. 데이터 전체를 한 바퀴 돌면 1 <b className="text-fg">epoch</b>.
          </p>
          <Analogy>
            안개 낀 산에서 내려오는 등산객입니다. 현재 위치에서 가장 가파른 내리막 방향(gradient)으로 보폭(learning
            rate)만큼 한 걸음씩 — 보폭이 너무 크면 골짜기를 지나쳐 반대편으로 튀고, 너무 작으면 해 지기 전에 못 내려옵니다.
          </Analogy>
          <p className="text-[13px] leading-relaxed text-mut">
            문제는 <b className="text-fg">오래 돌릴수록 좋은 게 아니라는 것</b>입니다. 학습 데이터에 대한 벌점은 계속
            떨어지지만, 어느 순간부터 모델은 일반 규칙 대신 <b className="text-fg">학습 데이터 자체를 암기</b>하기 시작합니다
            (<b className="text-fg">과적합</b>). 그 신호는 학습에 안 쓴 검증 데이터의 점수가 꺾이는 순간입니다.
          </p>
          <EarlyStopDiagram />
          <p className="text-[13px] leading-relaxed text-mut">
            그래서 이 랩의 <b className="text-fg">epochs는 “몇 번 돌지”가 아니라 천장(기본 12)</b>입니다. 매 epoch
            끝에 검증쌍으로 시험을 보고(val nDCG@10), 최고 기록이 갱신되면 그 시점의 가중치를 스냅샷하고,{" "}
            <b className="text-fg">patience</b>(기본 3)번 연속 갱신이 없으면 자동 중단합니다. 저장되는 건 마지막이 아니라{" "}
            <b className="text-fg">가장 좋았던 epoch의 모델</b> — 이름 끝의 <M>-e7</M>이 “7번째 epoch이
            최고였다”는 뜻입니다. epoch별 점수 이력 전체는 모델 폴더의 <M>train_meta.json</M>에 남습니다.
          </p>
        </Panel>
      </Topic>

      {/* ── 5. Losses ──────────────────────────────────────────────────── */}
      <Topic id="loss" title="Loss 함수 — 무엇을 '틀렸다'고 정의할 것인가" hint="4종 모두 목표는 같고, 오답의 출처가 다릅니다" delay={120}>
        <Panel className="space-y-4 p-5">
          <p className="text-[13px] leading-relaxed text-mut">
            <b className="text-fg">loss(손실 함수)는 “지금 모델이 얼마나 틀렸나”를 숫자 하나로 만드는 벌점
            함수</b>입니다. 학습은 오직 이 숫자를 줄이는 방향으로만 움직이므로,{" "}
            <b className="text-fg">loss의 설계가 곧 “모델이 무엇을 배우는가”의 정의</b>입니다. 검색 임베딩의 목표는
            공통입니다 — 쿼리↔정답은 가깝게, 쿼리↔오답은 멀게(<b className="text-fg">대조 학습</b>). 네 가지 loss의 차이는
            사실상 하나, <b className="text-fg">오답(negative)을 어디서 구해오느냐</b>입니다.
          </p>
          <div className="grid items-center gap-5 lg:grid-cols-[430px_1fr]">
            <MnrlMatrix />
            <div className="space-y-3">
              <Note tone="signal" title="이 행렬이 MNRL의 전부입니다">
                별도의 오답 라벨링 없이, 같은 batch 안 남의 정답이 내 오답이 됩니다. batch가 클수록 시험의 선택지가
                많아져서 학습 신호가 강해지고 — 그래서 <b className="text-fg">MNRL에선 batch size가 성능
                하이퍼파라미터</b>입니다.
              </Note>
              <p className="text-[12.5px] leading-relaxed text-mut">
                수식으로는 <M>−log( e^s(q,d⁺) / Σᵢ e^s(q,dᵢ) )</M> — softmax로 “정답이 뽑힐 확률”을 만들고 그
                확률이 낮을수록 벌점을 키웁니다. InfoNCE라고 부르는 형태로, 현대 검색 임베딩 학습의 사실상 표준입니다.
              </p>
            </div>
          </div>
          <div className="grid gap-4 sm:grid-cols-2">
            {LOSSES.map((l) => (
              <Panel key={l.name} className="flex flex-col gap-2.5 border-line2 bg-ink-880/40 p-5">
                <div className="flex flex-wrap items-center gap-2">
                  <h3 className="text-[14.5px] font-semibold text-fg">{l.name}</h3>
                  <Tag tone={l.tone}>{l.tag}</Tag>
                </div>
                <div className="mono text-[10.5px] text-faint">{l.full}</div>
                <p className="text-[12.5px] leading-relaxed text-mut">{l.how}</p>
                <div className="mt-auto space-y-1.5 border-t border-line/60 pt-2.5 text-[12px] leading-relaxed">
                  <div>
                    <b className="text-signal2">언제 </b>
                    <span className="text-mut">{l.when}</span>
                  </div>
                  <div>
                    <b className="text-amber">주의 </b>
                    <span className="text-mut">{l.caveat}</span>
                  </div>
                </div>
              </Panel>
            ))}
          </div>
          <Tbl
            head={["상황", "추천 loss"]}
            rows={[
              ["처음 시작 / 확신 없음", "MNRL — 표준이자 기본값. 여기서 출발해 나머지를 비교"],
              ["batch를 키우고 싶은데 메모리 부족(OOM)", "Cached MNRL — 같은 수학, 메모리 제약 해제"],
              ["비슷한 문서가 많은 코퍼스 (사내 위키 등)", "GIST — 가짜 오답을 걸러 신호를 깨끗하게"],
              ["hard negative 데이터 보유 + 미세 구분 학습", "Triplet — 헷갈리는 쌍을 정밀 타격"],
            ]}
          />
          <Note tone="signal" title="Matryoshka — loss를 감싸는 옵션">
            위 네 loss는 “무엇을 가깝게/멀게 할까”를 정합니다. <b className="text-fg">Matryoshka</b>는 그 위에 덧씌우는
            wrapper로, 같은 목표를 <b className="text-fg">여러 prefix 길이</b>(예: 1024·512·256·128·64)에서 동시에 학습합니다.
            결과 벡터는 <b className="text-fg">앞부분만 잘라 써도</b> 순위가 거의 유지돼서, 프로덕션이 저장·검색 비용을 줄이려
            짧은 벡터를 쓸 때 그대로 납품할 수 있는 dense 부품이 됩니다. 학습 탭의 loss 아래 체크박스로 켜며, 모델 이름에{" "}
            <M>-mrl</M>이 붙습니다. 효과는 <b className="text-fg">평가 탭의 “차원 절단”</b>으로 256·128·64차원 점수를 재서
            확인하세요 — 차원↓당 품질 손실 곡선이 보입니다. (여러 차원을 동시에 학습해 메모리를 많이 쓰니, OOM이면 차원
            수를 줄이세요.)
          </Note>
        </Panel>
      </Topic>

      {/* ── 6. Hyperparameters ─────────────────────────────────────────── */}
      <Topic id="knobs" title="하이퍼파라미터 — 레시피의 손잡이들" hint="배우는 '내용'(데이터)이 아니라 배우는 '방식'을 정합니다" delay={140}>
        <Panel className="space-y-4 p-5">
          <p className="text-[13px] leading-relaxed text-mut">
            <b className="text-fg">하이퍼파라미터</b>는 학습이 시작되기 전에 사람이 정하는 설정값입니다. 가중치는 모델이
            스스로 배우는 값, 하이퍼파라미터는 우리가 정하는 값 — 그리고 이 랩의 스윕은 “이 손잡이들을 어떻게 돌리는 게
            최선인가”를 실험으로 답하는 도구입니다.
          </p>
          <Tbl
            head={["손잡이", "직관", "기본값", "조절의 의미"]}
            rows={[
              [
                "learning rate",
                "한 걸음의 보폭",
                <M key="v">2e-5</M>,
                "크면 빨리 배우지만 발산 위험, 작으면 안전하지만 느리고 미적지근. 효과가 가장 큰 손잡이 — 스윕 1순위 (예: 1e-5 vs 2e-5 vs 1e-4)",
              ],
              [
                "batch size",
                "한 번에 보는 학습쌍 수",
                <M key="v">16</M>,
                "MNRL에선 batch = 오답 수 → 메모리가 허락하는 한 클수록 유리. OOM이면 줄이거나 Cached MNRL로 전환",
              ],
              [
                "epochs",
                "전체 데이터 회독 수의 천장",
                <M key="v">12</M>,
                "early stopping이 실제 종료를 정하므로 “넉넉히”가 정답. 보통 건드릴 필요 없음",
              ],
              [
                "dropout",
                "학습 중 뉴런을 끄는 비율",
                "모델 기본값",
                "과적합 신호(val 점수가 일찍 꺾임)가 보이면 0.05 → 0.1 → 0.2로 올려 비교. 너무 높이면 학습 자체가 둔해짐",
              ],
              [
                "seed",
                "난수 주사위 고정",
                <M key="v">42</M>,
                "같은 레시피의 재현용. 거꾸로 seed만 바꿔 여러 번 돌리면 ‘운에 의한 분산’을 잴 수 있음",
              ],
            ]}
          />
          <div className="space-y-3">
            <h3 className="text-[14px] font-semibold text-fg">dropout, 조금 더 깊게</h3>
            <p className="text-[13px] leading-relaxed text-mut">
              학습 중 매 step, 각 뉴런을 확률 p로 <b className="text-fg">임시로 꺼버립니다</b>(추론 때는 전부 켭니다). 효과는
              두 가지 — ① 특정 뉴런 몇 개가 “협업해서 정답을 암기”하는 회로를 만들지 못하게 하고, ② 매 step 조금씩
              다른 부분 신경망이 학습되므로 작은 모델 수천 개를 평균낸 <b className="text-fg">앙상블 효과</b>가 납니다. 둘 다
              결론은 같습니다: 암기를 어렵게 해서 <b className="text-fg">일반화를 돕는다</b>.
            </p>
            <Analogy>
              감독이 연습 때마다 무작위로 두세 명을 빼고 뛰게 하는 축구팀입니다. 에이스 한 명에게 의존하는 전술이 사라지고
              전원이 기본기를 갖추게 되죠. 시합(추론) 날에는 전원이 출전합니다.
            </Analogy>
            <Note tone="amber" title="사내 데이터처럼 학습쌍이 적을 때 특히 중요한 손잡이">
              데이터가 적을수록 암기가 쉬워 과적합이 빨리 옵니다. val 곡선이 2–3 epoch 만에 꺾이면 dropout을 올려 비교해
              보세요. 아키텍처마다 설정 키 이름이 달라서(Qwen3는 <M>attention_dropout</M>) 실제 적용된 키는 학습 로그에
              출력됩니다.
            </Note>
          </div>
          <Note tone="amber" title="스윕은 '한 변수' 탐침입니다 — 한계를 알고 쓰세요">
            이 랩의 스윕은 한 축만 바꾸고 나머지는 <b className="text-fg">고정</b>합니다. 읽기 쉬운 곡선을 주지만 두 가지를
            기억하세요. ① <b className="text-fg">상호작용</b> — learning rate는 batch·rank·loss와 얽혀서, 고정 LR에서 고른
            “최적 rank”가 LR이 바뀌면 더는 최적이 아닐 수 있습니다(비-LR 축은 <b className="text-fg">LR 동반(2축)</b>으로 함께
            변주). ② <b className="text-fg">노이즈</b> — 단발 점수는 학습 운에 흔들리니 <b className="text-fg">시드 반복</b>으로
            평균±편차를 보세요. 그리고 <span className="mono">비교</span> 탭의 유의성 검정은{" "}
            <b className="text-fg">평가셋 표집 노이즈</b>를 재는 것이지 학습 재현 노이즈가 아닙니다 — 둘은 다릅니다.
          </Note>
        </Panel>
      </Topic>

      {/* ── 7. Full vs LoRA ────────────────────────────────────────────── */}
      <Topic id="method" title="Full vs LoRA — 어디까지 고쳐 배울 것인가" hint="전부 다시 쓰기 vs 포스트잇 붙이기" delay={160}>
        <Panel className="space-y-4 p-5">
          <p className="text-[13px] leading-relaxed text-mut">
            <b className="text-fg">Full fine-tuning</b>은 모델의 모든 가중치(0.6B 모델이면 6억 개)를 수정 대상으로 삼습니다.
            표현력의 상한이 가장 높지만 비용도 큽니다 — 가중치에 더해 기울기·옵티마이저 상태까지 들고 있어야 해서 메모리가
            파라미터의 몇 배로 들고, 학습쌍이 적으면 6억 개의 손잡이가 수천 개의 예제를 <b className="text-fg">통째로
            외워버리는</b>(과적합) 위험이 있습니다.
          </p>
          <p className="text-[13px] leading-relaxed text-mut">
            <b className="text-fg">LoRA</b>(Low-Rank Adaptation)는 다르게 접근합니다 — 원본 가중치 W는{" "}
            <b className="text-fg">동결</b>하고, 각 층 옆에 작은 우회 행렬 두 개(A: d×r, B: r×d)만 학습합니다. 곱{" "}
            <M>B·A</M>가 “수정사항”이 되는데 r이 작아서 학습 파라미터는 전체의 1% 미만입니다. 배경에 있는 통찰:{" "}
            <b className="text-fg">도메인 적응에 필요한 변화는 대부분 ‘단순한 방향 몇 개’(저랭크)로
            충분하다</b>는 것.
          </p>
          <LoraDiagram />
          <Analogy>
            full은 교과서 전체를 다시 집필하는 것, LoRA는 교과서에 포스트잇을 붙이는 것입니다. 저장할 때 포스트잇 내용을
            본문에 반영해 다시 제본(merge)하므로, 받아보는 쪽(서빙)은 어느 방식으로 만들었는지 구분할 수 없습니다.
          </Analogy>
          <Tbl
            head={["LoRA 노브", "뜻", "기본값", "조절의 의미"]}
            rows={[
              [
                "r (rank)",
                "우회로의 폭 = 수정사항의 표현력",
                <M key="v">16</M>,
                "도메인이 멀거나 데이터가 많으면 ↑(32), 데이터가 적으면 ↓(8). LoRA 첫 스윕 축으로 추천 (8 vs 16 vs 32)",
              ],
              [
                "alpha (α)",
                "수정사항의 반영 강도 (실효 배율 α/r)",
                <M key="v">32</M>,
                "관례: α = 2r로 고정하고 r만 움직입니다 — 움직이는 축을 줄여야 실험 간 비교가 성립",
              ],
              [
                "lora_dropout",
                "어댑터에만 거는 dropout",
                <M key="v">0.05</M>,
                "본체는 동결이라 과적합은 어댑터에서 일어납니다 — 데이터가 적으면 0.1까지 올려 비교",
              ],
              [
                "target_modules",
                "포스트잇을 붙일 위치",
                <M key="v">all-linear</M>,
                "attention(q·k·v·o)만: 가볍고 빠름 / all-linear(FFN 포함): 대체로 더 좋다는 보고(QLoRA) — 그래서 기본값",
              ],
              [
                "learning rate",
                "(LoRA일 때의 보폭)",
                <M key="v">1e-4</M>,
                "전체를 다시 쓰는 게 아니라 작은 우회로만 조정하므로 full보다 ~5배 큰 보폭이 안전하고 관행",
              ],
            ]}
          />
          <Note tone="signal" title="뭘 골라야 하나">
            학습쌍이 수천~수만이고 빠르게 반복하고 싶다 → <b className="text-fg">LoRA부터</b>. 데이터가 많고 한계 성능을
            확인하고 싶다 → <b className="text-fg">full</b>도 시도. 둘이 비슷하면 가벼운 쪽(LoRA). 정답은 추측이 아니라{" "}
            <b className="text-fg">같은 데이터로 둘 다 돌려 dev에서 비교</b>하는 것 — 이 랩의 스윕이 그 비교를 자동화합니다.
            어느 쪽이든 저장 결과물은 병합된 일반 모델이라 서빙은 동일합니다.
          </Note>
        </Panel>
      </Topic>

      {/* ── 8. Evaluation ──────────────────────────────────────────────── */}
      <Topic id="eval" title="평가 — '좋아졌다'를 측정 가능하게" hint="같은 시험지, 공개된 채점 기준" delay={180}>
        <Panel className="space-y-4 p-5">
          <p className="text-[13px] leading-relaxed text-mut">
            평가는 단순하고 가차없습니다 — 평가셋의 모든 쿼리를 임베딩해 corpus 전체를 가까운 순으로 정렬하고, 채점표
            (qrels)와 대조해 점수를 냅니다. 핵심 지표는 세 개:
          </p>
          <Tbl
            head={["지표", "묻는 질문", "이 랩에서의 역할"]}
            rows={[
              [
                "recall@50",
                "정답이 상위 50 후보 안에 들었나",
                <>
                  <b className="text-signal2">주 지표 ★</b> — 뒤에 리랭커가 있으므로 임베더의 임무는 ‘후보에
                  넣기’. 순서는 리랭커의 일
                </>,
              ],
              ["nDCG@10", "상위 10에서 정답이 얼마나 위에 있나 (위치 가중)", "보조 — dense 단독의 순위 품질. 차이에 민감해 모델 선택 지표로도 사용"],
              ["MRR@10", "첫 정답이 평균 몇 등인가", "보조 — ‘첫 화면에 정답이 보이나’"],
            ]}
          />
          <Note tone="signal" title="왜 recall@50이 주 지표인가">
            프로덕션에서 임베더는 최종 정렬자가 아니라 <b className="text-fg">후보 추천자</b>입니다. 리랭커가 상위 후보를
            다시 정렬해 주므로 정답이 50등 안에만 들면 임무 완수 — 1등으로 올리는 건 리랭커가 더 잘합니다. 반대로 임베더가
            50 안에 못 넣은 정답은 <b className="text-fg">리랭커도 영영 보지 못합니다</b>. recall이 전체 검색 품질의
            천장인 이유입니다.
          </Note>
          <Note tone="amber" title="단, 작은 corpus에선 recall@50이 포화됩니다">
            recall@50은 <b className="text-fg">건초더미가 충분히 클 때만</b> 변별력이 있습니다. corpus가 300개면 상위 50은
            상위 16%라, 강한 모델은 거의 항상 정답을 담아 점수가 <b className="text-fg">1.0에 붙어버립니다</b>(모델 간 차이
            0). 이 PoC corpus를 300→1500으로 키우자 recall@1이 0.94(포화)에서 0.80으로 내려와 비로소 모델을 가렸지만,
            recall@50은 1500에서도 0.98로 평탄했습니다. <b className="text-fg">실전 규칙</b>: recall@50은 “프로덕션 후보
            커버리지” 지표로 보고하되, <b className="text-fg">모델 선택은 헤드룸이 있는 recall@5·recall@1·nDCG@10으로</b>{" "}
            하세요. recall@50 자체로 모델을 고르려면 corpus가 수천~만 단위는 돼야 합니다.
          </Note>
          <p className="text-[13px] leading-relaxed text-mut">
            여기에 두 가지 안전장치가 붙습니다. 첫째, <b className="text-fg">평가셋 지문(fingerprint)</b> — 평가셋 내용의
            해시가 모든 점수에 함께 기록되고, 평가셋이 한 글자라도 바뀌면 지문이 달라져 다른 지문끼리의 비교를 UI가
            차단합니다. “지난달 점수와 비교했는데 알고 보니 시험지가 달랐다” 사고의 원천 봉쇄. 둘째,{" "}
            <b className="text-fg">dev / final 분리</b>:
          </p>
          <Analogy>
            dev는 <b className="text-fg">모의고사</b>, final은 <b className="text-fg">수능</b>입니다. 여러 레시피 중 누가
            나은지는 모의고사(dev)로 몇 번이든 가려내되, 수능(final)은 뽑힌 승자가 마지막에 딱 한 번 봅니다. 같은 시험으로
            ‘고르기’와 ‘확정’을 다 하면, 실력이 아니라 <b className="text-fg">그 시험에 운 좋은
            모델</b>이 뽑히기 때문입니다(선택 편향). 모의고사 1등이 수능에서도 비슷한 점수면 — 그게 진짜 실력입니다.
          </Analogy>
        </Panel>
      </Topic>

      {/* ── 9. Compare ─────────────────────────────────────────────────── */}
      <Topic id="compare" title="비교 — 그 차이, 우연 아닌가요?" hint="평균 한 줄 뒤에 숨은 것들" delay={200}>
        <Panel className="space-y-4 p-5">
          <p className="text-[13px] leading-relaxed text-mut">
            쿼리 32개짜리 평가에서 nDCG +0.01은 쿼리 한두 개의 운으로도 만들어집니다. 그래서 실험 탭은 평균만 보여주지 않고
            세 가지를 함께 봅니다:
          </p>
          <div className="grid gap-3 lg:grid-cols-3">
            <Note tone="cyan" title="① 쿼리별 diff">
              어느 쿼리에서 이기고 어느 쿼리에서 졌는지, 진 쿼리에서는 실제로 무엇을 검색했는지 원문으로 확인합니다 —
              “평균은 올랐는데 VPN 쿼리들이 전부 망가졌다”를 잡아내는 장치.
            </Note>
            <Note tone="cyan" title="② p값 (paired 순열 검정)">
              “두 모델이 사실 같다면 쿼리별 승패의 방향은 동전던지기와 같다”는 가정 아래, 부호를 10,000번
              무작위로 뒤집어 지금 차이 이상이 우연히 나올 확률을 셉니다. p=0.03이면 “우연일 확률 3%”. 관례적
              기준은 0.05.
            </Note>
            <Note tone="cyan" title="③ 토픽 슬라이스">
              주제별(vpn, hr, …) 평균을 따로 봅니다 — 전체 평균이 가리는 특정 주제의 회귀(regression)를 노출합니다.
            </Note>
          </div>
          <Note tone="amber" title="p값이 안 작아져요 — 모델 탓이 아닐 수 있습니다">
            쿼리 수가 적으면 진짜 차이도 통계적으로 증명하기 어렵습니다(검정력 부족). 해법은 평가셋 키우기 — 쿼리 수가 곧
            현미경의 배율입니다. 이 랩이 평가셋 확장(실로그 가져오기·라벨링)을 데이터 탭에 둔 이유이기도 합니다.
          </Note>
        </Panel>
      </Topic>

      {/* ── 10. Handoff ────────────────────────────────────────────────── */}
      <Topic id="handoff" title="납품 — 마지막 1미터" hint="랩 점수가 프로덕션에서 증발하지 않게" delay={220}>
        <Panel className="space-y-4 p-5">
          <p className="text-[13px] leading-relaxed text-mut">
            임베딩 모델은 <b className="text-fg">입력을 만드는 방식까지가 모델</b>입니다. 서빙이 랩과 다르게 임베딩하면
            (쿼리 prefix 누락이 1순위 원인) 랩에서 올린 점수는 프로덕션에서 사라집니다. 그래서 승자 모델은 파일만 보내지
            않고, 계약서가 동봉된 <b className="text-fg">핸드오프 패키지</b>로 납품합니다:
          </p>
          <Code>{`쿼리   "Instruct: {instruction}\\nQuery: {사용자 쿼리}"   # prefix까지 똑같이
문서   "{title}\\n\\n{content}"
벡터   lasttoken pooling → L2 normalize → cosine
검증   동봉된 샘플 벡터 vs 서빙 재계산 → cosine ≥ 0.999`}</Code>
          <ul className="space-y-1.5 text-[12.5px] leading-relaxed text-mut">
            <li>
              <b className="text-fg">① 전체 재색인</b> — 새 모델의 벡터는 이전 모델의 벡터와 호환되지 않습니다(다른
              좌표계). 모델 교체 = 문서 벡터 전부 재계산.
            </li>
            <li>
              <b className="text-fg">② 패리티 확인</b> — 동봉된 샘플 입력을 서빙 파이프라인으로 임베딩해 벡터가 일치하는지
              (cosine ≥ 0.999) 확인. 통과 못 하면 전처리가 어딘가 다른 것.
            </li>
            <li>
              <b className="text-fg">③ A/B 테스트</b> — 트래픽 일부로 recall 개선이 클릭·이탈 같은 서비스 지표로
              이어지는지 최종 확인. 랩 점수는 필요조건이지 충분조건이 아닙니다.
            </li>
          </ul>
        </Panel>
      </Topic>

      {/* ── 11. Serving ────────────────────────────────────────────────── */}
      <Topic id="serving" title="서빙 — 납품한 모델이 실제로 검색하기까지" hint="벡터 DB 색인, 무중단 모델 교체, 그리고 자동화" delay={240}>
        <Panel className="space-y-4 p-5">
          <p className="text-[13px] leading-relaxed text-mut">
            학습·평가·납품이 끝난 모델이 실제 검색을 하려면 두 가지가 더 필요합니다. ①{" "}
            <b className="text-fg">색인(인덱싱)</b> — corpus의 모든 문서를 그 모델로 임베딩해{" "}
            <b className="text-fg">벡터 DB</b>에 저장해 두는 일, ② <b className="text-fg">검색</b> — 들어온 쿼리를{" "}
            <b className="text-fg">같은 모델</b>로 임베딩해 가장 가까운 문서 벡터들을 찾는 일. 이 랩의 벡터 DB는{" "}
            <M>Qdrant</M>이고, 임베딩 추론은 학습 산출물(<M>outputs/…</M>)을 변환 없이 그대로 로드하는
            sentence-transformers 인프로세스 방식입니다. <b className="text-fg">검색 탭</b>이 이 전체가 실제로
            돌아가는 곳입니다 — 인덱스 상태 확인, 재색인, 그리고 실검색까지.
          </p>
          <Analogy>
            색인은 <b className="text-fg">도서관 서가 정리</b>입니다. 임베딩 모델은 “이 책을 어느 서가에 꽂을지” 정하는
            사서, 벡터 DB는 서가고요. 사서가 바뀌면(모델 교체) 새 사서의 분류 기준으로{" "}
            <b className="text-fg">모든 책을 다시 꽂아야</b>(전면 재색인) 합니다 — 옛 배치를 그대로 두고 새 사서에게
            책을 찾아오라고 하면 엉뚱한 서가로 가거든요. 서로 다른 모델의 벡터는 호환되지 않는 좌표계라는 납품
            섹션의 이야기가 서빙에서 이렇게 나타납니다.
          </Analogy>
          <p className="text-[13px] leading-relaxed text-mut">
            그런데 전면 재색인은 문서 수천 건 기준으로도 수 분이 걸리는 일입니다. 그동안 검색이 멈추거나, 반쯤 만들어진
            인덱스가 사용자에게 보이면 안 됩니다. 이 랩은 <b className="text-fg">컬렉션 버저닝 + alias</b>로 이 문제를
            풉니다:
          </p>
          <Code>{`docs__outputs-embedding-ft-mnrl-e1-3__1024d__f61681b8872e   ← 버전 컬렉션 (모델·차원·corpus 지문을 이름에 인코딩)
docs-live                                                   ← 검색이 바라보는 유일한 이름 (alias)`}</Code>
          <ul className="space-y-1.5 text-[12.5px] leading-relaxed text-mut">
            <li>
              <b className="text-fg">① 새 컬렉션에 색인</b> — 새 모델용 컬렉션을 옆에 따로 만들어 채웁니다. 그동안
              검색은 기존 컬렉션으로 아무 일 없이 계속됩니다.
            </li>
            <li>
              <b className="text-fg">② alias 원자적 전환</b> — 색인이 <i>다 끝난 뒤에만</i> <M>docs-live</M>가
              가리키는 대상을 새 컬렉션으로 한 번에 바꿉니다. 검색이 반쯤 만든 인덱스를 보는 순간이 없고, 전환은
              무중단입니다.
            </li>
            <li>
              <b className="text-fg">③ 옛 컬렉션은 롤백용으로 보관</b> — 새 인덱스에 문제가 보이면 alias만 되돌리면
              끝. 확인이 끝나면 정리(prune)합니다.
            </li>
          </ul>
          <Note tone="signal" title="이름이 결정적이라 재실행이 안전합니다 (멱등)">
            컬렉션 이름이 (모델, 차원, corpus 내용의 지문)에서 <b className="text-fg">자동으로 유도</b>되므로, 같은
            조건으로 색인을 다시 걸면 “이미 완성돼 있음”을 감지하고 임베딩을 통째로 건너뜁니다. 버튼을 실수로 두 번
            눌러도, 자동화가 반복 실행해도 무해합니다 — 그래서 아래 자동화에 그대로 걸 수 있습니다.
          </Note>
          <Tbl
            head={["자동화", "무엇이 일어나나", "어디서"]}
            rows={[
              [
                "핸드오프 훅",
                "모델 탭에서 납품하는 순간 그 모델로 백그라운드 재색인이 자동 시작 — “이 모델이 라이브로 간다”는 결정을 인덱스가 사람 손 없이 따라갑니다",
                "모델 탭 → 납품",
              ],
              ["재색인 버튼", "모델을 골라 수동 재색인 + 진행률 표시. 실행 중 중복 시작은 거부됩니다", "검색 탭"],
              ["실검색", "쿼리를 서버의 임베더로 임베딩해 라이브 인덱스에서 검색 — 점수표가 아니라 실물 확인", "검색 탭"],
            ]}
          />
          <Note tone="amber" title="안전 가드 — 조용히 틀리는 대신 시끄럽게 멈춥니다">
            서빙에서 가장 위험한 사고는 <b className="text-fg">에러 없이 엉뚱한 결과가 나오는 것</b>입니다. 인덱스를
            만든 모델과 쿼리를 임베딩하는 모델이 다르면 검색은 “되지만” 순위가 무의미해지거든요. 그래서 차원이 다르면
            검색을 막고 재색인을 안내하며(HTTP 503), 같은 차원이라도 컬렉션 이름에 모델이 박혀 있어 상태 화면에서 눈으로
            대조할 수 있습니다. 색인의 문서와 검색의 쿼리는 학습 때와 <b className="text-fg">같은 포맷 코드</b>를
            통과합니다 — 납품 섹션의 “포맷 패리티” 계약이 서빙 경로에도 강제되는 것입니다.
          </Note>
          <p className="text-[13px] leading-relaxed text-mut">
            이 서빙 경로는 프로덕션 그 자체가 아니라 <b className="text-fg">레퍼런스 구현</b>입니다 — 프로덕션(hybrid +
            rerank)이 dense 부품을 교체할 때 따라야 할 절차(전면 재색인 → 패리티 확인 → 무중단 전환 → 롤백 대비)를 랩
            안에서 실제로 돌려보고 검증하는 축소판입니다. 납품받는 쪽은 이 탭을 그대로 절차서로 쓸 수 있습니다.
          </p>
        </Panel>
      </Topic>

      {/* ── 12. Report ─────────────────────────────────────────────────── */}
      <Topic id="report" title="보고 포인트 — 왜 이 결과를 믿어도 되는가" hint="이 섹션은 상급자에게 그대로 보여줘도 됩니다" delay={260}>
        <div className="space-y-4">
          <Panel className="relative overflow-hidden p-6">
            <div className="absolute -left-12 -top-12 h-44 w-44 rounded-full bg-signal/10 blur-3xl" />
            <p className="relative max-w-3xl text-[15px] font-medium leading-relaxed text-fg">
              “범용 임베딩 모델은 우리 회사의 용어와 사이트를 모릅니다. 이 랩은 사내 데이터로 임베딩 모델을 다시
              가르치고, 고정된 평가셋과 통계 검정으로 ‘진짜 좋아졌는지’를 증명한 뒤, 검증을 통과한 모델만
              패키지로 납품합니다. 프로덕션 검색은 부품 하나를 교체하듯 개선됩니다.”
            </p>
            <p className="relative mt-2.5 text-[12px] text-faint">— 이 랩을 한 문단으로 설명하면</p>
          </Panel>
          <Panel className="space-y-4 p-5">
            <div className="flex items-center gap-2">
              <ShieldCheck size={16} className="text-signal" />
              <h3 className="text-[14px] font-semibold text-fg">이 랩이 구조적으로 보장하는 것</h3>
            </div>
            <Tbl head={["주장", "이를 보장하는 장치", "확인 위치"]} rows={GUARANTEES} />
            <Note tone="cyan" title="보고 동선 추천 — 숫자 세 개면 충분합니다">
              <b className="text-fg">개요 탭</b>(최고 점수와 추이) → <b className="text-fg">실험 탭</b>(base 모델 대비
              쿼리별 diff와 p값, final 확정 ✓) → <b className="text-fg">모델 탭</b>(레시피와 핸드오프) →{" "}
              <b className="text-fg">검색 탭</b>(그 모델이 실제로 검색하는 라이브 데모). 보고에는{" "}
              <M>Δrecall@50</M>, <M>p값</M>, <M>final 점수</M> 세 숫자를 중심에 두세요 — 각각 “얼마나 좋아졌나 /
              우연인가 / 새 데이터에서도 유지되나”에 답합니다.
            </Note>
            <div className="flex flex-wrap gap-2.5">
              <Btn icon={<BarChart3 size={15} />} onClick={() => nav(PATH.compare)}>
                실험 결과 보기
              </Btn>
              <Btn variant="ghost" icon={<Package size={15} />} onClick={() => nav(PATH.models)}>
                모델 서가 · 납품
              </Btn>
            </div>
          </Panel>
        </div>
      </Topic>

      {/* ── 13. Glossary ───────────────────────────────────────────────── */}
      <Topic id="glossary" title="용어 사전" hint="본문에서 쓴 말들, 한 줄씩" delay={280}>
        <Panel className="p-5">
          <div className="grid gap-x-10 gap-y-2.5 sm:grid-cols-2">
            {GLOSSARY.map(([t, d]) => (
              <div key={t} className="text-[12.5px] leading-relaxed">
                <span className="mono font-semibold text-fg">{t}</span>
                <span className="text-mut"> — {d}</span>
              </div>
            ))}
          </div>
        </Panel>
      </Topic>
    </div>
  );
}
