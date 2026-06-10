import { useNavigate } from "react-router-dom";
import type { ReactNode } from "react";
import { ArrowRight, BarChart3, Boxes, Database, FlaskConical, Gauge, Layers, Scale, Target } from "lucide-react";

import { PATH } from "../lib/nav";
import { Btn, Panel, Section, SectionLabel, Tag } from "../components/ui";

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

/** Monospace snippet block (data schemas, formatting examples). */
function Code({ children }: { children: ReactNode }) {
  return (
    <pre className="mono overflow-x-auto rounded-xl border border-line bg-ink-950 p-3.5 text-[11.5px] leading-relaxed text-mut">
      {children}
    </pre>
  );
}

const STEPS = [
  { n: 1, icon: Database, title: "데이터 생성", desc: "training pairs(질문↔정답)와 BEIR 형식 eval set을 만듭니다.", tone: "signal" },
  { n: 2, icon: FlaskConical, title: "학습 (fine-tune)", desc: "base embedding model을 내 도메인 데이터로 contrastive 학습.", tone: "signal" },
  { n: 3, icon: Gauge, title: "평가 (eval)", desc: "eval set에서 recall@k · MRR · nDCG로 검색 정확도 측정.", tone: "cyan" },
  { n: 4, icon: BarChart3, title: "비교 (compare)", desc: "평가한 model들을 리더보드에서 나란히 비교.", tone: "cyan" },
] as const;

export default function About() {
  const nav = useNavigate();

  return (
    <div className="space-y-9">
      {/* ── Hero ───────────────────────────────────────────────────────── */}
      <Section>
        <Panel className="relative overflow-hidden p-7">
          <div className="absolute -right-16 -top-16 h-52 w-52 rounded-full bg-signal/10 blur-3xl" />
          <div className="relative max-w-2xl">
            <div className="flex items-center gap-2.5">
              <div className="grid h-9 w-9 place-items-center rounded-xl bg-signal text-ink-950 shadow-[0_0_26px_-6px_rgba(198,242,74,0.7)]">
                <Boxes size={18} strokeWidth={2.4} />
              </div>
              <h1 className="text-[24px] font-semibold tracking-tight text-fg">RAG Embedding Lab</h1>
            </div>
            <p className="mt-4 text-[14px] leading-relaxed text-mut">
              검색에 쓰는 <b className="text-fg">embedding model</b>을 우리 도메인에 맞게{" "}
              <b className="text-fg">fine-tuning</b>하고, 그 효과를 <b className="text-fg">정직하게 측정·비교</b>하는
              lab입니다. 데이터 생성부터 학습 · 평가 · 비교까지 한 화면에서 돕니다.
            </p>
            <p className="mt-3 text-[13px] leading-relaxed text-faint">
              일반 도메인에선 최신 base embedding이 이미 강력하지만, <b className="text-mut">특화 도메인(사내 문서 등)</b>
              에서는 도메인 데이터로 fine-tune한 model이 retrieval 품질을 눈에 띄게 끌어올립니다. 이 lab은 바로 그 향상이{" "}
              <b className="text-mut">실제로 일어났는지</b>를 숫자로 확인시켜 줍니다.
            </p>
            <div className="mt-6 flex flex-wrap gap-2.5">
              <Btn icon={<Database size={15} />} onClick={() => nav(PATH.data)}>
                데이터부터 시작
              </Btn>
              <Btn variant="ghost" icon={<Gauge size={15} />} onClick={() => nav(PATH.eval)}>
                모델 평가
              </Btn>
            </div>
          </div>
        </Panel>
      </Section>

      {/* ── The loop ──────────────────────────────────────────────────── */}
      <Section delay={60}>
        <SectionLabel hint="네 단계가 한 방향으로 흐릅니다">전체 흐름</SectionLabel>
        <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
          {STEPS.map((s) => {
            const Icon = s.icon;
            const accent = s.tone === "signal" ? "text-signal" : "text-cyan";
            return (
              <Panel key={s.n} className="relative p-4">
                <div className="flex items-center gap-2.5">
                  <span className="mono grid h-6 w-6 place-items-center rounded-md bg-ink-800 text-[11px] font-semibold text-mut">
                    {s.n}
                  </span>
                  <Icon size={17} className={accent} />
                </div>
                <div className="mt-3 text-[14px] font-semibold text-fg">{s.title}</div>
                <p className="mt-1.5 text-[12px] leading-relaxed text-mut">{s.desc}</p>
              </Panel>
            );
          })}
        </div>
      </Section>

      {/* ── Data ──────────────────────────────────────────────────────── */}
      <Section delay={90}>
        <SectionLabel hint="서로 다른 두 데이터를 혼동하지 않기">데이터</SectionLabel>
        <div className="grid gap-5 lg:grid-cols-2">
          <Panel className="p-5">
            <div className="mb-3 flex items-center gap-2">
              <Layers size={16} className="text-signal" />
              <h3 className="text-[15px] font-semibold text-fg">학습 데이터 — training pairs</h3>
              <Tag tone="signal">→ 학습</Tag>
            </div>
            <p className="text-[12.5px] leading-relaxed text-mut">
              모델이 배우는 (질문, 정답 문서) 쌍입니다. 한 줄에 한 레코드인 JSONL:
            </p>
            <Code>{`{"query": "비밀번호 재설정 방법",
 "positive": {"title": "...", "content": "..."},
 "negatives": [{"title": "...", "content": "..."}]}`}</Code>
            <ul className="mt-3 space-y-1.5 text-[12.5px] leading-relaxed text-mut">
              <li>
                <b className="text-fg">예제(toy)</b> — 파이프라인 점검용 미리 만든 샘플.
              </li>
              <li>
                <b className="text-fg">AI 자동(synthetic)</b> — LLM이 문서를 읽고 그 문서가 답하는 query를 생성. 더해서{" "}
                <b className="text-fg">hard negative</b>(헷갈리는 오답)를 mining해 대조 학습을 날카롭게.
              </li>
            </ul>
          </Panel>

          <Panel className="p-5">
            <div className="mb-3 flex items-center gap-2">
              <Target size={16} className="text-cyan" />
              <h3 className="text-[15px] font-semibold text-fg">평가셋 — eval set (BEIR)</h3>
              <Tag tone="cyan">→ 평가</Tag>
            </div>
            <p className="text-[12.5px] leading-relaxed text-mut">
              "needle in a haystack" — 큰 corpus 안에서 정답 문서를 위로 올리는지 봅니다. 표준 BEIR 레이아웃:
            </p>
            <Code>{`corpus.jsonl    {"_id","title","text"}   # 건초더미(전체 문서)
queries.jsonl   {"_id","text"}            # 사용자 질문
qrels/test.tsv  query-id  corpus-id  score # 정답 판정`}</Code>
            <p className="mt-3 text-[12.5px] leading-relaxed text-mut">
              corpus는 <b className="text-fg">정답 + 많은 distractor</b>로 충분히 커야 합니다. 정답 문서만 있으면 모든
              지표가 1.0에 포화돼 model을 구분하지 못합니다.
            </p>
          </Panel>
        </div>
        <div className="mt-4">
          <Note tone="amber" title="두 데이터의 query는 겹치면 안 됩니다 (leakage)">
            학습한 query로 평가하면 일반화가 아니라 <i>암기</i>를 재게 됩니다. 그래서 같은 topic이라도 학습용 query와
            평가용 query를 분리해 둡니다(<span className="mono">datagen/topics.py</span>).
          </Note>
        </div>
      </Section>

      {/* ── Training ──────────────────────────────────────────────────── */}
      <Section delay={120}>
        <SectionLabel hint="대조 학습으로 도메인에 맞춥니다">학습 방식</SectionLabel>
        <Panel className="space-y-3.5 p-5">
          <p className="text-[13px] leading-relaxed text-mut">
            <b className="text-fg">Contrastive fine-tuning</b>입니다. 손실 함수는{" "}
            <span className="mono text-[12px]">MultipleNegativesRankingLoss</span> (InfoNCE) — 같은 batch 안의 다른
            샘플들을 <b className="text-fg">in-batch negative</b>로 삼아 "정답을 오답들보다 위로" 당깁니다. 그래서{" "}
            <b className="text-fg">batch size가 곧 negative 수</b>이고, 클수록 신호가 강해집니다.
          </p>
          <div className="grid gap-3 sm:grid-cols-2">
            <Note tone="signal" title="parity — 학습 ≈ 평가 ≈ 서빙">
              query는 <span className="mono">Instruct: …\nQuery: …</span>, 문서는{" "}
              <span className="mono">title\n\ncontent</span>로 — 세 곳이 <b>똑같이</b> 임베딩하도록 한 곳에서 정의합니다.
              어긋나면 fine-tune 효과가 서빙에서 사라집니다.
            </Note>
            <Note tone="cyan" title="학습 방법 — full / LoRA">
              학습 탭에서 <b className="text-fg">전체(full) fine-tuning</b>(모든 parameter)과 <b className="text-fg">LoRA</b>
              (가벼운 부분 학습 — 저장 시 base에 <b className="text-fg">병합</b>되어 결과물은 일반 모델)를 골라
              학습하고, 같은 데이터로 각각 평가해 <span className="mono">실험</span> 탭에서 나란히 비교할 수 있습니다.
            </Note>
          </div>
        </Panel>
      </Section>

      {/* ── Evaluation ────────────────────────────────────────────────── */}
      <Section delay={150}>
        <SectionLabel hint="바꾼 것(embedder)만 격리해서 측정">평가 방식</SectionLabel>
        <Panel className="space-y-3.5 p-5">
          <p className="text-[13px] leading-relaxed text-mut">
            eval set의 모든 query를 embedding해 corpus 전체를 cosine으로 ranking하고, qrels와 대조해{" "}
            <span className="mono text-[12px]">recall@k · MRR@10 · nDCG@10</span>을 계산합니다(BEIR / trec_eval 규약).
          </p>
          <Note tone="signal" title="왜 dense-only로 재나 — 변수 격리">
            바꾸는 건 파이프라인에서 <b className="text-fg">dense embedder 하나</b>뿐입니다. 그 효과를 재려면 나머지를
            고정하고 그 변수만 측정해야 합니다. 처음부터 end-to-end(BM25 + dense + rerank)로 재면 BM25와 reranker가
            embedder의 변화를 가리거나 섞어버려 <b className="text-fg">귀속(attribution)이 불가능</b>합니다. 그래서
            dense-only는 한계가 아니라 <b className="text-fg">올바른 실험 설계</b>입니다.
          </Note>
          <div className="overflow-hidden rounded-xl border border-line">
            <table className="w-full text-left text-[12.5px]">
              <thead>
                <tr className="border-b border-line bg-ink-880/60 text-[11px] uppercase tracking-wider text-faint">
                  <th className="px-3.5 py-2.5 font-medium">질문</th>
                  <th className="px-3.5 py-2.5 font-medium">측정 방법</th>
                  <th className="px-3.5 py-2.5 font-medium">용도</th>
                </tr>
              </thead>
              <tbody className="text-mut">
                <tr className="border-b border-line/60">
                  <td className="px-3.5 py-2.5 text-fg">embedder가 좋아졌나?</td>
                  <td className="px-3.5 py-2.5">이 lab의 dense-only 평가</td>
                  <td className="px-3.5 py-2.5">빠르고 귀속 가능 — 반복 · 튜닝</td>
                </tr>
                <tr>
                  <td className="px-3.5 py-2.5 text-fg">검색 전체가 좋아졌나?</td>
                  <td className="px-3.5 py-2.5">서빙의 hybrid + rerank A/B</td>
                  <td className="px-3.5 py-2.5">느림 — 최종 출시 판정</td>
                </tr>
              </tbody>
            </table>
          </div>
          <p className="text-[12.5px] leading-relaxed text-faint">
            reranker가 뒤에 있다면 embedder의 임무는 <b className="text-mut">recall</b>(정답을 후보 풀에 넣기)입니다 —{" "}
            <span className="mono">recall@(rerank 후보 깊이)</span>를 보세요. 최종 정렬은 reranker가 합니다.
          </p>
        </Panel>
      </Section>

      {/* ── Meaning ───────────────────────────────────────────────────── */}
      <Section delay={180}>
        <SectionLabel hint="이 lab의 숫자가 프로덕션과 이어지는 지점">이게 어떤 의미인가</SectionLabel>
        <Panel className="space-y-3.5 p-5">
          <div className="flex items-start gap-3">
            <Scale size={18} className="mt-0.5 shrink-0 text-signal" />
            <p className="text-[13px] leading-relaxed text-mut">
              도메인 fine-tune은 <b className="text-fg">first-stage dense retriever</b>를 개선합니다. 그 model을 서빙(예:
              Elasticsearch + BM25 hybrid + reranker)에 꽂으면 first-stage <b className="text-fg">recall</b>이 올라가고,
              그게 전체 검색 품질로 이어집니다.
            </p>
          </div>
          <Note tone="amber" title="단, 서빙 formatting parity가 전제입니다">
            서빙 파이프라인이 이 lab과 <b>똑같은 방식</b>으로 임베딩하지 않으면(특히 query의 Instruct prefix) lab 점수가
            프로덕션으로 전이되지 않습니다 — fine-tune이 "lab에선 좋은데 프로덕션에선 그대로"인 #1 원인. 자세히는{" "}
            <span className="mono">docs/serving-parity.md</span>.
          </Note>
          <p className="text-[12.5px] leading-relaxed text-faint">
            정리: lab의 향상은 <b className="text-mut">필요조건</b>(여기서 안 오르면 프로덕션도 기대 못 함)이지만{" "}
            <b className="text-mut">충분조건은 아닙니다</b>. 실제 도메인 eval set으로 재고, 최종 확신은 서빙 A/B로.
          </p>
          <div className="flex flex-wrap gap-2.5 pt-1">
            <Btn variant="ghost" icon={<FlaskConical size={15} />} onClick={() => nav(PATH.train)}>
              학습 화면
            </Btn>
            <Btn variant="ghost" icon={<BarChart3 size={15} />} onClick={() => nav(PATH.compare)}>
              실험 비교
            </Btn>
            <span className="inline-flex items-center gap-1.5 self-center text-[12px] text-faint">
              더 자세히 <ArrowRight size={13} /> <span className="mono">docs/evaluation.md</span>
            </span>
          </div>
        </Panel>
      </Section>
    </div>
  );
}
