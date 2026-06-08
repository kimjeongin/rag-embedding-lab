// Navigation model — the steps, their routes, copy, and icons. Single source for the
// sidebar, the header title, and the router.
import { BarChart3, Database, FlaskConical, Gauge, LayoutDashboard } from "lucide-react";
import type { ComponentType } from "react";

export type Step = "overview" | "data" | "train" | "eval" | "compare";

/** URL path per step (overview is the index route). */
export const PATH: Record<Step, string> = {
  overview: "/",
  data: "/data",
  train: "/train",
  eval: "/eval",
  compare: "/compare",
};

/** Which step a URL path belongs to (for the header title). */
export const stepFromPath = (pathname: string): Step => {
  const seg = pathname.split("/")[1];
  return (["data", "train", "eval", "compare"] as const).find((s) => s === seg) ?? "overview";
};

export const META: Record<Step, { title: string; sub: string }> = {
  overview: { title: "개요", sub: "임베딩 모델 실험을 한눈에." },
  data: { title: "데이터", sub: "학습 데이터와 평가셋을 만들고 검수합니다." },
  train: { title: "학습", sub: "base 모델을 내 데이터로 fine-tuning 합니다." },
  eval: { title: "평가", sub: "모델의 검색 정확도를 측정합니다." },
  compare: { title: "실험", sub: "평가한 모델들을 한눈에 비교합니다." },
};

export const STEP_ICON: Record<Step, ComponentType<{ size?: number; className?: string }>> = {
  overview: LayoutDashboard,
  data: Database,
  train: FlaskConical,
  eval: Gauge,
  compare: BarChart3,
};

export const NAV_GROUPS: { label?: string; items: { id: Step; title: string; sub: string }[] }[] = [
  { items: [{ id: "overview", title: "개요", sub: "한눈에" }] },
  {
    label: "파이프라인",
    items: [
      { id: "data", title: "데이터", sub: "생성 · 검수" },
      { id: "train", title: "학습", sub: "fine-tune" },
      { id: "eval", title: "평가", sub: "recall · nDCG" },
    ],
  },
  { label: "분석", items: [{ id: "compare", title: "실험", sub: "모델 비교" }] },
];
