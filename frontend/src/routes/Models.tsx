// The saved-model shelf — every trained model with its recipe, size and scores, plus
// the two lifecycle actions: delete the losers (~1GB each) and hand off the winner.
import { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import { ClipboardCopy, Gauge, HardDrive, PackageCheck, Trash2 } from "lucide-react";
import { toast } from "sonner";

import { fmt, short } from "../lib/format";
import { PATH } from "../lib/nav";
import { useDeleteModel, useHandoff, useModelsDetail, useStatus } from "../lib/queries";
import type { ModelDetail } from "../lib/types";
import { Modal } from "../components/Modal";
import { Btn, ErrorNote, Info, Loading, Panel, Section, SectionLabel, Stat, Tag } from "../components/ui";

const gb = (bytes: number) => `${(bytes / 1e9).toFixed(2)}GB`;

function recipeLine(m: ModelDetail): string {
  const meta = m.meta as Record<string, unknown> | null | undefined;
  if (!meta) return "레시피 기록 없음 (예전 모델)";
  const parts: string[] = [];
  if (meta.loss) parts.push(String(meta.loss));
  if (meta.method) parts.push(meta.method === "lora" ? `lora r${meta.lora_r}` : "full");
  if (meta.learning_rate != null) parts.push(`lr ${meta.learning_rate}`);
  if (meta.saved_epoch != null) parts.push(`e${meta.saved_epoch}/${meta.epochs_ran}${meta.early_stopped ? " (early stop)" : ""}`);
  if (meta.train_data_fingerprint) parts.push(`data ${String(meta.train_data_fingerprint).slice(0, 6)}`);
  return parts.join(" · ");
}

function DeleteModelBtn({ path }: { path: string }) {
  const del = useDeleteModel();
  const [armed, setArmed] = useState(false);
  useEffect(() => {
    if (!armed) return;
    const t = setTimeout(() => setArmed(false), 3000);
    return () => clearTimeout(t);
  }, [armed]);
  if (armed) {
    return (
      <button
        onClick={() => del.mutate(path)}
        disabled={del.isPending}
        className="mono rounded-md bg-danger/15 px-2 py-1 text-[11px] font-semibold text-danger hover:bg-danger/25 disabled:opacity-40"
      >
        폴더 삭제?
      </button>
    );
  }
  return (
    <button onClick={() => setArmed(true)} title="모델 폴더 삭제 (평가 기록은 남음)" className="text-faint transition-colors hover:text-danger">
      <Trash2 size={15} />
    </button>
  );
}

export default function Models() {
  const nav = useNavigate();
  const { data, isLoading, error } = useModelsDetail();
  const status = useStatus();
  const handoff = useHandoff();
  const [handoffMd, setHandoffMd] = useState<{ path: string; markdown: string } | null>(null);

  if (isLoading) return <Loading label="모델 목록을 불러오는 중…" />;
  if (error) return <ErrorNote>{(error as Error).message}</ErrorNote>;

  const models = data?.models ?? [];
  const delivered = status.data?.handed_off?.model;

  return (
    <div className="space-y-9">
      <Section>
        <SectionLabel hint="런당 약 1GB — 지면 지우고, 이기면 납품">보관함</SectionLabel>
        <div className="grid grid-cols-2 gap-3 lg:grid-cols-4">
          <Stat label="저장된 모델" value={models.length} tone="signal" sub="outputs/" />
          <Stat label="디스크 사용량" value={gb(data?.disk_total_bytes ?? 0)} tone="cyan" sub="모델 폴더 합계" />
          <Stat
            label="최종 확정(final) 측정"
            value={models.filter((m) => m.eval_final).length}
            tone="signal"
            sub="held-out 확인 완료"
          />
          <Stat label="납품됨" value={delivered ? short(delivered) : "—"} tone="cyan" sub={status.data?.handed_off?.at ?? "핸드오프 전"} />
        </div>
      </Section>

      <Section delay={70}>
        <SectionLabel
          hint="핸드오프 = 서빙팀에 넘길 패키지 (계약·샘플 벡터·체크리스트)"
        >
          <span className="inline-flex items-center gap-1.5">
            모델 목록
            <Info title="납품(핸드오프)이란" align="left">
              랩의 결승선입니다. 프로덕션은 <b className="text-fg">하이브리드(BM25+dense)+리랭커</b> 파이프라인에서{" "}
              <b className="text-fg">dense 모델만 교체</b>하므로, 서빙팀에 필요한 건 모델 가중치 +{" "}
              <b className="text-fg">임베딩 계약</b>(instruction/문서 템플릿·pooling·normalize) +{" "}
              <b className="text-fg">패리티 검증용 샘플 벡터</b>(cosine ≥ 0.999) + 재색인 체크리스트. 버튼 한 번에
              모델 폴더 안 <span className="mono">HANDOFF.md / handoff.json</span>으로 만들어 드립니다. 그리고
              핸드오프는 곧 “이 모델이 라이브로 간다”는 결정이므로, <b className="text-fg">그 모델로 서빙 인덱스
              재색인이 자동 시작</b>됩니다 — 진행률은 검색 탭에서.
            </Info>
          </span>
        </SectionLabel>
        {models.length === 0 ? (
          <Panel className="p-10 text-center">
            <HardDrive size={26} className="mx-auto text-faint" />
            <p className="mt-4 text-[13px] text-mut">저장된 모델이 없습니다 — 학습 탭에서 첫 모델을 만들어 보세요.</p>
          </Panel>
        ) : (
          <Panel className="overflow-x-auto">
            <table className="w-full text-left text-[13px]">
              <thead>
                <tr className="border-b border-line bg-ink-880/60 text-[11px] uppercase tracking-wider text-faint">
                  <th className="px-4 py-3 font-medium">모델</th>
                  <th className="mono px-3 py-3 text-right font-medium normal-case">dev nDCG@10</th>
                  <th className="mono px-3 py-3 text-right font-medium normal-case">recall@50</th>
                  <th className="mono px-3 py-3 text-right font-medium normal-case">final</th>
                  <th className="mono px-3 py-3 text-right font-medium normal-case">dim</th>
                  <th className="mono px-3 py-3 text-right font-medium normal-case">size</th>
                  <th className="px-3 py-3" />
                </tr>
              </thead>
              <tbody>
                {models.map((m) => (
                  <tr key={m.path} className="border-b border-line/60 last:border-0 hover:bg-ink-880/40">
                    <td className="px-4 py-3">
                      <div className="flex flex-wrap items-center gap-1.5 font-medium text-fg">
                        <span className="mono text-[12.5px]">{short(m.path)}</span>
                        {m.handed_off && <Tag tone="signal">납품됨</Tag>}
                        {delivered === m.path && <Tag tone="cyan">현재 납품본</Tag>}
                      </div>
                      <div className="mono mt-0.5 text-[11px] text-faint">{recipeLine(m)}</div>
                      {(m.meta as Record<string, unknown> | null)?.note != null && (
                        <div className="truncate text-[11px] italic text-faint">
                          “{String((m.meta as Record<string, unknown>).note)}”
                        </div>
                      )}
                    </td>
                    <td className="mono px-3 py-3 text-right text-mut">
                      {m.eval_dev ? fmt(m.eval_dev.metrics["ndcg@10"] ?? 0) : "—"}
                    </td>
                    <td className="mono px-3 py-3 text-right text-mut">
                      {m.eval_dev?.metrics["recall@50"] != null ? fmt(m.eval_dev.metrics["recall@50"]) : "—"}
                    </td>
                    <td className="mono px-3 py-3 text-right">
                      {m.eval_final ? (
                        <span className="text-signal">{fmt(m.eval_final.metrics["ndcg@10"] ?? 0)} ✓</span>
                      ) : (
                        <span className="text-faint">—</span>
                      )}
                    </td>
                    <td className="mono px-3 py-3 text-right text-mut">{m.dim ?? "—"}</td>
                    <td className="mono px-3 py-3 text-right text-mut">{gb(m.size_bytes)}</td>
                    <td className="px-3 py-3">
                      <div className="flex items-center justify-end gap-2.5">
                        <button
                          title="이 모델 평가"
                          onClick={() => nav(PATH.eval, { state: { backend: "sentence-transformers", model: m.path } })}
                          className="text-faint transition-colors hover:text-cyan"
                        >
                          <Gauge size={15} />
                        </button>
                        <button
                          title="핸드오프 패키지 생성 (HANDOFF.md)"
                          disabled={handoff.isPending}
                          onClick={() =>
                            handoff.mutate(m.path, {
                              onSuccess: (res) => setHandoffMd({ path: res.path, markdown: res.markdown }),
                            })
                          }
                          className="text-faint transition-colors hover:text-signal disabled:opacity-40"
                        >
                          <PackageCheck size={15} />
                        </button>
                        <DeleteModelBtn path={m.path} />
                      </div>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </Panel>
        )}
        {handoff.isPending && <p className="mt-2 text-[12px] text-faint">패키지 생성 중 — 샘플 벡터 임베딩과 속도 측정에 잠시 걸립니다…</p>}
      </Section>

      <Modal open={!!handoffMd} onClose={() => setHandoffMd(null)} title={`핸드오프 — ${handoffMd ? short(handoffMd.path) : ""}`}>
        {handoffMd && (
          <>
            <div className="mb-3 flex items-center justify-between gap-3">
              <p className="text-[12.5px] text-mut">
                모델 폴더에 <span className="mono">HANDOFF.md</span>와 <span className="mono">handoff.json</span>
                (검증용 샘플 벡터 포함)이 저장됐어요 — 모델 폴더째 서빙팀에 전달하면 됩니다.
              </p>
              <Btn
                variant="ghost"
                icon={<ClipboardCopy size={14} />}
                onClick={() => {
                  navigator.clipboard.writeText(handoffMd.markdown);
                  toast.success("HANDOFF.md를 복사했어요");
                }}
              >
                복사
              </Btn>
            </div>
            <pre className="mono whitespace-pre-wrap rounded-xl border border-line bg-ink-950 p-4 text-[11.5px] leading-relaxed text-mut">
              {handoffMd.markdown}
            </pre>
          </>
        )}
      </Modal>
    </div>
  );
}
