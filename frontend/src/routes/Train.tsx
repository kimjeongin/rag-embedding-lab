import { useEffect, useRef, useState } from "react";
import { useNavigate } from "react-router-dom";
import { ArrowRight, Play, Square } from "lucide-react";

import { fmt } from "../lib/format";
import { PATH } from "../lib/nav";
import { useStatus } from "../lib/queries";
import { startTraining, stopTraining, useTrainState } from "../lib/trainStore";
import { LossCurve } from "../components/charts";
import { Btn, ErrorNote, Field, Input, Panel, Section, SectionLabel, Stat } from "../components/ui";

export default function Train() {
  const nav = useNavigate();
  const status = useStatus();
  const train = useTrainState();

  const [base, setBase] = useState("Qwen/Qwen3-Embedding-0.6B");
  const [out, setOut] = useState("outputs/embedding-ft");
  const [epochs, setEpochs] = useState(1);
  const [batch, setBatch] = useState(16);
  const [lr, setLr] = useState("2e-5");
  const [device, setDevice] = useState("");
  const [confirmStop, setConfirmStop] = useState(false);

  const logRef = useRef<HTMLDivElement>(null);
  useEffect(() => {
    if (logRef.current) logRef.current.scrollTop = logRef.current.scrollHeight;
  }, [train.log.length]);

  // arm the stop button for 3s, then relax back to the plain button
  useEffect(() => {
    if (!confirmStop) return;
    const t = setTimeout(() => setConfirmStop(false), 3000);
    return () => clearTimeout(t);
  }, [confirmStop]);

  const ready = status.data?.training_ready ?? true;
  const running = train.status === "running";
  const finished = train.status === "done" && train.exitCode === 0 && train.outputDir;

  const submit = () =>
    startTraining({
      base_model: base.trim(),
      output_dir: out.trim(),
      epochs,
      batch_size: batch,
      learning_rate: parseFloat(lr) || 2e-5,
      device: device.trim(),
    });

  const lossValues = train.loss.map((p) => p.loss);
  const lossLabel =
    lossValues.length > 1 ? `${lossValues[0].toFixed(3)} → ${lossValues[lossValues.length - 1].toFixed(3)}` : "—";
  const lastLoss = train.loss[train.loss.length - 1];

  return (
    <div className="space-y-9">
      <Section>
        <SectionLabel hint="base 모델 → 내 학습 데이터로 fine-tune">학습 설정</SectionLabel>
        <Panel className="p-5">
          {!ready && (
            <div className="mb-4 rounded-xl border border-amber/30 bg-amber/10 px-4 py-3 text-[12.5px] text-amber">
              ⚠️ 학습 라이브러리(torch 등)가 설치되어 있지 않습니다 —{" "}
              <code className="mono">uv sync --group training</code> 후 사용하세요. 그래도 시작하면 자세한 오류가 로그에 표시됩니다.
            </div>
          )}
          <div className="grid gap-4 sm:grid-cols-2">
            <Field label="base 모델" hint="HuggingFace">
              <Input value={base} onChange={(e) => setBase(e.target.value)} disabled={running} />
            </Field>
            <Field label="저장 폴더">
              <Input value={out} onChange={(e) => setOut(e.target.value)} disabled={running} />
            </Field>
          </div>
          <div className="mt-4 grid grid-cols-2 gap-4 sm:grid-cols-4">
            <Field label="epochs">
              <Input type="number" min={1} value={epochs} onChange={(e) => setEpochs(+e.target.value)} className="mono" disabled={running} />
            </Field>
            <Field label="batch size">
              <Input type="number" min={1} value={batch} onChange={(e) => setBatch(+e.target.value)} className="mono" disabled={running} />
            </Field>
            <Field label="learning rate">
              <Input value={lr} onChange={(e) => setLr(e.target.value)} className="mono" disabled={running} />
            </Field>
            <Field label="device" hint="빈칸=auto">
              <Input value={device} onChange={(e) => setDevice(e.target.value)} placeholder="auto" className="mono" disabled={running} />
            </Field>
          </div>
          <div className="mt-5">
            {running ? (
              confirmStop ? (
                <Btn
                  variant="ghost"
                  icon={<Square size={14} />}
                  className="border-danger/40 text-danger"
                  onClick={() => {
                    setConfirmStop(false);
                    stopTraining();
                  }}
                >
                  정말 중단할까요? — 진행분은 저장되지 않습니다
                </Btn>
              ) : (
                <Btn variant="ghost" icon={<Square size={14} />} onClick={() => setConfirmStop(true)}>
                  중단
                </Btn>
              )
            ) : (
              <Btn icon={<Play size={15} />} onClick={submit}>
                학습 시작
              </Btn>
            )}
          </div>
        </Panel>
      </Section>

      {finished && (
        <Section>
          <Panel className="flex flex-wrap items-center justify-between gap-3 border-signal/25 bg-signal/[0.06] p-5">
            <div>
              <div className="text-[14px] font-semibold text-fg">학습 완료</div>
              <div className="mono mt-1 text-[12px] text-mut">
                새 모델: {train.outputDir} — 평가셋으로 실측해야 진짜 점수를 알 수 있어요
              </div>
            </div>
            <Btn
              icon={<ArrowRight size={15} />}
              onClick={() =>
                nav(PATH.eval, { state: { backend: "sentence-transformers", model: train.outputDir } })
              }
            >
              이 모델 평가하기
            </Btn>
          </Panel>
        </Section>
      )}

      {(train.status !== "idle" || train.log.length > 0) && (
        <Section delay={70}>
          <SectionLabel
            hint={
              running
                ? `실시간 스트리밍 중…${lastLoss ? ` · epoch ${lastLoss.epoch} · step ${lastLoss.step}` : ""}`
                : "loss가 내려가면 정상"
            }
          >
            진행 상황
          </SectionLabel>
          <div className="grid gap-5 lg:grid-cols-[1.6fr_1fr]">
            <Panel className="p-5">
              <div className="mb-3 flex items-center justify-between">
                <span className="text-[13px] font-medium text-mut">training loss</span>
                <span className="mono text-[12px] text-signal2">{lossLabel}</span>
              </div>
              <LossCurve points={lossValues} />
            </Panel>
            <div className="grid grid-rows-2 gap-3">
              <Stat
                label="검증쌍 nDCG@10 · 학습 전"
                value={train.metrics.before != null ? fmt(train.metrics.before) : "—"}
                tone="cyan"
                sub="held-out 학습쌍"
              />
              <Stat
                label="검증쌍 nDCG@10 · 학습 후"
                value={train.metrics.after != null ? fmt(train.metrics.after) : "—"}
                tag={train.metrics.after != null ? "완료" : undefined}
                tone="signal"
                sub="실측은 평가 화면에서"
              />
            </div>
          </div>

          <div className="mt-5">
            <div className="mb-2 flex items-center gap-2 text-[11px] uppercase tracking-wider text-faint">
              로그
              {train.status === "done" && <span className="mono text-signal2">· exit {train.exitCode}</span>}
              {train.status === "stopped" && <span className="mono text-amber">· 중단됨</span>}
              {train.status === "error" && <span className="mono text-danger">· 오류</span>}
            </div>
            <div
              ref={logRef}
              className="mono max-h-72 overflow-auto whitespace-pre-wrap rounded-xl border border-line bg-ink-950 p-4 text-[11.5px] leading-relaxed text-mut"
            >
              {train.log.length ? train.log.join("\n") : "대기 중…"}
            </div>
            {train.error && <div className="mt-3"><ErrorNote>{train.error}</ErrorNote></div>}
          </div>
        </Section>
      )}
    </div>
  );
}
