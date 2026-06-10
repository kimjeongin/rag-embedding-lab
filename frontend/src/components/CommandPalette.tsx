import * as Dialog from "@radix-ui/react-dialog";
import { Command } from "cmdk";
import { useNavigate } from "react-router-dom";
import { BarChart3, BookOpen, Database, FlaskConical, Gauge, LayoutDashboard, Search } from "lucide-react";

import { PATH } from "../lib/nav";

const ITEMS = [
  { icon: LayoutDashboard, label: "개요", hint: "리더보드 · 최고 모델", to: PATH.overview },
  { icon: Database, label: "데이터", hint: "생성 · 검수", to: PATH.data },
  { icon: FlaskConical, label: "학습", hint: "fine-tune", to: PATH.train },
  { icon: Gauge, label: "평가", hint: "recall · nDCG", to: PATH.eval },
  { icon: BarChart3, label: "실험", hint: "모델 비교", to: PATH.compare },
  { icon: BookOpen, label: "소개", hint: "프로젝트 안내 · 동작 원리", to: PATH.about },
];

/** ⌘K palette — jump between screens. Radix Dialog (focus trap, ESC) + cmdk (fuzzy match). */
export function CommandPalette({ open, onOpenChange }: { open: boolean; onOpenChange: (o: boolean) => void }) {
  const nav = useNavigate();
  const go = (to: string) => {
    onOpenChange(false);
    nav(to);
  };

  return (
    <Dialog.Root open={open} onOpenChange={onOpenChange}>
      <Dialog.Portal>
        <Dialog.Overlay className="dlg-overlay fixed inset-0 z-50 bg-ink-950/70 backdrop-blur-sm" />
        <Dialog.Content
          className="dlg-content fixed left-1/2 top-[16%] z-50 w-[92vw] max-w-lg -translate-x-1/2 overflow-hidden rounded-2xl border border-line2 bg-ink-900 shadow-[0_30px_80px_-20px_rgba(0,0,0,0.85)]"
          aria-describedby={undefined}
        >
          <Dialog.Title className="sr-only">명령 팔레트</Dialog.Title>
          <Command className="text-fg">
            <div className="flex items-center gap-2.5 border-b border-line px-4">
              <Search size={15} className="shrink-0 text-faint" />
              <Command.Input
                autoFocus
                placeholder="어디로 갈까요? 화면 이름을 입력하세요…"
                className="w-full bg-transparent py-3.5 text-[14px] text-fg outline-none placeholder:text-faint"
              />
            </div>
            <Command.List className="max-h-[320px] overflow-auto p-2">
              <Command.Empty className="py-8 text-center text-[13px] text-faint">결과가 없어요</Command.Empty>
              <Command.Group heading="이동">
                {ITEMS.map((it) => {
                  const Icon = it.icon;
                  return (
                    <Command.Item
                      key={it.to}
                      value={it.label}
                      onSelect={() => go(it.to)}
                      className="flex cursor-pointer items-center gap-3 rounded-lg px-3 py-2.5 text-[13.5px] text-mut data-[selected=true]:bg-ink-800/70 data-[selected=true]:text-fg"
                    >
                      <Icon size={16} className="text-faint" />
                      <span className="flex-1">{it.label}</span>
                      <span className="text-[11px] text-faint">{it.hint}</span>
                    </Command.Item>
                  );
                })}
              </Command.Group>
            </Command.List>
          </Command>
        </Dialog.Content>
      </Dialog.Portal>
    </Dialog.Root>
  );
}
