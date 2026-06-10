import { NavLink } from "react-router-dom";
import { Boxes, Search } from "lucide-react";

import { cx } from "../lib/format";
import { NAV_GROUPS, PATH, STEP_ICON } from "../lib/nav";
import { useTrainStatus } from "../lib/trainStore";
import { Kbd } from "./ui";

/** Pulsing dot on the 학습 nav item while a fine-tune is streaming — the run keeps
 * going when you leave the screen, so the app must say so somewhere global. */
function TrainingDot() {
  return (
    <span className="relative flex h-2 w-2 shrink-0" title="학습 진행 중">
      <span className="absolute inline-flex h-full w-full animate-ping rounded-full bg-signal opacity-60" />
      <span className="relative inline-flex h-2 w-2 rounded-full bg-signal" />
    </span>
  );
}

export default function Sidebar({ onSearch }: { onSearch?: () => void }) {
  const training = useTrainStatus() === "running";
  return (
    <aside className="sticky top-0 z-30 flex h-screen w-[244px] shrink-0 flex-col border-r border-line bg-ink-925/80 backdrop-blur-xl">
      <div className="flex items-center gap-2.5 px-6 pb-4 pt-6">
        <div className="grid h-9 w-9 place-items-center rounded-xl bg-signal text-ink-950 shadow-[0_0_26px_-6px_rgba(198,242,74,0.7)]">
          <Boxes size={18} strokeWidth={2.4} />
        </div>
        <div className="leading-tight">
          <div className="mono text-[13.5px] font-semibold tracking-tight text-fg">
            RAG<span className="text-signal">·</span>LAB
          </div>
          <div className="text-[10.5px] text-faint">embedding studio</div>
        </div>
      </div>

      <button
        onClick={onSearch}
        className="mx-3 mb-2 flex items-center gap-2 rounded-xl border border-line bg-ink-900 px-3 py-2 text-[13px] text-faint transition-colors hover:border-line2 hover:text-mut"
      >
        <Search size={14} /> 검색
        <span className="ml-auto">
          <Kbd>⌘K</Kbd>
        </span>
      </button>

      <nav className="flex-1 overflow-y-auto px-3 py-1">
        {NAV_GROUPS.map((grp, gi) => (
          <div key={gi} className="mb-1">
            {grp.label && (
              <div className="px-3 pb-1 pt-3 text-[10px] font-semibold uppercase tracking-[0.16em] text-faint">
                {grp.label}
              </div>
            )}
            {grp.items.map((item) => {
              const Icon = STEP_ICON[item.id];
              return (
                <NavLink
                  key={item.id}
                  to={PATH[item.id]}
                  end={item.id === "overview"}
                  className={({ isActive }) =>
                    cx(
                      "group relative flex w-full items-center gap-3 rounded-xl px-3 py-2.5 text-left transition-colors",
                      isActive ? "bg-ink-800/70" : "hover:bg-ink-880/60",
                    )
                  }
                >
                  {({ isActive }) => (
                    <>
                      {isActive && (
                        <span className="absolute left-0 top-1/2 h-5 w-[3px] -translate-y-1/2 rounded-r-full bg-signal shadow-[0_0_8px] shadow-signal" />
                      )}
                      <Icon size={16} className={isActive ? "text-signal" : "text-faint group-hover:text-mut"} />
                      <span className="min-w-0 flex-1">
                        <span
                          className={cx(
                            "block text-[13.5px] font-medium",
                            isActive ? "text-fg" : "text-mut group-hover:text-fg",
                          )}
                        >
                          {item.title}
                        </span>
                        <span className="block text-[10.5px] text-faint">{item.sub}</span>
                      </span>
                      {item.id === "train" && training && <TrainingDot />}
                    </>
                  )}
                </NavLink>
              );
            })}
          </div>
        ))}
      </nav>

      <div className="mono border-t border-line px-6 py-4 text-[11px] text-faint">v0.4 · localhost</div>
    </aside>
  );
}
