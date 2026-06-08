import type { ReactNode } from "react";
import * as Dialog from "@radix-ui/react-dialog";
import { X } from "lucide-react";

/** Centered modal on Radix Dialog — focus-trapped, scroll-locked, closes on ESC or
 * backdrop click. Same API as before so callers don't change. */
export function Modal({
  open,
  onClose,
  title,
  children,
}: {
  open: boolean;
  onClose: () => void;
  title: string;
  children: ReactNode;
}) {
  return (
    <Dialog.Root open={open} onOpenChange={(o) => !o && onClose()}>
      <Dialog.Portal>
        <Dialog.Overlay className="dlg-overlay fixed inset-0 z-50 bg-ink-950/75 backdrop-blur-sm" />
        <Dialog.Content
          aria-describedby={undefined}
          className="dlg-content fixed left-1/2 top-[8vh] z-50 w-[92vw] max-w-4xl -translate-x-1/2 overflow-hidden rounded-2xl border border-line2 bg-ink-900 shadow-[0_30px_80px_-20px_rgba(0,0,0,0.8)] focus:outline-none"
        >
          <div className="flex items-center justify-between border-b border-line px-5 py-3.5">
            <Dialog.Title className="text-[15px] font-semibold text-fg">{title}</Dialog.Title>
            <Dialog.Close className="grid h-8 w-8 place-items-center rounded-lg text-mut transition-colors hover:bg-ink-800 hover:text-fg">
              <X size={17} />
            </Dialog.Close>
          </div>
          <div className="max-h-[70vh] overflow-auto p-5">{children}</div>
        </Dialog.Content>
      </Dialog.Portal>
    </Dialog.Root>
  );
}
