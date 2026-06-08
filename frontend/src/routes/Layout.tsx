import { useEffect, useState } from "react";
import { Outlet } from "react-router-dom";

import Header from "../components/Header";
import Sidebar from "../components/Sidebar";
import { CommandPalette } from "../components/CommandPalette";

export default function Layout() {
  const [paletteOpen, setPaletteOpen] = useState(false);

  // ⌘K / Ctrl-K toggles the command palette anywhere in the app.
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key.toLowerCase() === "k" && (e.metaKey || e.ctrlKey)) {
        e.preventDefault();
        setPaletteOpen((o) => !o);
      }
    };
    document.addEventListener("keydown", onKey);
    return () => document.removeEventListener("keydown", onKey);
  }, []);

  return (
    <div className="atmos relative min-h-screen">
      <div className="relative z-10 flex min-h-screen">
        <Sidebar onSearch={() => setPaletteOpen(true)} />
        <main className="min-w-0 flex-1">
          <Header />
          <div className="mx-auto max-w-[1180px] px-8 py-8">
            <Outlet />
          </div>
        </main>
      </div>
      <CommandPalette open={paletteOpen} onOpenChange={setPaletteOpen} />
    </div>
  );
}
