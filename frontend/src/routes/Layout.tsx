import { Outlet } from "react-router-dom";

import Header from "../components/Header";
import Sidebar from "../components/Sidebar";

export default function Layout() {
  return (
    <div className="atmos relative min-h-screen">
      <div className="relative z-10 flex min-h-screen">
        <Sidebar />
        <main className="min-w-0 flex-1">
          <Header />
          <div className="mx-auto max-w-[1180px] px-8 py-8">
            <Outlet />
          </div>
        </main>
      </div>
    </div>
  );
}
