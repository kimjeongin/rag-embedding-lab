import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { createBrowserRouter, RouterProvider } from "react-router-dom";
import { Toaster } from "sonner";

import "./index.css";
import Layout from "./routes/Layout";
import Overview from "./routes/Overview";
import Data from "./routes/Data";
import Train from "./routes/Train";
import Eval from "./routes/Eval";
import Compare from "./routes/Compare";
import Models from "./routes/Models";
import Search from "./routes/Search";
import Report from "./routes/Report";
import About from "./routes/About";

const queryClient = new QueryClient({
  defaultOptions: { queries: { staleTime: 10_000, retry: 1, refetchOnWindowFocus: false } },
});

const router = createBrowserRouter([
  {
    path: "/",
    element: <Layout />,
    children: [
      { index: true, element: <Overview /> },
      { path: "data", element: <Data /> },
      { path: "train", element: <Train /> },
      { path: "eval", element: <Eval /> },
      { path: "compare", element: <Compare /> },
      { path: "models", element: <Models /> },
      { path: "search", element: <Search /> },
      { path: "report", element: <Report /> },
      { path: "about", element: <About /> },
    ],
  },
]);

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <QueryClientProvider client={queryClient}>
      <RouterProvider router={router} />
      <Toaster
        theme="dark"
        position="bottom-right"
        toastOptions={{
          style: { background: "#15191f", border: "1px solid #ffffff1f", color: "#e9ecef" },
          className: "mono",
        }}
      />
    </QueryClientProvider>
  </StrictMode>,
);
