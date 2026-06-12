import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";

// /api is proxied to the FastAPI lab backend (wired in a later phase).
export default defineConfig({
  plugins: [react(), tailwindcss()],
  server: {
    port: Number(process.env.PORT) || 5273, // PORT lets preview tooling pick a free port
    proxy: { "/api": { target: "http://127.0.0.1:8800", changeOrigin: true } },
  },
});
