import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import tailwindcss from '@tailwindcss/vite';

export default defineConfig({
  plugins: [react(), tailwindcss()],
  server: {
    proxy: {
      '/api': 'http://localhost:8000',
    },
  },
  // Load env from the repo root (backend's .env is the single source of
  // truth).  Vite otherwise reads only frontend/.env — the root .env holds
  // every config (VITE_EMA_API_KEY for the API-key guard), and only
  // VITE_-prefixed vars are exposed to client code, so backend-only keys
  // never reach the bundle.
  envDir: '..',
});
