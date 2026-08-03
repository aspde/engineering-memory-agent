import { defineConfig } from 'vitest/config';
import { fileURLToPath } from 'node:url';
import react from '@vitejs/plugin-react';
import tailwindcss from '@tailwindcss/vite';

export default defineConfig({
  plugins: [react(), tailwindcss()],
  resolve: {
    // Force react / react-dom to a single instance. The app deps live in
    // frontend/node_modules while the test deps (@testing-library/react) are
    // hoisted to the repo root — without this, React hooks fail to render
    // under Vitest because the renderer and components use different copies.
    alias: [
      {
        find: 'react-dom/client',
        replacement: fileURLToPath(new URL('./node_modules/react-dom/client.js', import.meta.url)),
      },
      { find: 'react', replacement: fileURLToPath(new URL('./node_modules/react', import.meta.url)) },
      {
        find: 'react-dom',
        replacement: fileURLToPath(new URL('./node_modules/react-dom', import.meta.url)),
      },
    ],
  },
  test: {
    environment: 'jsdom',
    include: ['src/**/*.test.{ts,tsx}'],
    setupFiles: ['./src/test-setup.ts'],
    globals: true,
    css: false,
  },
});
