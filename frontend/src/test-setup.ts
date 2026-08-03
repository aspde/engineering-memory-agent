import { vi } from 'vitest';
import '@testing-library/jest-dom/vitest';

// Mock scrollIntoView — not implemented in jsdom
if (!Element.prototype.scrollIntoView) {
  Element.prototype.scrollIntoView = vi.fn();
}

// Mock crypto.randomUUID — used by AppContext initial state
if (!globalThis.crypto?.randomUUID) {
  Object.defineProperty(globalThis, 'crypto', {
    value: {
      randomUUID: () => '00000000-0000-0000-0000-000000000000',
    },
    writable: true,
  });
}
