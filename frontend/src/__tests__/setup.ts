/**
 * Vitest setup file for React Testing Library.
 *
 * Extends Vitest's expect with DOM-specific matchers from
 * @testing-library/jest-dom (e.g. toBeInTheDocument, toHaveTextContent).
 *
 * Also polyfills browser APIs missing from jsdom (ResizeObserver)
 * that third-party libraries like Recharts depend on.
 */
import '@testing-library/jest-dom/vitest';

// Recharts' ResponsiveContainer requires ResizeObserver which jsdom lacks.
globalThis.ResizeObserver = class ResizeObserver {
  observe() {}
  unobserve() {}
  disconnect() {}
};
