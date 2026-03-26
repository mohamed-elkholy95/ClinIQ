/**
 * Tests for the Layout component.
 *
 * Verifies header rendering, dark mode toggle, notification bell,
 * user avatar, and sidebar trigger button.  Uses MemoryRouter with
 * a simple child route to exercise the Outlet.
 */
import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, fireEvent } from '@testing-library/react';
import { MemoryRouter, Route, Routes } from 'react-router-dom';
import { Layout } from '../../components/Layout';

// Stub localStorage for theme persistence
const localStorageMock = (() => {
  let store: Record<string, string> = {};
  return {
    getItem: (key: string) => store[key] ?? null,
    setItem: (key: string, value: string) => { store[key] = value; },
    removeItem: (key: string) => { delete store[key]; },
    clear: () => { store = {}; },
  };
})();

Object.defineProperty(window, 'localStorage', { value: localStorageMock });

// Mock matchMedia for prefers-color-scheme
Object.defineProperty(window, 'matchMedia', {
  writable: true,
  value: vi.fn().mockImplementation((query: string) => ({
    matches: false,
    media: query,
    onchange: null,
    addListener: vi.fn(),
    removeListener: vi.fn(),
    addEventListener: vi.fn(),
    removeEventListener: vi.fn(),
    dispatchEvent: vi.fn(),
  })),
});

function renderLayout() {
  return render(
    <MemoryRouter initialEntries={['/']}>
      <Routes>
        <Route element={<Layout />}>
          <Route path="/" element={<div data-testid="child-page">Page Content</div>} />
        </Route>
      </Routes>
    </MemoryRouter>
  );
}

beforeEach(() => {
  localStorageMock.clear();
  document.documentElement.classList.remove('dark');
});

describe('Layout — header', () => {
  it('renders the platform subtitle', () => {
    renderLayout();
    expect(screen.getByText('Clinical NLP Platform')).toBeInTheDocument();
  });

  it('renders the user avatar with initials', () => {
    renderLayout();
    expect(screen.getByText('AD')).toBeInTheDocument();
  });

  it('renders the notification bell', () => {
    renderLayout();
    // Bell icon should be in a button
    const buttons = screen.getAllByRole('button');
    expect(buttons.length).toBeGreaterThanOrEqual(2);
  });
});

describe('Layout — dark mode toggle', () => {
  it('toggles dark class on documentElement', () => {
    renderLayout();
    const toggleBtn = screen.getByTitle(/Switch to dark mode/i);
    fireEvent.click(toggleBtn);
    expect(document.documentElement.classList.contains('dark')).toBe(true);
    expect(localStorageMock.getItem('cliniq_theme')).toBe('dark');
  });

  it('removes dark class when toggled back', () => {
    renderLayout();
    const toggleBtn = screen.getByTitle(/Switch to dark mode/i);
    fireEvent.click(toggleBtn); // -> dark
    const lightBtn = screen.getByTitle(/Switch to light mode/i);
    fireEvent.click(lightBtn); // -> light
    expect(document.documentElement.classList.contains('dark')).toBe(false);
    expect(localStorageMock.getItem('cliniq_theme')).toBe('light');
  });
});

describe('Layout — child route', () => {
  it('renders child page via Outlet', () => {
    renderLayout();
    expect(screen.getByTestId('child-page')).toBeInTheDocument();
    expect(screen.getByText('Page Content')).toBeInTheDocument();
  });
});

describe('Layout — sidebar integration', () => {
  it('renders the ClinIQ logo from Sidebar', () => {
    renderLayout();
    expect(screen.getByText('ClinIQ')).toBeInTheDocument();
  });
});
