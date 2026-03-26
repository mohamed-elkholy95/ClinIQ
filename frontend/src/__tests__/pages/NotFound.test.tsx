/**
 * Tests for the NotFound (404) page component.
 *
 * Verifies heading, descriptive text, navigation link,
 * icon rendering, and accessibility attributes.
 */
import { describe, it, expect } from 'vitest';
import { render, screen } from '@testing-library/react';
import { BrowserRouter } from 'react-router-dom';
import { NotFound } from '../../pages/NotFound';

function renderPage() {
  return render(
    <BrowserRouter>
      <NotFound />
    </BrowserRouter>
  );
}

describe('NotFound', () => {
  // --- Page structure ---

  it('renders the 404 heading', () => {
    renderPage();
    expect(screen.getByRole('heading', { level: 1 })).toHaveTextContent(
      'Page not found'
    );
  });

  it('renders descriptive body text', () => {
    renderPage();
    expect(
      screen.getByText(/doesn't exist or has been moved/)
    ).toBeInTheDocument();
  });

  it('renders a link back to the dashboard', () => {
    renderPage();
    const link = screen.getByRole('link', { name: /back to dashboard/i });
    expect(link).toBeInTheDocument();
    expect(link).toHaveAttribute('href', '/');
  });

  // --- Accessibility ---

  it('hides the decorative icon from screen readers', () => {
    renderPage();
    // The SVG icon should have aria-hidden="true" — we check there's
    // an element with that attribute inside the container
    const container = screen.getByRole('heading', { level: 1 }).parentElement;
    expect(container).not.toBeNull();
    const hiddenIcon = container!.querySelector('[aria-hidden="true"]');
    expect(hiddenIcon).toBeInTheDocument();
  });

  it('has a focus-visible link style class', () => {
    renderPage();
    const link = screen.getByRole('link', { name: /back to dashboard/i });
    expect(link.className).toContain('focus:ring');
  });
});
