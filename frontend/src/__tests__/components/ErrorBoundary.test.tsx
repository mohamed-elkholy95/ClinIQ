/**
 * Tests for ErrorBoundary component.
 *
 * Verifies fallback rendering on child errors, custom fallback support,
 * retry/reset behaviour, onError callback, and technical details display.
 */
import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, fireEvent } from '@testing-library/react';
import ErrorBoundary from '../../components/ErrorBoundary';

// Suppress React's default error boundary console.error during tests
const originalError = console.error;
beforeEach(() => {
  console.error = vi.fn();
  return () => {
    console.error = originalError;
  };
});

// A component that throws on render to trigger the boundary
function ThrowingChild({ message = 'Test error' }: { message?: string }) {
  throw new Error(message);
}

function GoodChild() {
  return <div>Healthy child</div>;
}

describe('ErrorBoundary', () => {
  it('renders children when no error occurs', () => {
    render(
      <ErrorBoundary>
        <GoodChild />
      </ErrorBoundary>
    );
    expect(screen.getByText('Healthy child')).toBeInTheDocument();
  });

  it('renders default fallback UI when child throws', () => {
    render(
      <ErrorBoundary>
        <ThrowingChild />
      </ErrorBoundary>
    );
    expect(screen.getByText('Something went wrong')).toBeInTheDocument();
    expect(
      screen.getByText(/An unexpected error occurred/)
    ).toBeInTheDocument();
  });

  it('shows "Try again" button in default fallback', () => {
    render(
      <ErrorBoundary>
        <ThrowingChild />
      </ErrorBoundary>
    );
    expect(screen.getByRole('button', { name: /try again/i })).toBeInTheDocument();
  });

  it('renders custom fallback when provided', () => {
    render(
      <ErrorBoundary fallback={<div>Custom fallback</div>}>
        <ThrowingChild />
      </ErrorBoundary>
    );
    expect(screen.getByText('Custom fallback')).toBeInTheDocument();
    expect(screen.queryByText('Something went wrong')).toBeNull();
  });

  it('shows error message in technical details', () => {
    render(
      <ErrorBoundary>
        <ThrowingChild message="Pipeline crashed" />
      </ErrorBoundary>
    );
    // The error message is inside a <pre> within a <details>
    expect(screen.getByText('Pipeline crashed')).toBeInTheDocument();
  });

  it('calls onError callback with error and errorInfo', () => {
    const onError = vi.fn();
    render(
      <ErrorBoundary onError={onError}>
        <ThrowingChild message="callback test" />
      </ErrorBoundary>
    );
    expect(onError).toHaveBeenCalledOnce();
    expect(onError.mock.calls[0][0]).toBeInstanceOf(Error);
    expect(onError.mock.calls[0][0].message).toBe('callback test');
    // Second arg is React's ErrorInfo object with componentStack
    expect(onError.mock.calls[0][1]).toHaveProperty('componentStack');
  });

  it('resets error state when "Try again" is clicked', () => {
    // We need a component that only throws on first render
    let shouldThrow = true;

    function ConditionalThrow() {
      if (shouldThrow) {
        throw new Error('First render error');
      }
      return <div>Recovered</div>;
    }

    render(
      <ErrorBoundary>
        <ConditionalThrow />
      </ErrorBoundary>
    );

    expect(screen.getByText('Something went wrong')).toBeInTheDocument();

    // Now tell the child not to throw
    shouldThrow = false;

    fireEvent.click(screen.getByRole('button', { name: /try again/i }));
    expect(screen.getByText('Recovered')).toBeInTheDocument();
  });

  it('renders technical details inside a collapsible details element', () => {
    const { container } = render(
      <ErrorBoundary>
        <ThrowingChild />
      </ErrorBoundary>
    );
    const details = container.querySelector('details');
    expect(details).toBeTruthy();
    const summary = container.querySelector('summary');
    expect(summary?.textContent).toContain('Technical details');
  });
});
