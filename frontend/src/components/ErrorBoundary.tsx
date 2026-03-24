/**
 * React Error Boundary for catching component-level rendering errors.
 *
 * Why a class component?  React's error boundary API (componentDidCatch,
 * getDerivedStateFromError) is only available on class components — there
 * is no hook equivalent as of React 18.  This boundary catches errors in
 * its child tree, displays a user-friendly fallback UI, and optionally
 * reports errors to a logging service.
 *
 * Usage:
 *   <ErrorBoundary fallback={<p>Something went wrong.</p>}>
 *     <YourComponent />
 *   </ErrorBoundary>
 */

import React, { Component, ErrorInfo, ReactNode } from "react";

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

interface ErrorBoundaryProps {
  /** Optional custom fallback UI rendered when an error is caught. */
  fallback?: ReactNode;
  /** Optional callback invoked with error details for logging/reporting. */
  onError?: (error: Error, errorInfo: ErrorInfo) => void;
  /** Child components to protect. */
  children: ReactNode;
}

interface ErrorBoundaryState {
  hasError: boolean;
  error: Error | null;
}

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

class ErrorBoundary extends Component<ErrorBoundaryProps, ErrorBoundaryState> {
  constructor(props: ErrorBoundaryProps) {
    super(props);
    this.state = { hasError: false, error: null };
  }

  /**
   * Update state so the next render shows the fallback UI.
   *
   * This static lifecycle method is called during the "render" phase,
   * so side effects (like logging) should go in componentDidCatch.
   */
  static getDerivedStateFromError(error: Error): ErrorBoundaryState {
    return { hasError: true, error };
  }

  /**
   * Called after an error has been thrown by a descendant component.
   *
   * This is the right place for side effects like error logging or
   * reporting to an external service (Sentry, Datadog, etc.).
   */
  componentDidCatch(error: Error, errorInfo: ErrorInfo): void {
    console.error("[ErrorBoundary] Caught error:", error, errorInfo);
    this.props.onError?.(error, errorInfo);
  }

  /** Reset the error state so the user can retry. */
  handleReset = (): void => {
    this.setState({ hasError: false, error: null });
  };

  render(): ReactNode {
    if (this.state.hasError) {
      // Custom fallback takes priority
      if (this.props.fallback) {
        return this.props.fallback;
      }

      // Default fallback UI — medical-app appropriate, non-alarming
      return (
        <div className="flex flex-col items-center justify-center min-h-[300px] p-8 text-center">
          <div className="rounded-full bg-red-100 dark:bg-red-900/30 p-4 mb-4">
            <svg
              className="w-8 h-8 text-red-600 dark:text-red-400"
              fill="none"
              viewBox="0 0 24 24"
              stroke="currentColor"
              strokeWidth={2}
              aria-hidden="true"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-2.5L13.732 4.5c-.77-.833-2.694-.833-3.464 0L3.34 16.5c-.77.833.192 2.5 1.732 2.5z"
              />
            </svg>
          </div>

          <h2 className="text-lg font-semibold text-gray-900 dark:text-gray-100 mb-2">
            Something went wrong
          </h2>

          <p className="text-sm text-gray-600 dark:text-gray-400 mb-4 max-w-md">
            An unexpected error occurred while rendering this section.
            Your data is safe — try refreshing or click below to retry.
          </p>

          {this.state.error && (
            <details className="mb-4 text-left w-full max-w-md">
              <summary className="text-xs text-gray-500 dark:text-gray-500 cursor-pointer hover:text-gray-700 dark:hover:text-gray-300">
                Technical details
              </summary>
              <pre className="mt-2 p-3 bg-gray-100 dark:bg-gray-800 rounded text-xs text-red-700 dark:text-red-300 overflow-auto max-h-32">
                {this.state.error.message}
              </pre>
            </details>
          )}

          <button
            onClick={this.handleReset}
            className="px-4 py-2 text-sm font-medium text-white bg-blue-600 hover:bg-blue-700 rounded-lg transition-colors focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2 dark:focus:ring-offset-gray-900"
            type="button"
          >
            Try again
          </button>
        </div>
      );
    }

    return this.props.children;
  }
}

export default ErrorBoundary;
