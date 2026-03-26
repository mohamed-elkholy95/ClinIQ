/**
 * Generic hook for clinical NLP analysis requests.
 *
 * Nearly every page in ClinIQ follows the same pattern:
 *  1. User enters/selects clinical text
 *  2. Clicks "Analyze"
 *  3. Page shows loading spinner → results or error
 *
 * This hook encapsulates that lifecycle — loading state, error handling,
 * result storage, and reset — so individual pages don't duplicate the
 * same 20 lines of useState/try/catch boilerplate.
 *
 * Design decisions:
 * - **Generic type parameter** — Each page has its own response shape;
 *   the hook is parameterised so TypeScript enforces correct usage.
 * - **Stable `analyze` reference** — useCallback prevents unnecessary
 *   re-renders when passed as a prop to child components.
 * - **Error as string** — Components typically display error messages
 *   in a banner; passing the raw Error object would require .message
 *   access everywhere.  The hook extracts the message (or falls back
 *   to a generic string for non-Error throws).
 * - **Reset function** — Allows clearing results when the user changes
 *   input text, preventing stale results from a previous analysis from
 *   showing alongside new input.
 *
 * @example
 * ```tsx
 * const { data, loading, error, analyze, reset } = useAnalysis(
 *   (text: string) => extractMedications({ text })
 * );
 * ```
 */

import { useState, useCallback, useRef } from 'react';

/**
 * Return type for the useAnalysis hook.
 *
 * @typeParam T - The shape of the successful analysis result.
 */
export interface UseAnalysisReturn<T> {
  /** The most recent successful result, or null before first analysis. */
  data: T | null;
  /** True while an analysis request is in flight. */
  loading: boolean;
  /** Human-readable error message from the last failed analysis, or null. */
  error: string | null;
  /** Trigger an analysis.  Accepts any arguments that the API function needs. */
  analyze: (...args: unknown[]) => Promise<T | null>;
  /** Clear data and error state (e.g., when user changes input). */
  reset: () => void;
}

/**
 * Hook that wraps a clinical analysis API call with loading/error/data state.
 *
 * @typeParam T - The response type returned by the API function.
 * @param apiFn - An async function that performs the analysis and returns T.
 * @returns An object with data, loading, error, analyze, and reset.
 */
export function useAnalysis<T>(
  apiFn: (...args: any[]) => Promise<T>
): UseAnalysisReturn<T> {
  const [data, setData] = useState<T | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Track the latest request to ignore stale responses when the user
  // fires multiple analyses in quick succession (e.g., rapid clicking).
  const requestIdRef = useRef(0);

  const analyze = useCallback(
    async (...args: unknown[]): Promise<T | null> => {
      const currentId = ++requestIdRef.current;
      setLoading(true);
      setError(null);

      try {
        const result = await apiFn(...args);

        // Only update state if this is still the latest request
        if (currentId === requestIdRef.current) {
          setData(result);
          setLoading(false);
        }
        return result;
      } catch (err) {
        if (currentId === requestIdRef.current) {
          const message =
            err instanceof Error
              ? err.message
              : 'An unexpected error occurred';
          setError(message);
          setData(null);
          setLoading(false);
        }
        return null;
      }
    },
    [apiFn]
  );

  const reset = useCallback(() => {
    setData(null);
    setError(null);
    setLoading(false);
  }, []);

  return { data, loading, error, analyze, reset };
}
