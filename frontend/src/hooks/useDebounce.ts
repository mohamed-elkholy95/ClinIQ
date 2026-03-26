/**
 * Debounce hook for delaying rapidly-changing values.
 *
 * Useful for search inputs, autocomplete queries, and live-preview
 * scenarios where firing an API call on every keystroke would be
 * wasteful.  The returned value only updates after the user stops
 * typing for `delayMs` milliseconds.
 *
 * @example
 * ```tsx
 * const [query, setQuery] = useState('');
 * const debouncedQuery = useDebounce(query, 300);
 *
 * useEffect(() => {
 *   if (debouncedQuery) searchDocuments(debouncedQuery);
 * }, [debouncedQuery]);
 * ```
 */

import { useState, useEffect } from 'react';

/**
 * Returns a debounced version of the provided value.
 *
 * @typeParam T - The type of the value being debounced.
 * @param value - The rapidly-changing input value.
 * @param delayMs - Milliseconds to wait before updating.  Defaults to 300.
 * @returns The debounced value, updated only after `delayMs` of inactivity.
 */
export function useDebounce<T>(value: T, delayMs: number = 300): T {
  const [debounced, setDebounced] = useState(value);

  useEffect(() => {
    const timer = setTimeout(() => setDebounced(value), delayMs);
    return () => clearTimeout(timer);
  }, [value, delayMs]);

  return debounced;
}
