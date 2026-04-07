/**
 * Tests for the useDebounce custom hook.
 *
 * Validates that rapidly-changing values are debounced and that
 * the timer is properly cleaned up on unmount.
 */
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { renderHook, act } from '@testing-library/react';
import { useDebounce } from '../../hooks/useDebounce';

describe('useDebounce', () => {
  beforeEach(() => {
    vi.useFakeTimers();
  });

  afterEach(() => {
    vi.useRealTimers();
  });

  it('returns the initial value immediately', () => {
    const { result } = renderHook(() => useDebounce('hello', 300));
    expect(result.current).toBe('hello');
  });

  it('does not update the value before the delay', () => {
    const { result, rerender } = renderHook(
      ({ value }) => useDebounce(value, 300),
      { initialProps: { value: 'A' } }
    );

    rerender({ value: 'B' });

    // Advance less than the delay
    act(() => {
      vi.advanceTimersByTime(200);
    });

    expect(result.current).toBe('A');
  });

  it('updates the value after the delay', () => {
    const { result, rerender } = renderHook(
      ({ value }) => useDebounce(value, 300),
      { initialProps: { value: 'A' } }
    );

    rerender({ value: 'B' });

    act(() => {
      vi.advanceTimersByTime(300);
    });

    expect(result.current).toBe('B');
  });

  it('resets the timer on rapid changes', () => {
    const { result, rerender } = renderHook(
      ({ value }) => useDebounce(value, 300),
      { initialProps: { value: 'A' } }
    );

    rerender({ value: 'B' });
    act(() => { vi.advanceTimersByTime(200); });

    rerender({ value: 'C' });
    act(() => { vi.advanceTimersByTime(200); });

    // 400ms total but timer was reset — should still be 'A'
    expect(result.current).toBe('A');

    // Wait for the full delay from the last change
    act(() => { vi.advanceTimersByTime(100); });
    expect(result.current).toBe('C');
  });

  it('uses default delay of 300ms', () => {
    const { result, rerender } = renderHook(
      ({ value }) => useDebounce(value),
      { initialProps: { value: 'start' } }
    );

    rerender({ value: 'end' });

    act(() => { vi.advanceTimersByTime(299); });
    expect(result.current).toBe('start');

    act(() => { vi.advanceTimersByTime(1); });
    expect(result.current).toBe('end');
  });

  it('works with non-string types', () => {
    const { result, rerender } = renderHook(
      ({ value }) => useDebounce(value, 100),
      { initialProps: { value: 0 } }
    );

    rerender({ value: 42 });

    act(() => { vi.advanceTimersByTime(100); });
    expect(result.current).toBe(42);
  });

  it('cleans up timer on unmount', () => {
    const clearSpy = vi.spyOn(globalThis, 'clearTimeout');
    const { unmount } = renderHook(() => useDebounce('test', 300));

    unmount();

    // clearTimeout should have been called during cleanup
    expect(clearSpy).toHaveBeenCalled();
    clearSpy.mockRestore();
  });
});

