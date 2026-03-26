/**
 * Tests for the useAnalysis custom hook.
 *
 * Validates loading states, data/error handling, reset behaviour,
 * and stale-request protection for concurrent analyses.
 */
import { describe, it, expect, vi } from 'vitest';
import { renderHook, act, waitFor } from '@testing-library/react';
import { useAnalysis } from '../../hooks/useAnalysis';

// Helper: create a controllable promise
function createDeferred<T>() {
  let resolve!: (value: T) => void;
  let reject!: (reason: Error) => void;
  const promise = new Promise<T>((res, rej) => {
    resolve = res;
    reject = rej;
  });
  return { promise, resolve, reject };
}

describe('useAnalysis', () => {
  // --- Initial state ---

  it('starts with null data, no error, and not loading', () => {
    const apiFn = vi.fn().mockResolvedValue({});
    const { result } = renderHook(() => useAnalysis(apiFn));
    expect(result.current.data).toBeNull();
    expect(result.current.error).toBeNull();
    expect(result.current.loading).toBe(false);
  });

  // --- Successful analysis ---

  it('sets loading=true while analysis is in flight', async () => {
    const deferred = createDeferred<string>();
    const apiFn = vi.fn().mockReturnValue(deferred.promise);
    const { result } = renderHook(() => useAnalysis(apiFn));

    act(() => {
      result.current.analyze('sample text');
    });

    expect(result.current.loading).toBe(true);

    await act(async () => {
      deferred.resolve('result');
    });

    expect(result.current.loading).toBe(false);
  });

  it('stores data on successful analysis', async () => {
    const mockResult = { entities: ['DRUG', 'DISEASE'] };
    const apiFn = vi.fn().mockResolvedValue(mockResult);
    const { result } = renderHook(() => useAnalysis(apiFn));

    await act(async () => {
      await result.current.analyze('patient has HTN');
    });

    expect(result.current.data).toEqual(mockResult);
    expect(result.current.error).toBeNull();
  });

  it('returns the result from analyze()', async () => {
    const mockResult = { score: 0.85 };
    const apiFn = vi.fn().mockResolvedValue(mockResult);
    const { result } = renderHook(() => useAnalysis(apiFn));

    let returned: unknown;
    await act(async () => {
      returned = await result.current.analyze();
    });

    expect(returned).toEqual(mockResult);
  });

  it('passes arguments through to the API function', async () => {
    const apiFn = vi.fn().mockResolvedValue('ok');
    const { result } = renderHook(() => useAnalysis(apiFn));

    await act(async () => {
      await result.current.analyze('arg1', 'arg2');
    });

    expect(apiFn).toHaveBeenCalledWith('arg1', 'arg2');
  });

  // --- Error handling ---

  it('captures Error message on failure', async () => {
    const apiFn = vi.fn().mockRejectedValue(new Error('Network timeout'));
    const { result } = renderHook(() => useAnalysis(apiFn));

    await act(async () => {
      await result.current.analyze();
    });

    expect(result.current.error).toBe('Network timeout');
    expect(result.current.data).toBeNull();
    expect(result.current.loading).toBe(false);
  });

  it('handles non-Error throws gracefully', async () => {
    const apiFn = vi.fn().mockRejectedValue('string error');
    const { result } = renderHook(() => useAnalysis(apiFn));

    await act(async () => {
      await result.current.analyze();
    });

    expect(result.current.error).toBe('An unexpected error occurred');
  });

  it('returns null from analyze() on failure', async () => {
    const apiFn = vi.fn().mockRejectedValue(new Error('fail'));
    const { result } = renderHook(() => useAnalysis(apiFn));

    let returned: unknown;
    await act(async () => {
      returned = await result.current.analyze();
    });

    expect(returned).toBeNull();
  });

  it('clears previous error on new successful analysis', async () => {
    const apiFn = vi
      .fn()
      .mockRejectedValueOnce(new Error('first fail'))
      .mockResolvedValueOnce({ ok: true });

    const { result } = renderHook(() => useAnalysis(apiFn));

    await act(async () => {
      await result.current.analyze();
    });
    expect(result.current.error).toBe('first fail');

    await act(async () => {
      await result.current.analyze();
    });
    expect(result.current.error).toBeNull();
    expect(result.current.data).toEqual({ ok: true });
  });

  // --- Reset ---

  it('reset clears data, error, and loading', async () => {
    const apiFn = vi.fn().mockResolvedValue({ value: 42 });
    const { result } = renderHook(() => useAnalysis(apiFn));

    await act(async () => {
      await result.current.analyze();
    });
    expect(result.current.data).not.toBeNull();

    act(() => {
      result.current.reset();
    });

    expect(result.current.data).toBeNull();
    expect(result.current.error).toBeNull();
    expect(result.current.loading).toBe(false);
  });

  // --- Stale request protection ---

  it('ignores stale responses when a newer request is pending', async () => {
    const deferred1 = createDeferred<string>();
    const deferred2 = createDeferred<string>();
    const apiFn = vi
      .fn()
      .mockReturnValueOnce(deferred1.promise)
      .mockReturnValueOnce(deferred2.promise);

    const { result } = renderHook(() => useAnalysis(apiFn));

    // Start first analysis
    act(() => {
      result.current.analyze('first');
    });

    // Start second analysis before first resolves
    act(() => {
      result.current.analyze('second');
    });

    // Resolve first (stale)
    await act(async () => {
      deferred1.resolve('stale result');
    });

    // First result should be ignored — data should still be null
    // (second hasn't resolved yet)
    expect(result.current.data).toBeNull();

    // Resolve second (current)
    await act(async () => {
      deferred2.resolve('current result');
    });

    expect(result.current.data).toBe('current result');
  });
});
