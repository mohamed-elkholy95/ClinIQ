/**
 * DriftMonitor page — Real-time model and data drift monitoring dashboard.
 *
 * Displays the current drift status for the overall system and per-model,
 * using the /drift/status endpoint.  Visual indicators make it easy to
 * spot degradation at a glance: green (stable), yellow (warning), red (drifted).
 *
 * Design decisions:
 * - Auto-refresh toggle lets users keep the dashboard live during monitoring
 *   sessions without manual polling.
 * - PSI (Population Stability Index) is shown with a visual gauge and
 *   interpretive thresholds aligned with industry standards (<0.1 stable,
 *   0.1–0.25 warning, >0.25 drifted).
 * - Per-model status is displayed as a grid of status cards for quick
 *   scan across all deployed models.
 */

import { useState, useEffect, useCallback, useRef } from 'react';
import { Activity, RefreshCw, AlertTriangle, CheckCircle2, XCircle, Clock, BarChart3, Cpu } from 'lucide-react';
import { getDriftStatus } from '../services/clinical';
import type { DriftStatusResponse, DriftStatus } from '../types/clinical';

/** Status configuration for visual rendering */
const STATUS_CONFIG: Record<DriftStatus, { label: string; color: string; bg: string; icon: typeof CheckCircle2 }> = {
  stable: {
    label: 'Stable',
    color: 'text-green-600 dark:text-green-400',
    bg: 'bg-green-50 dark:bg-green-900/30 border-green-200 dark:border-green-800',
    icon: CheckCircle2,
  },
  warning: {
    label: 'Warning',
    color: 'text-yellow-600 dark:text-yellow-400',
    bg: 'bg-yellow-50 dark:bg-yellow-900/30 border-yellow-200 dark:border-yellow-800',
    icon: AlertTriangle,
  },
  drifted: {
    label: 'Drifted',
    color: 'text-red-600 dark:text-red-400',
    bg: 'bg-red-50 dark:bg-red-900/30 border-red-200 dark:border-red-800',
    icon: XCircle,
  },
};

/** PSI interpretation thresholds */
function psiInterpretation(psi: number): { label: string; color: string } {
  if (psi < 0.1) return { label: 'No significant shift', color: 'text-green-600 dark:text-green-400' };
  if (psi < 0.25) return { label: 'Moderate shift detected', color: 'text-yellow-600 dark:text-yellow-400' };
  return { label: 'Significant distribution shift', color: 'text-red-600 dark:text-red-400' };
}

/** PSI bar fill percentage (capped at 100%) */
function psiFillPercent(psi: number): number {
  // Map 0–0.5 range to 0–100%
  return Math.min(100, (psi / 0.5) * 100);
}

export function DriftMonitor() {
  const [status, setStatus] = useState<DriftStatusResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [autoRefresh, setAutoRefresh] = useState(false);
  const [lastRefresh, setLastRefresh] = useState<Date | null>(null);
  const intervalRef = useRef<ReturnType<typeof setInterval> | null>(null);

  const fetchStatus = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const data = await getDriftStatus();
      setStatus(data);
      setLastRefresh(new Date());
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to fetch drift status');
    } finally {
      setLoading(false);
    }
  }, []);

  // Initial load
  useEffect(() => {
    fetchStatus();
  }, [fetchStatus]);

  // Auto-refresh
  useEffect(() => {
    if (autoRefresh) {
      intervalRef.current = setInterval(fetchStatus, 30_000);
    } else if (intervalRef.current) {
      clearInterval(intervalRef.current);
      intervalRef.current = null;
    }
    return () => {
      if (intervalRef.current) clearInterval(intervalRef.current);
    };
  }, [autoRefresh, fetchStatus]);

  const modelEntries = status?.model_drift ? Object.entries(status.model_drift) : [];

  return (
    <div className="max-w-4xl mx-auto space-y-6">
      {/* Page header */}
      <div className="flex items-start justify-between">
        <div>
          <h1 className="text-2xl font-bold text-text-primary flex items-center gap-2">
            <Activity className="w-7 h-7 text-primary-500" />
            Drift Monitor
          </h1>
          <p className="mt-1 text-sm text-text-muted">
            Track data distribution shifts and model performance degradation in real time.
          </p>
        </div>
        <div className="flex items-center gap-2">
          <label className="flex items-center gap-1.5 text-xs text-text-muted cursor-pointer">
            <input
              type="checkbox"
              checked={autoRefresh}
              onChange={(e) => setAutoRefresh(e.target.checked)}
              className="rounded border-border"
            />
            Auto-refresh (30s)
          </label>
          <button
            onClick={fetchStatus}
            disabled={loading}
            className="p-2 rounded-lg border border-border text-text-muted hover:text-text-primary hover:bg-gray-50 dark:hover:bg-gray-800 disabled:opacity-50 transition-colors"
            aria-label="Refresh status"
          >
            <RefreshCw className={`w-4 h-4 ${loading ? 'animate-spin' : ''}`} />
          </button>
        </div>
      </div>

      {/* Error */}
      {error && (
        <div className="p-4 rounded-lg bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 text-red-700 dark:text-red-400 text-sm" role="alert">
          {error}
        </div>
      )}

      {status && (
        <>
          {/* Overall status card */}
          <div className={`rounded-xl border p-6 ${STATUS_CONFIG[status.overall_status].bg}`} data-testid="overall-status">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-3">
                {(() => {
                  const Icon = STATUS_CONFIG[status.overall_status].icon;
                  return <Icon className={`w-8 h-8 ${STATUS_CONFIG[status.overall_status].color}`} />;
                })()}
                <div>
                  <h2 className="text-lg font-semibold text-text-primary">Overall System Status</h2>
                  <p className={`text-sm font-medium ${STATUS_CONFIG[status.overall_status].color}`}>
                    {STATUS_CONFIG[status.overall_status].label}
                  </p>
                </div>
              </div>
              {lastRefresh && (
                <div className="flex items-center gap-1 text-xs text-text-muted">
                  <Clock className="w-3 h-3" />
                  Updated {lastRefresh.toLocaleTimeString()}
                </div>
              )}
            </div>
          </div>

          {/* PSI gauge */}
          <div className="bg-surface rounded-xl border border-border p-5" data-testid="psi-gauge">
            <div className="flex items-center gap-2 mb-3">
              <BarChart3 className="w-5 h-5 text-text-muted" />
              <h3 className="font-semibold text-text-primary">Text Distribution PSI</h3>
            </div>
            <div className="flex items-center gap-4">
              <div className="flex-1">
                <div className="h-3 rounded-full bg-gray-200 dark:bg-gray-700 overflow-hidden">
                  <div
                    className={`h-full rounded-full transition-all duration-500 ${
                      status.text_distribution_psi < 0.1
                        ? 'bg-green-500'
                        : status.text_distribution_psi < 0.25
                        ? 'bg-yellow-500'
                        : 'bg-red-500'
                    }`}
                    style={{ width: `${psiFillPercent(status.text_distribution_psi)}%` }}
                    data-testid="psi-bar"
                  />
                </div>
                <div className="flex justify-between mt-1 text-[10px] text-text-muted">
                  <span>0.00</span>
                  <span>0.10</span>
                  <span>0.25</span>
                  <span>0.50</span>
                </div>
              </div>
              <div className="text-right">
                <p className="text-2xl font-bold text-text-primary">{status.text_distribution_psi.toFixed(3)}</p>
                <p className={`text-xs font-medium ${psiInterpretation(status.text_distribution_psi).color}`}>
                  {psiInterpretation(status.text_distribution_psi).label}
                </p>
              </div>
            </div>
            <div className="mt-3 grid grid-cols-3 gap-2 text-[10px]">
              <div className="flex items-center gap-1">
                <div className="w-2 h-2 rounded-full bg-green-500" />
                <span className="text-text-muted">&lt;0.10 Stable</span>
              </div>
              <div className="flex items-center gap-1">
                <div className="w-2 h-2 rounded-full bg-yellow-500" />
                <span className="text-text-muted">0.10–0.25 Warning</span>
              </div>
              <div className="flex items-center gap-1">
                <div className="w-2 h-2 rounded-full bg-red-500" />
                <span className="text-text-muted">&gt;0.25 Drifted</span>
              </div>
            </div>
          </div>

          {/* Per-model status grid */}
          {modelEntries.length > 0 && (
            <div>
              <h3 className="text-sm font-semibold text-text-primary mb-3 flex items-center gap-1.5">
                <Cpu className="w-4 h-4 text-text-muted" />
                Per-Model Status ({modelEntries.length})
              </h3>
              <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-3">
                {modelEntries.map(([model, modelStatus]) => {
                  const config = STATUS_CONFIG[modelStatus];
                  const Icon = config.icon;
                  return (
                    <div
                      key={model}
                      className={`rounded-lg border p-4 ${config.bg}`}
                      data-testid="model-status-card"
                    >
                      <div className="flex items-center gap-2">
                        <Icon className={`w-5 h-5 ${config.color}`} />
                        <div>
                          <p className="text-sm font-medium text-text-primary">{model}</p>
                          <p className={`text-xs font-medium ${config.color}`}>{config.label}</p>
                        </div>
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>
          )}

          {/* Last updated from server */}
          {status.last_updated && (
            <p className="text-xs text-text-muted text-center">
              Server data as of: {new Date(status.last_updated).toLocaleString()}
            </p>
          )}
        </>
      )}

      {/* Loading state */}
      {loading && !status && (
        <div className="text-center py-16 text-text-muted">
          <RefreshCw className="w-10 h-10 mx-auto mb-3 animate-spin opacity-40" />
          <p>Loading drift status...</p>
        </div>
      )}
    </div>
  );
}
