import { useState, useEffect, useCallback } from 'react';
import {
  MessageSquare,
  Plus,
  Trash2,
  RefreshCw,
  Clock,
  TrendingUp,
  Tags,
  FileCode,
  Users,
  BarChart3,
  ChevronDown,
  ChevronUp,
} from 'lucide-react';
import { clsx } from 'clsx';
import type {
  AddTurnResponse,
  ContextResponse,
  ConversationStats,
  SessionInfo,
} from '../types/clinical';
import {
  addConversationTurn,
  getConversationContext,
  clearConversationSession,
  getConversationStats,
  listConversationSessions,
} from '../services/clinical';

// ─── Sample clinical notes ───────────────────────────────────

const SAMPLE_NOTES = [
  {
    label: 'ER Chest Pain',
    text: 'Patient is a 58-year-old male presenting to the emergency department with acute onset chest pain radiating to the left arm. History of hypertension and type 2 diabetes mellitus. Current medications include metformin 1000mg BID and lisinopril 20mg daily. Vital signs: BP 158/94, HR 98, SpO2 96% on room air.',
    entities: [
      { text: 'chest pain', entity_type: 'SYMPTOM', confidence: 0.96 },
      { text: 'hypertension', entity_type: 'DISEASE', confidence: 0.94 },
      { text: 'type 2 diabetes mellitus', entity_type: 'DISEASE', confidence: 0.93 },
      { text: 'metformin', entity_type: 'MEDICATION', confidence: 0.97 },
      { text: 'lisinopril', entity_type: 'MEDICATION', confidence: 0.95 },
    ],
    icd_codes: [
      { code: 'R07.9', description: 'Chest pain, unspecified', confidence: 0.85 },
      { code: 'I10', description: 'Essential hypertension', confidence: 0.82 },
      { code: 'E11.9', description: 'Type 2 DM without complications', confidence: 0.80 },
    ],
    risk_score: 0.72,
    risk_level: 'high',
    summary: '58yo male with acute chest pain, HTN, and DM2 on metformin/lisinopril.',
  },
  {
    label: 'Follow-up Visit',
    text: 'Follow-up visit for chest pain evaluation. Troponin levels negative x2. EKG shows normal sinus rhythm. Chest pain has resolved. Patient reports compliance with medications. Blood pressure better controlled at 132/82. Continue current medications. Follow up in 2 weeks.',
    entities: [
      { text: 'chest pain', entity_type: 'SYMPTOM', confidence: 0.91 },
      { text: 'troponin', entity_type: 'LAB_TEST', confidence: 0.88 },
      { text: 'normal sinus rhythm', entity_type: 'FINDING', confidence: 0.92 },
    ],
    icd_codes: [
      { code: 'R07.9', description: 'Chest pain, unspecified', confidence: 0.70 },
      { code: 'Z09', description: 'Encounter for follow-up exam', confidence: 0.75 },
    ],
    risk_score: 0.35,
    risk_level: 'low',
    summary: 'Follow-up: chest pain resolved, troponins negative, NSR on EKG. Continue meds.',
  },
  {
    label: 'Dental Progress Note',
    text: 'Patient presents for periodontal maintenance. Probing depths improved from 6mm to 4mm in posterior quadrants. Generalized moderate plaque accumulation. Scaling and root planing performed in quadrants 1 and 4. Patient advised on improved oral hygiene techniques. History of diabetes may affect healing.',
    entities: [
      { text: 'periodontal maintenance', entity_type: 'PROCEDURE', confidence: 0.90 },
      { text: 'scaling and root planing', entity_type: 'PROCEDURE', confidence: 0.95 },
      { text: 'diabetes', entity_type: 'DISEASE', confidence: 0.88 },
      { text: 'plaque accumulation', entity_type: 'FINDING', confidence: 0.85 },
    ],
    icd_codes: [
      { code: 'K05.31', description: 'Chronic periodontitis, localized, moderate', confidence: 0.78 },
    ],
    risk_score: 0.45,
    risk_level: 'moderate',
    summary: 'Periodontal maintenance with improved probing depths. SRP Q1/Q4. DM comorbidity.',
  },
];

// ─── Component ───────────────────────────────────────────────

export function ConversationMemory() {
  const [sessionId, setSessionId] = useState('demo-session-1');
  const [stats, setStats] = useState<ConversationStats | null>(null);
  const [sessions, setSessions] = useState<SessionInfo[]>([]);
  const [context, setContext] = useState<ContextResponse | null>(null);
  const [lastTurn, setLastTurn] = useState<AddTurnResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [expandedTurn, setExpandedTurn] = useState<number | null>(null);
  const [refreshing, setRefreshing] = useState(false);

  // Load stats on mount
  const loadStats = useCallback(async () => {
    try {
      const s = await getConversationStats();
      setStats(s);
    } catch {
      // Stats may fail if no turns yet — that's fine
    }
  }, []);

  const loadSessions = useCallback(async () => {
    try {
      const s = await listConversationSessions();
      setSessions(s.sessions);
    } catch {
      // Ignore
    }
  }, []);

  const loadContext = useCallback(async () => {
    if (!sessionId.trim()) return;
    try {
      const ctx = await getConversationContext({
        session_id: sessionId,
        last_n: 10,
      });
      setContext(ctx);
    } catch {
      setContext(null);
    }
  }, [sessionId]);

  useEffect(() => {
    loadStats();
    loadSessions();
  }, [loadStats, loadSessions]);

  useEffect(() => {
    loadContext();
  }, [loadContext]);

  const handleAddTurn = async (sampleIndex: number) => {
    const sample = SAMPLE_NOTES[sampleIndex];
    setLoading(true);
    setError(null);
    try {
      const resp = await addConversationTurn({
        session_id: sessionId,
        text: sample.text,
        entities: sample.entities,
        icd_codes: sample.icd_codes,
        risk_score: sample.risk_score,
        risk_level: sample.risk_level,
        summary: sample.summary,
        document_id: `sample-${sampleIndex + 1}`,
        metadata: { source: 'demo', note_type: sample.label },
      });
      setLastTurn(resp);
      await Promise.all([loadContext(), loadStats(), loadSessions()]);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to add turn');
    } finally {
      setLoading(false);
    }
  };

  const handleClearSession = async () => {
    setLoading(true);
    setError(null);
    try {
      await clearConversationSession(sessionId);
      setContext(null);
      setLastTurn(null);
      await Promise.all([loadStats(), loadSessions()]);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to clear session');
    } finally {
      setLoading(false);
    }
  };

  const handleRefresh = async () => {
    setRefreshing(true);
    await Promise.all([loadContext(), loadStats(), loadSessions()]);
    setRefreshing(false);
  };

  const riskColor = (level: string | undefined) => {
    switch (level) {
      case 'critical':
        return 'text-red-600 bg-red-50 dark:text-red-400 dark:bg-red-900/30';
      case 'high':
        return 'text-orange-600 bg-orange-50 dark:text-orange-400 dark:bg-orange-900/30';
      case 'moderate':
        return 'text-yellow-600 bg-yellow-50 dark:text-yellow-400 dark:bg-yellow-900/30';
      case 'low':
        return 'text-green-600 bg-green-50 dark:text-green-400 dark:bg-green-900/30';
      default:
        return 'text-gray-600 bg-gray-50 dark:text-gray-400 dark:bg-gray-700';
    }
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-gray-900 dark:text-white flex items-center gap-2">
            <MessageSquare className="h-7 w-7 text-indigo-600" />
            Conversation Memory
          </h1>
          <p className="text-sm text-gray-500 dark:text-gray-400 mt-1">
            Session-scoped context tracking for sequential clinical analyses
          </p>
        </div>
        <button
          onClick={handleRefresh}
          disabled={refreshing}
          className="flex items-center gap-1.5 px-3 py-1.5 text-sm rounded-lg border border-gray-300 dark:border-gray-600 hover:bg-gray-50 dark:hover:bg-gray-700 transition-colors"
        >
          <RefreshCw className={clsx('h-4 w-4', refreshing && 'animate-spin')} />
          Refresh
        </button>
      </div>

      {/* Stats cards */}
      {stats && (
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          <div className="bg-white dark:bg-gray-800 rounded-xl p-4 border border-gray-200 dark:border-gray-700">
            <div className="flex items-center gap-2 text-sm text-gray-500 dark:text-gray-400 mb-1">
              <Users className="h-4 w-4" />
              Active Sessions
            </div>
            <p className="text-2xl font-bold text-gray-900 dark:text-white">
              {stats.active_sessions}
            </p>
            <p className="text-xs text-gray-400 mt-1">max {stats.max_sessions}</p>
          </div>
          <div className="bg-white dark:bg-gray-800 rounded-xl p-4 border border-gray-200 dark:border-gray-700">
            <div className="flex items-center gap-2 text-sm text-gray-500 dark:text-gray-400 mb-1">
              <BarChart3 className="h-4 w-4" />
              Total Turns
            </div>
            <p className="text-2xl font-bold text-gray-900 dark:text-white">
              {stats.total_turns}
            </p>
            <p className="text-xs text-gray-400 mt-1">
              max {stats.max_turns_per_session}/session
            </p>
          </div>
          <div className="bg-white dark:bg-gray-800 rounded-xl p-4 border border-gray-200 dark:border-gray-700">
            <div className="flex items-center gap-2 text-sm text-gray-500 dark:text-gray-400 mb-1">
              <Clock className="h-4 w-4" />
              Session TTL
            </div>
            <p className="text-2xl font-bold text-gray-900 dark:text-white">
              {Math.round(stats.session_ttl_seconds / 60)}m
            </p>
            <p className="text-xs text-gray-400 mt-1">idle expiry</p>
          </div>
          <div className="bg-white dark:bg-gray-800 rounded-xl p-4 border border-gray-200 dark:border-gray-700">
            <div className="flex items-center gap-2 text-sm text-gray-500 dark:text-gray-400 mb-1">
              <FileCode className="h-4 w-4" />
              Session ID
            </div>
            <input
              type="text"
              value={sessionId}
              onChange={(e) => setSessionId(e.target.value)}
              className="w-full text-sm font-mono bg-gray-50 dark:bg-gray-700 rounded px-2 py-1 border border-gray-200 dark:border-gray-600 mt-1"
              placeholder="Enter session ID..."
            />
          </div>
        </div>
      )}

      {/* Error */}
      {error && (
        <div className="bg-red-50 dark:bg-red-900/30 border border-red-200 dark:border-red-800 rounded-lg p-3 text-sm text-red-700 dark:text-red-300">
          {error}
        </div>
      )}

      {/* Add turn buttons */}
      <div className="bg-white dark:bg-gray-800 rounded-xl border border-gray-200 dark:border-gray-700 p-4">
        <h2 className="text-sm font-semibold text-gray-700 dark:text-gray-300 mb-3">
          Add Sample Analysis Turn
        </h2>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
          {SAMPLE_NOTES.map((sample, idx) => (
            <button
              key={idx}
              onClick={() => handleAddTurn(idx)}
              disabled={loading || !sessionId.trim()}
              className={clsx(
                'flex items-start gap-3 p-3 rounded-lg border text-left transition-all',
                'border-gray-200 dark:border-gray-600',
                'hover:border-indigo-400 hover:bg-indigo-50 dark:hover:bg-indigo-900/20',
                'disabled:opacity-50 disabled:cursor-not-allowed',
              )}
            >
              <Plus className="h-5 w-5 text-indigo-500 mt-0.5 shrink-0" />
              <div>
                <p className="font-medium text-sm text-gray-900 dark:text-white">
                  {sample.label}
                </p>
                <p className="text-xs text-gray-500 dark:text-gray-400 mt-0.5 line-clamp-2">
                  {sample.text.slice(0, 100)}…
                </p>
                <div className="flex gap-2 mt-1.5 flex-wrap">
                  <span className="text-xs px-1.5 py-0.5 rounded bg-blue-100 text-blue-700 dark:bg-blue-900/40 dark:text-blue-300">
                    {sample.entities.length} entities
                  </span>
                  <span className="text-xs px-1.5 py-0.5 rounded bg-purple-100 text-purple-700 dark:bg-purple-900/40 dark:text-purple-300">
                    {sample.icd_codes.length} ICD codes
                  </span>
                  <span
                    className={clsx(
                      'text-xs px-1.5 py-0.5 rounded',
                      riskColor(sample.risk_level),
                    )}
                  >
                    Risk: {sample.risk_level}
                  </span>
                </div>
              </div>
            </button>
          ))}
        </div>
        {lastTurn && (
          <p className="text-xs text-green-600 dark:text-green-400 mt-2">
            ✓ Turn #{lastTurn.turn_id} recorded ({lastTurn.turn_count} total in session)
          </p>
        )}
      </div>

      {/* Context display */}
      {context && context.turn_count > 0 && (
        <div className="bg-white dark:bg-gray-800 rounded-xl border border-gray-200 dark:border-gray-700 p-4 space-y-4">
          <div className="flex items-center justify-between">
            <h2 className="text-sm font-semibold text-gray-700 dark:text-gray-300">
              Session Context — {context.turn_count} turn{context.turn_count !== 1 ? 's' : ''}
            </h2>
            <button
              onClick={handleClearSession}
              disabled={loading}
              className="flex items-center gap-1 text-xs text-red-600 dark:text-red-400 hover:underline"
            >
              <Trash2 className="h-3.5 w-3.5" />
              Clear Session
            </button>
          </div>

          {/* Risk trend */}
          {context.overall_risk_trend.length > 0 && (
            <div>
              <h3 className="text-xs font-medium text-gray-500 dark:text-gray-400 mb-2 flex items-center gap-1">
                <TrendingUp className="h-3.5 w-3.5" />
                Risk Trend
              </h3>
              <div className="flex items-end gap-1 h-16">
                {context.overall_risk_trend.map((score, i) => (
                  <div
                    key={i}
                    className="flex-1 flex flex-col items-center gap-0.5"
                  >
                    <span className="text-[10px] text-gray-400">
                      {(score * 100).toFixed(0)}%
                    </span>
                    <div
                      className={clsx(
                        'w-full rounded-t',
                        score >= 0.7
                          ? 'bg-red-400'
                          : score >= 0.5
                            ? 'bg-yellow-400'
                            : 'bg-green-400',
                      )}
                      style={{ height: `${Math.max(score * 100, 5)}%` }}
                    />
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Aggregated entities */}
          {context.unique_entities.length > 0 && (
            <div>
              <h3 className="text-xs font-medium text-gray-500 dark:text-gray-400 mb-2 flex items-center gap-1">
                <Tags className="h-3.5 w-3.5" />
                Unique Entities ({context.unique_entities.length})
              </h3>
              <div className="flex flex-wrap gap-1.5">
                {context.unique_entities.map((ent) => (
                  <span
                    key={ent}
                    className="text-xs px-2 py-0.5 rounded-full bg-blue-100 text-blue-700 dark:bg-blue-900/40 dark:text-blue-300"
                  >
                    {ent}
                  </span>
                ))}
              </div>
            </div>
          )}

          {/* Aggregated ICD codes */}
          {context.unique_icd_codes.length > 0 && (
            <div>
              <h3 className="text-xs font-medium text-gray-500 dark:text-gray-400 mb-2 flex items-center gap-1">
                <FileCode className="h-3.5 w-3.5" />
                Unique ICD-10 Codes ({context.unique_icd_codes.length})
              </h3>
              <div className="flex flex-wrap gap-1.5">
                {context.unique_icd_codes.map((code) => (
                  <span
                    key={code}
                    className="text-xs px-2 py-0.5 rounded-full bg-purple-100 text-purple-700 dark:bg-purple-900/40 dark:text-purple-300 font-mono"
                  >
                    {code}
                  </span>
                ))}
              </div>
            </div>
          )}

          {/* Turn details */}
          <div>
            <h3 className="text-xs font-medium text-gray-500 dark:text-gray-400 mb-2">
              Recent Turns
            </h3>
            <div className="space-y-2">
              {context.turns.map((turn, i) => {
                const turnNum = (turn as Record<string, unknown>).turn as number;
                const isExpanded = expandedTurn === i;
                return (
                  <div
                    key={i}
                    className="border border-gray-200 dark:border-gray-600 rounded-lg"
                  >
                    <button
                      onClick={() => setExpandedTurn(isExpanded ? null : i)}
                      className="w-full flex items-center justify-between p-3 text-left"
                    >
                      <span className="text-sm font-medium text-gray-700 dark:text-gray-300">
                        Turn #{turnNum}
                      </span>
                      <div className="flex items-center gap-2">
                        {(turn as Record<string, unknown>).entity_count != null && (
                          <span className="text-xs text-gray-400">
                            {String((turn as Record<string, unknown>).entity_count)} entities
                          </span>
                        )}
                        {isExpanded ? (
                          <ChevronUp className="h-4 w-4 text-gray-400" />
                        ) : (
                          <ChevronDown className="h-4 w-4 text-gray-400" />
                        )}
                      </div>
                    </button>
                    {isExpanded && (
                      <div className="px-3 pb-3 text-xs">
                        <pre className="bg-gray-50 dark:bg-gray-900 rounded p-2 overflow-auto max-h-48 text-gray-600 dark:text-gray-400">
                          {JSON.stringify(turn, null, 2)}
                        </pre>
                      </div>
                    )}
                  </div>
                );
              })}
            </div>
          </div>
        </div>
      )}

      {/* Empty state */}
      {context && context.turn_count === 0 && (
        <div className="bg-white dark:bg-gray-800 rounded-xl border border-gray-200 dark:border-gray-700 p-8 text-center">
          <MessageSquare className="h-12 w-12 text-gray-300 dark:text-gray-600 mx-auto mb-3" />
          <p className="text-gray-500 dark:text-gray-400">
            No conversation history for this session. Add a sample turn above to
            get started.
          </p>
        </div>
      )}

      {/* Active sessions list */}
      {sessions.length > 0 && (
        <div className="bg-white dark:bg-gray-800 rounded-xl border border-gray-200 dark:border-gray-700 p-4">
          <h2 className="text-sm font-semibold text-gray-700 dark:text-gray-300 mb-3">
            Active Sessions ({sessions.length})
          </h2>
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="text-left text-xs text-gray-500 dark:text-gray-400 border-b border-gray-200 dark:border-gray-600">
                  <th className="pb-2 pr-4">Session ID</th>
                  <th className="pb-2 pr-4">Turns</th>
                  <th className="pb-2 pr-4">Turn Range</th>
                  <th className="pb-2">Last Active</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-gray-100 dark:divide-gray-700">
                {sessions.map((sess) => (
                  <tr key={sess.session_id}>
                    <td className="py-2 pr-4">
                      <button
                        onClick={() => setSessionId(sess.session_id)}
                        className="font-mono text-indigo-600 dark:text-indigo-400 hover:underline"
                      >
                        {sess.session_id}
                      </button>
                    </td>
                    <td className="py-2 pr-4">{sess.turn_count}</td>
                    <td className="py-2 pr-4 text-gray-500 dark:text-gray-400">
                      {sess.oldest_turn_id != null
                        ? `#${sess.oldest_turn_id}–#${sess.newest_turn_id}`
                        : '—'}
                    </td>
                    <td className="py-2 text-gray-500 dark:text-gray-400">
                      {new Date(sess.last_access * 1000).toLocaleTimeString()}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  );
}
