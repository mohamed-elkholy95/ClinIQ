/**
 * SearchExplorer page — Hybrid clinical document search interface.
 *
 * Provides a full-featured search UI that leverages the hybrid retrieval
 * pipeline (BM25 + TF-IDF cosine), optional query expansion, and neural
 * reranking.  Users can tune alpha (lexical ↔ semantic balance), top-k,
 * min-score threshold, and toggle expansion/reranking independently.
 *
 * Design decisions:
 * - Advanced options are collapsed by default to keep the primary search
 *   experience simple for casual users.
 * - Results are rendered as cards with highlighted snippets, document IDs,
 *   and relevance scores to support clinical document review workflows.
 * - Query expansion terms are displayed when active so users understand
 *   how the system interpreted their query.
 */

import { useState, useCallback } from 'react';
import { Search, SlidersHorizontal, FileText, Sparkles, ArrowUpDown, ChevronDown, ChevronUp, Clock } from 'lucide-react';
import { searchDocuments } from '../services/clinical';
import type { SearchResponse, SearchHit } from '../types/clinical';

/** Score colour based on relevance threshold */
function scoreColor(score: number): string {
  if (score >= 0.8) return 'text-green-600 bg-green-50 dark:bg-green-900/30 dark:text-green-400';
  if (score >= 0.5) return 'text-blue-600 bg-blue-50 dark:bg-blue-900/30 dark:text-blue-400';
  if (score >= 0.3) return 'text-yellow-600 bg-yellow-50 dark:bg-yellow-900/30 dark:text-yellow-400';
  return 'text-gray-500 bg-gray-50 dark:bg-gray-800 dark:text-gray-400';
}

export function SearchExplorer() {
  const [query, setQuery] = useState('');
  const [results, setResults] = useState<SearchResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [showAdvanced, setShowAdvanced] = useState(false);

  // Advanced options
  const [topK, setTopK] = useState(10);
  const [minScore, setMinScore] = useState(0.1);
  const [alpha, setAlpha] = useState(0.5);
  const [expandQuery, setExpandQuery] = useState(true);
  const [rerank, setRerank] = useState(true);

  const handleSearch = useCallback(async () => {
    if (!query.trim()) return;
    setLoading(true);
    setError(null);
    try {
      const response = await searchDocuments(query, {
        top_k: topK,
        min_score: minScore,
        alpha,
        expand_query: expandQuery,
        rerank,
      });
      setResults(response);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Search failed');
      setResults(null);
    } finally {
      setLoading(false);
    }
  }, [query, topK, minScore, alpha, expandQuery, rerank]);

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter') handleSearch();
  };

  return (
    <div className="max-w-4xl mx-auto space-y-6">
      {/* Page header */}
      <div>
        <h1 className="text-2xl font-bold text-text-primary flex items-center gap-2">
          <Search className="w-7 h-7 text-primary-500" />
          Clinical Document Search
        </h1>
        <p className="mt-1 text-sm text-text-muted">
          Hybrid retrieval with BM25 lexical matching, TF-IDF cosine similarity,
          medical query expansion, and neural reranking.
        </p>
      </div>

      {/* Search input */}
      <div className="bg-surface rounded-xl border border-border p-4 space-y-3">
        <div className="flex gap-2">
          <div className="relative flex-1">
            <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-text-muted" />
            <input
              type="text"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              onKeyDown={handleKeyDown}
              placeholder="Search clinical documents... (e.g., 'chest pain with elevated troponin')"
              className="w-full pl-10 pr-4 py-2.5 rounded-lg border border-border bg-surface-dim text-text-primary placeholder:text-text-muted focus:outline-none focus:ring-2 focus:ring-primary-500 focus:border-transparent"
              aria-label="Search query"
            />
          </div>
          <button
            onClick={handleSearch}
            disabled={loading || !query.trim()}
            className="px-5 py-2.5 rounded-lg bg-primary-500 text-white font-medium hover:bg-primary-600 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
          >
            {loading ? 'Searching...' : 'Search'}
          </button>
        </div>

        {/* Advanced options toggle */}
        <button
          onClick={() => setShowAdvanced(!showAdvanced)}
          className="flex items-center gap-1.5 text-sm text-text-muted hover:text-text-primary transition-colors"
        >
          <SlidersHorizontal className="w-3.5 h-3.5" />
          Advanced Options
          {showAdvanced ? <ChevronUp className="w-3.5 h-3.5" /> : <ChevronDown className="w-3.5 h-3.5" />}
        </button>

        {/* Advanced options panel */}
        {showAdvanced && (
          <div className="grid grid-cols-2 md:grid-cols-3 gap-4 pt-2 border-t border-border" data-testid="advanced-options">
            <div>
              <label className="block text-xs font-medium text-text-muted mb-1">
                Top K Results
              </label>
              <input
                type="number"
                value={topK}
                onChange={(e) => setTopK(Number(e.target.value))}
                min={1}
                max={100}
                className="w-full px-3 py-1.5 rounded border border-border bg-surface-dim text-sm text-text-primary"
                aria-label="Top K"
              />
            </div>
            <div>
              <label className="block text-xs font-medium text-text-muted mb-1">
                Min Score ({minScore.toFixed(2)})
              </label>
              <input
                type="range"
                value={minScore}
                onChange={(e) => setMinScore(Number(e.target.value))}
                min={0}
                max={1}
                step={0.05}
                className="w-full"
                aria-label="Minimum score"
              />
            </div>
            <div>
              <label className="block text-xs font-medium text-text-muted mb-1">
                Alpha — Lexical ↔ Semantic ({alpha.toFixed(2)})
              </label>
              <input
                type="range"
                value={alpha}
                onChange={(e) => setAlpha(Number(e.target.value))}
                min={0}
                max={1}
                step={0.05}
                className="w-full"
                aria-label="Alpha balance"
              />
            </div>
            <div className="flex items-center gap-2">
              <input
                type="checkbox"
                id="expand-query"
                checked={expandQuery}
                onChange={(e) => setExpandQuery(e.target.checked)}
                className="rounded border-border"
              />
              <label htmlFor="expand-query" className="text-sm text-text-primary flex items-center gap-1">
                <Sparkles className="w-3.5 h-3.5 text-amber-500" />
                Query Expansion
              </label>
            </div>
            <div className="flex items-center gap-2">
              <input
                type="checkbox"
                id="rerank"
                checked={rerank}
                onChange={(e) => setRerank(e.target.checked)}
                className="rounded border-border"
              />
              <label htmlFor="rerank" className="text-sm text-text-primary flex items-center gap-1">
                <ArrowUpDown className="w-3.5 h-3.5 text-indigo-500" />
                Neural Reranking
              </label>
            </div>
          </div>
        )}
      </div>

      {/* Error display */}
      {error && (
        <div className="p-4 rounded-lg bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 text-red-700 dark:text-red-400 text-sm" role="alert">
          {error}
        </div>
      )}

      {/* Results */}
      {results && (
        <div className="space-y-4">
          {/* Results summary */}
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <span className="text-sm font-medium text-text-primary">
                {results.total} result{results.total !== 1 ? 's' : ''}
              </span>
              <span className="flex items-center gap-1 text-xs text-text-muted">
                <Clock className="w-3 h-3" />
                {results.processing_time_ms.toFixed(0)}ms
              </span>
              {results.reranked && (
                <span className="px-2 py-0.5 rounded-full bg-indigo-50 dark:bg-indigo-900/30 text-indigo-600 dark:text-indigo-400 text-xs font-medium">
                  Reranked
                </span>
              )}
            </div>
          </div>

          {/* Query expansion info */}
          {results.query_expansion && results.query_expansion.expanded_terms.length > 0 && (
            <div className="p-3 rounded-lg bg-amber-50 dark:bg-amber-900/20 border border-amber-200 dark:border-amber-800" data-testid="query-expansion">
              <div className="flex items-center gap-1.5 text-xs font-medium text-amber-700 dark:text-amber-400 mb-1.5">
                <Sparkles className="w-3.5 h-3.5" />
                Query Expanded
              </div>
              <div className="flex flex-wrap gap-1.5">
                <span className="text-xs text-amber-600 dark:text-amber-300 font-medium">
                  "{results.query_expansion.original_query}"
                </span>
                <span className="text-xs text-amber-500">→</span>
                {results.query_expansion.expanded_terms.map((term, i) => (
                  <span key={i} className="px-1.5 py-0.5 rounded bg-amber-100 dark:bg-amber-800/40 text-xs text-amber-700 dark:text-amber-300">
                    {term}
                  </span>
                ))}
              </div>
            </div>
          )}

          {/* Result cards */}
          {results.hits.length === 0 ? (
            <div className="text-center py-12 text-text-muted">
              <Search className="w-10 h-10 mx-auto mb-2 opacity-40" />
              <p>No documents matched your query.</p>
            </div>
          ) : (
            <div className="space-y-3">
              {results.hits.map((hit: SearchHit, index: number) => (
                <div
                  key={hit.document_id}
                  className="bg-surface rounded-lg border border-border p-4 hover:border-primary-300 dark:hover:border-primary-700 transition-colors"
                  data-testid="search-result"
                >
                  <div className="flex items-start justify-between gap-3">
                    <div className="flex-1 min-w-0">
                      <div className="flex items-center gap-2 mb-1">
                        <span className="text-xs text-text-muted font-mono">#{index + 1}</span>
                        <FileText className="w-4 h-4 text-text-muted flex-shrink-0" />
                        <span className="font-medium text-text-primary truncate">
                          {hit.title || hit.document_id}
                        </span>
                      </div>
                      <p className="text-sm text-text-secondary mt-1 line-clamp-3">
                        {hit.snippet}
                      </p>
                      <p className="text-xs text-text-muted mt-1.5 font-mono">
                        ID: {hit.document_id}
                      </p>
                    </div>
                    <span className={`flex-shrink-0 px-2 py-1 rounded text-xs font-semibold ${scoreColor(hit.score)}`}>
                      {(hit.score * 100).toFixed(1)}%
                    </span>
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      )}

      {/* Empty state */}
      {!results && !loading && !error && (
        <div className="text-center py-16 text-text-muted">
          <Search className="w-12 h-12 mx-auto mb-3 opacity-30" />
          <p className="text-lg font-medium">Search Clinical Documents</p>
          <p className="text-sm mt-1">
            Enter a clinical query to search across your document corpus.
          </p>
        </div>
      )}
    </div>
  );
}
