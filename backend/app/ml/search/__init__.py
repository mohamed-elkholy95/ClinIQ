"""Hybrid document search module.

Provides BM25-based lexical search and TF-IDF-based semantic similarity
scoring for clinical documents, combined into a single hybrid ranking
function with tunable interpolation.
"""

from app.ml.search.hybrid import HybridSearchEngine, SearchResult

__all__ = ["HybridSearchEngine", "SearchResult"]
