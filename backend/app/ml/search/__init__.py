"""Clinical document search module.

Provides a multi-stage search pipeline for clinical documents:

1. **Hybrid retrieval** — BM25 lexical matching + TF-IDF cosine
   similarity for initial candidate generation.
2. **Query expansion** — Medical synonym and abbreviation expansion
   to improve recall (e.g. "htn" → "hypertension").
3. **Re-ranking** — Cross-encoder or rule-based re-scorer for
   precision refinement of the initial candidate set.
4. **Conversation memory** — Session-scoped context tracking for
   multi-turn clinical analysis workflows.
"""

from app.ml.search.conversation_memory import ConversationMemory, ConversationTurn, SessionContext
from app.ml.search.hybrid import HybridSearchEngine, SearchResult
from app.ml.search.query_expansion import ExpandedQuery, MedicalQueryExpander
from app.ml.search.reranker import (
                                               ClinicalRuleReRanker,
                                               ReRankCandidate,
                                               ReRankedResult,
                                               ReRanker,
                                               TransformerReRanker,
)

__all__ = [
    "HybridSearchEngine",
    "SearchResult",
    "MedicalQueryExpander",
    "ExpandedQuery",
    "ReRanker",
    "ClinicalRuleReRanker",
    "TransformerReRanker",
    "ReRankCandidate",
    "ReRankedResult",
    "ConversationMemory",
    "ConversationTurn",
    "SessionContext",
]
