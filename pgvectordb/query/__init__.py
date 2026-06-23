"""
Query module - LanceDB-style query builders

Provides fluent/chainable query API for pgVectorDB:
- VectorQueryBuilder: Vector similarity search
- FTSQueryBuilder: Full-text search
- HybridQueryBuilder: Combined vector + FTS with fusion
"""

from .builder import FTSQueryBuilder, HybridQueryBuilder, VectorQueryBuilder
from .builders import (
    HybridQueryBuilder as NewHybridQueryBuilder,
)
from .builders import (
    KeywordQueryBuilder,
    SemanticQueryBuilder,
    TrigramQueryBuilder,
)
from .builders import (
    VectorQueryBuilder as PureVectorQueryBuilder,
)
from .unified import SearchConfig, UnifiedQueryBuilder

__all__ = [
    # Legacy builders (from .builder - backward compatibility)
    "VectorQueryBuilder",
    "FTSQueryBuilder",
    "HybridQueryBuilder",
    # New extended builders (from .builders - v0.0.6)
    "SemanticQueryBuilder",
    "KeywordQueryBuilder",
    "TrigramQueryBuilder",
    "NewHybridQueryBuilder",
    "PureVectorQueryBuilder",
    # Unified builder (from .unified - v0.0.6)
    "UnifiedQueryBuilder",
    "SearchConfig",
]
