"""
Unified Query Builder - Single Entry Point for All Search Methods
================================================================

This module implements a unified query builder that supports all search methods
through a consistent API:

    await db.query("machine learning").search_mode(SearchMethod.SEMANTIC).limit(10).to_list()
    await db.query("machine learning").search_mode(SearchMethod.KEYWORD).bm25().limit(10).to_list()
    await db.query("machine learning").search_mode(SearchMethod.HYBRID).weights(0.7, 0.3).limit(10).to_list()

All search methods share common configuration:
- .where() for metadata filtering
- .limit() / .offset() for pagination
- .explain_plan() / .analyze_plan() for diagnostics
- .to_list() / .to_pandas() / .to_arrow() for output formats

Search-specific configs are available via:
- .search_config() - Generic config dict
- .bm25() - BM25-specific parameters
- .rrf() - RRF fusion parameters
- .ef() - HNSW parameters
- etc.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    List,
    Literal,
    Optional,
    Tuple,
    cast,
)

from ..base import KeywordSearchType, QueryResult, SearchMethod

if TYPE_CHECKING:
    import pandas as pd
    import pyarrow as pa

    from ..core import pgVectorDB


logger = logging.getLogger(__name__)


@dataclass
class SearchConfig:
    """Configuration container for search-specific parameters.

    Each search method can have its own config that gets validated
    and passed to the underlying implementation.
    """

    # Common config
    limit: int = 10
    offset: int = 0
    filter: Optional[Dict[str, Any]] = None
    columns: Optional[List[str]] = None

    # Vector search config
    ef: Optional[int] = None
    nprobes: Optional[int] = None
    refine_factor: Optional[int] = None
    distance_range: Optional[Tuple[float, float]] = None
    bypass_vector_index: bool = False

    # Keyword search config
    keyword_type: KeywordSearchType = KeywordSearchType.BM25
    bm25_k1: float = 1.2
    bm25_b: float = 0.75
    text_config: str = "english"
    phrase_query: bool = False
    universal_fields: Optional[List[str]] = None

    # Hybrid search config
    hybrid_mode: Literal["weighted", "rrf"] = "weighted"
    semantic_weight: float = 0.5
    keyword_weight: float = 0.5
    rrf_k: int = 60

    # Trigram search config
    trigram_threshold: float = 0.3
    case_sensitive: bool = False

    # Reranking config
    reranker: Optional[Callable] = None
    rerank_query: Optional[str] = None

    # Multimodal Spaces config
    spaces: Optional[List[Any]] = None
    space_weights: Optional[Dict[str, float]] = None
    active_space: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary, excluding None values."""
        return {k: v for k, v in self.__dict__.items() if v is not None}


@dataclass
class UnifiedQueryBuilder:
    """Unified query builder for all search methods.

    This is the primary query builder for pgVectorDB v0.0.6+.
    Use db.query("...") as the entry point.

    Examples:
        # Semantic search (default)
        results = await db.query("machine learning").limit(10).to_list()

        # Keyword search with BM25
        results = await (
            db.query("machine learning")
            .search_mode(SearchMethod.KEYWORD)
            .bm25_params(k1=1.2, b=0.75)
            .limit(10)
            .to_list()
        )

        # Hybrid with RRF
        results = await (
            db.query("machine learning")
            .search_mode(SearchMethod.HYBRID)
            .rrf(k=60)
            .limit(10)
            .to_list()
        )

        # Filtered search
        results = await (
            db.query("machine learning")
            .where({"category": "ai", "year": {"$gte": 2024}})
            .limit(10)
            .to_list()
        )

        # With query analysis
        metrics = await (
            db.query("machine learning")
            .where({"category": "ai"})
            .analyze_plan()
        )
    """

    db: "pgVectorDB" = field(repr=False)

    # Query input
    query_text: Optional[str] = None
    query_vector: Optional[List[float]] = None

    # Search configuration
    _search_method: SearchMethod = SearchMethod.SEMANTIC
    _config: SearchConfig = field(default_factory=SearchConfig)

    # Internal state
    _executed: bool = False
    _results: Optional[List[QueryResult]] = None

    def __post_init__(self):
        """Initialize config if not provided."""
        if self._config is None:
            self._config = SearchConfig()

    # ============================================================
    # Search Mode Selection
    # ============================================================

    def search_mode(self, method: SearchMethod) -> UnifiedQueryBuilder:
        """Set the search method.

        Args:
            method: One of SearchMethod.SEMANTIC, KEYWORD, HYBRID, TRIGRAM

        Returns:
            Self for chaining

        Examples:
            db.query("test").search_mode(SearchMethod.KEYWORD).limit(10).to_list()
            db.query("test").search_mode(SearchMethod.HYBRID).rrf().limit(10).to_list()
        """
        self._search_method = method
        return self

    def semantic(self) -> UnifiedQueryBuilder:
        """Set search mode to semantic (vector) search."""
        self._search_method = SearchMethod.SEMANTIC
        return self

    def keyword(self) -> UnifiedQueryBuilder:
        """Set search mode to keyword search."""
        self._search_method = SearchMethod.KEYWORD
        return self

    def hybrid(self) -> UnifiedQueryBuilder:
        """Set search mode to hybrid search."""
        self._search_method = SearchMethod.HYBRID
        return self

    def trigram(self) -> UnifiedQueryBuilder:
        """Set search mode to trigram (fuzzy) search."""
        self._search_method = SearchMethod.TRIGRAM
        return self

    # ============================================================
    # Multimodal Spaces Configuration
    # ============================================================

    def in_space(self, space: Any) -> UnifiedQueryBuilder:
        """Search within a single vector space.

        Args:
            space: VectorSpace instance (TextSpace, NumberSpace, CategorySpace, RecencySpace)

        Returns:
            Self for chaining

        Examples:
            from pgvectordb.spaces import TextSpace

            text_space = TextSpace(name="content", field="content")
            results = await (
                db.query("machine learning")
                .in_space(text_space)
                .limit(10)
                .to_list()
            )
        """
        self._config.spaces = [space]
        self._config.active_space = getattr(space, 'name', None)
        return self

    def across_spaces(
        self,
        spaces: List[Any],
        weights: Optional[Dict[str, float]] = None
    ) -> UnifiedQueryBuilder:
        """Search across multiple vector spaces with weighted fusion.

        Enables multimodal search combining text, numeric, categorical, and
        recency signals into a single ranking.

        Args:
            spaces: List of VectorSpace instances (TextSpace, NumberSpace, etc.)
            weights: Optional dict mapping space names to weights (must sum to 1.0)

        Returns:
            Self for chaining

        Examples:
            from pgvectordb.spaces import TextSpace, NumberSpace, CategorySpace

            text_space = TextSpace(name="description", field="content")
            price_space = NumberSpace(name="price", field="price",
                                       min_value=0, max_value=100000)
            category_space = CategorySpace(name="category", field="category",
                                          categories=["electronics", "books", "clothing"])

            results = await (
                db.query("laptop for gaming")
                .across_spaces(
                    [text_space, price_space, category_space],
                    weights={"description": 0.6, "price": 0.2, "category": 0.2}
                )
                .limit(10)
                .to_list()
            )

        Note:
            This requires the database to be initialized with multimodal support
            and documents to be added using add_documents_multimodal().
        """
        self._config.spaces = spaces
        self._config.space_weights = weights

        # Validate weights sum to approximately 1.0 if provided
        if weights is not None:
            total = sum(weights.values())
            if not (0.99 <= total <= 1.01):
                logger.warning(
                    f"Space weights sum to {total:.2f}, should be approximately 1.0"
                )

        return self

    # ============================================================
    # Common Configuration
    # ============================================================

    def limit(self, n: int) -> UnifiedQueryBuilder:
        """Set maximum number of results."""
        self._config.limit = n
        return self

    def offset(self, n: int) -> UnifiedQueryBuilder:
        """Set result offset for pagination."""
        self._config.offset = n
        return self

    def where(self, filter: Dict[str, Any]) -> UnifiedQueryBuilder:
        """Apply metadata filter using MongoDB-style syntax.

        Args:
            filter: Dict with operators ($eq, $gt, $in, $between, etc.)

        Examples:
            builder.where({"category": "ai"})
            builder.where({"year": {"$gte": 2024}})
            builder.where({"$and": [{"status": "active"}, {"priority": {"$gt": 5}}]})
        """
        self._config.filter = filter
        return self

    def select(self, columns: List[str]) -> UnifiedQueryBuilder:
        """Select specific columns to return."""
        self._config.columns = columns
        return self

    # ============================================================
    # Search-Specific Configuration
    # ============================================================

    def search_config(self, **kwargs) -> UnifiedQueryBuilder:
        """Set arbitrary search configuration parameters.

        Args:
            **kwargs: Configuration parameters that will be validated
                     and passed to the underlying search method

        Examples:
            builder.search_mode(SearchMethod.KEYWORD).search_config(k1=1.5, b=0.75)
            builder.search_config(use_exact_search=True, rrf_k=60)
        """
        for key, value in kwargs.items():
            if hasattr(self._config, key):
                setattr(self._config, key, value)
            else:
                logger.warning(f"Unknown search config parameter: {key}")
        return self

    # --------------------------------------------------------
    # Vector Search Parameters
    # --------------------------------------------------------

    def ef(self, n: int) -> UnifiedQueryBuilder:
        """Set HNSW ef_search parameter.

        Higher values improve recall at cost of latency.
        """
        self._config.ef = n
        return self

    def nprobes(self, n: int) -> UnifiedQueryBuilder:
        """Set IVF probes parameter."""
        self._config.nprobes = n
        return self

    def refine_factor(self, factor: int) -> UnifiedQueryBuilder:
        """Set refinement oversampling factor."""
        self._config.refine_factor = factor
        return self

    def distance_range(self, lower: float, upper: float) -> UnifiedQueryBuilder:
        """Filter by distance range."""
        self._config.distance_range = (lower, upper)
        return self

    def bypass_vector_index(self) -> UnifiedQueryBuilder:
        """Force exact (brute force) vector search."""
        self._config.bypass_vector_index = True
        return self

    # --------------------------------------------------------
    # Keyword Search Parameters
    # --------------------------------------------------------

    def bm25_params(self, k1: float = 1.2, b: float = 0.75) -> UnifiedQueryBuilder:
        """Set BM25 parameters for keyword search.

        Args:
            k1: Term frequency saturation (default: 1.2)
            b: Length normalization (default: 0.75)
        """
        self._config.keyword_type = KeywordSearchType.BM25
        self._config.bm25_k1 = k1
        self._config.bm25_b = b
        return self

    def fts(self, text_config: str = "english") -> UnifiedQueryBuilder:
        """Use PostgreSQL FTS (Full-Text Search)."""
        self._config.keyword_type = KeywordSearchType.FTS
        self._config.text_config = text_config
        return self

    def bm25(self) -> UnifiedQueryBuilder:
        """Use BM25 ranking (same as bm25_params with defaults)."""
        self._config.keyword_type = KeywordSearchType.BM25
        return self

    def universal(self, metadata_fields: List[str]) -> UnifiedQueryBuilder:
        """Enable universal keyword search with metadata boosting.

        Args:
            metadata_fields: Fields to check for query terms (e.g., ["title", "tags"])
        """
        self._config.universal_fields = metadata_fields
        return self

    def phrase(self, enabled: bool = True) -> UnifiedQueryBuilder:
        """Require exact phrase matching for keyword search."""
        self._config.phrase_query = enabled
        return self

    # --------------------------------------------------------
    # Hybrid Search Parameters
    # --------------------------------------------------------

    def weights(self, semantic: float, keyword: float) -> UnifiedQueryBuilder:
        """Set hybrid fusion weights.

        Args:
            semantic: Weight for semantic search (0.0-1.0)
            keyword: Weight for keyword search (0.0-1.0)
        """
        self._config.hybrid_mode = "weighted"
        self._config.semantic_weight = semantic
        self._config.keyword_weight = keyword
        return self

    def rrf(self, k: int = 60) -> UnifiedQueryBuilder:
        """Use Reciprocal Rank Fusion.

        Args:
            k: RRF constant (default: 60)
        """
        self._config.hybrid_mode = "rrf"
        self._config.rrf_k = k
        return self

    # --------------------------------------------------------
    # Trigram Search Parameters
    # --------------------------------------------------------

    def threshold(self, min_similarity: float) -> UnifiedQueryBuilder:
        """Set minimum trigram similarity threshold.

        Args:
            min_similarity: 0.0 to 1.0 (higher = stricter matching)
        """
        self._config.trigram_threshold = min_similarity
        return self

    def case_sensitive(self, enabled: bool = True) -> UnifiedQueryBuilder:
        """Enable case-sensitive trigram matching."""
        self._config.case_sensitive = enabled
        return self

    # ============================================================
    # Reranking
    # ============================================================

    def rerank(self, reranker: Callable, query: Optional[str] = None) -> UnifiedQueryBuilder:
        """Apply a reranker to results.

        Args:
            reranker: Callable that takes (query, [texts]) and returns scores
            query: Optional query string for reranking
        """
        self._config.reranker = reranker
        self._config.rerank_query = query or self.query_text
        return self

    # ============================================================
    # Execution & Output
    # ============================================================

    async def to_list(self) -> List[QueryResult]:
        """Execute the query and return results as a list."""
        self._ensure_executable()

        # Check for multimodal spaces search
        if self._config.spaces:
            results = await self._execute_multimodal()
        else:
            # Build execution arguments
            args = self._build_execution_args()

            # Execute based on search method
            if self._search_method == SearchMethod.SEMANTIC:
                results = await self._execute_semantic(**args)
            elif self._search_method == SearchMethod.KEYWORD:
                results = await self._execute_keyword(**args)
            elif self._search_method == SearchMethod.HYBRID:
                results = await self._execute_hybrid(**args)
            elif self._search_method == SearchMethod.TRIGRAM:
                results = await self._execute_trigram(**args)
            else:
                raise ValueError(f"Unknown search method: {self._search_method}")

        # Apply post-processing
        results = self._apply_post_processing(results)

        return results

    async def to_pandas(self) -> "pd.DataFrame":
        """Execute and return as pandas DataFrame."""
        import pandas as pd

        results = await self.to_list()
        return pd.DataFrame(results)

    async def to_arrow(self) -> "pa.Table":
        """Execute and return as PyArrow Table."""
        import pyarrow as pa

        results = await self.to_list()
        if not results:
            return pa.table({})
        return pa.Table.from_pylist(results)

    # ============================================================
    # Query Analysis
    # ============================================================

    def explain_plan(self, verbose: bool = False) -> Dict[str, Any]:
        """Generate query execution plan without running."""
        self.db._ensure_initialized()

        # Check for multimodal spaces search
        if self._config.spaces:
            return self._build_multimodal_explain(verbose)

        # Determine which method would be called
        if self._search_method == SearchMethod.SEMANTIC:
            return self._build_semantic_explain(verbose)
        elif self._search_method == SearchMethod.KEYWORD:
            return self._build_keyword_explain(verbose)
        elif self._search_method == SearchMethod.HYBRID:
            return {"plan": "Hybrid search runs two queries in parallel"}
        else:
            return {"plan": f"{self._search_method.value} search"}

    async def analyze_plan(self) -> Dict[str, Any]:
        """Execute with EXPLAIN ANALYZE and return metrics."""
        import time

        start = time.time()
        results = await self.to_list()
        elapsed = (time.time() - start) * 1000

        result = {
            "execution_time_ms": elapsed,
            "rows_returned": len(results),
            "config": self._config.to_dict(),
        }

        # Add multimodal info if applicable
        if self._config.spaces:
            result["search_method"] = "multimodal"
            result["num_spaces"] = len(self._config.spaces)
        else:
            result["search_method"] = self._search_method.value

        return result

    # ============================================================
    # Internal Methods
    # ============================================================

    def _ensure_executable(self) -> None:
        """Ensure query can be executed."""
        if not self.query_text and not self.query_vector:
            raise ValueError("Either query_text or query_vector is required")
        self.db._ensure_initialized()

    def _build_execution_args(self) -> Dict[str, Any]:
        """Build execution arguments for the search method."""
        args: Dict[str, Any] = {
            "k": self._config.limit + self._config.offset,
        }

        if self._config.filter:
            args["filter"] = self._config.filter

        return args

    async def _execute_semantic(self, **args) -> List[QueryResult]:
        """Execute semantic search."""
        query = self.query_text or ""

        # Use appropriate method based on filter
        if self._config.filter:
            results = await self.db.metadata_semantic_search(
                query=query,
                k=args.get("k", 10),
                filter=self._config.filter,
                use_exact_search=self._config.bypass_vector_index,
            )
        else:
            results = await self.db.semantic_search(
                query=query,
                k=args.get("k", 10),
                use_exact_search=self._config.bypass_vector_index,
            )

        return results

    async def _execute_keyword(self, **args) -> List[QueryResult]:
        """Execute keyword search."""
        query = self.query_text or ""

        # Wrap in quotes if phrase query
        if self._config.phrase_query and not (query.startswith('"') and query.endswith('"')):
            query = f'"{query}"'

        if self._config.universal_fields:
            results = await self.db.universal_keyword_search(
                query=query,
                k=args.get("k", 10),
                metadata_fields=self._config.universal_fields,
            )
        elif self._config.filter:
            results = await self.db.metadata_keyword_search(
                query=query,
                k=args.get("k", 10),
                filter=self._config.filter,
                search_type=self._config.keyword_type,
                k1=self._config.bm25_k1,
                b=self._config.bm25_b,
                text_config=self._config.text_config,
            )
        else:
            results = await self.db.keyword_search(
                query=query,
                k=args.get("k", 10),
                search_type=self._config.keyword_type,
                k1=self._config.bm25_k1,
                b=self._config.bm25_b,
                text_config=self._config.text_config,
            )

        return results

    async def _execute_hybrid(self, **args) -> List[QueryResult]:
        """Execute hybrid search."""
        query = self.query_text or ""

        results = await self.db.hybrid_search(
            query=query,
            k=args.get("k", 10),
            weights=(self._config.semantic_weight, self._config.keyword_weight),
            use_rrf=(self._config.hybrid_mode == "rrf"),
            rrf_k=self._config.rrf_k,
            keyword_type=self._config.keyword_type,
            bm25_k1=self._config.bm25_k1,
            bm25_b=self._config.bm25_b,
            text_config=self._config.text_config,
        )

        return results

    async def _execute_trigram(self, **args) -> List[QueryResult]:
        """Execute trigram search."""
        query = self.query_text or ""

        if self._config.filter:
            results = await self.db.metadata_trigram_search(
                query=query,
                k=args.get("k", 10),
                filter=self._config.filter,
                threshold=self._config.trigram_threshold,
            )
        else:
            results = await self.db.trigram_search(
                query=query,
                k=args.get("k", 10),
                threshold=self._config.trigram_threshold,
            )

        return results

    async def _execute_multimodal(self) -> List[QueryResult]:
        """Execute multimodal search across configured spaces.

        Uses the pgVectorDB multimodal search capabilities when spaces are configured.
        Falls back to semantic search if multimodal is not available.
        """
        if not self._config.spaces:
            raise ValueError("No spaces configured for multimodal search")

        # Check if db has multimodal search capability
        if not hasattr(self.db, 'multimodal_search'):
            logger.warning(
                "Multimodal search not available on this database instance. "
                "Falling back to standard semantic search."
            )
            # Fall back to semantic search
            return await self._execute_semantic(k=self._config.limit)

        try:
            # Build query parameters from spaces
            query_params = {}

            # If single space (in_space), use query text for that space
            if len(self._config.spaces) == 1 and self.query_text:
                space = self._config.spaces[0]
                space_name = getattr(space, 'name', getattr(space, 'field', 'default'))
                query_params[space_name] = self.query_text
            elif self.query_text:
                # For multiple spaces, try to use query for text spaces
                for space in self._config.spaces:
                    from ..spaces import TextSpace
                    if isinstance(space, TextSpace) or hasattr(space, 'embed_query'):
                        space_name = getattr(space, 'name', getattr(space, 'field', 'default'))
                        query_params[space_name] = self.query_text
                        break

            if not query_params:
                raise ValueError("Could not determine query parameters for multimodal search")

            # Execute multimodal search
            results = await self.db.multimodal_search(
                query_params=query_params,
                weights=self._config.space_weights,
                k=self._config.limit + self._config.offset,
            )

            return results

        except Exception as e:
            logger.warning(f"Multimodal search failed: {e}. Falling back to semantic search.")
            return await self._execute_semantic(k=self._config.limit)

    def _apply_post_processing(self, results: List[QueryResult]) -> List[QueryResult]:
        """Apply offset, limit, column selection, and reranking."""
        # Apply offset
        if self._config.offset > 0:
            results = results[self._config.offset :]

        # Apply limit
        results = results[: self._config.limit]

        # Apply reranking
        if self._config.reranker and results:
            query = self._config.rerank_query or self.query_text or ""
            texts = [r.get("content", "") for r in results]
            try:
                scores = self._config.reranker(query, texts)
                scored = list(zip(results, scores))
                scored.sort(key=lambda x: x[1], reverse=True)
                results = [r for r, _ in scored]
            except Exception as e:
                logger.warning(f"Reranking failed: {e}")

        # Apply column selection
        if self._config.columns:
            results = cast(List[QueryResult], [
                {k: v for k, v in r.items() if k in self._config.columns or k == "id"}
                for r in results
            ])

        return results

    def _build_semantic_explain(self, verbose: bool) -> Dict[str, Any]:
        """Build explain plan for semantic search."""
        return {
            "search_method": "semantic",
            "query": self.query_text,
            "filter": self._config.filter,
            "limit": self._config.limit,
            "index_type": self.db.index_type.value if hasattr(self.db, "index_type") else "unknown",
        }

    def _build_keyword_explain(self, verbose: bool) -> Dict[str, Any]:
        """Build explain plan for keyword search."""
        return {
            "search_method": "keyword",
            "query": self.query_text,
            "type": self._config.keyword_type.value,
            "filter": self._config.filter,
            "limit": self._config.limit,
        }

    def _build_multimodal_explain(self, verbose: bool) -> Dict[str, Any]:
        """Build explain plan for multimodal search."""
        space_info = []
        for space in (self._config.spaces or []):
            space_name = getattr(space, 'name', getattr(space, 'field', 'unknown'))
            space_type = type(space).__name__
            space_info.append({"name": space_name, "type": space_type})

        return {
            "search_method": "multimodal",
            "query": self.query_text,
            "spaces": space_info,
            "weights": self._config.space_weights,
            "limit": self._config.limit,
        }
        return {
            "search_method": "keyword",
            "query": self.query_text,
            "type": self._config.keyword_type.value,
            "filter": self._config.filter,
            "limit": self._config.limit,
        }
