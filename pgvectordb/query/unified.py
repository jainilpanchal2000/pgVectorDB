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
    Literal,
)

from ..base import KeywordSearchType, QueryResult, SearchMethod
from ._postprocessing import post_process_results, results_to_arrow, results_to_pandas

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
    filter: dict[str, Any] | None = None
    columns: list[str] | None = None

    # Vector search config
    ef: int | None = None
    nprobes: int | None = None
    refine_factor: int | None = None
    distance_range: tuple[float, float] | None = None
    bypass_vector_index: bool = False
    exact_search: bool = False

    # Filter strategy
    filter_strategy: Literal["pre", "post", "auto"] = "auto"
    fetch_multiplier: float = 2.0

    # Keyword search config
    keyword_type: KeywordSearchType = KeywordSearchType.BM25
    bm25_k1: float = 1.2
    bm25_b: float = 0.75
    text_config: str = "english"
    phrase_query: bool = False
    universal_fields: list[str] | None = None

    # Hybrid search config
    hybrid_mode: Literal["weighted", "rrf"] = "weighted"
    semantic_weight: float = 0.5
    keyword_weight: float = 0.5
    rrf_k: int = 60

    # Trigram search config
    trigram_threshold: float = 0.3
    case_sensitive: bool = False

    # Reranking config
    reranker: Any | None = None
    rerank_query: str | None = None

    # Multimodal Spaces config
    spaces: list[Any] | None = None
    space_weights: dict[str, float] | None = None
    active_space: str | None = None

    # DiskANN label filtering
    label_filter: list[int] | None = None

    def to_dict(self) -> dict[str, Any]:
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

    db: pgVectorDB = field(repr=False)

    # Query input
    query_text: str | None = None
    query_vector: list[float] | None = None

    # Search configuration
    _search_method: SearchMethod = SearchMethod.SEMANTIC
    _config: SearchConfig = field(default_factory=SearchConfig)

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

    def ensemble(self) -> UnifiedQueryBuilder:
        """Set search mode to ensemble (filtered hybrid) search.

        This is equivalent to hybrid search but requires a filter to be set
        via .where(). Performs both semantic and keyword search on the
        filtered subset and fuses results.

        Returns:
            Self for chaining

        Examples:
            results = await (
                db.query("machine learning")
                .ensemble()
                .where({"category": "ai"})
                .weights(0.7, 0.3)
                .limit(10)
                .to_list()
            )

        Note:
            Requires a filter to be set via .where(). Without a filter,
            this falls back to standard hybrid search.
        """
        self._search_method = SearchMethod.HYBRID
        # Note: ensemble is essentially hybrid with mandatory filter
        return self

    def trigram(self) -> UnifiedQueryBuilder:
        """Set search mode to trigram (fuzzy) search."""
        self._search_method = SearchMethod.TRIGRAM
        return self

    def metadata_only(self) -> UnifiedQueryBuilder:
        """Set search mode to pure metadata filtering (no text search).

        Requires a filter to be set via .where(). Query text is optional
        for this search mode.

        Examples:
            # Filter documents by metadata only
            results = await db.query("")
                .metadata_only()
                .where({"category": "ai", "status": "active"})
                .limit(10)
                .to_list()
        """
        self._search_method = SearchMethod.METADATA_FILTER
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
        self._config.active_space = getattr(space, "name", None)
        return self

    def across_spaces(
        self, spaces: list[Any], weights: dict[str, float] | None = None
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
                logger.warning(f"Space weights sum to {total:.2f}, should be approximately 1.0")

        return self

    def labels(self, label_ids: list[int]) -> UnifiedQueryBuilder:
        """Set DiskANN label filter for semantic search.

        Only applicable when using DiskANN index type. Filters results to
        only include documents with the specified labels.

        Args:
            label_ids: List of label IDs to filter by

        Returns:
            Self for chaining

        Examples:
            # Filter to documents with labels 1 or 2
            results = await (
                db.query("machine learning")
                .labels([1, 2])
                .limit(10)
                .to_list()
            )

        Note:
            Only works with IndexType.DISKANN. Other index types will ignore this.
        """
        self._config.label_filter = label_ids
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

    def where(self, filter: dict[str, Any]) -> UnifiedQueryBuilder:
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

    def select(self, columns: list[str]) -> UnifiedQueryBuilder:
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
        """Force exact (brute force) vector search.

        Deprecated: Use exact_search() instead for clearer semantics.
        """
        self._config.bypass_vector_index = True
        self._config.exact_search = True
        return self

    def exact_search(self, exact: bool = True) -> UnifiedQueryBuilder:
        """Force exact (brute force) search bypassing ANN index.

        When exact=True:
        - HNSW: SET LOCAL hnsw.ef_search = 100000 (effectively exact)
        - IVFFlat: Skip list probes, use exact distance calculation
        - DiskANN: Use exact mode or very high ef

        Returns best possible recall at cost of latency.

        Args:
            exact: Whether to use exact search (default: True)

        Returns:
            Self for chaining

        Examples:
            # Force exact search for best recall
            results = await db.query("test").exact_search().limit(10).to_list()

            # Override previous exact setting
            results = await db.query("test").exact_search(False).limit(10).to_list()
        """
        self._config.exact_search = exact
        self._config.bypass_vector_index = exact
        return self

    def within_distance(self, radius: float) -> UnifiedQueryBuilder:
        """Filter to results within distance radius from query vector.

        Only applies to semantic/hybrid search. Uses distance operator
        in WHERE clause for pre-filtering when possible.

        Note: Distance metric depends on index type:
        - HNSW/IVFFlat: Cosine distance (0=same, 2=opposite)
        - Inner product indexes: Lower is better

        Args:
            radius: Maximum distance from query vector (inclusive)

        Returns:
            Self for chaining

        Examples:
            # Find similar documents with similarity >= 0.95 (cosine dist <= 0.05)
            results = await db.query("machine learning")
                .semantic()
                .within_distance(0.05)
                .limit(10)
                .to_list()

            # With metadata filter
            results = await db.query("database")
                .semantic()
                .where({"category": "tech"})
                .within_distance(0.1)
                .limit(10)
                .to_list()
        """
        self._config.distance_range = (0.0, radius)
        return self

    def pre_filter(self) -> UnifiedQueryBuilder:
        """Force pre-filtering: apply metadata filter before vector search.

        Best for: Highly selective filters (few results expected)
        Trade-off: May reduce recall if filter excludes relevant vectors
        that were approximated out by the vector index.

        Returns:
            Self for chaining

        Examples:
            # Pre-filter for selective metadata
            results = await db.query("ml")
                .semantic()
                .pre_filter()
                .where({"category": "ai", "priority": {"$gt": 8}})
                .limit(10)
                .to_list()
        """
        self._config.filter_strategy = "pre"
        return self

    def post_filter(self) -> UnifiedQueryBuilder:
        """Force post-filtering: apply metadata filter after vector search.

        Best for: Non-selective filters (many results expected)
        Trade-off: Higher latency (fetches extra results), but perfect recall
        within the returned set.

        Returns:
            Self for chaining

        Examples:
            # Post-filter for common metadata values
            results = await db.query("technology")
                .semantic()
                .post_filter()
                .where({"status": "active"})  # Many active docs
                .limit(10)
                .to_list()
        """
        self._config.filter_strategy = "post"
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

    def universal(self, metadata_fields: list[str]) -> UnifiedQueryBuilder:
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

    def rerank(self, reranker: Any, query: str | None = None) -> UnifiedQueryBuilder:
        """Apply a reranker to results.

        Args:
            reranker: Callable scorer or object with rerank(query, documents, top_k)
            query: Optional query string for reranking
        """
        self._config.reranker = reranker
        self._config.rerank_query = query or self.query_text
        return self

    # ============================================================
    # Execution & Output
    # ============================================================

    async def to_list(self) -> list[QueryResult]:
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
            elif self._search_method == SearchMethod.METADATA_FILTER:
                results = await self._execute_metadata_filter(**args)
            else:
                raise ValueError(f"Unknown search method: {self._search_method}")

        # Apply post-processing
        results = self._apply_post_processing(results)

        return results

    async def to_pandas(self) -> pd.DataFrame:
        """Execute and return as pandas DataFrame."""
        results = await self.to_list()
        return results_to_pandas(results)

    async def to_arrow(self) -> pa.Table:
        """Execute and return as PyArrow Table."""
        results = await self.to_list()
        return results_to_arrow(results)

    # ============================================================
    # Query Analysis
    # ============================================================

    def explain_plan(self, verbose: bool = False) -> dict[str, Any]:
        """Generate query execution plan without executing.

        Uses PostgreSQL EXPLAIN (FORMAT JSON) to get structured plan.

        Args:
            verbose: Include verbose output

        Returns:
            Structured plan information with query details, index usage,
            estimated costs, and filter conditions.
        """
        self.db._ensure_initialized()

        # Build the query based on search method
        query_text = self.query_text or ""
        qualified_table = f'"{self.db.schema_name}"."{self.db.table_name}"'

        # Build filter clause if present
        where_clause = ""
        filter_info = None
        if self._config.filter:
            from ..query.filters import MetadataFilterCompiler

            filter_sql, _ = MetadataFilterCompiler().build(self._config.filter)
            where_clause = f"WHERE {filter_sql}"
            filter_info = self._config.filter

        # Build query info based on search method
        if self._search_method == SearchMethod.SEMANTIC:
            query_info = {
                "search_method": "semantic",
                "query": query_text,
                "filter": filter_info,
                "limit": self._config.limit,
                "index_type": self.db.index_type.value,
                "ef_search": self._config.ef,
                "nprobes": self._config.nprobes,
                "bypass_index": self._config.bypass_vector_index,
                "sql_preview": f"SELECT ... FROM {qualified_table} {where_clause} ORDER BY embedding <=> $1 LIMIT {self._config.limit}",
            }
        elif self._search_method == SearchMethod.KEYWORD:
            query_info = {
                "search_method": "keyword",
                "query": query_text,
                "keyword_type": self._config.keyword_type.value,
                "filter": filter_info,
                "limit": self._config.limit,
                "bm25_k1": self._config.bm25_k1
                if self._config.keyword_type.value == "bm25"
                else None,
                "bm25_b": self._config.bm25_b
                if self._config.keyword_type.value == "bm25"
                else None,
                "sql_preview": f"SELECT ... FROM {qualified_table} {where_clause} ORDER BY ts_rank(...) DESC LIMIT {self._config.limit}",
            }
        elif self._search_method == SearchMethod.HYBRID:
            query_info = {
                "search_method": "hybrid",
                "query": query_text,
                "hybrid_mode": self._config.hybrid_mode,
                "semantic_weight": self._config.semantic_weight,
                "keyword_weight": self._config.keyword_weight,
                "rrf_k": self._config.rrf_k if self._config.hybrid_mode == "rrf" else None,
                "filter": filter_info,
                "limit": self._config.limit,
                "note": "Hybrid search executes both semantic and keyword queries in parallel",
            }
        elif self._search_method == SearchMethod.TRIGRAM:
            query_info = {
                "search_method": "trigram",
                "query": query_text,
                "threshold": self._config.trigram_threshold,
                "filter": filter_info,
                "limit": self._config.limit,
            }
        elif self._search_method == SearchMethod.METADATA_FILTER:
            query_info = {
                "search_method": "metadata_filter",
                "filter": filter_info,
                "limit": self._config.limit,
                "note": "Pure metadata filtering without text search",
            }
        else:
            query_info = {"plan": f"{self._search_method.value} search"}

        # Check for multimodal spaces
        if self._config.spaces:
            space_info = []
            for space in self._config.spaces:
                space_name = getattr(space, "name", getattr(space, "field", "unknown"))
                space_type = type(space).__name__
                space_info.append({"name": space_name, "type": space_type})
            query_info["spaces"] = space_info
            query_info["space_weights"] = self._config.space_weights

        return query_info

    async def analyze_plan(self) -> dict[str, Any]:
        """Execute with EXPLAIN ANALYZE and return metrics.

        Uses PostgreSQL EXPLAIN (ANALYZE, BUFFERS, FORMAT JSON, TIMING) to get
        actual execution statistics including timing and I/O.

        Returns:
            Dictionary with actual PostgreSQL execution metrics:
            - execution_time_ms: Actual query execution time from PostgreSQL
            - planning_time_ms: Query planning time
            - rows_returned: Number of rows returned
            - rows_scanned: Number of rows scanned
            - total_cost: Estimated total cost
            - index_used: Which index was used
            - shared_hit_blocks: Shared buffer hits
            - shared_read_blocks: Shared buffer reads
        """
        self.db._ensure_initialized()

        # Build the query based on search method
        query_text = self.query_text or ""
        embedding: list[float] | None = None

        if self._search_method in (SearchMethod.SEMANTIC, SearchMethod.HYBRID):
            if query_text:
                embedding = self.db.embedding_model.embed_query(query_text)

        # Build base query
        qualified_table = f'"{self.db.schema_name}"."{self.db.table_name}"'

        # Build filter clause if present
        where_clause = ""
        params: dict[str, Any] = {}
        if self._config.filter:
            from ..query.filters import MetadataFilterCompiler

            filter_sql, filter_params = MetadataFilterCompiler().build(self._config.filter)
            where_clause = f"WHERE {filter_sql}"
            params.update(filter_params)

        # Build query based on search method
        if self._search_method == SearchMethod.SEMANTIC and embedding is not None:
            query_sql = f"""
                SELECT "langchain_id", "content", "langchain_metadata",
                       "embedding" <=> :embedding AS distance
                FROM {qualified_table}
                {where_clause}
                ORDER BY distance
                LIMIT :k
            """
            params["embedding"] = str(embedding)
            params["k"] = self._config.limit
        elif self._search_method == SearchMethod.KEYWORD:
            query_sql = f"""
                SELECT "langchain_id", "content", "langchain_metadata",
                       ts_rank(content_tsvector, plainto_tsquery('english', :query)) as rank
                FROM {qualified_table}
                {where_clause}
                {"AND" if where_clause else "WHERE"} content_tsvector @@ plainto_tsquery('english', :query)
                ORDER BY rank DESC
                LIMIT :k
            """
            params["query"] = query_text
            params["k"] = self._config.limit
        else:
            # For other methods, fall back to timing to_list()
            import time

            start = time.time()
            results = await self.to_list()
            elapsed = (time.time() - start) * 1000
            return {
                "execution_time_ms": elapsed,
                "rows_returned": len(results),
                "config": self._config.to_dict(),
                "search_method": self._search_method.value,
                "note": "EXPLAIN ANALYZE not implemented for this search method",
            }

        # Run EXPLAIN ANALYZE
        explain_query = f"EXPLAIN (ANALYZE, BUFFERS, FORMAT JSON, TIMING) {query_sql}"

        try:
            from sqlalchemy import text

            async with self.db.sqlalchemy_engine.connect() as conn:
                # Apply query parameters (ef, nprobes, etc.)
                self._sync_search_config_to_query_params()

                # Apply stored query params
                if hasattr(self.db, "_apply_query_params"):
                    await self.db._apply_query_params(conn)

                # Bypass index if requested
                if self._config.bypass_vector_index:
                    await conn.execute(text("SET LOCAL enable_indexscan = off"))

                # Execute EXPLAIN ANALYZE
                result = await conn.execute(text(explain_query), params)
                rows = result.fetchall()

                # Parse the JSON output
                if rows:
                    plan_json = rows[0][0]
                    if isinstance(plan_json, str):
                        import json

                        plan_data = json.loads(plan_json)
                    else:
                        plan_data = plan_json

                    # Extract key metrics
                    plan = plan_data[0] if isinstance(plan_data, list) else plan_data
                    plan_info = plan.get("Plan", {})

                    return {
                        "plan": plan_data,
                        "execution_time_ms": plan.get("Execution Time", 0.0),
                        "planning_time_ms": plan.get("Planning Time", 0.0),
                        "rows_returned": plan_info.get("Actual Rows", 0),
                        "rows_scanned": plan_info.get("Actual Rows", 0),
                        "total_cost": plan_info.get("Total Cost", 0.0),
                        "index_used": self.db.index_type.value,
                        "shared_hit_blocks": plan.get("Shared Hit Blocks", 0),
                        "shared_read_blocks": plan.get("Shared Read Blocks", 0),
                        "search_method": self._search_method.value,
                        "config": self._config.to_dict(),
                    }
                else:
                    return {
                        "plan": None,
                        "error": "No plan returned",
                        "search_method": self._search_method.value,
                    }

        except Exception as e:
            logger.warning(f"Failed to run EXPLAIN ANALYZE: {e}")
            # Fall back to timing to_list()
            import time

            start = time.time()
            results = await self.to_list()
            elapsed = (time.time() - start) * 1000
            return {
                "execution_time_ms": elapsed,
                "rows_returned": len(results),
                "config": self._config.to_dict(),
                "search_method": self._search_method.value,
                "error": f"EXPLAIN ANALYZE failed: {e}",
            }

    # ============================================================
    # Internal Methods
    # ============================================================

    def _ensure_executable(self) -> None:
        """Ensure query can be executed."""
        # METADATA_FILTER does not require query_text or query_vector
        if self._search_method == SearchMethod.METADATA_FILTER:
            if not self._config.filter:
                raise ValueError(
                    "METADATA_FILTER search method requires a filter. Use .where() to specify filter criteria."
                )
            self.db._ensure_initialized()
            return

        if not self.query_text and not self.query_vector:
            raise ValueError("Either query_text or query_vector is required")
        self.db._ensure_initialized()

    def _build_execution_args(self) -> dict[str, Any]:
        """Build execution arguments for the search method."""
        args: dict[str, Any] = {
            "k": self._config.limit + self._config.offset,
        }

        if self._config.filter:
            args["filter"] = self._config.filter

        return args

    def _sync_search_config_to_query_params(self) -> None:
        """Sync SearchConfig parameters to db._query_params for connection-level settings.

        This ensures ef, nprobes, and other index-specific parameters are applied
        before executing the search.
        """
        from ..base import IndexType

        # Apply ef for HNSW
        if self._config.ef is not None and self.db.index_type == IndexType.HNSW:
            self.db._query_params["hnsw.ef_search"] = self._config.ef

        # Apply nprobes for IVFFlat
        if self._config.nprobes is not None and self.db.index_type == IndexType.IVFFLAT:
            self.db._query_params["ivfflat.probes"] = self._config.nprobes

    async def _execute_semantic(self, **args) -> list[QueryResult]:
        """Execute semantic search."""
        query = self.query_text or ""

        # Sync SearchConfig params to db._query_params
        self._sync_search_config_to_query_params()

        # Calculate k with refine_factor if specified
        k = args.get("k", 10)
        if self._config.refine_factor is not None and self._config.refine_factor > 1:
            k = k * self._config.refine_factor

        # Use appropriate method based on filter
        if self._config.filter:
            results = await self.db.metadata_semantic_search(
                query=query,
                k=k,
                filter=self._config.filter,
                use_exact_search=self._config.bypass_vector_index,
            )
        else:
            results = await self.db.semantic_search(
                query=query,
                k=k,
                label_filter=self._config.label_filter,
                use_exact_search=self._config.bypass_vector_index,
            )

        return results

    async def _execute_keyword(self, **args) -> list[QueryResult]:
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

    async def _execute_hybrid(self, **args) -> list[QueryResult]:
        """Execute hybrid search."""
        query = self.query_text or ""

        results = await self.db.hybrid_search(
            query=query,
            k=args.get("k", 10),
            weights=(self._config.semantic_weight, self._config.keyword_weight),
            label_filter=self._config.label_filter,
            use_rrf=(self._config.hybrid_mode == "rrf"),
            rrf_k=self._config.rrf_k,
            keyword_type=self._config.keyword_type,
            bm25_k1=self._config.bm25_k1,
            bm25_b=self._config.bm25_b,
            text_config=self._config.text_config,
            filter=args.get("filter"),
        )

        return results

    async def _execute_trigram(self, **args) -> list[QueryResult]:
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

    async def _execute_metadata_filter(self, **args) -> list[QueryResult]:
        """Execute pure metadata filtering without text search."""
        if not self._config.filter:
            raise ValueError(
                "METADATA_FILTER search method requires a filter. Use .where() to specify filter criteria."
            )

        results = await self.db.metadata_filter(
            filter=self._config.filter,
            k=args.get("k", 10),
        )

        return results

    async def _execute_multimodal(self) -> list[QueryResult]:
        """Execute multimodal search across configured spaces.

        Uses the pgVectorDB multimodal search capabilities when spaces are configured.
        Falls back to semantic search if multimodal is not available.
        """
        if not self._config.spaces:
            raise ValueError("No spaces configured for multimodal search")

        # Check if db has multimodal search capability
        if not hasattr(self.db, "multimodal_search"):
            logger.warning(
                "Multimodal search not available on this database instance. "
                "Falling back to standard semantic search."
            )
            # Fall back to semantic search
            return await self._execute_semantic(k=self._config.limit)

        query_params = {}

        # If single space (in_space), use query text for that space
        if len(self._config.spaces) == 1 and self.query_text:
            space = self._config.spaces[0]
            space_name = getattr(space, "name", getattr(space, "field", "default"))
            query_params[space_name] = self.query_text
        elif self.query_text:
            # For multiple spaces, try to use query for text spaces
            for space in self._config.spaces:
                from ..spaces import TextSpace

                if isinstance(space, TextSpace) or hasattr(space, "embed_query"):
                    space_name = getattr(space, "name", getattr(space, "field", "default"))
                    query_params[space_name] = self.query_text
                    break

        if not query_params:
            raise ValueError("Could not determine query parameters for multimodal search")

        return await self.db.multimodal_search(
            query_params=query_params,
            weights=self._config.space_weights,
            k=self._config.limit + self._config.offset,
        )

    def _apply_post_processing(self, results: list[QueryResult]) -> list[QueryResult]:
        """Apply offset, limit, column selection, and reranking."""
        return post_process_results(
            results,
            offset=self._config.offset,
            limit=self._config.limit,
            columns=self._config.columns,
            reranker=self._config.reranker,
            rerank_query=self._config.rerank_query or self.query_text or "",
        )

    def _build_semantic_explain(self, verbose: bool) -> dict[str, Any]:
        """Build explain plan for semantic search."""
        return {
            "search_method": "semantic",
            "query": self.query_text,
            "filter": self._config.filter,
            "limit": self._config.limit,
            "index_type": self.db.index_type.value if hasattr(self.db, "index_type") else "unknown",
        }

    def _build_keyword_explain(self, verbose: bool) -> dict[str, Any]:
        """Build explain plan for keyword search."""
        return {
            "search_method": "keyword",
            "query": self.query_text,
            "type": self._config.keyword_type.value,
            "filter": self._config.filter,
            "limit": self._config.limit,
        }

    def _build_multimodal_explain(self, verbose: bool) -> dict[str, Any]:
        """Build explain plan for multimodal search."""
        space_info = []
        for space in self._config.spaces or []:
            space_name = getattr(space, "name", getattr(space, "field", "unknown"))
            space_type = type(space).__name__
            space_info.append({"name": space_name, "type": space_type})

        return {
            "search_method": "multimodal",
            "query": self.query_text,
            "spaces": space_info,
            "weights": self._config.space_weights,
            "limit": self._config.limit,
        }
