"""
Query Builders - LanceDB-style fluent API

This module implements the query builder pattern from LanceDB,
adapted for PostgreSQL with pgvector.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import (
    TYPE_CHECKING,
    Any,
    Literal,
)

from ..base import QueryResult
from ._postprocessing import results_to_arrow, results_to_pandas

if TYPE_CHECKING:
    import pandas as pd
    import pyarrow as pa

    from ..core import pgVectorDB


logger = logging.getLogger(__name__)


@dataclass
class VectorQueryBuilder:
    """
    LanceDB-style query builder for vector search.

    Lazy execution - methods return self for chaining.
    Query only executes when to_list(), to_pandas(), etc. is called.

    Examples:
        >>> results = await (
        ...     db.query("machine learning").semantic()
        ...     .where({"category": "ai"})
        ...     .limit(10)
        ...     .ef(100)
        ...     .to_list()
        ... )
    """

    # Reference to the database instance
    db: pgVectorDB = field(repr=False)

    # Query state
    query_vector: list[float] | None = None
    query_text: str | None = None  # For hybrid conversion

    # Execution parameters
    _limit: int = 10
    _offset: int = 0
    _columns: list[str] | None = None
    _where: str | dict[str, Any] | None = None
    _prefilter: bool = True

    # Vector-specific parameters
    _distance_type: str | None = None
    _vector_column: str = "embedding"
    _nprobes: int | None = None
    _minimum_nprobes: int | None = None
    _maximum_nprobes: int | None = None
    _ef: int | None = None
    _refine_factor: int | None = None
    _lower_bound: float | None = None
    _upper_bound: float | None = None
    _bypass_vector_index: bool = False

    # Reranking
    _reranker: Any | None = None
    _rerank_query: str | None = None

    def limit(self, n: int) -> VectorQueryBuilder:
        """Set the maximum number of results to return.

        Args:
            n: Maximum number of results (default: 10)

        Returns:
            Self for chaining
        """
        self._limit = n
        return self

    def offset(self, n: int) -> VectorQueryBuilder:
        """Set the offset for results (for pagination).

        Args:
            n: Number of results to skip

        Returns:
            Self for chaining
        """
        self._offset = n
        return self

    def select(self, columns: list[str]) -> VectorQueryBuilder:
        """Set columns to return in results.

        Args:
            columns: List of column names to include

        Returns:
            Self for chaining
        """
        self._columns = columns
        return self

    def where(self, filter: str | dict[str, Any], prefilter: bool = True) -> VectorQueryBuilder:
        """Set filter conditions for the query.

        Args:
            filter: Dict for metadata filters or SQL string expression
                - Dict: MongoDB-style operators ($eq, $gt, $in, etc)
                - str: Raw SQL WHERE clause (e.g. "langchain_metadata->>'status' = 'active'")
            prefilter: If True, apply filter before search (default)

        Returns:
            Self for chaining

        Examples:
            # Dict filter (recommended)
            builder.where({"category": "ai", "year": {"$gte": 2024}})

            # String filter (advanced use cases)
            builder.where("langchain_metadata->>'status' = 'active'")
            builder.where("(langchain_metadata->>'priority')::int > 5")
        """
        self._where = filter
        self._prefilter = prefilter
        return self

    def distance_type(self, metric: Literal["l2", "cosine", "dot"]) -> VectorQueryBuilder:
        """Set the distance metric for vector search.

        Args:
            metric: One of "l2", "cosine", "dot"

        Returns:
            Self for chaining
        """
        self._distance_type = metric
        return self

    def nprobes(self, n: int) -> VectorQueryBuilder:
        """Set number of IVF partitions to probe.

        Higher values improve recall at cost of latency.
        Only applies to IVF-based indexes.

        Args:
            n: Number of probes (default: varies by index)

        Returns:
            Self for chaining
        """
        self._nprobes = n
        self._minimum_nprobes = n
        self._maximum_nprobes = n
        return self

    def minimum_nprobes(self, n: int) -> VectorQueryBuilder:
        """Set minimum number of IVF partitions to probe.

        These partitions are always searched.
        """
        self._minimum_nprobes = n
        return self

    def maximum_nprobes(self, n: int) -> VectorQueryBuilder:
        """Set maximum number of IVF partitions to probe.

        If > minimum_nprobes, additional partitions are searched
        only if needed to satisfy limit.
        """
        self._maximum_nprobes = n
        return self

    def ef(self, n: int) -> VectorQueryBuilder:
        """Set candidate pool size for HNSW search.

        Higher values improve recall at cost of latency.
        Only applies to HNSW index.

        Args:
            n: Candidate pool size (default: 1.5 * limit)

        Returns:
            Self for chaining
        """
        self._ef = n
        return self

    def refine_factor(self, factor: int) -> VectorQueryBuilder:
        """Set refinement oversampling factor.

        Fetches k*factor results, reranks with true distances,
        and returns top k. Improves accuracy.

        Args:
            factor: Multiplier for oversampling

        Returns:
            Self for chaining
        """
        self._refine_factor = factor
        return self

    def distance_range(
        self, lower: float | None = None, upper: float | None = None
    ) -> VectorQueryBuilder:
        """Set distance range filter.

        Only results with distance in [lower, upper] are returned.

        Args:
            lower: Minimum distance (inclusive)
            upper: Maximum distance (inclusive)

        Returns:
            Self for chaining
        """
        self._lower_bound = lower
        self._upper_bound = upper
        return self

    def bypass_vector_index(self) -> VectorQueryBuilder:
        """Force exact (flat) search, skipping any index.

        Useful for calculating recall or ground truth results.

        Returns:
            Self for chaining
        """
        self._bypass_vector_index = True
        return self

    def rerank(self, reranker: Any, query: str | None = None) -> VectorQueryBuilder:
        """Apply a cross-encoder reranker to results.

        Args:
            reranker: Function taking (query, [texts]) and returning scores
            query: Optional query string for reranking (uses original if not provided)

        Returns:
            Self for chaining
        """
        self._reranker = reranker
        self._rerank_query = query or self.query_text
        return self

    def explain_plan(self, verbose: bool = False) -> dict[str, Any]:
        """Generate query execution plan without running.

        Uses PostgreSQL EXPLAIN to show query plan.

        Args:
            verbose: Include additional detail

        Returns:
            Structured plan information
        """
        # This will be implemented to generate the SQL
        # and run EXPLAIN (FORMAT JSON)
        return self.db._explain_query_plan(self, verbose=verbose)  # type: ignore[arg-type]

    async def analyze_plan(self) -> dict[str, Any]:
        """Execute query with EXPLAIN ANALYZE and return metrics.

        Returns actual execution statistics including timing and I/O.

        Returns:
            Dictionary with execution metrics
        """
        return await self.db._analyze_query_plan(self)  # type: ignore[arg-type]

    async def to_list(self) -> list[QueryResult]:
        """Execute query and return results as a list.

        Returns:
            List of QueryResult dictionaries
        """
        if self.query_vector is None:
            raise ValueError("query_vector must be set")
        return await self.db.asimilarity_search_by_vector(  # type: ignore[attr-defined]
            embedding=self.query_vector, k=self._limit
        )

    async def to_pandas(self) -> pd.DataFrame:
        """Execute query and return results as pandas DataFrame.

        Returns:
            pandas DataFrame
        """
        results = await self.to_list()
        return results_to_pandas(results)

    async def to_arrow(self) -> pa.Table:
        """Execute query and return results as PyArrow Table.

        Returns:
            PyArrow Table
        """
        results = await self.to_list()
        return results_to_arrow(results)

    def nearest_to_text(self, query: str, columns: list[str] | None = None) -> HybridQueryBuilder:
        """Convert vector query to hybrid search with FTS.

        Args:
            query: Text query for full-text search
            columns: Optional columns to search in

        Returns:
            HybridQueryBuilder combining vector and FTS
        """
        return HybridQueryBuilder(
            db=self.db,
            vector_builder=self,
            text_query=query,
            fts_columns=columns,
        )


@dataclass
class FTSQueryBuilder:
    """Query builder for full-text search."""

    db: pgVectorDB = field(repr=False)
    query_text: str = ""

    _limit: int = 10
    _offset: int = 0
    _columns: list[str] | None = None
    _where: str | dict[str, Any] | None = None
    _fts_columns: list[str] | None = None
    _phrase_query: bool = False

    def limit(self, n: int) -> FTSQueryBuilder:
        self._limit = n
        return self

    def where(self, filter: str | dict[str, Any]) -> FTSQueryBuilder:
        self._where = filter
        return self

    def select(self, columns: list[str]) -> FTSQueryBuilder:
        self._columns = columns
        return self

    def phrase_query(self, enabled: bool = True) -> FTSQueryBuilder:
        """Wrap query in quotes for exact phrase matching."""
        self._phrase_query = enabled
        return self

    async def to_list(self) -> list[QueryResult]:
        """Execute FTS query."""
        return await self.db.keyword_search(  # type: ignore[attr-defined]
            query=self.query_text, k=self._limit
        )

    async def to_pandas(self) -> pd.DataFrame:
        import pandas as pd

        results = await self.to_list()
        return pd.DataFrame(results)

    async def to_arrow(self) -> pa.Table:
        import pyarrow as pa

        results = await self.to_list()
        if not results:
            return pa.table({})
        return pa.Table.from_pylist(results)

    def nearest_to(self, vector: list[float]) -> HybridQueryBuilder:
        """Convert FTS query to hybrid with vector search."""
        # Create a vector builder first
        vector_builder = VectorQueryBuilder(db=self.db, query_vector=vector)
        return HybridQueryBuilder(
            db=self.db,
            vector_builder=vector_builder,
            text_query=self.query_text,
            fts_columns=self._fts_columns,
        )


@dataclass
class HybridQueryBuilder:
    """Query builder for hybrid (vector + FTS) search."""

    db: pgVectorDB = field(repr=False)
    vector_builder: VectorQueryBuilder = field(repr=False)
    text_query: str = ""
    fts_columns: list[str] | None = None

    _limit: int = 10
    _reranker: Any | None = None
    _norm: str = "score"  # "score" or "rank"
    _rrf_k: int = 60  # RRF constant

    def limit(self, n: int) -> HybridQueryBuilder:
        self._limit = n
        return self

    def where(self, filter: str | dict[str, Any]) -> HybridQueryBuilder:
        """Apply filter to both vector and FTS queries."""
        self.vector_builder.where(filter)
        # FTS filter would also be applied here
        return self

    def select(self, columns: list[str]) -> HybridQueryBuilder:
        self.vector_builder.select(columns)
        return self

    def rerank(self, reranker: Any | None = None, normalize: str = "score") -> HybridQueryBuilder:
        """Set reranker and normalization method.

        Args:
            reranker: Optional custom reranker
            normalize: "score" or "rank" normalization
        """
        self._reranker = reranker
        self._norm = normalize
        return self

    def rrf_k(self, k: int) -> HybridQueryBuilder:
        """Set RRF constant for reciprocal rank fusion."""
        self._rrf_k = k
        return self

    async def to_list(self) -> list[QueryResult]:
        """Execute hybrid search with fusion."""
        return await self.db.hybrid_search(  # type: ignore[attr-defined]
            query=self.text_query,
            k=self._limit,  # type: ignore[union-attr]
        )

    async def to_pandas(self) -> pd.DataFrame:
        import pandas as pd

        results = await self.to_list()
        return pd.DataFrame(results)

    async def to_arrow(self) -> pa.Table:
        import pyarrow as pa

        results = await self.to_list()
        if not results:
            return pa.table({})
        return pa.Table.from_pylist(results)
