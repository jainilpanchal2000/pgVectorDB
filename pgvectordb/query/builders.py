"""
Query Builders - LanceDB-style fluent API (Extended)

This module implements comprehensive query builders for all search methods:
- SemanticQueryBuilder: Vector/semantic search
- KeywordQueryBuilder: FTS/BM25 keyword search
- HybridQueryBuilder: Combined vector + keyword
- TrigramQueryBuilder: Fuzzy text search
- VectorQueryBuilder: Direct vector input

All builders use lazy execution - methods return self for chaining.
Query only executes when to_list(), to_pandas(), etc. is called.
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
    Optional,
    Self,
    Tuple,
    cast,
)

from ..base import KeywordSearchType, QueryResult

if TYPE_CHECKING:
    import pandas as pd
    import pyarrow as pa

    from ..core import pgVectorDB


logger = logging.getLogger(__name__)


@dataclass
class BaseQueryBuilder:
    """Base class for all query builders with common functionality."""

    db: "pgVectorDB" = field(repr=False)

    _limit: int = 10
    _offset: int = 0
    _columns: Optional[List[str]] = None
    _where: Optional[Dict[str, Any]] = None

    # Execution tracking
    _executed: bool = False
    _results: Optional[List[QueryResult]] = None

    def limit(self, n: int) -> Self:
        """Set the maximum number of results to return."""
        self._limit = n
        return self

    def offset(self, n: int) -> Self:
        """Set the offset for results (for pagination)."""
        self._offset = n
        return self

    def where(self, filter: Dict[str, Any]) -> Self:
        """Apply metadata filter using MongoDB-style syntax.

        Examples:
            builder.where({"category": "ai"})
            builder.where({"year": {"$gte": 2024}})
            builder.where({"$and": [{"status": "active"}, {"priority": {"$gt": 5}}]})
        """
        self._where = filter
        return self

    def select(self, columns: List[str]) -> Self:
        """Select specific columns to return."""
        self._columns = columns
        return self

    async def to_list(self) -> List[QueryResult]:
        """Execute query and return results as list.

        Must be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses must implement to_list()")

    async def to_pandas(self) -> "pd.DataFrame":
        """Execute query and return results as pandas DataFrame."""
        import pandas as pd

        results = await self.to_list()
        return pd.DataFrame(results)

    async def to_arrow(self) -> "pa.Table":
        """Execute query and return results as PyArrow Table."""
        import pyarrow as pa

        results = await self.to_list()
        if not results:
            return pa.table({})
        return pa.Table.from_pylist(results)


@dataclass
class SemanticQueryBuilder(BaseQueryBuilder):
    """Query builder for semantic (vector) search.

    Examples:
        # Basic search
        results = await db.semantic("machine learning").limit(10).to_list()

        # With filtering
        results = await (
            db.semantic("machine learning")
            .where({"category": "ai"})
            .limit(10)
            .to_list()
        )

        # With parameter tuning
        results = await (
            db.semantic("machine learning")
            .ef(100)
            .refine_factor(2)
            .limit(10)
            .to_list()
        )
    """

    query_text: str = ""

    # Vector parameters
    _ef: Optional[int] = None
    _nprobes: Optional[int] = None
    _refine_factor: Optional[int] = None
    _lower_bound: Optional[float] = None
    _upper_bound: Optional[float] = None
    _bypass_vector_index: bool = False

    # Reranking
    _reranker: Optional[Callable] = None

    def ef(self, n: int) -> SemanticQueryBuilder:
        """Set HNSW ef_search parameter for better recall.

        Args:
            n: Candidate pool size (higher = better recall, slower)
        """
        self._ef = n
        return self

    def nprobes(self, n: int) -> SemanticQueryBuilder:
        """Set IVF probes parameter for better recall.

        Only applies to IVFFlat indexes.
        """
        self._nprobes = n
        return self

    def refine_factor(self, factor: int) -> SemanticQueryBuilder:
        """Set refinement oversampling factor.

        Fetches k*factor candidates, reranks with exact distances,
        returns top k. Improves accuracy at cost of speed.
        """
        self._refine_factor = factor
        return self

    def distance_range(
        self, lower: Optional[float] = None, upper: Optional[float] = None
    ) -> SemanticQueryBuilder:
        """Filter by distance range."""
        self._lower_bound = lower
        self._upper_bound = upper
        return self

    def bypass_vector_index(self) -> SemanticQueryBuilder:
        """Force exact (brute force) search.

        Useful for recall calculation or small datasets.
        """
        self._bypass_vector_index = True
        return self

    def rerank(self, reranker: Callable) -> SemanticQueryBuilder:
        """Apply a reranker to results.

        Args:
            reranker: Callable that takes (query, [texts]) and returns scores
        """
        self._reranker = reranker
        return self

    def explain_plan(self, verbose: bool = False) -> Dict[str, Any]:
        """Generate query execution plan without running.

        Returns:
            Dictionary with plan information
        """
        return {
            "search_method": "semantic",
            "query": self.query_text,
            "filter": self._where,
            "limit": self._limit,
            "ef": self._ef,
            "index_type": self.db.index_type.value if hasattr(self.db, "index_type") else "unknown",
            "plan_type": "Index Scan" if not self._bypass_vector_index else "Seq Scan",
        }

    async def analyze_plan(self) -> Dict[str, Any]:
        """Execute with timing and return metrics."""
        import time

        start = time.time()
        results = await self.to_list()
        elapsed = (time.time() - start) * 1000

        return {
            "execution_time_ms": elapsed,
            "rows_returned": len(results),
            "search_method": "semantic",
        }

    async def to_list(self) -> List[QueryResult]:
        """Execute semantic search."""
        if not self.query_text:
            raise ValueError("Query text is required")

        # Determine which method to use based on configuration
        if self._where:
            if isinstance(self._where, dict):
                # Dict filter - use metadata_semantic_search
                results = await self.db.metadata_semantic_search(
                    query=self.query_text,
                    filter=self._where,
                    k=self._limit * (self._refine_factor or 1),
                    use_exact_search=self._bypass_vector_index,
                )
            elif isinstance(self._where, str):
                # String SQL filter - use SQL-based filtering
                results = await self.db._semantic_search_with_sql_filter(
                    embedding=self.db.embedding_model.embed_query(self.query_text),
                    sql_filter=self._where,
                    k=self._limit * (self._refine_factor or 1),
                    use_exact_search=self._bypass_vector_index,
                )
            else:
                raise ValueError(f"Filter must be dict or str, got {type(self._where)}")
        else:
            # Unfiltered semantic search
            results = await self.db.semantic_search(
                query=self.query_text,
                k=self._limit * (self._refine_factor or 1),
                use_exact_search=self._bypass_vector_index,
            )

        # Apply offset and limit
        if self._offset > 0:
            results = results[self._offset:]
        results = results[: self._limit]

        # Apply reranking if configured
        if self._reranker and results:
            texts = [r.get("content", "") for r in results]
            try:
                scores = self._reranker(self.query_text, texts)
                scored = list(zip(results, scores))
                scored.sort(key=lambda x: x[1], reverse=True)
                results = [r for r, _ in scored]
            except Exception as e:
                logger.warning(f"Reranking failed: {e}")

        # Apply column selection
        if self._columns:
            results = [
                {k: v for k, v in r.items() if k in self._columns or k == "id"}
                for r in results
            ]

        return cast(List[QueryResult], results)


@dataclass
class KeywordQueryBuilder(BaseQueryBuilder):
    """Query builder for keyword (FTS/BM25) search.

    Examples:
        # BM25 search
        results = await db.keyword("machine learning").bm25().limit(10).to_list()

        # FTS search
        results = await db.keyword("machine learning").fts().limit(10).to_list()

        # Universal search with metadata boosting
        results = await (
            db.keyword("machine learning")
            .universal(["title", "tags"])
            .limit(10)
            .to_list()
        )
    """

    query_text: str = ""

    _search_type: KeywordSearchType = KeywordSearchType.FTS
    _bm25_k1: float = 1.2
    _bm25_b: float = 0.75
    _text_config: str = "english"
    _universal_fields: Optional[List[str]] = None
    _phrase_query: bool = False

    def bm25(self, k1: float = 1.2, b: float = 0.75) -> KeywordQueryBuilder:
        """Use BM25 ranking with optional parameters.

        Args:
            k1: Term frequency saturation (default: 1.2)
            b: Length normalization (default: 0.75)
        """
        self._search_type = KeywordSearchType.BM25
        self._bm25_k1 = k1
        self._bm25_b = b
        return self

    def fts(self, text_config: str = "english") -> KeywordQueryBuilder:
        """Use PostgreSQL FTS (Full-Text Search).

        Args:
            text_config: Text search configuration (default: "english")
        """
        self._search_type = KeywordSearchType.FTS
        self._text_config = text_config
        return self

    def universal(self, metadata_fields: List[str]) -> KeywordQueryBuilder:
        """Enable universal search - boosts score if query found in metadata fields.

        Args:
            metadata_fields: List of metadata fields to check (e.g., ["title", "tags"])
        """
        self._universal_fields = metadata_fields
        return self

    def phrase(self, enabled: bool = True) -> KeywordQueryBuilder:
        """Require exact phrase matching."""
        self._phrase_query = enabled
        return self

    async def to_list(self) -> List[QueryResult]:
        """Execute keyword search."""
        if not self.query_text:
            raise ValueError("Query text is required")

        # Wrap in quotes if phrase query
        query = self.query_text
        if self._phrase_query and not (query.startswith('"') and query.endswith('"')):
            query = f'"{query}"'

        if self._universal_fields:
            # Universal keyword search
            results = await self.db.universal_keyword_search(
                query=query,
                k=self._limit + self._offset,
                metadata_fields=self._universal_fields,
            )
        elif self._where:
            # Filtered keyword search
            results = await self.db.metadata_keyword_search(
                query=query,
                filter=self._where,
                k=self._limit + self._offset,
                search_type=self._search_type,
                k1=self._bm25_k1,
                b=self._bm25_b,
                text_config=self._text_config,
            )
        else:
            # Plain keyword search
            results = await self.db.keyword_search(
                query=query,
                k=self._limit + self._offset,
                search_type=self._search_type,
                k1=self._bm25_k1,
                b=self._bm25_b,
                text_config=self._text_config,
            )

        # Apply offset and limit
        if self._offset > 0:
            results = results[self._offset:]
        results = results[: self._limit]

        # Apply column selection
        if self._columns:
            results = [
                {k: v for k, v in r.items() if k in self._columns or k == "id"}
                for r in results
            ]

        return cast(List[QueryResult], results)


@dataclass
class TrigramQueryBuilder(BaseQueryBuilder):
    """Query builder for trigram (fuzzy) search.

    Examples:
        # Fuzzy search (typo-tolerant)
        results = await db.trigram("machin lerning").limit(10).to_list()

        # With threshold
        results = await db.trigram("query").threshold(0.3).limit(10).to_list()
    """

    query_text: str = ""
    _threshold: float = 0.3
    _case_sensitive: bool = False

    def threshold(self, min_similarity: float) -> TrigramQueryBuilder:
        """Set minimum similarity threshold.

        Args:
            min_similarity: 0.0 to 1.0 (higher = stricter matching)
        """
        self._threshold = min_similarity
        return self

    def case_sensitive(self, enabled: bool = True) -> TrigramQueryBuilder:
        """Enable case-sensitive matching."""
        self._case_sensitive = enabled
        return self

    async def to_list(self) -> List[QueryResult]:
        """Execute trigram search."""
        if not self.query_text:
            raise ValueError("Query text is required")

        if self._where:
            results = await self.db.metadata_trigram_search(
                query=self.query_text,
                filter=self._where,
                k=self._limit + self._offset,
                threshold=self._threshold,
            )
        else:
            results = await self.db.trigram_search(
                query=self.query_text,
                k=self._limit + self._offset,
                threshold=self._threshold,
            )

        # Apply offset and limit
        if self._offset > 0:
            results = results[self._offset:]
        results = results[: self._limit]

        # Apply column selection
        if self._columns:
            results = [
                {k: v for k, v in r.items() if k in self._columns or k == "id"}
                for r in results
            ]

        return cast(List[QueryResult], results)


@dataclass
class HybridQueryBuilder(BaseQueryBuilder):
    """Query builder for hybrid (vector + keyword) search.

    Examples:
        # Weighted fusion
        results = await (
            db.hybrid("machine learning")
            .weights(semantic=0.7, keyword=0.3)
            .limit(10)
            .to_list()
        )

        # RRF fusion
        results = await (
            db.hybrid("machine learning")
            .rrf(k=60)
            .limit(10)
            .to_list()
        )

        # With BM25 keyword search
        results = await (
            db.hybrid("machine learning")
            .weights(0.8, 0.2)
            .bm25_params(k1=1.2, b=0.75)
            .limit(10)
            .to_list()
        )
    """

    query_text: str = ""

    _weights: Tuple[float, float] = (0.5, 0.5)
    _use_rrf: bool = False
    _rrf_k: int = 60
    _keyword_type: KeywordSearchType = KeywordSearchType.FTS
    _bm25_k1: float = 1.2
    _bm25_b: float = 0.75
    _text_config: str = "english"

    def weights(self, semantic: float, keyword: float) -> HybridQueryBuilder:
        """Set fusion weights.

        Args:
            semantic: Weight for semantic search (0.0 to 1.0)
            keyword: Weight for keyword search (0.0 to 1.0)
        """
        self._weights = (semantic, keyword)
        self._use_rrf = False
        return self

    def rrf(self, k: int = 60) -> HybridQueryBuilder:
        """Use Reciprocal Rank Fusion instead of weighted fusion.

        Args:
            k: RRF constant (default: 60)
        """
        self._use_rrf = True
        self._rrf_k = k
        return self

    def bm25_params(self, k1: float, b: float) -> HybridQueryBuilder:
        """Set BM25 parameters for keyword component."""
        self._keyword_type = KeywordSearchType.BM25
        self._bm25_k1 = k1
        self._bm25_b = b
        return self

    async def to_list(self) -> List[QueryResult]:
        """Execute hybrid search."""
        if not self.query_text:
            raise ValueError("Query text is required")

        results = await self.db.hybrid_search(
            query=self.query_text,
            k=self._limit + self._offset,
            weights=self._weights,
            use_rrf=self._use_rrf,
            rrf_k=self._rrf_k,
            keyword_type=self._keyword_type,
            bm25_k1=self._bm25_k1,
            bm25_b=self._bm25_b,
            text_config=self._text_config,
        )

        # Apply offset and limit
        if self._offset > 0:
            results = results[self._offset:]
        results = results[: self._limit]

        # Apply column selection
        if self._columns:
            results = [
                {k: v for k, v in r.items() if k in self._columns or k == "id"}
                for r in results
            ]

        return cast(List[QueryResult], results)


@dataclass
class VectorQueryBuilder(BaseQueryBuilder):
    """Query builder for pre-computed vector search.

    Examples:
        # Search with pre-computed embedding
        embedding = model.encode("query")
        results = await db.vector(embedding).limit(10).to_list()

        # With filtering
        results = await (
            db.vector(embedding)
            .where({"category": "ai"})
            .limit(10)
            .to_list()
        )
    """

    query_vector: Optional[List[float]] = None

    # Vector parameters
    _ef: Optional[int] = None
    _refine_factor: Optional[int] = None
    _bypass_vector_index: bool = False

    def ef(self, n: int) -> VectorQueryBuilder:
        """Set HNSW ef_search parameter."""
        self._ef = n
        return self

    def refine_factor(self, factor: int) -> VectorQueryBuilder:
        """Set refinement oversampling factor."""
        self._refine_factor = factor
        return self

    def bypass_vector_index(self) -> VectorQueryBuilder:
        """Force exact (brute force) search."""
        self._bypass_vector_index = True
        return self

    async def to_list(self) -> List[QueryResult]:
        """Execute vector search."""
        if not self.query_vector:
            raise ValueError("Query vector is required")

        results = cast(
            List[QueryResult],
            await self.db.asimilarity_search_by_vector(
                embedding=self.query_vector,
                k=self._limit + self._offset,
            ),
        )

        # Apply offset and limit
        if self._offset > 0:
            results = results[self._offset:]
        results = results[: self._limit]

        # Apply column selection
        if self._columns:
            results = cast(List[QueryResult], [
                {k: v for k, v in r.items() if k in self._columns or k == "id"}
                for r in results
            ])

        return results
