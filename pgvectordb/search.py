"""
Search Module for pgVectorDB
============================

This module implements the SearchMixin class, which provides all search functionality
for the pgVectorDB system. It includes:
- Keyword search (FTS & BM25)
- Semantic search (Vector similarity)
- Hybrid search (Weighted & RRF fusion)
- Metadata filtering
- Trigram fuzzy search
"""

import logging
from typing import Any

from sqlalchemy import text

from .base import (
    DatabaseError,
    IndexType,
    InitializationError,
    KeywordSearchType,
    QueryResult,
    ValidationError,
)
from .mixins._base import MixinBase
from .options import HybridSearchOptions
from .query.filters import MetadataFilterCompiler

logger = logging.getLogger(__name__)


class SearchMixin(MixinBase):
    """
    Mixin class providing search capabilities to pgVectorDB.

    Expected attributes on self:
    - schema_name: str
    - table_name: str
    - sqlalchemy_engine: AsyncEngine
    - embedding_model: Embeddings
    - index_type: IndexType
    - vector_size: int
    - _extensions: Optional[ExtensionManager]

    Expected methods on self:
    - _ensure_initialized() -> None
    - _apply_query_params(conn) -> None
    """

    # ==================== VALIDATION HELPERS ====================

    def _validate_search_params(self, query: str, k: int) -> None:
        """Validate common search parameters."""
        if not query or not isinstance(query, str) or not query.strip():
            raise ValidationError("query must be a non-empty string")
        if k <= 0:
            raise ValidationError("k must be positive")

    def _validate_weights(self, weights: tuple[float, float]) -> None:
        """Validate hybrid search weights."""
        if len(weights) != 2:
            raise ValidationError("weights must be a tuple of 2 floats")
        if not all(isinstance(w, (int, float)) and w >= 0 for w in weights):
            raise ValidationError("weights must be non-negative numbers")
        weight_sum = sum(weights)
        if not (0.99 <= weight_sum <= 1.01):
            raise ValidationError(f"weights must sum to 1.0, got {weight_sum}")

    # ==================== FILTER BUILDERS ====================

    def _build_filter_clauses(self, filter: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        """Build SQL WHERE clauses from filter dictionary."""
        return MetadataFilterCompiler().build(filter)

    def _parse_filter(
        self, filter: dict[str, Any], params: dict[str, Any], counter: int
    ) -> tuple[str, dict[str, Any], int]:
        """Recursively parse filter conditions."""
        return MetadataFilterCompiler().parse(filter, params, counter)

    def _build_single_condition(
        self, key: str, operator: str, value: Any, params: dict[str, Any], counter: int
    ) -> tuple[str, dict[str, Any], int]:
        """Build a single filter condition with proper type handling."""
        return MetadataFilterCompiler().build_single_condition(
            key, operator, value, params, counter
        )

    def _build_filter_clauses_wrapper(self, filter: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        """Wrapper for backward compatibility."""
        return self._build_filter_clauses(filter)

    # ==================== SCORING & FUSION ====================

    def _normalize_scores(
        self, scores: dict[str, float], inverse: bool = False
    ) -> dict[str, float]:
        """Normalize scores to 0-1 range."""
        if not scores:
            return {}

        values = list(scores.values())
        min_score = min(values)
        max_score = max(values)

        if max_score == min_score:
            return {k: 1.0 for k in scores.keys()}

        if inverse:
            return {k: 1.0 - (v - min_score) / (max_score - min_score) for k, v in scores.items()}
        else:
            return {k: (v - min_score) / (max_score - min_score) for k, v in scores.items()}

    def _fuse_results(
        self,
        semantic_results: list[QueryResult],
        keyword_results: list[QueryResult],
        weights: tuple[float, float],
        k: int,
    ) -> list[QueryResult]:
        """Common fusion logic for hybrid and ensemble search using weighted scores."""
        semantic_scores = self._normalize_scores(
            {r["id"]: r["score"] for r in semantic_results}, inverse=True
        )
        keyword_scores = self._normalize_scores(
            {r["id"]: r["score"] for r in keyword_results}, inverse=False
        )

        combined_scores: dict[str, float] = {}
        doc_map: dict[str, QueryResult] = {}

        for res in semantic_results:
            doc_map[res["id"]] = res
            combined_scores[res["id"]] = semantic_scores.get(res["id"], 0.0) * weights[0]

        for res in keyword_results:
            doc_map[res["id"]] = res
            if res["id"] in combined_scores:
                combined_scores[res["id"]] += keyword_scores.get(res["id"], 0.0) * weights[1]
            else:
                combined_scores[res["id"]] = keyword_scores.get(res["id"], 0.0) * weights[1]

        sorted_ids = sorted(combined_scores.keys(), key=lambda x: combined_scores[x], reverse=True)

        return [
            QueryResult(
                id=doc_id,
                content=doc_map[doc_id]["content"],
                metadata=doc_map[doc_id]["metadata"],
                score=combined_scores[doc_id],
            )
            for doc_id in sorted_ids[:k]
        ]

    def _fuse_results_rrf(
        self,
        semantic_results: list[QueryResult],
        keyword_results: list[QueryResult],
        k: int,
        rrf_k: int = 60,
    ) -> list[QueryResult]:
        """Reciprocal Rank Fusion (RRF) scoring for hybrid searches."""
        combined_scores: dict[str, float] = {}
        doc_map: dict[str, QueryResult] = {}

        # Add semantic search rankings
        for rank, res in enumerate(semantic_results, start=1):
            doc_map[res["id"]] = res
            combined_scores[res["id"]] = 1.0 / (rrf_k + rank)

        # Add keyword search rankings
        for rank, res in enumerate(keyword_results, start=1):
            doc_map[res["id"]] = res
            if res["id"] in combined_scores:
                combined_scores[res["id"]] += 1.0 / (rrf_k + rank)
            else:
                combined_scores[res["id"]] = 1.0 / (rrf_k + rank)

        sorted_ids = sorted(combined_scores.keys(), key=lambda x: combined_scores[x], reverse=True)

        return [
            QueryResult(
                id=doc_id,
                content=doc_map[doc_id]["content"],
                metadata=doc_map[doc_id]["metadata"],
                score=combined_scores[doc_id],
            )
            for doc_id in sorted_ids[:k]
        ]

    # ==================== SEARCH METHODS ====================

    async def _keyword_search_fts(self, query: str, k: int) -> list[QueryResult]:
        """Internal method for FTS (Full-Text Search) using PostgreSQL ts_rank."""
        try:
            sanitized_words = [w for w in query.split() if w.isalnum()]
            if not sanitized_words:
                ts_query_str = ""
            else:
                ts_query_str = " | ".join(sanitized_words)

            full_query = text(f"""
                SELECT "langchain_id", "content", "langchain_metadata",
                       ts_rank(content_tsvector, to_tsquery('english', :query)) as rank
                FROM "{self.schema_name}"."{self.table_name}"
                WHERE content_tsvector @@ to_tsquery('english', :query)
                ORDER BY rank DESC LIMIT :k
            """)

            async with self.sqlalchemy_engine.connect() as conn:
                if not ts_query_str:
                    return []

                result = await conn.execute(full_query, {"query": ts_query_str, "k": k})
                return [
                    QueryResult(
                        id=str(row[0]),
                        content=row[1],
                        metadata=row[2] or {},
                        score=float(row[3]),
                    )
                    for row in result.fetchall()
                ]
        except Exception as e:
            raise DatabaseError(f"FTS search failed: {e}") from e

    async def _keyword_search_bm25(
        self, query: str, k: int, k1: float, b: float, text_config: str
    ) -> list[QueryResult]:
        """Internal method for BM25 search using pg_textsearch."""
        # Check extension availability (v2.2.0)
        if hasattr(self, "_extensions") and self._extensions is not None:
            self._extensions.require_pg_textsearch("BM25 search")

        try:
            index_name = f"idx_{self.table_name}_bm25"
            qualified_index = f'"{self.schema_name}"."{index_name}"'

            full_query = text(f"""
                SELECT "langchain_id", "content", "langchain_metadata",
                       -(content <@> to_bm25query(:query, '{qualified_index}')) as score
                FROM "{self.schema_name}"."{self.table_name}"
                ORDER BY content <@> to_bm25query(:query, '{qualified_index}') ASC
                LIMIT :k
            """)

            async with self.sqlalchemy_engine.connect() as conn:
                result = await conn.execute(full_query, {"query": query, "k": k})
                return [
                    QueryResult(
                        id=str(row[0]),
                        content=row[1],
                        metadata=row[2] or {},
                        score=float(row[3]),
                    )
                    for row in result.fetchall()
                ]
        except Exception as e:
            # Handle case where extension check was bypassed or failed in query
            if "pg_textsearch" in str(e) or "to_bm25query" in str(e):
                raise InitializationError(
                    "BM25 search failed. Is pg_textsearch extension installed?"
                ) from e
            raise DatabaseError(f"BM25 search failed: {e}") from e

    async def keyword_search(
        self,
        query: str,
        k: int = 4,
        filter: dict[str, Any] | None = None,
        search_type: KeywordSearchType = KeywordSearchType.FTS,
        k1: float = 1.2,
        b: float = 0.75,
        text_config: str = "english",
    ) -> list[QueryResult]:
        """METHOD 1: Keyword search using FTS or BM25."""
        self._ensure_initialized()
        self._validate_search_params(query, k)

        if filter:
            return await self.metadata_keyword_search(
                query, filter, k, search_type, k1, b, text_config
            )

        if search_type == KeywordSearchType.BM25:
            return await self._keyword_search_bm25(query, k, k1, b, text_config)
        else:
            return await self._keyword_search_fts(query, k)

    async def universal_keyword_search(
        self,
        query: str,
        k: int = 4,
        metadata_fields: list[str] | None = None,
        search_type: KeywordSearchType = KeywordSearchType.FTS,
        k1: float = 1.2,
        b: float = 0.75,
        text_config: str = "english",
    ) -> list[QueryResult]:
        """METHOD 2: Searches keywords in both content and metadata."""
        self._ensure_initialized()
        self._validate_search_params(query, k)

        try:
            params = {"query": query, "k": k}
            if search_type == KeywordSearchType.BM25:
                # Check extension (v2.2.0)
                if hasattr(self, "_extensions") and self._extensions is not None:
                    self._extensions.require_pg_textsearch("BM25 search")

                index_name = f"idx_{self.table_name}_bm25"
                qualified_index = f'"{self.schema_name}"."{index_name}"'

                select_clause = f"""
                    "langchain_id", "content", "langchain_metadata",
                    (-(content <@> to_bm25query(:query, '{qualified_index}')))
                """

                if metadata_fields:
                    params["like_query"] = f"%{query}%"
                    bonus_parts = []
                    for field in metadata_fields:
                        if not field.replace("_", "").isalnum():
                            raise ValidationError(f"Invalid field name: {field}")
                        bonus_parts.append(
                            f"(CASE WHEN (langchain_metadata->>'{field}') ILIKE :like_query THEN 1.0 ELSE 0.0 END)"
                        )

                    if bonus_parts:
                        select_clause += " + " + " + ".join(bonus_parts)

                select_clause += " as score"

                full_query = text(f"""
                    SELECT {select_clause}
                    FROM "{self.schema_name}"."{self.table_name}"
                    ORDER BY score DESC
                    LIMIT :k
                """)

            else:
                # FTS Logic
                where_conditions = ["content_tsvector @@ plainto_tsquery('english', :query)"]

                if metadata_fields:
                    if not isinstance(metadata_fields, list):
                        raise ValidationError("metadata_fields must be a list")

                    params["like_query"] = f"%{query}%"
                    for field in metadata_fields:
                        if not field.replace("_", "").isalnum():
                            raise ValidationError(f"Invalid field name: {field}")
                        where_conditions.append(
                            f"(langchain_metadata->>'{field}') ILIKE :like_query"
                        )

                full_where_clause = " OR ".join(where_conditions)

                full_query = text(f"""
                    SELECT "langchain_id", "content", "langchain_metadata",
                           ts_rank(content_tsvector, plainto_tsquery('english', :query)) as rank
                    FROM "{self.schema_name}"."{self.table_name}"
                    WHERE {full_where_clause}
                    ORDER BY rank DESC NULLS LAST LIMIT :k
                """)

            async with self.sqlalchemy_engine.connect() as conn:
                result = await conn.execute(full_query, params)
                return [
                    QueryResult(
                        id=str(row[0]),
                        content=row[1],
                        metadata=row[2] or {},
                        score=float(row[3]) if row[3] is not None else 0.0,
                    )
                    for row in result.fetchall()
                ]
        except Exception as e:
            # Handle case where extension check was bypassed or failed in query
            if "pg_textsearch" in str(e) or "to_bm25query" in str(e):
                raise InitializationError(
                    "BM25 search failed. Is pg_textsearch extension installed?"
                ) from e
            raise DatabaseError(f"Universal keyword search failed: {e}") from e

    async def semantic_search(
        self,
        query: str,
        k: int = 4,
        label_filter: list[int] | None = None,
        filter: dict[str, Any] | None = None,
        use_exact_search: bool = False,
    ) -> list[QueryResult]:
        """METHOD 3: Semantic search using vector embeddings."""
        self._ensure_initialized()
        self._validate_search_params(query, k)

        if filter:
            return await self.metadata_semantic_search(
                query, filter, k, use_exact_search=use_exact_search
            )

        try:
            embedding = self.embedding_model.embed_query(query)

            where_clause = ""
            params = {"embedding": str(embedding), "k": k}

            if label_filter is not None and self.index_type == IndexType.DISKANN:
                where_clause = "WHERE labels && :labels"
                params["labels"] = label_filter

            full_query = text(f"""
                SELECT "langchain_id", "content", "langchain_metadata",
                       "embedding" <=> :embedding AS distance
                FROM "{self.schema_name}"."{self.table_name}"
                {where_clause}
                ORDER BY distance LIMIT :k
            """)

            async with self.sqlalchemy_engine.connect() as conn:
                await self._apply_query_params(conn)
                if use_exact_search:
                    await conn.execute(text("SET LOCAL enable_indexscan = off"))

                result = await conn.execute(full_query, params)
                return [
                    QueryResult(
                        id=str(row[0]),
                        content=row[1],
                        metadata=row[2] or {},
                        score=float(row[3]),
                    )
                    for row in result.fetchall()
                ]
        except Exception as e:
            raise DatabaseError(f"Semantic search failed: {e}") from e

    async def asimilarity_search_by_vector(
        self,
        embedding: list[float],
        k: int = 4,
        label_filter: list[int] | None = None,
        use_exact_search: bool = False,
    ) -> list[QueryResult]:
        """Search using pre-computed embeddings."""
        self._ensure_initialized()

        if not embedding or not isinstance(embedding, list):
            raise ValidationError("embedding must be a non-empty list of floats")
        if len(embedding) != self.vector_size:
            raise ValidationError(
                f"embedding dimension {len(embedding)} doesn't match vector_size {self.vector_size}"
            )
        if k <= 0:
            raise ValidationError("k must be positive")

        try:
            where_clause = ""
            params = {"embedding": str(embedding), "k": k}

            if label_filter is not None and self.index_type == IndexType.DISKANN:
                where_clause = "WHERE labels && :labels"
                params["labels"] = label_filter

            full_query = text(f"""
                SELECT "langchain_id", "content", "langchain_metadata",
                       "embedding" <=> :embedding AS distance
                FROM "{self.schema_name}"."{self.table_name}"
                {where_clause}
                ORDER BY distance LIMIT :k
            """)

            async with self.sqlalchemy_engine.connect() as conn:
                await self._apply_query_params(conn)
                if use_exact_search:
                    await conn.execute(text("SET LOCAL enable_indexscan = off"))

                result = await conn.execute(full_query, params)
                return [
                    QueryResult(
                        id=str(row[0]),
                        content=row[1],
                        metadata=row[2] or {},
                        score=float(row[3]),
                    )
                    for row in result.fetchall()
                ]
        except Exception as e:
            raise DatabaseError(f"Similarity search by vector failed: {e}") from e

    async def asimilarity_search_with_score(
        self,
        query: str,
        k: int = 4,
        label_filter: list[int] | None = None,
        filter: dict[str, Any] | None = None,
    ) -> list[tuple[QueryResult, float]]:
        """Semantic search returning (document, score) tuples."""
        results = await self.semantic_search(query, k, label_filter, filter=filter)
        return [(result, result["score"]) for result in results]

    async def metadata_filter(
        self,
        filter: dict[str, Any],
        k: int = 4,
        order_by: str | None = None,
        ascending: bool = True,
    ) -> list[QueryResult]:
        """METHOD 4: Pure metadata filtering without any search query."""
        self._ensure_initialized()

        if k <= 0:
            raise ValidationError("k must be positive")

        if not filter:
            raise ValidationError("filter cannot be empty")

        try:
            filter_clauses, params = self._build_filter_clauses_wrapper(filter)
            params["k"] = k

            if order_by:
                if not order_by.replace("_", "").isalnum():
                    raise ValidationError(f"Invalid field name: {order_by}")
                direction = "ASC" if ascending else "DESC"
                order_clause = f"ORDER BY (langchain_metadata->>'{order_by}') {direction}"
            else:
                order_clause = "ORDER BY langchain_id"

            full_query = text(f"""
                SELECT "langchain_id", "content", "langchain_metadata", 1.0 as score
                FROM "{self.schema_name}"."{self.table_name}"
                WHERE {filter_clauses}
                {order_clause}
                LIMIT :k
            """)

            async with self.sqlalchemy_engine.connect() as conn:
                result = await conn.execute(full_query, params)
                return [
                    QueryResult(id=str(row[0]), content=row[1], metadata=row[2] or {}, score=1.0)
                    for row in result.fetchall()
                ]
        except Exception as e:
            raise DatabaseError(f"Metadata filter failed: {e}") from e

    async def count_by_metadata(self, filter: dict[str, Any] | None = None) -> int:
        """Count documents matching filter criteria without retrieval."""
        self._ensure_initialized()

        try:
            if filter:
                filter_clauses, params = self._build_filter_clauses_wrapper(filter)
                full_query = text(f"""
                    SELECT COUNT(*)
                    FROM "{self.schema_name}"."{self.table_name}"
                    WHERE {filter_clauses}
                """)
            else:
                params = {}
                full_query = text(f"""
                    SELECT COUNT(*)
                    FROM "{self.schema_name}"."{self.table_name}"
                """)

            async with self.sqlalchemy_engine.connect() as conn:
                result = await conn.execute(full_query, params)
                return result.scalar() or 0
        except Exception as e:
            raise DatabaseError(f"Count by metadata failed: {e}") from e

    async def metadata_keyword_search(
        self,
        query: str,
        filter: dict[str, Any],
        k: int = 4,
        search_type: KeywordSearchType = KeywordSearchType.FTS,
        k1: float = 1.2,
        b: float = 0.75,
        text_config: str = "english",
    ) -> list[QueryResult]:
        """METHOD 5: MANDATORY metadata filtering FIRST, then keyword search."""
        self._ensure_initialized()

        if not query or not query.strip():
            logger.warning(
                "No query provided for metadata_keyword_search, falling back to metadata_filter"
            )
            return await self.metadata_filter(filter, k)

        self._validate_search_params(query, k)

        if not filter:
            logger.warning(
                "No filter provided for metadata_keyword_search, falling back to keyword_search"
            )
            return await self.keyword_search(query, k)

        if search_type == KeywordSearchType.BM25:
            if hasattr(self, "_extensions") and self._extensions is not None:
                self._extensions.require_pg_textsearch("BM25 search")

        try:
            filter_clauses, params = self._build_filter_clauses_wrapper(filter)
            params.update({"query": query, "k": k})

            full_query = text(f"""
                WITH filtered_docs AS (
                    SELECT "langchain_id", "content", "langchain_metadata", content_tsvector
                    FROM "{self.schema_name}"."{self.table_name}"
                    WHERE {filter_clauses}
                )
                SELECT "langchain_id", "content", "langchain_metadata",
                       ts_rank(content_tsvector, plainto_tsquery('english', :query)) as rank
                FROM filtered_docs
                WHERE content_tsvector @@ plainto_tsquery('english', :query)
                ORDER BY rank DESC LIMIT :k
            """)

            async with self.sqlalchemy_engine.connect() as conn:
                result = await conn.execute(full_query, params)
                return [
                    QueryResult(
                        id=str(row[0]),
                        content=row[1],
                        metadata=row[2] or {},
                        score=float(row[3]),
                    )
                    for row in result.fetchall()
                ]
        except Exception as e:
            raise DatabaseError(f"Metadata keyword search failed: {e}") from e

    async def metadata_semantic_search(
        self,
        query: str,
        filter: dict[str, Any],
        k: int = 4,
        use_exact_search: bool = False,
    ) -> list[QueryResult]:
        """METHOD 6: MANDATORY metadata filtering FIRST, then semantic search."""
        self._ensure_initialized()

        if not query or not query.strip():
            logger.warning(
                "No query provided for metadata_semantic_search, falling back to metadata_filter"
            )
            return await self.metadata_filter(filter, k)

        self._validate_search_params(query, k)

        if not filter:
            logger.warning(
                "No filter provided for metadata_semantic_search, falling back to semantic_search"
            )
            return await self.semantic_search(query, k, use_exact_search=use_exact_search)

        try:
            embedding = self.embedding_model.embed_query(query)
            filter_clauses, params = self._build_filter_clauses_wrapper(filter)
            params.update({"embedding": str(embedding), "k": k})

            full_query = text(f"""
                WITH filtered_docs AS (
                    SELECT "langchain_id", "content", "langchain_metadata", "embedding"
                    FROM "{self.schema_name}"."{self.table_name}"
                    WHERE {filter_clauses}
                )
                SELECT "langchain_id", "content", "langchain_metadata",
                       "embedding" <=> :embedding AS distance
                FROM filtered_docs
                ORDER BY distance LIMIT :k
            """)

            async with self.sqlalchemy_engine.connect() as conn:
                await self._apply_query_params(conn)
                if use_exact_search:
                    await conn.execute(text("SET LOCAL enable_indexscan = off"))

                result = await conn.execute(full_query, params)
                return [
                    QueryResult(
                        id=str(row[0]),
                        content=row[1],
                        metadata=row[2] or {},
                        score=float(row[3]),
                    )
                    for row in result.fetchall()
                ]
        except Exception as e:
            raise DatabaseError(f"Metadata semantic search failed: {e}") from e

    async def hybrid_search(
        self,
        query: str,
        k: int = 4,
        weights: tuple[float, float] = (0.5, 0.5),
        label_filter: list[int] | None = None,
        use_rrf: bool = False,
        rrf_k: int = 60,
        keyword_type: KeywordSearchType = KeywordSearchType.FTS,
        bm25_k1: float = 1.2,
        bm25_b: float = 0.75,
        text_config: str = "english",
        options: HybridSearchOptions | None = None,
        filter: dict[str, Any] | None = None,
    ) -> list[QueryResult]:
        """METHOD 7: Combines keyword and semantic search."""
        self._ensure_initialized()

        if options is not None:
            k = options.k
            weights = options.weights
            label_filter = options.label_filter
            use_rrf = options.use_rrf
            rrf_k = options.rrf_k
            keyword_type = options.keyword_type
            bm25_k1 = options.bm25_k1
            bm25_b = options.bm25_b
            text_config = options.text_config
            filter = options.filter

        if not query or not query.strip():
            raise ValidationError("hybrid_search requires a non-empty query")

        self._validate_search_params(query, k)

        if not use_rrf:
            self._validate_weights(weights)

        try:
            if filter:
                semantic_results = await self.metadata_semantic_search(query, filter, k=k * 2)
                keyword_results = await self.metadata_keyword_search(
                    query,
                    filter,
                    k=k * 2,
                    search_type=keyword_type,
                    k1=bm25_k1,
                    b=bm25_b,
                    text_config=text_config,
                )
            else:
                semantic_results = await self.semantic_search(
                    query, k=k * 2, label_filter=label_filter
                )
                keyword_results = await self.keyword_search(
                    query,
                    k=k * 2,
                    search_type=keyword_type,
                    k1=bm25_k1,
                    b=bm25_b,
                    text_config=text_config,
                )

            if use_rrf:
                return self._fuse_results_rrf(semantic_results, keyword_results, k, rrf_k)
            else:
                return self._fuse_results(semantic_results, keyword_results, weights, k)
        except Exception as e:
            raise DatabaseError(f"Hybrid search failed: {e}") from e

    async def ensemble_search(
        self,
        query: str,
        filter: dict[str, Any],
        k: int = 4,
        weights: tuple[float, float] = (0.5, 0.5),
        use_rrf: bool = False,
        rrf_k: int = 60,
        keyword_type: KeywordSearchType = KeywordSearchType.FTS,
        bm25_k1: float = 1.2,
        bm25_b: float = 0.75,
        text_config: str = "english",
        options: HybridSearchOptions | None = None,
    ) -> list[QueryResult]:
        """METHOD 8: Ensemble search (filtered hybrid)."""
        self._ensure_initialized()

        if options is not None:
            k = options.k
            weights = options.weights
            use_rrf = options.use_rrf
            rrf_k = options.rrf_k
            keyword_type = options.keyword_type
            bm25_k1 = options.bm25_k1
            bm25_b = options.bm25_b
            text_config = options.text_config

        self._validate_search_params(query, k)

        if not use_rrf:
            self._validate_weights(weights)

        if not filter:
            return await self.hybrid_search(
                query,
                k,
                weights,
                use_rrf=use_rrf,
                rrf_k=rrf_k,
                keyword_type=keyword_type,
                bm25_k1=bm25_k1,
                bm25_b=bm25_b,
                text_config=text_config,
                options=options,
            )

        try:
            semantic_results = await self.metadata_semantic_search(query, filter, k=k * 2)
            keyword_results = await self.metadata_keyword_search(
                query,
                filter,
                k=k * 2,
                search_type=keyword_type,
                k1=bm25_k1,
                b=bm25_b,
                text_config=text_config,
            )

            if use_rrf:
                return self._fuse_results_rrf(semantic_results, keyword_results, k, rrf_k)
            else:
                return self._fuse_results(semantic_results, keyword_results, weights, k)
        except Exception as e:
            raise DatabaseError(f"Ensemble search failed: {e}") from e

    async def trigram_search(
        self,
        query: str,
        k: int = 4,
        threshold: float = 0.3,
        filter: dict[str, Any] | None = None,
    ) -> list[QueryResult]:
        """METHOD 9: Fuzzy text matching using trigram similarity."""
        self._ensure_initialized()
        self._validate_search_params(query, k)

        if filter:
            return await self.metadata_trigram_search(query, filter, k, threshold)

        if threshold < 0.0 or threshold > 1.0:
            raise ValidationError("threshold must be between 0.0 and 1.0")

        try:
            full_query = text(f"""
                SELECT "langchain_id", "content", "langchain_metadata",
                       similarity("content", :query) as score
                FROM "{self.schema_name}"."{self.table_name}"
                WHERE similarity("content", :query) > :threshold
                ORDER BY score DESC LIMIT :k
            """)

            async with self.sqlalchemy_engine.connect() as conn:
                result = await conn.execute(
                    full_query, {"query": query, "threshold": threshold, "k": k}
                )
                return [
                    QueryResult(
                        id=str(row[0]),
                        content=row[1],
                        metadata=row[2] or {},
                        score=float(row[3]),
                    )
                    for row in result.fetchall()
                ]
        except Exception as e:
            raise DatabaseError(f"Trigram search failed: {e}") from e

    async def metadata_trigram_search(
        self, query: str, filter: dict[str, Any], k: int = 4, threshold: float = 0.3
    ) -> list[QueryResult]:
        """METHOD 10: Metadata filtering + fuzzy text matching."""
        self._ensure_initialized()
        self._validate_search_params(query, k)

        if not filter:
            return await self.trigram_search(query, k, threshold)

        if threshold < 0.0 or threshold > 1.0:
            raise ValidationError("threshold must be between 0.0 and 1.0")

        try:
            filter_clauses, params = self._build_filter_clauses_wrapper(filter)
            params.update({"query": query, "threshold": threshold, "k": k})

            full_query = text(f"""
                WITH filtered_docs AS (
                    SELECT "langchain_id", "content", "langchain_metadata"
                    FROM "{self.schema_name}"."{self.table_name}"
                    WHERE {filter_clauses}
                )
                SELECT "langchain_id", "content", "langchain_metadata",
                       similarity("content", :query) as score
                FROM filtered_docs
                WHERE similarity("content", :query) > :threshold
                ORDER BY score DESC LIMIT :k
            """)

            async with self.sqlalchemy_engine.connect() as conn:
                result = await conn.execute(full_query, params)
                return [
                    QueryResult(
                        id=str(row[0]),
                        content=row[1],
                        metadata=row[2] or {},
                        score=float(row[3]),
                    )
                    for row in result.fetchall()
                ]
        except Exception as e:
            raise DatabaseError(f"Metadata trigram search failed: {e}") from e
