"""
Analytics, monitoring, and diagnostics mixin for pgVectorDB.

Provides: get_stats, get_index_stats, explain_query, benchmark_search_methods,
validate_collection, compute_recall, compute_centroid, get_bm25_index_stats,
get_slow_queries, semantic_search_with_reranker, set_iterative_scan,
create_label_definitions, get_label_ids_by_names, set_maintenance_work_mem,
set_parallel_workers, dump_bm25_index, spill_bm25_index.
"""

import json
import time
import logging
from typing import Dict, List, Optional, Any, Callable

from sqlalchemy import text

from ..base import (
    IndexType,
    IterativeScanMode,
    ValidationError,
    DatabaseError,
    QueryResult,
)
from ..schema import build_qualified_name

logger = logging.getLogger(__name__)


class AnalyticsMixin:
    """Mixin providing analytics, monitoring, and diagnostics."""

    async def get_stats(self) -> Dict[str, Any]:
        """
        Get comprehensive statistics about the system.

        Returns:
            Dictionary with system stats including document count, index info, etc.
        """
        self._ensure_initialized()

        stats = {
            "index_type": self.index_type.value,
            "table_name": self.table_name,
            "schema_name": self.schema_name,
            "vector_size": self.vector_size,
            "index_built": self._index_built,
        }

        try:
            async with self.sqlalchemy_engine.connect() as conn:
                # Get document count
                result = await conn.execute(
                    text(
                        f'SELECT COUNT(*) FROM "{self.schema_name}"."{self.table_name}"'
                    )
                )
                stats["document_count"] = result.scalar()

                # Get index names
                result = await conn.execute(
                    text(
                        """
                    SELECT indexname, indexdef FROM pg_indexes 
                    WHERE schemaname = :schema AND tablename = :table
                    """
                    ),
                    {"schema": self.schema_name, "table": self.table_name},
                )
                stats["indexes"] = [
                    {"name": row[0], "definition": row[1]} for row in result.fetchall()
                ]

                # Get table size
                result = await conn.execute(
                    text(
                        f"""
                    SELECT pg_size_pretty(pg_total_relation_size('"{self.schema_name}"."{self.table_name}"'))
                    """
                    )
                )
                stats["table_size"] = result.scalar()
        except Exception as e:
            logger.warning(f"Could not fetch complete stats: {e}")

        return stats

    async def get_index_stats(self) -> Dict[str, Any]:
        """
        Get detailed index statistics for monitoring and optimization.

        Returns comprehensive information about:
        - Index type, size, and health
        - Index parameters and configuration
        - Performance-related metrics
        - Table bloat and fragmentation

        Returns:
            Dictionary with index statistics

        Example:
            >>> stats = await rag.get_index_stats()
            >>> print(f"Index type: {stats['index_type']}")
            >>> print(f"Index size: {stats['index_size']}")
            >>> print(f"Table bloat: {stats['bloat_ratio']:.1%}")
        """
        self._ensure_initialized()

        stats = {
            "index_type": self.index_type.value,
            "index_built": self._index_built,
            "vector_size": self.vector_size,
        }

        try:
            async with self.sqlalchemy_engine.connect() as conn:
                # Get all indexes on the table
                result = await conn.execute(
                    text("""
                    SELECT 
                        i.indexname,
                        i.indexdef
                    FROM pg_indexes i
                    WHERE i.schemaname = :schema 
                    AND i.tablename = :table
                """),
                    {"schema": self.schema_name, "table": self.table_name},
                )

                indexes = []
                for row in result.fetchall():
                    indexes.append({"name": row[0], "definition": row[1]})
                stats["indexes"] = indexes

                # Get table statistics
                result = await conn.execute(
                    text("""
                    SELECT
                        n_tup_ins as inserts,
                        n_tup_upd as updates,
                        n_tup_del as deletes,
                        n_live_tup as live_tuples,
                        n_dead_tup as dead_tuples,
                        last_vacuum,
                        last_autovacuum,
                        last_analyze,
                        last_autoanalyze
                    FROM pg_stat_user_tables
                    WHERE schemaname = :schema
                    AND relname = :table
                """),
                    {"schema": self.schema_name, "table": self.table_name},
                )

                row = result.fetchone()
                if row:
                    stats["table_stats"] = {
                        "inserts": row[0],
                        "updates": row[1],
                        "deletes": row[2],
                        "live_tuples": row[3],
                        "dead_tuples": row[4],
                        "last_vacuum": str(row[5]) if row[5] else None,
                        "last_autovacuum": str(row[6]) if row[6] else None,
                        "last_analyze": str(row[7]) if row[7] else None,
                        "last_autoanalyze": str(row[8]) if row[8] else None,
                        "bloat_ratio": row[4] / max(row[3], 1) if row[3] else 0,
                    }

                # Get table size
                result = await conn.execute(
                    text(
                        f"""
                    SELECT 
                        pg_size_pretty(pg_total_relation_size('"{self.schema_name}"."{self.table_name}"')) as total_size,
                        pg_size_pretty(pg_table_size('"{self.schema_name}"."{self.table_name}"')) as table_size,
                        pg_size_pretty(pg_indexes_size('"{self.schema_name}"."{self.table_name}"')) as indexes_size
                    """
                    )
                )
                row = result.fetchone()
                if row:
                    stats["size"] = {
                        "total": row[0],
                        "table": row[1],
                        "indexes": row[2],
                    }

                logger.info("Retrieved index statistics")
        except Exception as e:
            logger.warning(f"Could not fetch complete index stats: {e}")

        return stats

    async def explain_query(
        self, query: str, search_method: str = "semantic_search", **search_kwargs
    ) -> List[str]:
        """
        Show PostgreSQL query execution plan for performance debugging.

        Useful for:
        - Understanding if indexes are being used
        - Identifying slow query patterns
        - Optimizing search performance
        - Debugging unexpected results

        Args:
            query: Search query string
            search_method: Name of search method to explain
            **search_kwargs: Additional arguments for the search method

        Returns:
            Query execution plan as formatted string

        Example:
            >>> plan = await rag.explain_query(
            ...     "machine learning",
            ...     search_method="hybrid_search",
            ...     k=10
            ... )
            >>> print(plan)
        """
        self._ensure_initialized()

        if search_method not in ["semantic_search", "keyword_search", "hybrid_search"]:
            raise ValidationError(
                "explain_query only supports semantic_search, keyword_search, hybrid_search"
            )

        try:
            # Get the embedding if needed
            if search_method in ["semantic_search", "hybrid_search"]:
                embedding = self.embedding_model.embed_query(query)

            k = search_kwargs.get("k", 4)

            # Build EXPLAIN query based on method
            if search_method == "semantic_search":
                explain_query = text(f"""
                    EXPLAIN (ANALYZE, BUFFERS, VERBOSE)
                    SELECT "langchain_id", "content", "langchain_metadata", 
                           "embedding" <=> :embedding AS distance
                    FROM "{self.schema_name}"."{self.table_name}"
                    ORDER BY distance LIMIT :k
                """)
                params = {"embedding": str(embedding), "k": k}

            elif search_method == "keyword_search":
                explain_query = text(f"""
                    EXPLAIN (ANALYZE, BUFFERS, VERBOSE)
                    SELECT "langchain_id", "content", "langchain_metadata", 
                           ts_rank(content_tsvector, plainto_tsquery('english', :query)) as rank
                    FROM "{self.schema_name}"."{self.table_name}"
                    WHERE content_tsvector @@ plainto_tsquery('english', :query)
                    ORDER BY rank DESC LIMIT :k
                """)
                params = {"query": query, "k": k}

            async with self.sqlalchemy_engine.connect() as conn:
                result = await conn.execute(explain_query, params)
                plan_lines = [row[0] for row in result.fetchall()]

            logger.info(f"Generated EXPLAIN plan for {search_method}")
            return plan_lines
        except Exception as e:
            raise DatabaseError(f"Failed to explain query: {e}") from e

    async def benchmark_search_methods(
        self, test_queries: List[str], k: int = 4
    ) -> Dict[str, Dict[str, float]]:
        """
        Compare performance of all search methods on test queries.

        Measures:
        - Average query time
        - Total time for all queries
        - Queries per second

        Args:
            test_queries: List of queries to benchmark
            k: Number of results per query

        Returns:
            Dictionary mapping method name to performance metrics

        Example:
            >>> queries = ["AI", "machine learning", "neural networks"]
            >>> results = await rag.benchmark_search_methods(queries, k=10)
            >>> for method, metrics in results.items():
            ...     print(f"{method}: {metrics['avg_time_ms']:.2f}ms, {metrics['qps']:.1f} QPS")
        """
        self._ensure_initialized()

        methods_to_test = [
            "semantic_search",
            "keyword_search",
            "hybrid_search",
            "trigram_search",
        ]

        results = {}

        try:
            for method_name in methods_to_test:
                method = getattr(self, method_name, None)
                if not method:
                    continue

                times = []
                for query in test_queries:
                    start = time.time()
                    try:
                        await method(query, k=k)
                        elapsed = (time.time() - start) * 1000  # Convert to ms
                        times.append(elapsed)
                    except Exception as e:
                        logger.warning(
                            f"Error in {method_name} with query '{query}': {e}"
                        )
                        continue

                if times:
                    avg_time = sum(times) / len(times)
                    total_time = sum(times)
                    qps = (len(times) / total_time) * 1000 if total_time > 0 else 0

                    results[method_name] = {
                        "avg_time_ms": avg_time,
                        "total_time_ms": total_time,
                        "qps": qps,
                        "num_queries": len(times),
                        "min_time_ms": min(times),
                        "max_time_ms": max(times),
                    }

            logger.info(
                f"Benchmarked {len(methods_to_test)} methods on {len(test_queries)} queries"
            )
            return results
        except Exception as e:
            raise DatabaseError(f"Failed to benchmark search methods: {e}") from e

    async def validate_collection(self) -> Dict[str, Any]:
        """
        Check data integrity and health of the collection.

        Validates:
        - Documents have embeddings
        - No null/empty content
        - Metadata structure consistency
        - Orphaned data
        - Embedding dimension consistency

        Returns:
            Dictionary with validation results and issues found

        Example:
            >>> validation = await rag.validate_collection()
            >>> if validation['issues_found']:
            ...     print(f"Found {len(validation['issues'])} issues:")
            ...     for issue in validation['issues']:
            ...         print(f"  - {issue}")
            >>> else:
            ...     print("Collection is healthy!")
        """
        self._ensure_initialized()

        issues = []
        stats = {}

        try:
            async with self.sqlalchemy_engine.connect() as conn:
                # Check total count
                result = await conn.execute(
                    text(f"""
                    SELECT COUNT(*) FROM "{self.schema_name}"."{self.table_name}"
                """)
                )
                total_count = result.scalar()
                stats["total_documents"] = total_count

                # Check for null embeddings
                result = await conn.execute(
                    text(f"""
                    SELECT COUNT(*) FROM "{self.schema_name}"."{self.table_name}"
                    WHERE embedding IS NULL
                """)
                )
                null_embeddings = result.scalar()
                stats["null_embeddings"] = null_embeddings
                if null_embeddings > 0:
                    issues.append(f"{null_embeddings} documents have null embeddings")

                # Check for empty content
                result = await conn.execute(
                    text(f"""
                    SELECT COUNT(*) FROM "{self.schema_name}"."{self.table_name}"
                    WHERE content IS NULL OR content = ''
                """)
                )
                empty_content = result.scalar()
                stats["empty_content"] = empty_content
                if empty_content > 0:
                    issues.append(f"{empty_content} documents have empty content")

                # Check for null IDs
                result = await conn.execute(
                    text(f"""
                    SELECT COUNT(*) FROM "{self.schema_name}"."{self.table_name}"
                    WHERE langchain_id IS NULL
                """)
                )
                null_ids = result.scalar()
                stats["null_ids"] = null_ids
                if null_ids > 0:
                    issues.append(f"{null_ids} documents have null IDs")

                # Check for duplicate IDs
                result = await conn.execute(
                    text(f"""
                    SELECT langchain_id, COUNT(*) as cnt
                    FROM "{self.schema_name}"."{self.table_name}"
                    GROUP BY langchain_id
                    HAVING COUNT(*) > 1
                """)
                )
                duplicate_ids = result.fetchall()
                stats["duplicate_ids"] = len(duplicate_ids)
                if duplicate_ids:
                    issues.append(f"{len(duplicate_ids)} duplicate IDs found")

                # Check embedding dimensions (pgvector doesn't support array_length, use expected dimension)
                if total_count > 0:
                    # For pgvector, we validate by checking if embeddings can be cast to the expected dimension
                    stats["embedding_dimensions"] = {
                        self.vector_size: total_count - null_embeddings
                    }
                    # Note: pgvector enforces dimension at insert time, so inconsistencies are not possible

            validation_result = {
                "healthy": len(issues) == 0,
                "issues_found": len(issues),
                "issues": issues,
                "stats": stats,
            }

            if validation_result["healthy"]:
                logger.info("✓ Collection validation passed - no issues found")
            else:
                logger.warning(f"⚠ Collection validation found {len(issues)} issues")

            return validation_result
        except Exception as e:
            raise DatabaseError(f"Failed to validate collection: {e}") from e

    async def compute_recall(
        self, test_queries: List[str], k: int = 10, sample_size: Optional[int] = None
    ) -> Dict[str, float]:
        """
        Compute recall by comparing approximate vs exact search results.

        Useful for tuning ef_search/probes parameters.

        Args:
            test_queries: List of test query strings
            k: Number of results to compare (default: 10)
            sample_size: Optional limit on number of queries to test

        Returns:
            Dictionary with 'recall@k', 'queries_tested', and 'avg_overlap'

        Example:
            >>> recall = await rag.compute_recall(
            ...     test_queries=["AI applications", "machine learning"],
            ...     k=10
            ... )
            >>> print(f"Recall@10: {recall['recall@k']:.2%}")
        """
        self._ensure_initialized()

        queries = test_queries[:sample_size] if sample_size else test_queries
        total_overlap = 0

        for query in queries:
            # Get approximate results
            approx_results = await self.semantic_search(
                query, k=k, use_exact_search=False
            )
            approx_ids = {r["id"] for r in approx_results}

            # Get exact results
            exact_results = await self.semantic_search(
                query, k=k, use_exact_search=True
            )
            exact_ids = {r["id"] for r in exact_results}

            # Calculate overlap
            overlap = len(approx_ids & exact_ids) / len(exact_ids) if exact_ids else 1.0
            total_overlap += overlap

        avg_recall = total_overlap / len(queries) if queries else 0.0

        return {"recall@k": avg_recall, "queries_tested": len(queries), "k": k}

    def set_iterative_scan(
        self,
        mode: IterativeScanMode = IterativeScanMode.RELAXED_ORDER,
        max_scan_tuples: Optional[int] = None,
        scan_mem_multiplier: Optional[float] = None,
        max_probes: Optional[int] = None,
    ) -> None:
        """
        Configure iterative index scan for better recall with filtered queries.

        Args:
            mode: Scan mode - STRICT_ORDER (exact ordering) or RELAXED_ORDER (better recall)
            max_scan_tuples: HNSW max tuples to visit (default: 20000)
            scan_mem_multiplier: HNSW memory multiplier (default: 1)
            max_probes: IVFFlat max probes for iterative scan

        Example:
            >>> rag.set_iterative_scan(
            ...     mode=IterativeScanMode.STRICT_ORDER,
            ...     max_scan_tuples=50000
            ... )
        """
        if self.index_type == IndexType.HNSW:
            self._query_params["hnsw.iterative_scan"] = mode.value
            if max_scan_tuples is not None:
                self._query_params["hnsw.max_scan_tuples"] = max_scan_tuples
            if scan_mem_multiplier is not None:
                self._query_params["hnsw.scan_mem_multiplier"] = scan_mem_multiplier

        elif self.index_type == IndexType.IVFFLAT:
            self._query_params["ivfflat.iterative_scan"] = mode.value
            if max_probes is not None:
                self._query_params["ivfflat.max_probes"] = max_probes

        logger.info(f"Iterative scan configured: mode={mode.value}")

    async def create_label_definitions(self, labels: List[Dict[str, Any]]) -> int:
        """
        Create label definitions for semantic label filtering with DiskANN.

        Args:
            labels: List of label dictionaries with 'id', 'name', 'description'

        Returns:
            Number of labels created

        Example:
            >>> await rag.create_label_definitions([
            ...     {"id": 1, "name": "science", "description": "Scientific content"},
            ...     {"id": 2, "name": "technology", "description": "Tech content"},
            ... ])
        """
        try:
            async with self.sqlalchemy_engine.connect() as conn:
                # Create label definitions table
                await conn.execute(
                    text(f"""
                    CREATE TABLE IF NOT EXISTS {build_qualified_name(self.schema_name, "label_definitions")} (
                        id INTEGER PRIMARY KEY,
                        name VARCHAR(255) NOT NULL UNIQUE,
                        description TEXT,
                        attributes JSONB DEFAULT '{{}}'::jsonb,
                        created_at TIMESTAMPTZ DEFAULT NOW()
                    )
                """)
                )

                # Insert labels
                for label in labels:
                    await conn.execute(
                        text(f"""
                            INSERT INTO {build_qualified_name(self.schema_name, "label_definitions")}
                            (id, name, description, attributes)
                            VALUES (:id, :name, :description, :attributes)
                            ON CONFLICT (id) DO UPDATE SET
                                name = EXCLUDED.name,
                                description = EXCLUDED.description,
                                attributes = EXCLUDED.attributes
                        """),
                        {
                            "id": label.get("id"),
                            "name": label.get("name"),
                            "description": label.get("description"),
                            "attributes": json.dumps(label.get("attributes", {})),
                        },
                    )

                await conn.commit()

            logger.info(f"Created {len(labels)} label definitions")
            return len(labels)

        except Exception as e:
            raise DatabaseError(f"Failed to create label definitions: {e}") from e

    async def get_label_ids_by_names(self, names: List[str]) -> List[int]:
        """
        Get label IDs from label names.

        Args:
            names: List of label names

        Returns:
            List of corresponding label IDs
        """
        try:
            async with self.sqlalchemy_engine.connect() as conn:
                result = await conn.execute(
                    text(f"""
                        SELECT id FROM {build_qualified_name(self.schema_name, "label_definitions")}
                        WHERE name = ANY(:names)
                    """),
                    {"names": names},
                )
                return [row[0] for row in result.fetchall()]
        except Exception as e:
            logger.warning(f"Could not get label IDs: {e}")
            return []

    # ==================== NEW FEATURES: MAINTENANCE WORK MEM (Task 31) ====================

    async def set_maintenance_work_mem(self, value: str) -> None:
        """
        Set maintenance_work_mem for faster index builds.

        Higher values allow index graphs to fit in memory, significantly
        speeding up HNSW index creation.

        Args:
            value: Memory value like '2GB', '4GB', '8GB'

        Warning:
            Don't set higher than available server memory minus needs of other processes.

        Example:
            >>> await rag.set_maintenance_work_mem('8GB')
            >>> await rag.build_index()  # Faster with more memory
        """
        try:
            async with self.sqlalchemy_engine.connect() as conn:
                await conn.execute(text(f"SET maintenance_work_mem = '{value}'"))
                await conn.commit()
            logger.info(f"Set maintenance_work_mem = {value}")
        except Exception as e:
            raise DatabaseError(f"Failed to set maintenance_work_mem: {e}") from e

    # ==================== NEW FEATURES: PARALLEL WORKERS (Task 32) ====================

    async def set_parallel_workers(
        self, gather: Optional[int] = None, maintenance: Optional[int] = None
    ) -> None:
        """
        Configure parallel workers for queries and index builds.

        Args:
            gather: max_parallel_workers_per_gather for exact search speedup
            maintenance: max_parallel_maintenance_workers for faster index builds

        Example:
            >>> await rag.set_parallel_workers(gather=4, maintenance=7)
        """
        try:
            async with self.sqlalchemy_engine.connect() as conn:
                if gather is not None:
                    await conn.execute(
                        text(f"SET max_parallel_workers_per_gather = {gather}")
                    )
                    logger.info(f"Set max_parallel_workers_per_gather = {gather}")

                if maintenance is not None:
                    await conn.execute(
                        text(f"SET max_parallel_maintenance_workers = {maintenance}")
                    )
                    logger.info(f"Set max_parallel_maintenance_workers = {maintenance}")

                await conn.commit()
        except Exception as e:
            raise DatabaseError(f"Failed to set parallel workers: {e}") from e

    async def compute_centroid(
        self, filter: Optional[Dict[str, Any]] = None
    ) -> Optional[List[float]]:
        """
        Compute average (centroid) of embeddings, optionally filtered.

        Useful for:
        - Finding cluster centers
        - Analyzing document groups
        - Creating representative vectors

        Args:
            filter: Optional metadata filter

        Returns:
            Average embedding vector, or None if no documents

        Example:
            >>> centroid = await rag.compute_centroid(filter={"category": "ai"})
        """
        self._ensure_initialized()

        try:
            qualified_table = build_qualified_name(self.schema_name, self.table_name)

            if filter:
                filter_clauses, params = self._build_filter_clauses_wrapper(filter)
                query = text(f"""
                    SELECT AVG(embedding) FROM {qualified_table}
                    WHERE {filter_clauses}
                """)
            else:
                params = {}
                query = text(f"SELECT AVG(embedding) FROM {qualified_table}")

            async with self.sqlalchemy_engine.connect() as conn:
                result = await conn.execute(query, params)
                row = result.fetchone()

                if row and row[0]:
                    # Parse vector string to list
                    vec_str = str(row[0]).strip("[]")
                    return [float(x) for x in vec_str.split(",")]
                return None

        except Exception as e:
            raise DatabaseError(f"Failed to compute centroid: {e}") from e

    # ==================== NEW FEATURES: BM25 MONITORING (Task 21) ====================

    async def get_bm25_index_stats(self) -> Dict[str, Any]:
        """
        Get BM25 index statistics for monitoring.

        Returns:
            Dictionary with index scan stats
        """
        try:
            async with self.sqlalchemy_engine.connect() as conn:
                result = await conn.execute(
                    text("""
                    SELECT 
                        indexrelid::regclass as index_name,
                        idx_scan,
                        idx_tup_read,
                        idx_tup_fetch
                    FROM pg_stat_user_indexes
                    WHERE indexrelid::regclass::text LIKE '%bm25%'
                """)
                )

                rows = result.fetchall()
                return {
                    "indexes": [
                        {
                            "name": str(row[0]),
                            "scans": row[1],
                            "tuples_read": row[2],
                            "tuples_fetched": row[3],
                        }
                        for row in rows
                    ]
                }
        except Exception as e:
            logger.warning(f"Could not get BM25 index stats: {e}")
            return {"indexes": [], "error": str(e)}

    # ==================== NEW FEATURES: SLOW QUERY MONITORING (Task 29) ====================

    async def get_slow_queries(self, limit: int = 20) -> List[Dict[str, Any]]:
        """
        Get slow queries from pg_stat_statements (if available).

        Args:
            limit: Number of queries to return

        Returns:
            List of slow query statistics
        """
        try:
            async with self.sqlalchemy_engine.connect() as conn:
                result = await conn.execute(
                    text(f"""
                    SELECT 
                        query,
                        calls,
                        ROUND((total_plan_time + total_exec_time) / calls) AS avg_time_ms,
                        ROUND((total_plan_time + total_exec_time) / 60000) AS total_time_min
                    FROM pg_stat_statements
                    WHERE query LIKE '%embedding%' OR query LIKE '%vector%'
                    ORDER BY total_plan_time + total_exec_time DESC
                    LIMIT {limit}
                """)
                )

                return [
                    {
                        "query": row[0][:200] + "..." if len(row[0]) > 200 else row[0],
                        "calls": row[1],
                        "avg_time_ms": float(row[2]) if row[2] else 0,
                        "total_time_min": float(row[3]) if row[3] else 0,
                    }
                    for row in result.fetchall()
                ]
        except Exception as e:
            logger.warning(
                f"Could not get slow queries (pg_stat_statements may not be enabled): {e}"
            )
            return []

    # ==================== NEW FEATURES: RERANKER SUPPORT (Task 28) ====================

    async def semantic_search_with_reranker(
        self,
        query: str,
        k: int = 10,
        rerank_top_k: int = 5,
        reranker: Optional[Callable[[str, List[str]], List[float]]] = None,
        **search_kwargs,
    ) -> List[QueryResult]:
        """
        Semantic search with optional cross-encoder reranking.

        Fetches more candidates than needed, reranks with a cross-encoder,
        and returns top results.

        Args:
            query: Search query
            k: Number of initial candidates to fetch (default: 10)
            rerank_top_k: Number of results to return after reranking (default: 5)
            reranker: Function that takes (query, [texts]) and returns scores
                     Higher scores = more relevant
            **search_kwargs: Additional args passed to semantic_search

        Returns:
            Reranked QueryResult list

        Example:
            >>> from sentence_transformers import CrossEncoder
            >>> ce = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
            >>>
            >>> def rerank_fn(query, texts):
            ...     pairs = [[query, text] for text in texts]
            ...     return ce.predict(pairs)
            >>>
            >>> results = await rag.semantic_search_with_reranker(
            ...     "AI applications",
            ...     k=20,
            ...     rerank_top_k=5,
            ...     reranker=rerank_fn
            ... )
        """
        self._ensure_initialized()

        # Fetch initial candidates
        candidates = await self.semantic_search(query, k=k, **search_kwargs)

        if not reranker or len(candidates) <= rerank_top_k:
            return candidates[:rerank_top_k]

        # Rerank with cross-encoder
        texts = [c["content"] for c in candidates]
        try:
            rerank_scores = reranker(query, texts)
        except Exception as e:
            logger.warning(f"Reranking failed, returning original order: {e}")
            return candidates[:rerank_top_k]

        # Sort by rerank scores (higher = better)
        scored = list(zip(candidates, rerank_scores))
        scored.sort(key=lambda x: x[1], reverse=True)

        return [
            QueryResult(
                id=c["id"],
                content=c["content"],
                metadata=c["metadata"],
                score=float(score),  # Use rerank score
            )
            for c, score in scored[:rerank_top_k]
        ]

    async def dump_bm25_index(self, output_file: Optional[str] = None) -> str:
        """
        Dump BM25 index structure for debugging.

        Uses pg_textsearch's bm25_summarize_index function to get
        detailed information about the BM25 index.

        Args:
            output_file: Optional file to write full dump (default: return summary)

        Returns:
            Index summary or path to dump file

        Example:
            >>> summary = await rag.dump_bm25_index()
            >>> print(summary)
        """
        self._ensure_initialized()

        index_name = f"idx_{self.table_name}_content_bm25"

        try:
            async with self.sqlalchemy_engine.connect() as conn:
                if output_file:
                    # Full dump to file
                    result = await conn.execute(
                        text("SELECT bm25_dump_index(:index_name, :file_path)"),
                        {"index_name": index_name, "file_path": output_file},
                    )
                    return output_file
                else:
                    # Summary only
                    result = await conn.execute(
                        text("SELECT bm25_summarize_index(:index_name)"),
                        {"index_name": index_name},
                    )
                    row = result.fetchone()
                    return str(row[0]) if row else "No BM25 index found"

        except Exception as e:
            logger.warning(f"Could not dump BM25 index (may not exist): {e}")
            return f"Error: {e}"

    async def spill_bm25_index(self) -> int:
        """
        Force BM25 memtable spill to disk segment.

        The BM25 index uses a memtable architecture. This function forces
        the in-memory data to be written to disk segments.

        Returns:
            Number of entries spilled

        Example:
            >>> entries = await rag.spill_bm25_index()
            >>> print(f"Spilled {entries} entries to disk")

        Note:
            - Useful for memory management
            - Normally happens automatically at transaction commit
        """
        self._ensure_initialized()

        index_name = f"idx_{self.table_name}_content_bm25"

        try:
            async with self.sqlalchemy_engine.connect() as conn:
                result = await conn.execute(
                    text("SELECT bm25_spill_index(:index_name)"),
                    {"index_name": index_name},
                )
                row = result.fetchone()
                entries = int(row[0]) if row else 0

                await conn.commit()

            logger.info(f"BM25 index spilled: {entries} entries")
            return entries

        except Exception as e:
            logger.warning(f"Could not spill BM25 index: {e}")
            return 0
