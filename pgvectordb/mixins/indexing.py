"""
Index building and management mixin for pgVectorDB.

Provides: build_index, build_bm25_index, build_index_concurrent,
build_index_with_subvectors, build_index_binary_quantized, areindex,
adrop_vector_index, vacuum_analyze, set_query_params, set_diskann_build_params,
create_metadata_index, and related search methods.
"""

import json
import logging
import re
from typing import Any, Literal

from langchain_postgres.v2.indexes import HNSWIndex, IVFFlatIndex
from sqlalchemy import inspect, text

from ..base import (
    ALLOWED_TEXT_CONFIGS,
    VALID_QUERY_PARAMS,
    DatabaseError,
    DistanceMetric,
    IndexType,
    QueryResult,
    StorageLayout,
    ValidationError,
)
from ..options import ConcurrentIndexBuildOptions, IndexBuildOptions
from ..schema import build_qualified_name
from ._base import MixinBase

logger = logging.getLogger(__name__)


class IndexingMixin(MixinBase):
    """Mixin providing index build, tune, and management operations."""

    async def create_metadata_index(self, columns: list[str]) -> None:
        """
        Create GIN indexes on metadata JSONB fields for faster filtering.

        Args:
            columns: List of metadata field names to index

        Raises:
            InitializationError: If system not initialized
            ValidationError: If columns list is empty
            DatabaseError: If index creation fails
        """
        self._ensure_initialized()

        if not columns:
            raise ValidationError("columns list cannot be empty")

        try:
            async with self.sqlalchemy_engine.connect() as conn:
                # pg_trgm extension already created during initialize()

                for column in columns:
                    index_name = f"idx_{self.table_name}_{column}_metadata"
                    query = text(
                        f'CREATE INDEX IF NOT EXISTS "{index_name}" '
                        f'ON "{self.schema_name}"."{self.table_name}" '
                        f"USING GIN ((langchain_metadata->>'{column}') gin_trgm_ops);"
                    )
                    await conn.execute(query)
                    logger.info(f"✓ Metadata index created for column: {column}")

                await conn.commit()
            logger.info(f"Metadata indexes created for: {columns}")
        except Exception as e:
            raise DatabaseError(f"Failed to create metadata indexes: {e}") from e

    def _get_distance_ops(self, metric: DistanceMetric = DistanceMetric.COSINE) -> str:
        """Get the appropriate operator class for distance metric."""
        if metric == DistanceMetric.COSINE:
            return "vector_cosine_ops"
        elif metric == DistanceMetric.L2:
            return "vector_l2_ops"
        elif metric == DistanceMetric.INNER_PRODUCT:
            return "vector_ip_ops"
        else:
            raise ValidationError(f"Unsupported distance metric: {metric}")

    async def build_index(
        self,
        metric: DistanceMetric = DistanceMetric.COSINE,
        # HNSW parameters
        m: int = 16,
        ef_construction: int = 64,
        # IVFFlat parameters
        lists: int | None = None,
        # DiskANN parameters
        num_neighbors: int = 50,
        search_list_size: int = 100,
        max_alpha: float = 1.2,
        storage_layout: StorageLayout = StorageLayout.MEMORY_OPTIMIZED,
        num_dimensions: int = 0,
        num_bits_per_dimension: int | None = None,
        include_labels: bool = False,
        options: IndexBuildOptions | None = None,
    ) -> None:
        """
        Build vector index based on the selected index type.

        Note: BM25 indexes are created separately using build_bm25_index() method.

        Args:
            metric: Distance metric (cosine, l2, or inner_product)

            HNSW parameters:
            - m: Connections per layer (default: 16)
            - ef_construction: Construction quality (default: 64)

            IVFFlat parameters:
            - lists: Number of inverted lists (default: rows/1000, min 10)

            DiskANN parameters:
            - num_neighbors: Max neighbors per node (default: 50)
            - search_list_size: S parameter for construction (default: 100)
            - max_alpha: Alpha parameter (default: 1.2)
            - storage_layout: memory_optimized (SBQ) or plain (default: memory_optimized)
            - num_dimensions: Dimensions to index, 0 for all (default: 0)
            - num_bits_per_dimension: Bits per dimension for SBQ (default: auto)
            - include_labels: Include labels column for filtering (default: False)

        Raises:
            InitializationError: If system not initialized
            ValidationError: If parameters are invalid
            DatabaseError: If index build fails
        """
        self._ensure_initialized()

        if options is not None:
            metric = options.metric
            m = options.m
            ef_construction = options.ef_construction
            lists = options.lists
            num_neighbors = options.num_neighbors
            search_list_size = options.search_list_size
            max_alpha = options.max_alpha
            storage_layout = options.storage_layout
            num_dimensions = options.num_dimensions
            num_bits_per_dimension = options.num_bits_per_dimension
            include_labels = options.include_labels

        if self._index_built:
            logger.warning(f"{self.index_type.value} index already built, rebuilding...")

        try:
            if self.index_type == IndexType.HNSW:
                await self._build_hnsw_index(metric, m, ef_construction)
            elif self.index_type == IndexType.IVFFLAT:
                await self._build_ivfflat_index(metric, lists)
            elif self.index_type == IndexType.DISKANN:
                await self._build_diskann_index(
                    metric,
                    num_neighbors,
                    search_list_size,
                    max_alpha,
                    storage_layout,
                    num_dimensions,
                    num_bits_per_dimension,
                    include_labels,
                )

            self._index_built = True
            logger.info(f"{self.index_type.value} index built successfully")
        except Exception as e:
            raise DatabaseError(f"Failed to build {self.index_type.value} index: {e}") from e

    async def _build_hnsw_index(self, metric: DistanceMetric, m: int, ef_construction: int) -> None:
        """Build HNSW index using pgvector."""
        if m <= 0 or ef_construction <= 0:
            raise ValidationError("m and ef_construction must be positive")

        if self._vector_store is None:
            raise RuntimeError("Vector store is not initialized")
        index = HNSWIndex(m=m, ef_construction=ef_construction)
        await self._vector_store.aapply_vector_index(index)
        logger.info(f"HNSW index built (m={m}, ef_construction={ef_construction})")

    async def _build_ivfflat_index(self, metric: DistanceMetric, lists: int | None) -> None:
        """Build IVFFlat index using pgvector."""
        if lists is None:
            async with self.sqlalchemy_engine.connect() as conn:
                result = await conn.execute(
                    text(f'SELECT COUNT(*) FROM "{self.schema_name}"."{self.table_name}"')
                )
                row_count = result.scalar()
                lists = max(10, (row_count or 0) // 1000)

        assert lists is not None
        if lists <= 0:
            raise ValidationError("lists must be positive")

        if self._vector_store is None:
            raise RuntimeError("Vector store is not initialized")
        index = IVFFlatIndex(lists=lists)
        await self._vector_store.aapply_vector_index(index)
        logger.info(f"IVFFlat index built (lists={lists})")

    async def _build_diskann_index(
        self,
        metric: DistanceMetric,
        num_neighbors: int,
        search_list_size: int,
        max_alpha: float,
        storage_layout: StorageLayout,
        num_dimensions: int,
        num_bits_per_dimension: int | None,
        include_labels: bool,
    ) -> None:
        """Build DiskANN index using pgvectorscale."""
        if num_neighbors <= 0 or search_list_size <= 0 or max_alpha <= 0:
            raise ValidationError("DiskANN parameters must be positive")

        if num_bits_per_dimension is None:
            num_bits_per_dimension = 2 if self.vector_size < 900 else 1

        ops_class = self._get_distance_ops(metric)
        index_name = f"idx_{self.table_name}_diskann"

        with_params = [
            f"storage_layout={storage_layout.value}",
            f"num_neighbors={num_neighbors}",
            f"search_list_size={search_list_size}",
            f"max_alpha={max_alpha}",
        ]
        if num_dimensions > 0:
            with_params.append(f"num_dimensions={num_dimensions}")
        if storage_layout == StorageLayout.MEMORY_OPTIMIZED:
            with_params.append(f"num_bits_per_dimension={num_bits_per_dimension}")

        with_clause = ", ".join(with_params)

        if include_labels:
            create_index_sql = f"""
            CREATE INDEX "{index_name}" ON "{self.schema_name}"."{self.table_name}"
            USING diskann (embedding {ops_class}, labels) WITH ({with_clause});
            """
        else:
            create_index_sql = f"""
            CREATE INDEX "{index_name}" ON "{self.schema_name}"."{self.table_name}"
            USING diskann (embedding {ops_class}) WITH ({with_clause});
            """

        async with self.sqlalchemy_engine.connect() as conn:
            # Apply build-time parameters if any (DiskANN specific)
            if self._diskann_build_params:
                for param, value in self._diskann_build_params.items():
                    await conn.execute(text(f"SET LOCAL {param} = :value"), {"value": value})
                    logger.info(f"Applied build param: {param}={value}")

            await conn.execute(text(f'DROP INDEX IF EXISTS "{self.schema_name}"."{index_name}"'))
            await conn.execute(text(create_index_sql))
            await conn.commit()

        logger.info(
            f"DiskANN index built (neighbors={num_neighbors}, "
            f"search_list={search_list_size}, storage={storage_layout.value}, "
            f"labels={include_labels})"
        )

    async def set_query_params(
        self,
        probes: int | None = None,
        ef_search: int | None = None,
        query_search_list_size: int | None = None,
        query_rescore: int | None = None,
        iterative_scan: Literal["strict_order", "relaxed_order"] | None = None,
        max_scan_tuples: int | None = None,
        scan_mem_multiplier: int | None = None,
        max_probes: int | None = None,
    ) -> None:
        """
        Set query-time parameters for the active index type.

        Args:
            IVFFlat: probes - Number of lists to search (default: 1)
            HNSW: ef_search - Dynamic candidate list size (default: 40)
            DiskANN: query_search_list_size - Additional candidates (default: 100)
            DiskANN: query_rescore - Elements to rescore, 0 to disable (default: 50)
            iterative_scan: Scan mode (strict_order or relaxed_order) - applies to HNSW/IVFFlat
            max_scan_tuples: Max tuples to scan (HNSW)
            scan_mem_multiplier: Memory multiplier for scan (HNSW)
            max_probes: Max probes for scan (IVFFlat)
        """
        # Store params to be applied before each search
        if probes is not None:
            if probes <= 0:
                raise ValidationError("probes must be positive")
            self._query_params["ivfflat.probes"] = probes
            logger.info(f"Set ivfflat.probes = {probes}")

        if ef_search is not None:
            if ef_search <= 0:
                raise ValidationError("ef_search must be positive")
            self._query_params["hnsw.ef_search"] = ef_search
            logger.info(f"Set hnsw.ef_search = {ef_search}")

        if query_search_list_size is not None:
            if query_search_list_size <= 0:
                raise ValidationError("query_search_list_size must be positive")
            self._query_params["diskann.query_search_list_size"] = query_search_list_size
            logger.info(f"Set diskann.query_search_list_size = {query_search_list_size}")

        if query_rescore is not None:
            if query_rescore < 0:
                raise ValidationError("query_rescore must be non-negative")
            self._query_params["diskann.query_rescore"] = query_rescore
            logger.info(f"Set diskann.query_rescore = {query_rescore}")

        if iterative_scan is not None:
            if iterative_scan == "strict_order":
                self._query_params["hnsw.iterative_scan"] = iterative_scan
                logger.info(f"Set hnsw.iterative_scan = {iterative_scan}")
            elif iterative_scan == "relaxed_order":
                self._query_params["hnsw.iterative_scan"] = iterative_scan
                self._query_params["ivfflat.iterative_scan"] = iterative_scan
                logger.info(f"Set hnsw/ivfflat iterative_scan = {iterative_scan}")
            else:
                raise ValidationError("iterative_scan must be 'strict_order' or 'relaxed_order'")

        if max_scan_tuples is not None:
            if max_scan_tuples <= 0:
                raise ValidationError("max_scan_tuples must be positive")
            self._query_params["hnsw.max_scan_tuples"] = max_scan_tuples
            logger.info(f"Set hnsw.max_scan_tuples = {max_scan_tuples}")

        if scan_mem_multiplier is not None:
            if scan_mem_multiplier <= 0:
                raise ValidationError("scan_mem_multiplier must be positive")
            self._query_params["hnsw.scan_mem_multiplier"] = scan_mem_multiplier
            logger.info(f"Set hnsw.scan_mem_multiplier = {scan_mem_multiplier}")

        if max_probes is not None:
            if max_probes <= 0:
                raise ValidationError("max_probes must be positive")
            self._query_params["ivfflat.max_probes"] = max_probes
            logger.info(f"Set ivfflat.max_probes = {max_probes}")

    async def _apply_query_params(self, conn: Any) -> None:
        """Apply stored query parameters to the current connection."""
        if not self._query_params:
            return

        for param, value in self._query_params.items():
            # Validate parameter is in allowlist (security)
            if param not in VALID_QUERY_PARAMS:
                raise ValidationError(f"Unknown query parameter: {param}")
            # Use SET LOCAL for transaction-scoped settings with safe interpolation
            # asyncpg does not support parameters in SET commands
            if isinstance(value, str) and "'" in value:
                # Should not happen with validated params, but defensive coding
                raise ValidationError(f"Invalid characters in query parameter value: {value}")

            await conn.execute(text(f"SET LOCAL {param} = '{value}'"))

    async def set_diskann_build_params(
        self,
        force_parallel_workers: int | None = None,
        min_vectors_for_parallel_build: int | None = None,
        parallel_flush_interval: int | None = None,
        parallel_initial_start_nodes_count: int | None = None,
    ) -> None:
        """
        Set session-level parameters for DiskANN parallel index build.
        These are applied via SET LOCAL before the CREATE INDEX command.
        """
        if force_parallel_workers is not None:
            if force_parallel_workers < 0:
                raise ValidationError("Workers must be non-negative")
            self._diskann_build_params["diskann.force_parallel_workers"] = force_parallel_workers

        if min_vectors_for_parallel_build is not None:
            if min_vectors_for_parallel_build < 0:
                raise ValidationError("Min vectors must be non-negative")
            self._diskann_build_params["diskann.min_vectors_for_parallel_build"] = (
                min_vectors_for_parallel_build
            )

        if parallel_flush_interval is not None:
            if parallel_flush_interval < 0:
                raise ValidationError("Flush interval must be non-negative")
            self._diskann_build_params["diskann.parallel_flush_interval"] = parallel_flush_interval

        if parallel_initial_start_nodes_count is not None:
            if parallel_initial_start_nodes_count < 0:
                raise ValidationError("Start nodes must be non-negative")
            self._diskann_build_params["diskann.parallel_initial_start_nodes_count"] = (
                parallel_initial_start_nodes_count
            )

        logger.info(f"Set DiskANN build params: {self._diskann_build_params}")

    async def build_bm25_index(
        self,
        text_config: str = "english",
        k1: float = 1.2,
        b: float = 0.75,
        max_parallel_maintenance_workers: int | None = None,
    ) -> None:
        """
        Build native BM25 index using pg_textsearch extension.

        Args:
            text_config: PostgreSQL text search configuration (default: 'english')
            k1: Term frequency saturation parameter (default: 1.2, range: 0.1-10.0)
            b: Length normalization parameter (default: 0.75, range: 0.0-1.0)
            max_parallel_maintenance_workers: Hint to use parallel index builds (pg_textsearch >= 0.5.0)

        Raises:
            InitializationError: If system not initialized
            ValidationError: If parameters are invalid
            DatabaseError: If index build fails
        """
        self._ensure_initialized()

        if not 0.1 <= k1 <= 10.0:
            raise ValidationError("k1 must be between 0.1 and 10.0")
        if not 0.0 <= b <= 1.0:
            raise ValidationError("b must be between 0.0 and 1.0")

        # Validate text_config against allowlist to prevent SQL injection
        if text_config.lower() not in ALLOWED_TEXT_CONFIGS:
            raise ValidationError(
                f"text_config must be one of: {', '.join(sorted(ALLOWED_TEXT_CONFIGS))}"
            )

        try:
            index_name = f"idx_{self.table_name}_bm25"

            async with self.sqlalchemy_engine.connect() as conn:
                # Check if pg_textsearch extension is available
                result = await conn.execute(
                    text("SELECT * FROM pg_available_extensions WHERE name = 'pg_textsearch';")
                )
                if result.fetchone() is None:
                    raise DatabaseError(
                        "pg_textsearch extension not available. "
                        "Install from: https://github.com/timescale/pg_textsearch"
                    )

                # Drop existing BM25 index if exists
                await conn.execute(
                    text(f'DROP INDEX IF EXISTS "{self.schema_name}"."{index_name}"')
                )

                # Set parallel workers hint if provided
                if max_parallel_maintenance_workers is not None:
                    if max_parallel_maintenance_workers < 0:
                        raise ValidationError(
                            "max_parallel_maintenance_workers must be non-negative"
                        )
                    await conn.execute(
                        text(
                            f"SET LOCAL max_parallel_maintenance_workers = {max_parallel_maintenance_workers};"
                        )
                    )
                    logger.info(
                        f"Set parallel maintenance workers: {max_parallel_maintenance_workers}"
                    )

                # Create BM25 index
                create_index_sql = f"""
                CREATE INDEX "{index_name}"
                ON "{self.schema_name}"."{self.table_name}"
                USING bm25(content)
                WITH (text_config='{text_config}', k1={k1}, b={b})
                """
                await conn.execute(text(create_index_sql))
                await conn.commit()

            logger.info(f"BM25 index built (text_config={text_config}, k1={k1}, b={b})")
        except Exception as e:
            raise DatabaseError(f"Failed to build BM25 index: {e}") from e

    async def areindex(self, index_name: str | None = None) -> None:
        """
        Rebuild vector index using existing data.

        Important for IVFFlat and DiskANN after adding significant amounts of new data.
        HNSW indexes don't require reindexing.

        Args:
            index_name: Optional custom index name. If None, uses default naming pattern.

        Raises:
            InitializationError: If system not initialized
            DatabaseError: If reindex operation fails

        Examples:
            >>> # Add 100k new documents
            >>> await pgvdb.add_documents(new_docs)
            >>> # Rebuild index for optimal performance
            >>> await pgvdb.areindex()
        """
        self._ensure_initialized()

        if not self._index_built:
            logger.warning("No index has been built yet. Use build_index() first.")
            return

        try:
            if index_name is None:
                if self.index_type == IndexType.DISKANN:
                    index_name = f"idx_{self.table_name}_diskann"
                else:
                    # For HNSW and IVFFlat, get index name from pg_indexes
                    async with self.sqlalchemy_engine.connect() as conn:
                        result = await conn.execute(
                            text("""
                            SELECT indexname FROM pg_indexes
                            WHERE schemaname = :schema
                            AND tablename = :table
                            AND indexdef LIKE '%embedding%'
                            AND indexdef LIKE '%USING%'
                            LIMIT 1
                        """),
                            {"schema": self.schema_name, "table": self.table_name},
                        )
                        row = result.fetchone()
                        if row:
                            index_name = row[0]
                        else:
                            raise DatabaseError("No vector index found to reindex")

            async with self.sqlalchemy_engine.connect() as conn:
                logger.info(f"Reindexing '{index_name}'...")
                await conn.execute(text(f'REINDEX INDEX "{self.schema_name}"."{index_name}"'))
                await conn.commit()
                logger.info(f"✓ Index '{index_name}' rebuilt successfully")
        except Exception as e:
            raise DatabaseError(f"Failed to reindex: {e}") from e

    async def adrop_vector_index(self, index_name: str | None = None) -> None:
        """
        Remove vector index while keeping all data.

        Useful for:
        - Switching to a different index type
        - Saving disk space when index not needed
        - Testing different index configurations

        Args:
            index_name: Optional custom index name. If None, drops the primary vector index.

        Raises:
            InitializationError: If system not initialized
            DatabaseError: If drop operation fails

        Examples:
            >>> # Switch from HNSW to DiskANN
            >>> await pgvdb.adrop_vector_index()
            >>> pgvdb.index_type = IndexType.DISKANN
            >>> await pgvdb.build_index()
        """
        self._ensure_initialized()

        try:
            if index_name is None:
                if self.index_type == IndexType.DISKANN:
                    index_name = f"idx_{self.table_name}_diskann"
                else:
                    # Get index name from pg_indexes
                    async with self.sqlalchemy_engine.connect() as conn:
                        result = await conn.execute(
                            text("""
                            SELECT indexname FROM pg_indexes
                            WHERE schemaname = :schema
                            AND tablename = :table
                            AND indexdef LIKE '%embedding%'
                            AND (indexdef LIKE '%hnsw%' OR indexdef LIKE '%ivfflat%')
                            LIMIT 1
                        """),
                            {"schema": self.schema_name, "table": self.table_name},
                        )
                        row = result.fetchone()
                        if row:
                            index_name = row[0]
                        else:
                            logger.warning("No vector index found to drop")
                            return

            async with self.sqlalchemy_engine.connect() as conn:
                await conn.execute(
                    text(f'DROP INDEX IF EXISTS "{self.schema_name}"."{index_name}"')
                )
                await conn.commit()
                logger.info(f"✓ Dropped index '{index_name}'")
                self._index_built = False
        except Exception as e:
            raise DatabaseError(f"Failed to drop vector index: {e}") from e

    async def vacuum_analyze(self, full: bool = False) -> None:
        """
        PostgreSQL maintenance to optimize performance.

        VACUUM reclaims storage from deleted rows.
        ANALYZE updates statistics for query planner.

        Run after:
        - Large batch inserts/updates/deletes
        - Noticing slow query performance
        - Before benchmarking

        Args:
            full: If True, runs VACUUM FULL (more thorough but locks table). Default: False

        Raises:
            InitializationError: If system not initialized
            DatabaseError: If maintenance fails

        Examples:
            >>> # After bulk operations
            >>> await pgvdb.add_documents_batch(large_docs)
            >>> await pgvdb.vacuum_analyze()
            >>>
            >>> # Deep maintenance (locks table)
            >>> await pgvdb.vacuum_analyze(full=True)
        """
        self._ensure_initialized()

        try:
            # VACUUM cannot run inside a transaction block — use autocommit.
            # NOTE: execution_options() returns a *new* connection-like proxy;
            # it does NOT mutate the connection in place. We must create the
            # connection using begin_nested=False and then apply the option via
            # the engine-level helper so that the underlying asyncpg connection
            # truly runs outside a transaction.
            async with self.sqlalchemy_engine.connect() as conn:
                # Escape any active implicit transaction so asyncpg accepts VACUUM
                await conn.execute(text("COMMIT"))

                qualified_table = f'"{self.schema_name}"."{self.table_name}"'

                if full:
                    logger.info(
                        "Running VACUUM FULL ANALYZE (this may take a while and locks table)..."
                    )
                    await conn.execute(text(f"VACUUM FULL ANALYZE {qualified_table}"))
                else:
                    logger.info("Running VACUUM ANALYZE...")
                    await conn.execute(text(f"VACUUM ANALYZE {qualified_table}"))

                logger.info("✓ Maintenance completed")
        except Exception as e:
            raise DatabaseError(f"Failed to run vacuum/analyze: {e}") from e

    # ==================== NEW FEATURES: SQLALCHEMY INSPECTOR (Task 26) ====================

    async def _index_exists(self, index_name: str) -> bool:
        """
        Check if an index exists using SQLAlchemy inspector (AGNO pattern).

        More robust than querying pg_indexes directly.

        Args:
            index_name: Name of the index to check

        Returns:
            True if index exists, False otherwise
        """
        try:
            async with self.sqlalchemy_engine.connect() as conn:

                def check_sync(sync_conn):
                    inspector = inspect(sync_conn)
                    indexes = inspector.get_indexes(self.table_name, schema=self.schema_name)
                    return any(idx["name"] == index_name for idx in indexes)

                return await conn.run_sync(check_sync)
        except Exception as e:
            logger.warning(f"Could not check index existence via inspector: {e}")
            # Fallback to pg_indexes query
            async with self.sqlalchemy_engine.connect() as conn:
                result = await conn.execute(
                    text("""
                        SELECT 1 FROM pg_indexes
                        WHERE schemaname = :schema
                        AND tablename = :table
                        AND indexname = :index_name
                    """),
                    {
                        "schema": self.schema_name,
                        "table": self.table_name,
                        "index_name": index_name,
                    },
                )
                return result.fetchone() is not None

    async def build_index_concurrent(
        self,
        # Index type override
        index_type: IndexType | None = None,
        # HNSW parameters
        m: int = 16,
        ef_construction: int = 64,
        # IVFFlat parameters
        lists: int | None = None,
        # DiskANN parameters
        num_neighbors: int = 50,
        search_list_size: int = 100,
        max_alpha: float = 1.2,
        storage_layout: StorageLayout = StorageLayout.MEMORY_OPTIMIZED,
        include_labels: bool = False,
        # Distance metric
        distance: DistanceMetric = DistanceMetric.COSINE,
        options: ConcurrentIndexBuildOptions | None = None,
    ) -> None:
        """
        Build vector index CONCURRENTLY (non-blocking writes).

        Uses CREATE INDEX CONCURRENTLY to avoid blocking writes during index build.
        Takes longer than regular index creation but allows concurrent operations.

        Args:
            m: HNSW max connections per layer (default: 16)
            ef_construction: HNSW construction candidate list size (default: 64)
            lists: IVFFlat number of lists (default: auto-calculated)
            num_neighbors: DiskANN neighbors per node (default: 50)
            search_list_size: DiskANN search list size (default: 100)
            max_alpha: DiskANN alpha parameter (default: 1.2)
            storage_layout: DiskANN storage layout (default: memory_optimized)
            include_labels: Include labels column in DiskANN index (default: False)
            distance: Distance metric (default: cosine)

        Note:
            - Cannot be run inside a transaction
            - Takes extra time and disk space
            - May fail if there are long-running transactions
        """
        self._ensure_initialized()

        if options is not None:
            index_type = options.index_type
            m = options.m
            ef_construction = options.ef_construction
            lists = options.lists
            num_neighbors = options.num_neighbors
            search_list_size = options.search_list_size
            max_alpha = options.max_alpha
            storage_layout = options.storage_layout
            include_labels = options.include_labels
            distance = options.distance

        idx_type = index_type or self.index_type
        index_name = f"idx_{self.table_name}_{idx_type.value}"
        qualified_table = build_qualified_name(self.schema_name, self.table_name)

        # Get operator class for distance metric
        ops_class = (
            f"vector_{distance.value}_ops"
            if distance != DistanceMetric.INNER_PRODUCT
            else "vector_ip_ops"
        )
        if distance == DistanceMetric.L1:
            ops_class = "vector_l1_ops"

        try:
            # Drop existing index if exists
            if await self._index_exists(index_name):
                async with self.sqlalchemy_engine.connect() as conn:
                    await conn.execute(
                        text(
                            f"DROP INDEX CONCURRENTLY IF EXISTS {build_qualified_name(self.schema_name, index_name)}"
                        )
                    )
                    await conn.commit()

            # Build index based on type
            async with self.sqlalchemy_engine.connect() as conn:
                # Set autocommit for CONCURRENTLY (required)
                await conn.execute(text("COMMIT"))

                if idx_type == IndexType.HNSW:
                    await conn.execute(
                        text(f'''
                        CREATE INDEX CONCURRENTLY "{index_name}"
                        ON {qualified_table} USING hnsw (embedding {ops_class})
                        WITH (m = {m}, ef_construction = {ef_construction})
                    ''')
                    )

                elif idx_type == IndexType.IVFFLAT:
                    if lists is None:
                        # Auto-calculate lists based on row count
                        result = await conn.execute(text(f"SELECT COUNT(*) FROM {qualified_table}"))
                        row_count = result.scalar() or 1000
                        lists = (
                            max(int(row_count / 1000), 1)
                            if row_count < 1000000
                            else int(row_count**0.5)
                        )

                    await conn.execute(
                        text(f'''
                        CREATE INDEX CONCURRENTLY "{index_name}"
                        ON {qualified_table} USING ivfflat (embedding {ops_class})
                        WITH (lists = {lists})
                    ''')
                    )

                elif idx_type == IndexType.DISKANN:
                    label_clause = ", labels" if include_labels else ""
                    await conn.execute(
                        text(f'''
                        CREATE INDEX CONCURRENTLY "{index_name}"
                        ON {qualified_table} USING diskann (embedding {ops_class}{label_clause})
                        WITH (
                            num_neighbors = {num_neighbors},
                            search_list_size = {search_list_size},
                            max_alpha = {max_alpha},
                            storage_layout = '{storage_layout.value}'
                        )
                    ''')
                    )

            self._index_built = True
            logger.info(f"✓ Concurrent {idx_type.value} index '{index_name}' created")

        except Exception as e:
            raise DatabaseError(f"Failed to build concurrent index: {e}") from e

    # ==================== NEW FEATURES: INDEX BUILD PROGRESS (Task 13) ====================

    async def get_index_build_progress(self) -> dict[str, Any] | None:
        """
        Get index build progress for ongoing index creation.

        Returns:
            Dictionary with 'phase' and 'percent' if build in progress, None otherwise

        Examples:
            >>> progress = await pgvdb.get_index_build_progress()
            >>> if progress:
            ...     print(f"Phase: {progress['phase']}, Progress: {progress['percent']:.1f}%")
        """
        try:
            async with self.sqlalchemy_engine.connect() as conn:
                result = await conn.execute(
                    text("""
                    SELECT
                        phase,
                        ROUND(100.0 * blocks_done / NULLIF(blocks_total, 0), 1) AS percent
                    FROM pg_stat_progress_create_index
                """)
                )
                row = result.fetchone()

                if row:
                    return {
                        "phase": row[0],
                        "percent": float(row[1]) if row[1] else 0.0,
                    }
                return None
        except Exception as e:
            logger.warning(f"Could not get index build progress: {e}")
            return None

    async def build_index_with_subvectors(
        self,
        subvector_dims: int,
        start_dim: int = 1,
        index_type: IndexType | None = None,
        distance: DistanceMetric = DistanceMetric.COSINE,
        m: int = 16,
        ef_construction: int = 64,
    ) -> str:
        """
        Build index on subvectors (first N dimensions) for faster queries.

        Subvector indexing allows:
        - Faster queries by indexing fewer dimensions
        - Re-ranking with full vectors for better recall
        - Support for Matryoshka embeddings

        Args:
            subvector_dims: Number of dimensions to index (e.g., 256 for first 256 dims)
            start_dim: Starting dimension (1-indexed, default: 1)
            index_type: Index type to use (default: current index_type)
            distance: Distance metric (default: cosine)
            m: HNSW m parameter (default: 16)
            ef_construction: HNSW ef_construction (default: 64)

        Returns:
            Name of the created index

        Examples:
            >>> # Index first 256 dimensions of 1536-dim embeddings
            >>> index_name = await pgvdb.build_index_with_subvectors(subvector_dims=256)
            >>>
            >>> # Query must also use subvector
            >>> # SELECT * FROM docs ORDER BY subvector(embedding, 1, 256)::vector(256) <=> query LIMIT 10

        Note:
            - Best for Matryoshka embeddings (OpenAI, Nomic)
            - Re-rank with full vectors for best recall
        """
        self._ensure_initialized()

        idx_type = index_type or self.index_type
        qualified_table = build_qualified_name(self.schema_name, self.table_name)
        index_name = f"idx_{self.table_name}_subvec_{subvector_dims}"

        # Get operator class
        ops_map = {
            DistanceMetric.COSINE: "cosine_ops",
            DistanceMetric.L2: "l2_ops",
            DistanceMetric.INNER_PRODUCT: "ip_ops",
        }
        ops_class = f"vector_{ops_map.get(distance, 'cosine_ops')}"

        try:
            async with self.sqlalchemy_engine.connect() as conn:
                # Drop existing subvector index
                await conn.execute(
                    text(
                        f"DROP INDEX IF EXISTS {build_qualified_name(self.schema_name, index_name)}"
                    )
                )

                if idx_type == IndexType.HNSW:
                    await conn.execute(
                        text(f'''
                        CREATE INDEX "{index_name}"
                        ON {qualified_table} USING hnsw (
                            (subvector(embedding, {start_dim}, {subvector_dims})::vector({subvector_dims})) {ops_class}
                        )
                        WITH (m = {m}, ef_construction = {ef_construction})
                    ''')
                    )
                elif idx_type == IndexType.IVFFLAT:
                    # Calculate lists based on row count
                    result = await conn.execute(text(f"SELECT COUNT(*) FROM {qualified_table}"))
                    row_count = result.scalar() or 1000
                    lists = max(int(row_count / 1000), 1)

                    await conn.execute(
                        text(f'''
                        CREATE INDEX "{index_name}"
                        ON {qualified_table} USING ivfflat (
                            (subvector(embedding, {start_dim}, {subvector_dims})::vector({subvector_dims})) {ops_class}
                        )
                        WITH (lists = {lists})
                    ''')
                    )

                await conn.commit()

            logger.info(
                f"✓ Created subvector index: {index_name} (dims {start_dim}-{start_dim + subvector_dims - 1})"
            )
            return index_name

        except Exception as e:
            raise DatabaseError(f"Failed to create subvector index: {e}") from e

    async def search_with_subvector_rerank(
        self,
        query: str,
        subvector_dims: int,
        k: int = 10,
        rerank_top: int = 20,
        start_dim: int = 1,
    ) -> list[QueryResult]:
        """
        Search using subvector index with full-vector re-ranking for better recall.

        Two-stage search:
        1. Fast search using subvector index (more candidates)
        2. Re-rank with full vectors (better precision)

        Args:
            query: Search query
            subvector_dims: Dimensions used in subvector index
            k: Final number of results (default: 10)
            rerank_top: Candidates to fetch before reranking (default: 20)
            start_dim: Starting dimension (1-indexed, default: 1)

        Returns:
            Re-ranked results with scores based on full vector similarity
        """
        self._ensure_initialized()

        query_embedding = self.embedding_model.embed_query(query)
        query_subvec = query_embedding[start_dim - 1 : start_dim - 1 + subvector_dims]
        qualified_table = build_qualified_name(self.schema_name, self.table_name)

        try:
            async with self.sqlalchemy_engine.connect() as conn:
                # Two-stage query with CTE
                result = await conn.execute(
                    text(f"""
                        WITH subvec_results AS (
                            SELECT langchain_id, content, langchain_metadata, embedding
                            FROM {qualified_table}
                            ORDER BY subvector(embedding, {start_dim}, {subvector_dims})::vector({subvector_dims}) <=> :subvec_query
                            LIMIT :rerank_top
                        )
                        SELECT langchain_id, content, langchain_metadata,
                               1 - (embedding <=> :full_query) as score
                        FROM subvec_results
                        ORDER BY embedding <=> :full_query
                        LIMIT :k
                    """),
                    {
                        "subvec_query": str(query_subvec),
                        "full_query": str(query_embedding),
                        "rerank_top": rerank_top,
                        "k": k,
                    },
                )

                return [
                    QueryResult(
                        id=str(row[0]),
                        content=row[1],
                        metadata=row[2] or {},
                        score=float(row[3]) if row[3] else 0.0,
                    )
                    for row in result.fetchall()
                ]

        except Exception as e:
            raise DatabaseError(f"Subvector search failed: {e}") from e

    # ==================== REMAINING TASK 5: BINARY QUANTIZATION INDEX (Task 5) ====================

    async def build_index_binary_quantized(
        self,
        distance: DistanceMetric = DistanceMetric.COSINE,
        m: int = 16,
        ef_construction: int = 64,
    ) -> str:
        """
        Build index using binary quantization for 87.5% storage savings.

        Binary quantization converts float32 vectors to single bits,
        dramatically reducing storage and enabling fast Hamming distance search.

        Args:
            distance: Distance metric (Hamming for binary, or original for re-rank)
            m: HNSW m parameter (default: 16)
            ef_construction: HNSW ef_construction (default: 64)

        Returns:
            Name of the created index

        Examples:
            >>> index_name = await pgvdb.build_index_binary_quantized()
            >>> # Searches will use binary index, re-rank with full vectors

        Note:
            - 87.5% smaller than full precision (1 bit vs 32 bits per dim)
            - Best with re-ranking for high recall
            - Uses Hamming distance for fast initial search
        """
        self._ensure_initialized()

        qualified_table = build_qualified_name(self.schema_name, self.table_name)
        index_name = f"idx_{self.table_name}_bq"

        try:
            async with self.sqlalchemy_engine.connect() as conn:
                # Drop existing binary index
                await conn.execute(
                    text(
                        f"DROP INDEX IF EXISTS {build_qualified_name(self.schema_name, index_name)}"
                    )
                )

                # Create binary quantized index using expression indexing
                await conn.execute(
                    text(f'''
                    CREATE INDEX "{index_name}"
                    ON {qualified_table} USING hnsw (
                        (binary_quantize(embedding)::bit({self.vector_size})) bit_hamming_ops
                    )
                    WITH (m = {m}, ef_construction = {ef_construction})
                ''')
                )

                await conn.commit()

            logger.info(f"✓ Created binary quantized index: {index_name}")
            return index_name

        except Exception as e:
            raise DatabaseError(f"Failed to create binary quantized index: {e}") from e

    async def search_with_binary_rerank(
        self, query: str, k: int = 10, rerank_top: int = 50
    ) -> list[QueryResult]:
        """
        Search using binary quantized index with full-vector re-ranking.

        Two-stage search for high recall with binary quantization:
        1. Fast Hamming distance search on binary index (many candidates)
        2. Re-rank with original vectors (better precision)

        Args:
            query: Search query
            k: Final number of results (default: 10)
            rerank_top: Candidates to fetch before reranking (default: 50)

        Returns:
            Re-ranked results with cosine similarity scores
        """
        self._ensure_initialized()

        query_embedding = self.embedding_model.embed_query(query)
        qualified_table = build_qualified_name(self.schema_name, self.table_name)

        # Embed the query vector directly in the SQL string to avoid asyncpg
        # parameter substitution issues: asyncpg converts named params (:foo)
        # to positional ($N), but `:query::vector(N)` is ambiguous — the `::`
        # cast operator confuses the substitution, leaving a bare `:` that
        # Postgres rejects.  The embedding is a list of floats so interpolation
        # is safe (no SQL-injection risk).
        query_literal = str(query_embedding)

        try:
            async with self.sqlalchemy_engine.connect() as conn:
                result = await conn.execute(
                    text(f"""
                        SELECT * FROM (
                            SELECT langchain_id, content, langchain_metadata, embedding
                            FROM {qualified_table}
                            ORDER BY binary_quantize(embedding::vector({self.vector_size}))::bit({self.vector_size}) <~>
                                     binary_quantize('{query_literal}'::vector({self.vector_size}))::bit({self.vector_size})
                            LIMIT :rerank_top
                        ) subq
                        ORDER BY embedding <=> '{query_literal}'::vector({self.vector_size})
                        LIMIT :k
                    """),
                    {"rerank_top": rerank_top, "k": k},
                )

                rows = result.fetchall()
                return [
                    QueryResult(
                        id=str(row[0]),
                        content=row[1],
                        metadata=row[2] or {},
                        score=1.0 - float(i) / len(rows),  # Rank-based score
                    )
                    for i, row in enumerate(rows)
                ]

        except Exception as e:
            raise DatabaseError(f"Binary quantized search failed: {e}") from e

    # ==================== NEW: Scalar Index Methods (v0.0.6) ====================

    async def create_scalar_index(
        self,
        column: str,
        index_type: Literal["btree", "bitmap", "gin", "labellist"] = "btree",
        unique: bool = False,
        concurrently: bool = False,
    ) -> str:
        """
        Create PostgreSQL scalar index on metadata or table column.

        Scalar indexes speed up metadata filtering and range queries.
        Different index types are optimal for different use cases:

        - **btree**: Best for equality and range queries on ordered data (default)
        - **bitmap**: For low-cardinality categorical data (falls back to GIN in PG)
        - **gin**: For array/containment queries
        - **labellist**: For label-based filtering with arrays

        Args:
            column: Column name or metadata key (e.g., "price", "category")
            index_type: Type of index (btree, bitmap, gin, labellist)
            unique: Whether index should enforce uniqueness (BTree only)
            concurrently: Use CREATE INDEX CONCURRENTLY to avoid locking

        Returns:
            Name of the created index

        Raises:
            ValidationError: If parameters are invalid
            DatabaseError: If index creation fails

        Examples:
            >>> # BTree index for range queries
            >>> await db.create_scalar_index("price", index_type="btree")

            >>> # Index for categorical data (maps to GIN in PostgreSQL)
            >>> await db.create_scalar_index("category", index_type="bitmap")

            >>> # Array containment index
            >>> await db.create_scalar_index("tags", index_type="gin")
        """
        self._ensure_initialized()

        # Validate column name for security
        if not column or not isinstance(column, str):
            raise ValidationError("column must be a non-empty string")
        if not re.match(r"^[a-zA-Z0-9_]+$", column):
            raise ValidationError(
                f"Invalid column name: '{column}'. Use alphanumeric characters only."
            )

        valid_types = ["btree", "bitmap", "gin", "labellist"]
        if index_type not in valid_types:
            raise ValidationError(
                f"Invalid index_type: {index_type}. Must be one of: {valid_types}"
            )

        # Generate index name
        index_name = f"idx_{self.table_name}_{column}_{index_type}"
        qualified_table = build_qualified_name(self.schema_name, self.table_name)

        try:
            async with self.sqlalchemy_engine.connect() as conn:
                # Map index type to PostgreSQL index method
                pg_index_type = self._map_scalar_index_type(index_type, column)

                # Build the index expression based on index type
                if index_type == "labellist":
                    # For arrays, index directly
                    index_expr = column
                elif index_type == "gin":
                    # For arrays in metadata
                    index_expr = f"(langchain_metadata->>'{column}')"
                else:
                    # For regular metadata fields
                    # Detect type from a sample value
                    col_type = await self._detect_metadata_type(column, conn)
                    if col_type in ("integer", "numeric", "float"):
                        index_expr = f"((langchain_metadata->>'{column}')::numeric)"
                    elif col_type == "boolean":
                        index_expr = f"((langchain_metadata->>'{column}')::boolean)"
                    else:
                        index_expr = f"((langchain_metadata->>'{column}'))"

                # Build CREATE INDEX statement - create using raw SQL to avoid escaping issues
                concurrently_clause = "CONCURRENTLY " if concurrently else ""
                unique_clause = "UNIQUE " if unique and pg_index_type == "btree" else ""

                create_sql = (
                    f'CREATE {unique_clause}INDEX {concurrently_clause}IF NOT EXISTS "{index_name}" '
                    f"ON {qualified_table} "
                    f"USING {pg_index_type} ({index_expr})"
                )

                await conn.execute(text(create_sql))
                await conn.commit()

            logger.info(f"✓ Created {index_type} index on column: {column}")
            return index_name

        except Exception as e:
            raise DatabaseError(f"Failed to create scalar index: {e}") from e

    def _map_scalar_index_type(self, index_type: str, column: str) -> str:
        """Map requested index type to PostgreSQL index method.

        Args:
            index_type: Requested index type (btree, bitmap, gin, labellist)
            column: Column name

        Returns:
            PostgreSQL index method string
        """
        mapping = {
            "btree": "btree",
            "gin": "gin",
            "labellist": "gin",
            # PostgreSQL doesn't have native bitmap indexes
            # Use GIN for bitmap approximation on low-cardinality data
            "bitmap": "btree",  # Fall back to BTree with partial index
        }
        return mapping.get(index_type, "btree")

    async def _detect_metadata_type(self, column: str, conn: Any) -> str:
        """Detect PostgreSQL type for a metadata column from sample data.

        Args:
            column: Metadata key to check
            conn: Database connection

        Returns:
            Detected type: 'text', 'integer', 'numeric', 'boolean', 'array'
        """
        try:
            # Sample up to 100 rows to detect type
            result = await conn.execute(
                text(f"""
                    SELECT langchain_metadata->>'{column}' as sample
                    FROM "{self.schema_name}"."{self.table_name}"
                    WHERE langchain_metadata->>'{column}' IS NOT NULL
                    LIMIT 100
                """)
            )
            rows = result.fetchall()

            if not rows:
                return "text"  # Default to text if no data

            # Analyze samples to determine most probable type
            samples = [row[0] for row in rows if row[0] is not None]
            if not samples:
                return "text"

            # Check if all samples are integers
            try:
                for s in samples:
                    if s is not None and not isinstance(json.loads(s), int):
                        break
                else:
                    return "integer"
            except (json.JSONDecodeError, ValueError):
                pass

            # Check if numeric/float
            try:
                for s in samples:
                    if s is not None:
                        float(s)
            except ValueError:
                pass

            # Check if boolean
            bool_values = {"true", "false", "1", "0", "yes", "no"}
            if all(s.lower() in bool_values for s in samples if s):
                return "boolean"

            return "text"

        except Exception:
            return "text"  # Default to text on any error
