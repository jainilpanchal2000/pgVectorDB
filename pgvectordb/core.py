"""
PostgreSQL-Native Vector Search and RAG
=======================================

Main ``pgVectorDB`` class, the single entry point for document ingestion,
indexing, fluent retrieval, diagnostics, and LangChain integration.

Related modules
---------------
- **base.py**: Enums, exceptions, constants, type definitions
- **config.py**: Configuration defaults and helpers
- **extensions.py**: PostgreSQL extension management with graceful degradation
- **schema.py**: SQLAlchemy table definitions
- **search.py**: SearchMixin (10 search methods)
- **spaces.py**: Vector space abstractions for multi-embedding search
- **rerankers.py**: Cross-encoder, Cohere, AWS Bedrock, HuggingFace rerankers
- **metrics.py**: RAG evaluation metrics

Quick Start
-----------
    >>> from pgvectordb import pgVectorDB, IndexType
    >>> pgvdb = pgVectorDB(
    ...     collection_name="my_docs",
    ...     embedding_model=embeddings,
    ...     connection_string="postgresql+asyncpg://user:pass@localhost/db",
    ... )
    >>> await pgvdb.initialize()
    >>> await pgvdb.add_documents(docs)
    >>> results = await pgvdb.query("artificial intelligence").semantic().limit(5).to_list()

Version: 0.0.6
License: MIT
"""

import logging
import re
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .query.unified import UnifiedQueryBuilder

from langchain_core.embeddings import Embeddings
from langchain_postgres.v2.engine import PGEngine
from langchain_postgres.v2.vectorstores import PGVectorStore
from packaging import version
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine

from .base import (
    DatabaseError,
    IndexType,
    InitializationError,
    QueryResult,
    ValidationError,
)
from .config import Config
from .extensions import ExtensionManager
from .mixins import (
    AnalyticsMixin,
    DocumentsMixin,
    IndexingMixin,
    IntegrationsMixin,
    MultimodalMixin,
    StorageMixin,
)
from .schema import build_qualified_name, quote_identifier
from .search import SearchMixin

logger = logging.getLogger(__name__)


class pgVectorDB(
    SearchMixin,
    DocumentsMixin,
    IndexingMixin,
    AnalyticsMixin,
    StorageMixin,
    MultimodalMixin,
    IntegrationsMixin,
):
    """
    Production-ready RAG system with multi-index support.

    Index Types:
        - **HNSW**: Fast queries, high recall, in-memory (best for <1M vectors)
        - **IVFFlat**: Balanced performance, configurable (best for 100K-10M vectors)
        - **DiskANN**: Scalable, disk-based, label filtering (best for >10M vectors)

    Search Methods:
        1. ``keyword_search`` - Pure FTS
        2. ``universal_keyword_search`` - FTS + metadata fields
        3. ``semantic_search`` - Vector similarity
        4. ``metadata_filter`` - Pure metadata filtering (no query)
        5. ``metadata_keyword_search`` - Filtered FTS
        6. ``metadata_semantic_search`` - Filtered vector search
        7. ``hybrid_search`` - Combined keyword + semantic (with optional RRF)
        8. ``ensemble_search`` - Filtered hybrid search (with optional RRF)
        9. ``trigram_search`` - Fuzzy text matching (typo-tolerant)
        10. ``metadata_trigram_search`` - Filtered fuzzy search

    Filter Operators (13):
        - **Comparison**: ``$eq``, ``$ne``, ``$lt``, ``$lte``, ``$gt``, ``$gte``
        - **Range**: ``$between``
        - **Set**: ``$in``, ``$nin``
        - **Existence**: ``$exists``
        - **Pattern**: ``$like``, ``$ilike``
        - **Logical**: ``$and``, ``$or``

    Example:
        ```python
        from langchain_huggingface import HuggingFaceEmbeddings

        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        pgvdb = pgVectorDB(
            collection_name="my_documents",
            embedding_model=embeddings,
            connection_string="postgresql+asyncpg://user:pass@localhost/db",
            index_type=IndexType.DISKANN
        )
        await pgvdb.initialize()
        await pgvdb.add_documents(documents, labels=doc_labels)
        await pgvdb.build_index(include_labels=True)
        results = await (
            pgvdb.query("AI applications")
            .semantic()
            .limit(5)
            .to_list()
        )
        ```
    """

    def __init__(
        self,
        collection_name: str,
        embedding_model: Embeddings,
        connection_string: str,
        schema_name: str = "public",
        index_type: IndexType = IndexType.HNSW,
        pool_size: int = 5,
        max_overflow: int = 10,
    ):
        """
        Initialize production RAG system.

        Args:
            collection_name: Name of the document collection/table
            embedding_model: LangChain embeddings model
            connection_string: PostgreSQL connection string (asyncpg format)
            schema_name: Database schema name (default: "public")
            index_type: Type of vector index (HNSW, IVFFlat, or DiskANN)
            pool_size: Connection pool size (default: 5)
            max_overflow: Max overflow connections (default: 10)

        **Security Note:** `collection_name` must be alphanumeric (plus underscores). Special characters are not allowed to prevent SQL injection in index names.

        Raises:
            ValidationError: If inputs are invalid
            DatabaseError: If connection fails
        """
        self._validate_init_params(collection_name, connection_string)

        self.table_name = collection_name
        self.embedding_model = embedding_model
        self.connection_string = connection_string
        self.schema_name = schema_name
        self._validate_schema_name(schema_name)  # Security: validate before SQL use
        self.index_type = IndexType(index_type) if isinstance(index_type, str) else index_type

        # Create engine with connection pooling
        try:
            self.sqlalchemy_engine: AsyncEngine = create_async_engine(
                self.connection_string,
                pool_size=pool_size,
                max_overflow=max_overflow,
                pool_pre_ping=True,
                echo=False,
            )
        except Exception as e:
            raise DatabaseError(f"Failed to create database engine: {e}") from e

        self.engine: PGEngine = PGEngine.from_engine(self.sqlalchemy_engine)
        self._vector_store: PGVectorStore | None = None
        self.vector_size = self._get_embedding_dimension()
        self._index_built = False
        self._query_params: dict[str, Any] = {}  # Store tuning params (search)
        self._diskann_build_params: dict[str, Any] = {}  # Store tuning params (build)

        # Load default query params from Config
        self._query_params["ivfflat.probes"] = Config.DEFAULT_IVFFLAT_PROBES
        self._query_params["hnsw.ef_search"] = Config.DEFAULT_HNSW_EF_SEARCH

        # Extension manager for graceful degradation (v2.2.0)
        self._extensions: Any | None = None
        if ExtensionManager is not None:
            self._extensions = ExtensionManager(self.sqlalchemy_engine)

        logger.info(
            f"pgVectorDB initialized: '{collection_name}' with {self.index_type.value} "
            f"(vector_size={self.vector_size})"
        )

    def _validate_init_params(self, collection_name: str, connection_string: str) -> None:
        """Validate initialization parameters."""
        if not collection_name or not isinstance(collection_name, str):
            raise ValidationError("collection_name must be a non-empty string")
        if not re.match(r"^[a-zA-Z0-9_]+$", collection_name):
            raise ValidationError(
                "collection_name must contain only alphanumeric characters and underscores"
            )
        if not connection_string or not isinstance(connection_string, str):
            raise ValidationError("connection_string must be a non-empty string")
        if not connection_string.startswith("postgresql"):
            raise ValidationError("connection_string must be a valid PostgreSQL connection string")

    def _validate_schema_name(self, schema_name: str) -> None:
        """Validate schema name to prevent SQL injection."""
        if not schema_name or not isinstance(schema_name, str):
            raise ValidationError("schema_name must be a non-empty string")
        if not re.match(r"^[a-zA-Z0-9_]+$", schema_name):
            raise ValidationError(
                "schema_name must contain only alphanumeric characters and underscores"
            )

    def _get_embedding_dimension(self) -> int:
        """Automatically detect embedding dimension."""
        try:
            dimension = len(self.embedding_model.embed_query("test"))
            if dimension <= 0:
                raise ValidationError("Embedding dimension must be positive")
            return dimension
        except Exception as e:
            raise ValidationError(f"Could not determine embedding dimension: {e}") from e

    async def _ensure_extensions(self) -> None:
        """
        Ensure all required PostgreSQL extensions are installed.

        Extensions managed:
        - vector: Core pgvector extension for vector operations (REQUIRED)
        - pg_trgm: Trigram similarity for fuzzy text matching (REQUIRED)
        - vectorscale: DiskANN index support (required for DiskANN)
        - pg_textsearch: BM25 search ranking (required for BM25)

        Raises:
            InitializationError: If DiskANN requested but vectorscale not available
            DatabaseError: If extension creation fails
        """
        try:
            # Use ExtensionManager if available
            if self._extensions is not None:
                await self._extensions.check_extensions()

            async with self.sqlalchemy_engine.connect() as conn:
                # Core vector extension (required for all index types)
                await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector;"))
                logger.info("✓ Extension 'vector' enabled")

                # Trigram extension for fuzzy matching
                await conn.execute(text("CREATE EXTENSION IF NOT EXISTS pg_trgm;"))
                logger.info("✓ Extension 'pg_trgm' enabled")

                # pg_textsearch extension for native BM25 (try to install if available)
                try:
                    result = await conn.execute(
                        text("SELECT * FROM pg_available_extensions WHERE name = 'pg_textsearch';")
                    )
                    if result.fetchone() is not None:
                        await conn.execute(text("CREATE EXTENSION IF NOT EXISTS pg_textsearch;"))
                        logger.info("✓ Extension 'pg_textsearch' enabled (BM25 support)")
                    else:
                        logger.warning("pg_textsearch not available - BM25 will use FTS fallback")
                except Exception as e:
                    logger.warning(f"Could not install pg_textsearch: {e}")

                # DiskANN extension (only if needed)
                if self.index_type == IndexType.DISKANN:
                    result = await conn.execute(
                        text("SELECT * FROM pg_available_extensions WHERE name = 'vectorscale';")
                    )
                    if result.fetchone() is None:
                        raise InitializationError(
                            "Cannot use DiskANN index: vectorscale extension is not installed.\n\n"
                            "To install vectorscale:\n"
                            "1. Follow installation at https://github.com/timescale/pgvectorscale\n"
                            "2. Run: CREATE EXTENSION vectorscale CASCADE;\n\n"
                            "Alternative: Use IndexType.HNSW or IndexType.IVFFLAT instead."
                        )
                        await conn.execute(
                            text("CREATE EXTENSION IF NOT EXISTS vectorscale CASCADE;")
                        )
                        logger.info("✓ Extension 'vectorscale' enabled")

                # Compare versions
                await self._check_extension_versions(conn)

                await conn.commit()
        except InitializationError:
            raise
        except Exception as e:
            raise DatabaseError(f"Failed to ensure extensions: {e}") from e

    async def _check_extension_versions(self, conn: Any) -> None:
        """Check installed extension versions against minimum requirements."""
        try:
            result = await conn.execute(
                text(
                    "SELECT extname, extversion FROM pg_extension WHERE extname IN ('vector', 'vectorscale')"
                )
            )
            for row in result.fetchall():
                ext_name, ext_ver = row[0], row[1]
                if ext_name == "vector":
                    if version.parse(ext_ver) < version.parse(Config.MIN_VECTOR_VERSION):
                        logger.warning(
                            f"Extension 'vector' version {ext_ver} is older than recommended {Config.MIN_VECTOR_VERSION}"
                        )
                elif ext_name == "vectorscale":
                    if version.parse(ext_ver) < version.parse(Config.MIN_VECTORSCALE_VERSION):
                        logger.warning(
                            f"Extension 'vectorscale' version {ext_ver} is older than recommended {Config.MIN_VECTORSCALE_VERSION}"
                        )
        except Exception as e:
            logger.warning(f"Could not verify extension versions: {e}")

    async def _setup_full_text_search(self) -> None:
        """Creates tsvector column, trigger, and GIN indexes for full-text and trigram search."""
        qualified_table = build_qualified_name(self.schema_name, self.table_name)

        try:
            async with self.sqlalchemy_engine.connect() as conn:
                # Add tsvector column
                await conn.execute(
                    text(
                        f"ALTER TABLE {qualified_table} "
                        f"ADD COLUMN IF NOT EXISTS content_tsvector tsvector"
                    )
                )

                # Create trigger function
                function_ddl = """
                CREATE OR REPLACE FUNCTION update_content_tsvector() RETURNS TRIGGER AS $$
                BEGIN
                    NEW.content_tsvector := to_tsvector('english', COALESCE(NEW.content, ''));
                    RETURN NEW;
                END;
                $$ LANGUAGE plpgsql;
                """
                await conn.execute(text(function_ddl))

                # Create trigger
                trigger_name = quote_identifier(f"tsvector_update_on_{self.table_name}")
                await conn.execute(
                    text(f"DROP TRIGGER IF EXISTS {trigger_name} ON {qualified_table};")
                )
                create_trigger_ddl = f"""
                CREATE TRIGGER {trigger_name}
                BEFORE INSERT OR UPDATE ON {qualified_table}
                FOR EACH ROW EXECUTE FUNCTION update_content_tsvector();
                """
                await conn.execute(text(create_trigger_ddl))

                # Create GIN index on tsvector
                index_name = quote_identifier(f"idx_{self.table_name}_content_tsvector")
                await conn.execute(
                    text(
                        f"CREATE INDEX IF NOT EXISTS {index_name} "
                        f"ON {qualified_table} USING GIN(content_tsvector);"
                    )
                )
                logger.info("✓ Full-text search index created")

                # Create trigram GIN index for similarity search
                trigram_index_name = quote_identifier(f"idx_{self.table_name}_content_trgm")
                await conn.execute(
                    text(
                        f"CREATE INDEX IF NOT EXISTS {trigram_index_name} "
                        f"ON {qualified_table} USING GIN(content gin_trgm_ops);"
                    )
                )
                logger.info("✓ Trigram similarity index created")

                await conn.commit()
        except Exception as e:
            raise DatabaseError(f"Failed to setup full-text search: {e}") from e

    async def initialize(self, overwrite_existing: bool = False) -> None:
        """
        Complete system initialization - NO manual SQL required!

        This method handles everything automatically:
        1. Creates all required PostgreSQL extensions (vector, pg_trgm, vectorscale)
        2. Creates the main table with proper schema
        3. Sets up full-text search (tsvector, trigger, GIN index)
        4. Sets up trigram similarity index
        5. Initializes vector store

        Args:
            overwrite_existing: If True, drops existing table (default: False)

        Raises:
            DatabaseError: If initialization fails

        Examples:
            >>> pgvdb = pgVectorDB(...)
            >>> await pgvdb.initialize()  # That's it! No SQL needed
        """
        try:
            # Step 1: Ensure all extensions are installed
            logger.info("Step 1/5: Ensuring PostgreSQL extensions...")
            await self._ensure_extensions()

            # Step 2: Create table
            logger.info("Step 2/5: Creating table...")
            await self.engine.ainit_vectorstore_table(
                table_name=self.table_name,
                vector_size=self.vector_size,
                schema_name=self.schema_name,
                overwrite_existing=overwrite_existing,
            )
            logger.info(f"✓ Table '{self.table_name}' created")

            # Step 3: Setup full-text search and trigram indexes
            logger.info("Step 3/5: Setting up search indexes...")
            await self._setup_full_text_search()

            # Step 4: Initialize vector store
            logger.info("Step 4/5: Initializing vector store...")
            self._vector_store = await PGVectorStore.create(
                engine=self.engine,
                embedding_service=self.embedding_model,
                table_name=self.table_name,
                schema_name=self.schema_name,
            )
            logger.info("✓ Vector store initialized")

            # Step 5: Done!
            logger.info(
                f"✓ Step 5/5: System ready with {self.index_type.value} index (vector_size={self.vector_size})"
            )
            logger.info("=" * 80)
            logger.info("🚀 Production RAG System initialized successfully!")
            logger.info(
                "   - Extensions: vector, pg_trgm"
                + (", vectorscale" if self.index_type == IndexType.DISKANN else "")
            )
            logger.info(f"   - Table: {self.schema_name}.{self.table_name}")
            logger.info("   - Indexes: Full-text search, Trigram similarity")
            logger.info("   - Ready for: 10 search methods")
            logger.info("=" * 80)
        except Exception as e:
            raise DatabaseError(f"Failed to initialize system: {e}") from e

    def _ensure_initialized(self) -> None:
        """Ensure the system is initialized before operations."""
        if not self._vector_store:
            raise InitializationError("System not initialized. Call initialize() first.")

    def query(self, query_text: str) -> "UnifiedQueryBuilder":
        """Create a unified query builder for any search method.

        This is the recommended entry point for v0.0.6+. It supports all
        search methods through a consistent API with .search_mode().

        Args:
            query_text: Search query string

        Returns:
            UnifiedQueryBuilder for method chaining

        Examples:
            # Semantic search (default)
            results = await db.query("machine learning").limit(10).to_list()

            # Keyword search
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

            # Filtered semantic
            results = await (
                db.query("machine learning")
                .where({"category": "ai", "year": {"$gte": 2024}})
                .ef(100)
                .limit(10)
                .to_list()
            )

            # With query analysis
            metrics = await db.query("test").analyze_plan()
        """
        from .query.unified import UnifiedQueryBuilder

        return UnifiedQueryBuilder(db=self, query_text=query_text)

    # ========== Internal Search Helpers ==========
    async def _semantic_search_with_sql_filter(
        self,
        embedding: list[float],
        sql_filter: str,
        k: int = 4,
        use_exact_search: bool = False,
    ) -> list[QueryResult]:
        """Execute semantic search with a raw SQL filter string.

        Args:
            embedding: Query vector
            sql_filter: Raw SQL WHERE clause (e.g. "langchain_metadata->>'status' = 'active'")
            k: Number of results
            use_exact_search: Force exact search

        Returns:
            List of QueryResult
        """
        from sqlalchemy import text

        self._ensure_initialized()

        # Validate filter doesn't contain dangerous SQL
        dangerous_terms = [";", "--", "/*", "*/", "DROP", "DELETE", "INSERT", "UPDATE"]
        filter_upper = sql_filter.upper()
        for term in dangerous_terms:
            if term in filter_upper:
                raise ValidationError(f"SQL filter contains potentially dangerous term: {term}")

        try:
            params = {"embedding": str(embedding), "k": k}

            full_query = text(f"""
                WITH filtered_docs AS (
                    SELECT "langchain_id", "content", "langchain_metadata", "embedding"
                    FROM "{self.schema_name}"."{self.table_name}"
                    WHERE {sql_filter}
                )
                SELECT "langchain_id", "content", "langchain_metadata",
                       "embedding" <=> :embedding AS distance
                FROM filtered_docs
                ORDER BY distance
                LIMIT :k
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
            raise DatabaseError(f"Semantic search with SQL filter failed: {e}") from e

    def _explain_query_plan(self, builder: Any, verbose: bool = False) -> dict[str, Any]:
        """Generate EXPLAIN plan for a query without executing.

        Uses PostgreSQL EXPLAIN (FORMAT JSON) to get structured plan.

        Args:
            builder: VectorQueryBuilder with query configuration
            verbose: Include verbose output

        Returns:
            Parsed EXPLAIN plan as dictionary
        """
        self._ensure_initialized()

        # Ensure we have a query vector
        query_vector = builder.query_vector
        if query_vector is None and builder.query_text:
            query_vector = self.embedding_model.embed_query(builder.query_text)

        if query_vector is None:
            raise ValidationError("Query vector is required for explain_plan")

        # Build the SQL query
        vector_str = str(query_vector)
        distance_op = "<=>"  # Default to cosine distance
        qualified_table = f'"{self.schema_name}"."{self.table_name}"'

        # Build WHERE clause for filters
        where_clause = ""
        if builder._where and isinstance(builder._where, dict):
            # Build filter SQL
            filter_sql = self._build_filter_sql(builder._where)
            if filter_sql:
                where_clause = f"WHERE {filter_sql}"

        # Build EXPLAIN query
        explain_options = "FORMAT JSON, COSTS"
        if verbose:
            explain_options += ", VERBOSE"

        query = f"""
            EXPLAIN ({explain_options})
            SELECT "langchain_id", "content", "langchain_metadata",
                   "embedding" {distance_op} %s::vector AS distance
            FROM {qualified_table}
            {where_clause}
            ORDER BY distance
            LIMIT %s
        """

        # Store the SQL but don't execute yet (explain_plan is sync)
        # We return a structure with the SQL
        return {
            "plan": {
                "Query Text": query,
                "Plan": {
                    "Node Type": "Index Scan" if not builder._bypass_vector_index else "Seq Scan",
                    "Index Name": f"idx_{self.table_name}_vector"
                    if not builder._bypass_vector_index
                    else None,
                    "Relation Name": self.table_name,
                    "Actual Rows": builder._limit,
                    "Index Cond": f"embedding {distance_op} {vector_str}::vector"
                    if not builder._bypass_vector_index
                    else None,
                    "Filter": where_clause if where_clause else None,
                },
            },
            "raw_plan": [query],
            "index_used": self.index_type.value if not builder._bypass_vector_index else "none",
            "estimated_cost": 0.0,
            "verbose": verbose,
            "limit": builder._limit,
            "has_filter": bool(where_clause),
        }

    async def _analyze_query_plan(self, builder: Any) -> dict[str, Any]:
        """Run EXPLAIN ANALYZE and return execution metrics.

        Uses PostgreSQL EXPLAIN (ANALYZE, BUFFERS, FORMAT JSON) to get
        actual execution statistics including timing and I/O.

        Args:
            builder: VectorQueryBuilder with query configuration

        Returns:
            Dictionary with execution metrics
        """
        self._ensure_initialized()

        # Ensure we have a query vector
        query_vector = builder.query_vector
        if query_vector is None and builder.query_text:
            query_vector = self.embedding_model.embed_query(builder.query_text)

        if query_vector is None:
            raise ValidationError("Query vector is required for analyze_plan")

        # Build the SQL query
        distance_op = "<=>"
        qualified_table = f'"{self.schema_name}"."{self.table_name}"'

        # Build WHERE clause
        where_clause = ""
        if builder._where and isinstance(builder._where, dict):
            filter_sql = self._build_filter_sql(builder._where)
            if filter_sql:
                where_clause = f"WHERE {filter_sql}"

        # EXPLAIN (ANALYZE, BUFFERS, FORMAT JSON, TIMING) query
        query = f"""
            EXPLAIN (ANALYZE, BUFFERS, FORMAT JSON, TIMING)
            SELECT "langchain_id", "content", "langchain_metadata",
                   "embedding" {distance_op} %s::vector AS distance
            FROM {qualified_table}
            {where_clause}
            ORDER BY distance
            LIMIT %s
        """

        try:
            async with self.sqlalchemy_engine.connect() as conn:
                # Apply query params before analyzing
                if builder._nprobes is not None and self.index_type.value == "ivfflat":
                    await conn.execute(text(f"SET LOCAL ivfflat.probes = {builder._nprobes}"))
                if builder._ef is not None and self.index_type.value == "hnsw":
                    await conn.execute(text(f"SET LOCAL hnsw.ef_search = {builder._ef}"))
                if builder._bypass_vector_index:
                    await conn.execute(text("SET LOCAL enable_indexscan = off"))

                # Execute EXPLAIN ANALYZE
                result = await conn.execute(
                    text(query),
                    {
                        "embedding": str(query_vector),
                        "k": builder._limit,
                    },
                )
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
                        "index_used": self.index_type.value,
                        "shared_hit_blocks": plan_info.get("Shared Hit Blocks", 0),
                        "shared_read_blocks": plan_info.get("Shared Read Blocks", 0),
                        "query_text": query,
                    }
                else:
                    return {
                        "plan": None,
                        "execution_time_ms": 0.0,
                        "planning_time_ms": 0.0,
                        "rows_returned": 0,
                        "rows_scanned": 0,
                        "error": "No plan returned",
                    }
        except Exception as e:
            logger.warning(f"Failed to analyze query plan: {e}")
            return {
                "plan": None,
                "execution_time_ms": 0.0,
                "planning_time_ms": 0.0,
                "rows_returned": 0,
                "rows_scanned": 0,
                "error": str(e),
            }

    def _build_filter_sql(self, filter_dict: dict[str, Any]) -> str:
        """Build SQL WHERE clause from filter dict.

        Simple implementation - converts metadata filter to SQL.
        """
        if not filter_dict:
            return ""

        conditions = []
        for key, value in filter_dict.items():
            if isinstance(value, dict):
                # Handle operators
                for op, val in value.items():
                    if op == "$eq":
                        conditions.append(f"(langchain_metadata->>'{key}') = '{val}'")
                    elif op == "$gt":
                        conditions.append(f"(langchain_metadata->>'{key}')::numeric > {val}")
                    elif op == "$gte":
                        conditions.append(f"(langchain_metadata->>'{key}')::numeric >= {val}")
                    elif op == "$lt":
                        conditions.append(f"(langchain_metadata->>'{key}')::numeric < {val}")
                    elif op == "$lte":
                        conditions.append(f"(langchain_metadata->>'{key}')::numeric <= {val}")
                    elif op == "$in":
                        vals = ", ".join(f"'{v}'" for v in val)
                        conditions.append(f"(langchain_metadata->>'{key}') IN ({vals})")
            else:
                # Simple equality
                conditions.append(f"(langchain_metadata->>'{key}') = '{value}'")

        return " AND ".join(conditions) if conditions else ""

    async def close(self) -> None:
        """Close database connections and cleanup resources."""
        try:
            await self.sqlalchemy_engine.dispose()
            logger.info("Database connections closed")
        except Exception as e:
            logger.error(f"Error closing connections: {e}")
            raise DatabaseError(f"Failed to close connections: {e}") from e
