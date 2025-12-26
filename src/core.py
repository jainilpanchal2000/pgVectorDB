"""
Production-Ready Multi-Index RAG System
========================================

Combines all features from exhaustive_retrieval_demo and multi_index_system:
- 3 Index Types: HNSW, IVFFlat, DiskANN
- 10 Search Methods: Keyword, Universal Keyword, Semantic, Metadata Filter, Metadata+Keyword, Metadata+Semantic, Hybrid, Ensemble, Trigram, Metadata+Trigram
- 13 Filter Operators: $eq, $ne, $lt, $lte, $gt, $gte, $in, $nin, $between, $exists, $like, $ilike, $and, $or
- Connection pooling, error handling, validation, logging
- Label-based filtering for DiskANN
- Query parameter tuning for all index types
- Reciprocal Rank Fusion (RRF) scoring for hybrid searches
"""

import uuid
import logging
from typing import Dict, List, Optional, Tuple, TypedDict, Any, Literal
from enum import Enum

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_postgres.v2.indexes import HNSWIndex, IVFFlatIndex
from langchain_postgres.v2.vectorstores import PGVectorStore
from langchain_postgres.v2.engine import PGEngine
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine
from sqlalchemy import text

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ==================== Enums ====================
class IndexType(str, Enum):
    """Supported vector index types."""
    HNSW = "hnsw"
    IVFFLAT = "ivfflat"
    DISKANN = "diskann"


class KeywordSearchType(str, Enum):
    """Keyword search implementations."""
    FTS = "fts"      # PostgreSQL ts_rank (full-text search)
    BM25 = "bm25"    # pg_textsearch native BM25


class StorageLayout(str, Enum):
    """Storage layout for DiskANN index."""
    MEMORY_OPTIMIZED = "memory_optimized"  # Uses SBQ compression
    PLAIN = "plain"  # Uncompressed


class DistanceMetric(str, Enum):
    """Distance metrics for vector operations."""
    COSINE = "cosine"
    L2 = "l2"
    INNER_PRODUCT = "inner_product"


# ==================== Custom Exceptions ====================
class RetrievalSystemError(Exception):
    """Base exception for retrieval system errors."""
    pass


class InitializationError(RetrievalSystemError):
    """Raised when system is not properly initialized."""
    pass


class ValidationError(RetrievalSystemError):
    """Raised when input validation fails."""
    pass


class DatabaseError(RetrievalSystemError):
    """Raised when database operations fail."""
    pass


# ==================== Type Definitions ====================
class QueryResult(TypedDict):
    """Structured result with score and metadata."""
    id: str
    content: str
    metadata: Dict[str, Any]
    score: float


# ==================== Production RAG System ====================
class pgVectorDB:
    """
    Production-ready RAG system with multi-index support.
    
    **Index Types:**
    - HNSW: Fast queries, high recall, in-memory (best for <1M vectors)
    - IVFFlat: Balanced performance, configurable (best for 100K-10M vectors)
    - DiskANN: Scalable, disk-based, label filtering (best for >10M vectors)
    
    **Search Methods:**
    1. keyword_search - Pure FTS
    2. universal_keyword_search - FTS + metadata fields
    3. semantic_search - Vector similarity
    4. metadata_filter - Pure metadata filtering (no query)
    5. metadata_keyword_search - Filtered FTS
    6. metadata_semantic_search - Filtered vector search
    7. hybrid_search - Combined keyword + semantic (with optional RRF)
    8. ensemble_search - Filtered hybrid search (with optional RRF)
    9. trigram_search - Fuzzy text matching (typo-tolerant)
    10. metadata_trigram_search - Filtered fuzzy search
    
    **Filter Operators (13):**
    Comparison: $eq, $ne, $lt, $lte, $gt, $gte
    Range: $between
    Set: $in, $nin
    Existence: $exists
    Pattern: $like, $ilike
    Logical: $and, $or
    
    **Example:**
        >>> from langchain_huggingface import HuggingFaceEmbeddings
        >>> 
        >>> embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        >>> rag = pgVectorDB(
        ...     collection_name="my_documents",
        ...     embedding_model=embeddings,
        ...     connection_string="postgresql+asyncpg://user:pass@localhost/db",
        ...     index_type=IndexType.DISKANN
        ... )
        >>> await rag.initialize()
        >>> await rag.add_documents(documents, labels=doc_labels)
        >>> await rag.build_index(include_labels=True)
        >>> results = await rag.semantic_search("AI applications", k=5, label_filter=[1, 2])
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
            
        Raises:
            ValidationError: If inputs are invalid
            DatabaseError: If connection fails
        """
        self._validate_init_params(collection_name, connection_string)
        
        self.table_name = collection_name
        self.embedding_model = embedding_model
        self.connection_string = connection_string
        self.schema_name = schema_name
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
        self._vector_store: Optional[PGVectorStore] = None
        self.vector_size = self._get_embedding_dimension()
        self._index_built = False
        
        logger.info(
            f"pgVectorDB initialized: '{collection_name}' with {self.index_type.value} "
            f"(vector_size={self.vector_size})"
        )

    def _validate_init_params(self, collection_name: str, connection_string: str) -> None:
        """Validate initialization parameters."""
        if not collection_name or not isinstance(collection_name, str):
            raise ValidationError("collection_name must be a non-empty string")
        if not connection_string or not isinstance(connection_string, str):
            raise ValidationError("connection_string must be a non-empty string")
        if not connection_string.startswith("postgresql"):
            raise ValidationError("connection_string must be a valid PostgreSQL connection string")

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
        
        Extensions created:
        - vector: Core pgvector extension for vector operations
        - pg_trgm: Trigram similarity for fuzzy text matching
        - vectorscale: DiskANN index support (only if using DiskANN)
        """
        try:
            async with self.sqlalchemy_engine.connect() as conn:
                # Core vector extension (required for all index types)
                await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector;"))
                logger.info("✓ Extension 'vector' enabled")
                
                # Trigram extension for fuzzy matching
                await conn.execute(text("CREATE EXTENSION IF NOT EXISTS pg_trgm;"))
                logger.info("✓ Extension 'pg_trgm' enabled")
                
                # pg_textsearch extension for native BM25
                result = await conn.execute(text(
                    "SELECT * FROM pg_available_extensions WHERE name = 'pg_textsearch';"
                ))
                if result.fetchone() is not None:
                    await conn.execute(text("CREATE EXTENSION IF NOT EXISTS pg_textsearch;"))
                    logger.info("✓ Extension 'pg_textsearch' enabled (BM25 support)")
                else:
                    logger.warning("pg_textsearch extension not available. BM25 search will not be supported.")
                
                # DiskANN extension (only if needed)
                if self.index_type == IndexType.DISKANN:
                    result = await conn.execute(text(
                        "SELECT * FROM pg_available_extensions WHERE name = 'vectorscale';"
                    ))
                    if result.fetchone() is None:
                        raise DatabaseError(
                            "pgvectorscale extension not available. "
                            "Install from: https://github.com/timescale/pgvectorscale"
                        )
                    await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vectorscale CASCADE;"))
                    logger.info("✓ Extension 'vectorscale' enabled")
                
                await conn.commit()
        except Exception as e:
            raise DatabaseError(f"Failed to ensure extensions: {e}") from e

    async def _setup_full_text_search(self) -> None:
        """Creates tsvector column, trigger, and GIN indexes for full-text and trigram search."""
        try:
            async with self.sqlalchemy_engine.connect() as conn:
                # Add tsvector column
                await conn.execute(text(
                    f'ALTER TABLE "{self.schema_name}"."{self.table_name}" '
                    f'ADD COLUMN IF NOT EXISTS content_tsvector tsvector'
                ))
                
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
                trigger_name = f"tsvector_update_on_{self.table_name}"
                await conn.execute(text(
                    f'DROP TRIGGER IF EXISTS "{trigger_name}" ON "{self.schema_name}"."{self.table_name}";'
                ))
                create_trigger_ddl = f"""
                CREATE TRIGGER "{trigger_name}" 
                BEFORE INSERT OR UPDATE ON "{self.schema_name}"."{self.table_name}"
                FOR EACH ROW EXECUTE FUNCTION update_content_tsvector();
                """
                await conn.execute(text(create_trigger_ddl))
                
                # Create GIN index on tsvector
                index_name = f"idx_{self.table_name}_content_tsvector"
                await conn.execute(text(
                    f'CREATE INDEX IF NOT EXISTS "{index_name}" '
                    f'ON "{self.schema_name}"."{self.table_name}" USING GIN(content_tsvector);'
                ))
                logger.info("✓ Full-text search index created")
                
                # Create trigram GIN index for similarity search
                trigram_index_name = f"idx_{self.table_name}_content_trgm"
                await conn.execute(text(
                    f'CREATE INDEX IF NOT EXISTS "{trigram_index_name}" '
                    f'ON "{self.schema_name}"."{self.table_name}" USING GIN(content gin_trgm_ops);'
                ))
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
            
        Example:
            >>> rag = pgVectorDB(...)
            >>> await rag.initialize()  # That's it! No SQL needed
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
            logger.info(f"✓ Step 5/5: System ready with {self.index_type.value} index (vector_size={self.vector_size})")
            logger.info("=" * 80)
            logger.info("🚀 Production RAG System initialized successfully!")
            logger.info("   - Extensions: vector, pg_trgm" + (", vectorscale" if self.index_type == IndexType.DISKANN else ""))
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

    async def add_documents(
        self, 
        documents: List[Document],
        labels: Optional[List[List[int]]] = None
    ) -> List[str]:
        """
        Add documents with optional labels for DiskANN filtering.
        
        Args:
            documents: List of LangChain Document objects
            labels: Optional list of label arrays for DiskANN (one per document)
            
        Returns:
            List of document IDs
            
        Raises:
            InitializationError: If system not initialized
            ValidationError: If documents list is empty or labels mismatch
            DatabaseError: If document insertion fails
        """
        self._ensure_initialized()
        
        if not documents:
            raise ValidationError("documents list cannot be empty")
        
        if labels is not None:
            if len(labels) != len(documents):
                raise ValidationError(f"labels length ({len(labels)}) must match documents length ({len(documents)})")
            for i, label_list in enumerate(labels):
                for label in label_list:
                    if not isinstance(label, int) or label < -32768 or label > 32767:
                        raise ValidationError(
                            f"Label {label} in document {i} is outside smallint range (-32768 to 32767)"
                        )
        
        try:
            for i, doc in enumerate(documents):
                if "langchain_id" not in doc.metadata:
                    doc.metadata["langchain_id"] = str(uuid.uuid4())
                if labels is not None and self.index_type == IndexType.DISKANN:
                    doc.metadata["labels"] = labels[i]
            
            doc_ids = await self._vector_store.aadd_documents(documents)
            
            if labels is not None and self.index_type == IndexType.DISKANN:
                await self._add_labels_column(doc_ids, labels)
            
            logger.info(f"Added {len(doc_ids)} documents")
            return doc_ids
        except Exception as e:
            raise DatabaseError(f"Failed to add documents: {e}") from e

    async def _add_labels_column(self, doc_ids: List[str], labels: List[List[int]]) -> None:
        """Add labels column for DiskANN filtering."""
        try:
            async with self.sqlalchemy_engine.connect() as conn:
                await conn.execute(text(
                    f'ALTER TABLE "{self.schema_name}"."{self.table_name}" '
                    f'ADD COLUMN IF NOT EXISTS labels SMALLINT[]'
                ))
                
                for doc_id, label_list in zip(doc_ids, labels):
                    await conn.execute(
                        text(
                            f'UPDATE "{self.schema_name}"."{self.table_name}" '
                            f'SET labels = :labels WHERE langchain_id = :doc_id'
                        ),
                        {"labels": label_list, "doc_id": doc_id}
                    )
                
                await conn.commit()
            logger.info("Labels added for DiskANN filtering")
        except Exception as e:
            raise DatabaseError(f"Failed to add labels column: {e}") from e

    async def create_metadata_index(self, columns: List[str]) -> None:
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
        lists: Optional[int] = None,
        # DiskANN parameters
        num_neighbors: int = 50,
        search_list_size: int = 100,
        max_alpha: float = 1.2,
        storage_layout: StorageLayout = StorageLayout.MEMORY_OPTIMIZED,
        num_dimensions: int = 0,
        num_bits_per_dimension: Optional[int] = None,
        include_labels: bool = False,
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
        
        if self._index_built:
            logger.warning(f"{self.index_type.value} index already built, rebuilding...")
        
        try:
            if self.index_type == IndexType.HNSW:
                await self._build_hnsw_index(metric, m, ef_construction)
            elif self.index_type == IndexType.IVFFLAT:
                await self._build_ivfflat_index(metric, lists)
            elif self.index_type == IndexType.DISKANN:
                await self._build_diskann_index(
                    metric, num_neighbors, search_list_size, max_alpha,
                    storage_layout, num_dimensions, num_bits_per_dimension, include_labels
                )
            
            self._index_built = True
            logger.info(f"{self.index_type.value} index built successfully")
        except Exception as e:
            raise DatabaseError(f"Failed to build {self.index_type.value} index: {e}") from e

    async def _build_hnsw_index(
        self, 
        metric: DistanceMetric, 
        m: int, 
        ef_construction: int
    ) -> None:
        """Build HNSW index using pgvector."""
        if m <= 0 or ef_construction <= 0:
            raise ValidationError("m and ef_construction must be positive")
        
        index = HNSWIndex(m=m, ef_construction=ef_construction)
        await self._vector_store.aapply_vector_index(index)
        logger.info(f"HNSW index built (m={m}, ef_construction={ef_construction})")

    async def _build_ivfflat_index(
        self, 
        metric: DistanceMetric, 
        lists: Optional[int]
    ) -> None:
        """Build IVFFlat index using pgvector."""
        if lists is None:
            async with self.sqlalchemy_engine.connect() as conn:
                result = await conn.execute(text(
                    f'SELECT COUNT(*) FROM "{self.schema_name}"."{self.table_name}"'
                ))
                row_count = result.scalar()
                lists = max(10, row_count // 1000)
        
        if lists <= 0:
            raise ValidationError("lists must be positive")
        
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
        num_bits_per_dimension: Optional[int],
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
        probes: Optional[int] = None,
        ef_search: Optional[int] = None,
        query_search_list_size: Optional[int] = None,
        query_rescore: Optional[int] = None,
    ) -> None:
        """
        Set query-time parameters for the active index type.
        
        Args:
            IVFFlat: probes - Number of lists to search (default: 1)
            HNSW: ef_search - Dynamic candidate list size (default: 40)
            DiskANN: query_search_list_size - Additional candidates (default: 100)
            DiskANN: query_rescore - Elements to rescore, 0 to disable (default: 50)
        """
        try:
            async with self.sqlalchemy_engine.connect() as conn:
                if self.index_type == IndexType.IVFFLAT and probes is not None:
                    if probes <= 0:
                        raise ValidationError("probes must be positive")
                    await conn.execute(text(f"SET ivfflat.probes = {probes}"))
                    logger.info(f"IVFFlat: probes={probes}")
                
                elif self.index_type == IndexType.HNSW and ef_search is not None:
                    if ef_search <= 0:
                        raise ValidationError("ef_search must be positive")
                    await conn.execute(text(f"SET hnsw.ef_search = {ef_search}"))
                    logger.info(f"HNSW: ef_search={ef_search}")
                
                elif self.index_type == IndexType.DISKANN:
                    if query_search_list_size is not None:
                        if query_search_list_size <= 0:
                            raise ValidationError("query_search_list_size must be positive")
                        await conn.execute(text(f"SET diskann.query_search_list_size = {query_search_list_size}"))
                    if query_rescore is not None:
                        if query_rescore < 0:
                            raise ValidationError("query_rescore must be non-negative")
                        await conn.execute(text(f"SET diskann.query_rescore = {query_rescore}"))
                    logger.info(f"DiskANN: search_list={query_search_list_size}, rescore={query_rescore}")
                
                await conn.commit()
        except Exception as e:
            raise DatabaseError(f"Failed to set query parameters: {e}") from e

    async def build_bm25_index(
        self,
        text_config: str = 'english',
        k1: float = 1.2,
        b: float = 0.75
    ) -> None:
        """
        Build native BM25 index using pg_textsearch extension.
        
        Args:
            text_config: PostgreSQL text search configuration (default: 'english')
            k1: Term frequency saturation parameter (default: 1.2, range: 0.1-10.0)
            b: Length normalization parameter (default: 0.75, range: 0.0-1.0)
            
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
        
        try:
            index_name = f"idx_{self.table_name}_bm25"
            
            async with self.sqlalchemy_engine.connect() as conn:
                # Check if pg_textsearch extension is available
                result = await conn.execute(text(
                    "SELECT * FROM pg_available_extensions WHERE name = 'pg_textsearch';"
                ))
                if result.fetchone() is None:
                    raise DatabaseError(
                        "pg_textsearch extension not available. "
                        "Install from: https://github.com/timescale/pg_textsearch"
                    )
                
                # Drop existing BM25 index if exists
                await conn.execute(text(
                    f'DROP INDEX IF EXISTS "{self.schema_name}"."{index_name}"'
                ))
                
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

    # ==================== SEARCH METHODS ====================

    def _validate_search_params(self, query: str, k: int) -> None:
        """Validate common search parameters."""
        if not query or not isinstance(query, str):
            raise ValidationError("query must be a non-empty string")
        if k <= 0:
            raise ValidationError("k must be positive")

    async def _keyword_search_fts(
        self, 
        query: str, 
        k: int
    ) -> List[QueryResult]:
        """
        Internal method for FTS (Full-Text Search) using PostgreSQL ts_rank.
        
        Uses traditional PostgreSQL full-text search with tsvector/tsquery.
        """
        try:
            full_query = text(f"""
                SELECT "langchain_id", "content", "langchain_metadata", 
                       ts_rank(content_tsvector, plainto_tsquery('english', :query)) as rank
                FROM "{self.schema_name}"."{self.table_name}"
                WHERE content_tsvector @@ plainto_tsquery('english', :query)
                ORDER BY rank DESC LIMIT :k
            """)
            
            async with self.sqlalchemy_engine.connect() as conn:
                result = await conn.execute(full_query, {"query": query, "k": k})
                return [
                    QueryResult(
                        id=str(row[0]),
                        content=row[1],
                        metadata=row[2] or {},
                        score=float(row[3])
                    )
                    for row in result.fetchall()
                ]
        except Exception as e:
            raise DatabaseError(f"FTS search failed: {e}") from e

    async def _keyword_search_bm25(
        self, 
        query: str, 
        k: int, 
        k1: float, 
        b: float, 
        text_config: str
    ) -> List[QueryResult]:
        """
        Internal method for BM25 search using pg_textsearch.
        
        Uses native BM25 ranking with configurable parameters:
        - k1: Term frequency saturation (1.2 default, range 0.1-10.0)
        - b: Length normalization (0.75 default, range 0.0-1.0)
        - text_config: Language configuration (english, french, german, etc.)
        
        Note: BM25 <@> operator returns NEGATIVE scores (lower = better).
        We negate them for consistency (higher = better).
        """
        try:
            index_name = f"idx_{self.table_name}_bm25"
            
            # <@> returns negative scores, negate for consistency
            full_query = text(f"""
                SELECT "langchain_id", "content", "langchain_metadata", 
                       -(content <@> to_bm25query(:query, '{index_name}')) as score
                FROM "{self.schema_name}"."{self.table_name}"
                ORDER BY content <@> to_bm25query(:query, '{index_name}')
                LIMIT :k
            """)
            
            async with self.sqlalchemy_engine.connect() as conn:
                result = await conn.execute(full_query, {"query": query, "k": k})
                return [
                    QueryResult(
                        id=str(row[0]),
                        content=row[1],
                        metadata=row[2] or {},
                        score=float(row[3])
                    )
                    for row in result.fetchall()
                ]
        except Exception as e:
            raise DatabaseError(f"BM25 search failed: {e}") from e

    async def keyword_search(
        self, 
        query: str, 
        k: int = 4,
        search_type: KeywordSearchType = KeywordSearchType.FTS,
        # BM25-specific parameters (only used when search_type='bm25')
        k1: float = 1.2,
        b: float = 0.75,
        text_config: str = 'english'
    ) -> List[QueryResult]:
        """
        METHOD 1: Pure keyword search using FTS or BM25.
        
        Args:
            query: Search query text
            k: Number of results to return
            search_type: 'fts' for PostgreSQL ts_rank or 'bm25' for native BM25
            k1: BM25 term frequency saturation (only for BM25, default: 1.2)
            b: BM25 length normalization (only for BM25, default: 0.75)
            text_config: Text search configuration (only for BM25, default: 'english')
        
        Returns documents ranked by keyword relevance.
        
        **FTS vs BM25:**
        - FTS: PostgreSQL's ts_rank, simple and fast
        - BM25: Industry-standard (Elasticsearch, Lucene), better ranking quality
        """
        self._ensure_initialized()
        self._validate_search_params(query, k)
        
        # Route to appropriate search implementation
        if search_type == KeywordSearchType.BM25:
            return await self._keyword_search_bm25(query, k, k1, b, text_config)
        else:
            return await self._keyword_search_fts(query, k)

    async def universal_keyword_search(
        self,
        query: str,
        k: int = 4,
        metadata_fields: Optional[List[str]] = None,
        search_type: KeywordSearchType = KeywordSearchType.FTS,
        # BM25-specific parameters
        k1: float = 1.2,
        b: float = 0.75,
        text_config: str = 'english'
    ) -> List[QueryResult]:
        """
        METHOD 2: Searches keywords in both content (FTS/BM25) and metadata fields (ILIKE).
        
        Useful for searching across document content AND metadata like author, title, etc.
        """
        self._ensure_initialized()
        self._validate_search_params(query, k)
        
        try:
            params = {"query": query, "k": k}
            where_conditions = ["content_tsvector @@ plainto_tsquery('english', :query)"]
            
            if metadata_fields:
                if not isinstance(metadata_fields, list):
                    raise ValidationError("metadata_fields must be a list")
                    
                params["like_query"] = f"%{query}%"
                for field in metadata_fields:
                    if not field.replace('_', '').isalnum():
                        raise ValidationError(f"Invalid field name: {field}")
                    where_conditions.append(f"(langchain_metadata->>'{field}') ILIKE :like_query")
            
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
                        score=float(row[3]) if row[3] is not None else 0.0
                    )
                    for row in result.fetchall()
                ]
        except Exception as e:
            raise DatabaseError(f"Universal keyword search failed: {e}") from e

    async def semantic_search(
        self, 
        query: str, 
        k: int = 4,
        label_filter: Optional[List[int]] = None
    ) -> List[QueryResult]:
        """
        METHOD 3: Pure semantic search using vector embeddings.
        
        Args:
            query: Search query string
            k: Number of results
            label_filter: Optional labels for DiskANN filtering (uses && operator)
        
        Returns documents ranked by semantic similarity (cosine distance).
        """
        self._ensure_initialized()
        self._validate_search_params(query, k)
        
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
                result = await conn.execute(full_query, params)
                return [
                    QueryResult(
                        id=str(row[0]),
                        content=row[1],
                        metadata=row[2] or {},
                        score=float(row[3])
                    )
                    for row in result.fetchall()
                ]
        except Exception as e:
            raise DatabaseError(f"Semantic search failed: {e}") from e

    async def metadata_filter(
        self,
        filter: Dict[str, Any],
        k: int = 4,
        order_by: Optional[str] = None,
        ascending: bool = True
    ) -> List[QueryResult]:
        """
        METHOD 4: Pure metadata filtering without any search query.
        
        Returns documents matching metadata criteria, ordered by specified field or insertion order.
        
        Args:
            filter: Metadata filter dictionary using filter operators
            k: Number of results to return
            order_by: Optional metadata field to order by (default: None = insertion order)
            ascending: Sort order direction (default: True)
        
        Returns:
            List of documents matching the filter criteria
        
        Example:
            >>> # Get recent high-priority documents
            >>> results = await rag.metadata_filter(
            ...     filter={"priority": {"$gte": 8}, "status": "active"},
            ...     k=10,
            ...     order_by="created_date",
            ...     ascending=False
            ... )
        """
        self._ensure_initialized()
        
        if k <= 0:
            raise ValidationError("k must be positive")
        
        if not filter:
            raise ValidationError("filter cannot be empty")
        
        try:
            filter_clauses, params = self._build_filter_clauses_wrapper(filter)
            params["k"] = k
            
            # Build ORDER BY clause
            if order_by:
                if not order_by.replace('_', '').isalnum():
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
                    QueryResult(
                        id=str(row[0]),
                        content=row[1],
                        metadata=row[2] or {},
                        score=1.0
                    )
                    for row in result.fetchall()
                ]
        except Exception as e:
            raise DatabaseError(f"Metadata filter failed: {e}") from e

    async def metadata_keyword_search(
        self, 
        query: str, 
        filter: Dict[str, Any], 
        k: int = 4,
        search_type: KeywordSearchType = KeywordSearchType.FTS,
        # BM25-specific parameters
        k1: float = 1.2,
        b: float = 0.75,
        text_config: str = 'english'
    ) -> List[QueryResult]:
        """
        METHOD 5: MANDATORY metadata filtering FIRST, then keyword search (FTS or BM25).
        
        Execution order (enforced via CTE):
        1. Filter documents by metadata criteria
        2. Perform full-text search on filtered results only
        
        This ensures metadata constraints are always respected.
        """
        self._ensure_initialized()
        
        if not query or not query.strip():
            logger.warning("No query provided for metadata_keyword_search, falling back to metadata_filter")
            return await self.metadata_filter(filter, k)
        
        self._validate_search_params(query, k)
        
        if not filter:
            logger.warning("No filter provided for metadata_keyword_search, falling back to keyword_search")
            return await self.keyword_search(query, k)
        
        try:
            filter_clauses, params = self._build_filter_clauses_wrapper(filter)
            params.update({"query": query, "k": k})
            
            # Use CTE to enforce metadata filtering FIRST, then keyword search
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
                        score=float(row[3])
                    )
                    for row in result.fetchall()
                ]
        except Exception as e:
            raise DatabaseError(f"Metadata keyword search failed: {e}") from e

    async def metadata_semantic_search(
        self, 
        query: str, 
        filter: Dict[str, Any], 
        k: int = 4
    ) -> List[QueryResult]:
        """
        METHOD 6: MANDATORY metadata filtering FIRST, then semantic search.
        
        Execution order (enforced via CTE):
        1. Filter documents by metadata criteria
        2. Perform vector similarity search on filtered results only
        
        This ensures metadata constraints are always respected.
        """
        self._ensure_initialized()
        
        if not query or not query.strip():
            logger.warning("No query provided for metadata_semantic_search, falling back to metadata_filter")
            return await self.metadata_filter(filter, k)
        
        self._validate_search_params(query, k)
        
        if not filter:
            logger.warning("No filter provided for metadata_semantic_search, falling back to semantic_search")
            return await self.semantic_search(query, k)
        
        try:
            embedding = self.embedding_model.embed_query(query)
            filter_clauses, params = self._build_filter_clauses_wrapper(filter)
            params.update({"embedding": str(embedding), "k": k})
            
            # Use CTE to enforce metadata filtering FIRST, then semantic search
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
                result = await conn.execute(full_query, params)
                return [
                    QueryResult(
                        id=str(row[0]),
                        content=row[1],
                        metadata=row[2] or {},
                        score=float(row[3])
                    )
                    for row in result.fetchall()
                ]
        except Exception as e:
            raise DatabaseError(f"Metadata semantic search failed: {e}") from e

    def _validate_weights(self, weights: Tuple[float, float]) -> None:
        """Validate hybrid search weights."""
        if len(weights) != 2:
            raise ValidationError("weights must be a tuple of 2 floats")
        if not all(isinstance(w, (int, float)) and w >= 0 for w in weights):
            raise ValidationError("weights must be non-negative numbers")
        weight_sum = sum(weights)
        if not (0.99 <= weight_sum <= 1.01):
            raise ValidationError(f"weights must sum to 1.0, got {weight_sum}")

    def _fuse_results(
        self,
        semantic_results: List[QueryResult],
        keyword_results: List[QueryResult],
        weights: Tuple[float, float],
        k: int
    ) -> List[QueryResult]:
        """Common fusion logic for hybrid and ensemble search using weighted scores."""
        semantic_scores = self._normalize_scores(
            {r['id']: r['score'] for r in semantic_results},
            inverse=True
        )
        keyword_scores = self._normalize_scores(
            {r['id']: r['score'] for r in keyword_results},
            inverse=False
        )
        
        combined_scores: Dict[str, float] = {}
        doc_map: Dict[str, QueryResult] = {}
        
        for res in semantic_results:
            doc_map[res['id']] = res
            combined_scores[res['id']] = semantic_scores.get(res['id'], 0.0) * weights[0]
        
        for res in keyword_results:
            doc_map[res['id']] = res
            if res['id'] in combined_scores:
                combined_scores[res['id']] += keyword_scores.get(res['id'], 0.0) * weights[1]
            else:
                combined_scores[res['id']] = keyword_scores.get(res['id'], 0.0) * weights[1]
        
        sorted_ids = sorted(combined_scores.keys(), key=lambda x: combined_scores[x], reverse=True)
        
        return [
            QueryResult(
                id=doc_id,
                content=doc_map[doc_id]['content'],
                metadata=doc_map[doc_id]['metadata'],
                score=combined_scores[doc_id]
            )
            for doc_id in sorted_ids[:k]
        ]

    def _fuse_results_rrf(
        self,
        semantic_results: List[QueryResult],
        keyword_results: List[QueryResult],
        k: int,
        rrf_k: int = 60
    ) -> List[QueryResult]:
        """
        Reciprocal Rank Fusion (RRF) scoring for hybrid searches.
        
        RRF Formula: score = sum(1 / (k + rank)) for each retrieval method
        
        Args:
            semantic_results: Results from semantic search
            keyword_results: Results from keyword search
            k: Number of final results
            rrf_k: RRF constant (default: 60, from original paper)
        
        Returns:
            Fused results ranked by RRF score
        """
        combined_scores: Dict[str, float] = {}
        doc_map: Dict[str, QueryResult] = {}
        
        # Add semantic search rankings
        for rank, res in enumerate(semantic_results, start=1):
            doc_map[res['id']] = res
            combined_scores[res['id']] = 1.0 / (rrf_k + rank)
        
        # Add keyword search rankings
        for rank, res in enumerate(keyword_results, start=1):
            doc_map[res['id']] = res
            if res['id'] in combined_scores:
                combined_scores[res['id']] += 1.0 / (rrf_k + rank)
            else:
                combined_scores[res['id']] = 1.0 / (rrf_k + rank)
        
        sorted_ids = sorted(combined_scores.keys(), key=lambda x: combined_scores[x], reverse=True)
        
        return [
            QueryResult(
                id=doc_id,
                content=doc_map[doc_id]['content'],
                metadata=doc_map[doc_id]['metadata'],
                score=combined_scores[doc_id]
            )
            for doc_id in sorted_ids[:k]
        ]

    async def hybrid_search(
        self, 
        query: str, 
        k: int = 4, 
        weights: Tuple[float, float] = (0.5, 0.5),
        label_filter: Optional[List[int]] = None,
        use_rrf: bool = False,
        rrf_k: int = 60,
        keyword_type: KeywordSearchType = KeywordSearchType.FTS,
        # BM25-specific parameters
        bm25_k1: float = 1.2,
        bm25_b: float = 0.75,
        text_config: str = 'english'
    ) -> List[QueryResult]:
        """
        METHOD 7: Combines keyword (FTS or BM25) and semantic search using weighted fusion or RRF.
        
        Args:
            query: Search query
            k: Number of results
            weights: (semantic_weight, keyword_weight) must sum to 1.0 (ignored if use_rrf=True)
            label_filter: Optional labels for DiskANN filtering
            use_rrf: Use Reciprocal Rank Fusion instead of weighted scoring (default: False)
            rrf_k: RRF constant (default: 60)
        
        Best for: Balancing exact keyword matching with semantic understanding.
        
        **Scoring Methods:**
        - Weighted (use_rrf=False): Normalized scores with custom weights
        - RRF (use_rrf=True): Reciprocal Rank Fusion, no weight tuning needed
        """
        self._ensure_initialized()
        
        if not query or not query.strip():
            logger.warning("No query provided for hybrid_search, cannot perform search without query")
            raise ValidationError("hybrid_search requires a non-empty query")
        
        self._validate_search_params(query, k)
        
        if not use_rrf:
            self._validate_weights(weights)
        
        try:
            semantic_results = await self.semantic_search(query, k=k*2, label_filter=label_filter)
            keyword_results = await self.keyword_search(
                query, k=k*2, search_type=keyword_type,
                k1=bm25_k1, b=bm25_b, text_config=text_config
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
        filter: Dict[str, Any], 
        k: int = 4,
        weights: Tuple[float, float] = (0.5, 0.5),
        use_rrf: bool = False,
        rrf_k: int = 60,
        keyword_type: KeywordSearchType = KeywordSearchType.FTS,
        # BM25-specific parameters
        bm25_k1: float = 1.2,
        bm25_b: float = 0.75,
        text_config: str = 'english'
    ) -> List[QueryResult]:
        """
        METHOD 8: Filter by metadata, then combine keyword (FTS or BM25) and semantic search.
        
        Most comprehensive search: metadata filtering + hybrid (keyword + semantic).
        
        Args:
            query: Search query
            filter: Metadata filter dictionary
            k: Number of results
            weights: (semantic_weight, keyword_weight) must sum to 1.0 (ignored if use_rrf=True)
            use_rrf: Use Reciprocal Rank Fusion instead of weighted scoring (default: False)
            rrf_k: RRF constant (default: 60)
            keyword_type: 'fts' or 'bm25' for keyword search implementation
        """
        self._ensure_initialized()
        
        if not query or not query.strip():
            logger.warning("No query provided for ensemble_search, falling back to metadata_filter")
            return await self.metadata_filter(filter, k)
        
        self._validate_search_params(query, k)
        
        if not use_rrf:
            self._validate_weights(weights)
        
        if not filter:
            logger.warning("No filter provided for ensemble_search, falling back to hybrid_search")
            return await self.hybrid_search(
                query, k, weights, use_rrf=use_rrf, rrf_k=rrf_k,
                keyword_type=keyword_type, bm25_k1=bm25_k1, bm25_b=bm25_b, text_config=text_config
            )
        
        try:
            semantic_results = await self.metadata_semantic_search(query, filter, k=k*2)
            keyword_results = await self.metadata_keyword_search(
                query, filter, k=k*2, search_type=keyword_type,
                k1=bm25_k1, b=bm25_b, text_config=text_config
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
        threshold: float = 0.3
    ) -> List[QueryResult]:
        """
        METHOD 9: Fuzzy text matching using trigram similarity.
        
        Uses PostgreSQL pg_trgm for typo-tolerant searching. Good for handling
        spelling variations, partial matches, and user input errors.
        
        Args:
            query: Search query string
            k: Number of results
            threshold: Minimum similarity score (0.0-1.0, default: 0.3)
        
        Returns:
            Documents ranked by trigram similarity score
        
        Example:
            >>> # Finds "machine learning" even with typos like "machin lerning"
            >>> results = await rag.trigram_search("artifical inteligence", k=5, threshold=0.4)
        """
        self._ensure_initialized()
        self._validate_search_params(query, k)
        
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
                result = await conn.execute(full_query, {"query": query, "threshold": threshold, "k": k})
                return [
                    QueryResult(
                        id=str(row[0]),
                        content=row[1],
                        metadata=row[2] or {},
                        score=float(row[3])
                    )
                    for row in result.fetchall()
                ]
        except Exception as e:
            raise DatabaseError(f"Trigram search failed: {e}") from e

    async def metadata_trigram_search(
        self, 
        query: str, 
        filter: Dict[str, Any], 
        k: int = 4,
        threshold: float = 0.3
    ) -> List[QueryResult]:
        """
        METHOD 10: MANDATORY metadata filtering FIRST, then fuzzy text matching.
        
        Execution order (enforced via CTE):
        1. Filter documents by metadata criteria
        2. Perform trigram similarity search on filtered results only
        
        This ensures metadata constraints are always respected.
        
        Args:
            query: Search query string
            filter: Metadata filter dictionary (REQUIRED)
            k: Number of results
            threshold: Minimum similarity score (0.0-1.0, default: 0.3)
        
        Returns:
            Documents matching filter criteria, ranked by trigram similarity
        
        Example:
            >>> # Fuzzy search within a specific category
            >>> results = await rag.metadata_trigram_search(
            ...     query="neurla netwrk",  # typos handled
            ...     filter={"category": "ai", "year": {"$gte": 2020}},
            ...     k=5,
            ...     threshold=0.4
            ... )
        """
        self._ensure_initialized()
        
        if not query or not query.strip():
            logger.warning("No query provided for metadata_trigram_search, falling back to metadata_filter")
            return await self.metadata_filter(filter, k)
        
        self._validate_search_params(query, k)
        
        if not filter:
            logger.warning("No filter provided for metadata_trigram_search, falling back to trigram_search")
            return await self.trigram_search(query, k, threshold)
        
        if threshold < 0.0 or threshold > 1.0:
            raise ValidationError("threshold must be between 0.0 and 1.0")
        
        try:
            filter_clauses, params = self._build_filter_clauses_wrapper(filter)
            params.update({"query": query, "threshold": threshold, "k": k})
            
            # Use CTE to enforce metadata filtering FIRST, then trigram search
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
                        score=float(row[3])
                    )
                    for row in result.fetchall()
                ]
        except Exception as e:
            raise DatabaseError(f"Metadata trigram search failed: {e}") from e

    # ==================== FILTER BUILDING METHODS ====================

    def _build_filter_clauses(self, filter: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
        """Build SQL WHERE clauses from filter dictionary."""
        if not filter:
            return "1=1", {}
        where_clause, params, _ = self._parse_filter(filter, {}, 0)
        return where_clause, params
    
    def _parse_filter(
        self, 
        filter: Dict[str, Any], 
        params: Dict[str, Any], 
        counter: int
    ) -> Tuple[str, Dict[str, Any], int]:
        """Recursively parse filter conditions."""
        filter_expressions = []
        
        for key, condition in filter.items():
            if key == "$and":
                and_clauses = []
                for sub_filter in condition:
                    clause, params, counter = self._parse_filter(sub_filter, params, counter)
                    and_clauses.append(f"({clause})")
                filter_expressions.append(" AND ".join(and_clauses))
                continue
            
            if key == "$or":
                or_clauses = []
                for sub_filter in condition:
                    clause, params, counter = self._parse_filter(sub_filter, params, counter)
                    or_clauses.append(f"({clause})")
                filter_expressions.append("(" + " OR ".join(or_clauses) + ")")
                continue
            
            if isinstance(condition, dict):
                for op_key, value in condition.items():
                    expr, params, counter = self._build_single_condition(
                        key, op_key, value, params, counter
                    )
                    filter_expressions.append(expr)
            else:
                expr, params, counter = self._build_single_condition(
                    key, "$eq", condition, params, counter
                )
                filter_expressions.append(expr)
        
        where_clause = " AND ".join(filter_expressions) if filter_expressions else "1=1"
        return where_clause, params, counter
    
    def _build_single_condition(
        self, 
        key: str, 
        operator: str, 
        value: Any, 
        params: Dict[str, Any], 
        counter: int
    ) -> Tuple[str, Dict[str, Any], int]:
        """Build a single filter condition with proper type handling."""
        param_name = f"param_{counter}"
        counter += 1
        
        is_numeric = isinstance(value, (int, float))
        if operator == "$between" and isinstance(value, (list, tuple)) and len(value) > 0:
            is_numeric = isinstance(value[0], (int, float))
        
        field_expr = f"(langchain_metadata->>'{key}')::numeric" if is_numeric else f"langchain_metadata->>'{key}'"
        
        op_map = {"$eq": "=", "$ne": "!=", "$lt": "<", "$lte": "<=", "$gt": ">", "$gte": ">="}
        if operator in op_map:
            params[param_name] = value
            return f"{field_expr} {op_map[operator]} :{param_name}", params, counter
        
        elif operator == "$in":
            if not isinstance(value, (list, tuple)):
                value = [value]
            if not is_numeric:
                value = [str(v) for v in value]
            params[param_name] = tuple(value)
            return f"{field_expr} = ANY(:{param_name})", params, counter
        
        elif operator == "$nin":
            if not isinstance(value, (list, tuple)):
                value = [value]
            if not is_numeric:
                value = [str(v) for v in value]
            params[param_name] = tuple(value)
            return f"{field_expr} != ALL(:{param_name})", params, counter
        
        elif operator == "$between":
            if not isinstance(value, (list, tuple)) or len(value) != 2:
                raise ValidationError(f"$between requires a list/tuple of 2 values, got: {value}")
            param_name_2 = f"param_{counter}"
            counter += 1
            params[param_name] = value[0]
            params[param_name_2] = value[1]
            return f"{field_expr} BETWEEN :{param_name} AND :{param_name_2}", params, counter
        
        elif operator == "$exists":
            condition = "IS NOT NULL" if value else "IS NULL"
            return f"langchain_metadata->>'{key}' {condition}", params, counter
        
        elif operator == "$like":
            params[param_name] = value
            return f"langchain_metadata->>'{key}' LIKE :{param_name}", params, counter
        
        elif operator == "$ilike":
            params[param_name] = value
            return f"langchain_metadata->>'{key}' ILIKE :{param_name}", params, counter
        
        else:
            raise ValidationError(f"Unsupported operator: {operator}")
    
    def _build_filter_clauses_wrapper(self, filter: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
        """Wrapper for backward compatibility."""
        where_clause, params, _ = self._parse_filter(filter, {}, 0)
        return where_clause, params

    def _normalize_scores(
        self, 
        scores: Dict[str, float], 
        inverse: bool = False
    ) -> Dict[str, float]:
        """Normalize scores to 0-1 range."""
        if not scores:
            return {}
        
        values = list(scores.values())
        min_score = min(values)
        max_score = max(values)
        
        if max_score == min_score:
            return {k: 1.0 for k in scores.keys()}
        
        if inverse:
            return {
                k: 1.0 - (v - min_score) / (max_score - min_score)
                for k, v in scores.items()
            }
        else:
            return {
                k: (v - min_score) / (max_score - min_score)
                for k, v in scores.items()
            }

    # ==================== UTILITY METHODS ====================

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
                result = await conn.execute(text(
                    f'SELECT COUNT(*) FROM "{self.schema_name}"."{self.table_name}"'
                ))
                stats["document_count"] = result.scalar()
                
                # Get index names
                result = await conn.execute(text(
                    """
                    SELECT indexname, indexdef FROM pg_indexes 
                    WHERE schemaname = :schema AND tablename = :table
                    """
                ), {"schema": self.schema_name, "table": self.table_name})
                stats["indexes"] = [{"name": row[0], "definition": row[1]} for row in result.fetchall()]
                
                # Get table size
                result = await conn.execute(text(
                    f"""
                    SELECT pg_size_pretty(pg_total_relation_size('"{self.schema_name}"."{self.table_name}"'))
                    """
                ))
                stats["table_size"] = result.scalar()
        except Exception as e:
            logger.warning(f"Could not fetch complete stats: {e}")
        
        return stats

    async def close(self) -> None:
        """Close database connections and cleanup resources."""
        try:
            await self.sqlalchemy_engine.dispose()
            logger.info("Database connections closed")
        except Exception as e:
            logger.error(f"Error closing connections: {e}")
            raise DatabaseError(f"Failed to close connections: {e}") from e
