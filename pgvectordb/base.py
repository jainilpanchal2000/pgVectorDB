"""
Base Module - Enums, Exceptions, Constants, and Type Definitions
=================================================================

This module contains foundational components used throughout pgVectorDB:

- **Enums**: IndexType, KeywordSearchType, StorageLayout, DistanceMetric,
             VectorPrecision, IterativeScanMode
- **Exceptions**: RetrievalSystemError, InitializationError, ValidationError,
                  DatabaseError, RateLimitError
- **Constants**: ALLOWED_TEXT_CONFIGS, VALID_QUERY_PARAMS
- **Type Definitions**: QueryResult

Usage:
    >>> from pgvectordb.base import IndexType, ValidationError, QueryResult
    >>> index = IndexType.HNSW
    >>> raise ValidationError("Invalid input")
"""

from enum import Enum
from typing import Dict, Any, TypedDict


# ==================== Enums ====================


class IndexType(str, Enum):
    """
    Supported vector index types for approximate nearest neighbor search.

    Each index type has different performance characteristics:

    Attributes:
        HNSW: Hierarchical Navigable Small World graph.
            - Best for: <1M vectors
            - Pros: Fast queries, high recall, no training required
            - Cons: Higher memory usage, slower inserts

        IVFFLAT: Inverted File with Flat quantization.
            - Best for: 100K-10M vectors
            - Pros: Balanced performance, lower memory than HNSW
            - Cons: Requires training step, lower recall

        DISKANN: Disk-based Approximate Nearest Neighbor (via pgvectorscale).
            - Best for: >10M vectors
            - Pros: Scalable, low memory, label filtering support
            - Cons: Requires vectorscale extension
            - Extension: Requires 'vectorscale' PostgreSQL extension

    Example:
        >>> from pgvectordb.base import IndexType
        >>> rag = pgVectorDB(
        ...     collection_name="docs",
        ...     index_type=IndexType.HNSW,  # Fast for small datasets
        ...     ...
        ... )
    """

    HNSW = "hnsw"
    IVFFLAT = "ivfflat"
    DISKANN = "diskann"


class KeywordSearchType(str, Enum):
    """
    Keyword search implementations for full-text search.

    Attributes:
        FTS: PostgreSQL Full-Text Search using ts_rank.
            - Uses: Built-in PostgreSQL text search
            - Ranking: ts_rank function
            - No additional extension required

        BM25: Best Matching 25 algorithm (via pg_textsearch).
            - Uses: Native BM25 ranking (like Elasticsearch, Lucene)
            - Ranking: BM25 formula with configurable k1 and b
            - Extension: Requires 'pg_textsearch' PostgreSQL extension

    Example:
        >>> from pgvectordb.base import KeywordSearchType
        >>> results = await rag.keyword_search(
        ...     "machine learning",
        ...     search_type=KeywordSearchType.BM25  # Better relevance ranking
        ... )
    """

    FTS = "fts"  # PostgreSQL ts_rank (full-text search)
    BM25 = "bm25"  # pg_textsearch native BM25


class StorageLayout(str, Enum):
    """
    Storage layout options for DiskANN index (pgvectorscale).

    Attributes:
        MEMORY_OPTIMIZED: Uses Statistical Binary Quantization (SBQ).
            - Storage: Compressed vectors (significant savings)
            - Performance: Slightly slower queries
            - Memory: Much lower memory usage
            - Best for: Large datasets where memory is constrained

        PLAIN: Uncompressed vector storage.
            - Storage: Full precision vectors
            - Performance: Fastest queries
            - Memory: Highest memory usage
            - Best for: When query speed is critical

    Example:
        >>> from pgvectordb.base import StorageLayout
        >>> await rag.build_index(
        ...     storage_layout=StorageLayout.MEMORY_OPTIMIZED  # 75% less memory
        ... )
    """

    MEMORY_OPTIMIZED = "memory_optimized"  # Uses SBQ compression
    PLAIN = "plain"  # Uncompressed


class DistanceMetric(str, Enum):
    """
    Distance metrics for vector similarity operations.

    Different metrics are suited for different embedding models and use cases.

    Attributes:
        COSINE: Cosine distance (1 - cosine similarity).
            - Operator: <=>
            - Best for: Normalized embeddings (most common)
            - Range: 0 (identical) to 2 (opposite)

        L2: Euclidean distance (L2 norm).
            - Operator: <->
            - Best for: When magnitude matters
            - Range: 0 to infinity

        INNER_PRODUCT: Negative inner product.
            - Operator: <#>
            - Best for: Dot product similarity (normalized vectors)
            - Note: Negated, so ORDER BY ASC gives highest similarity

        L1: Manhattan distance (L1 norm / taxicab distance).
            - Operator: <+>
            - Best for: Sparse features, grid-like data
            - Range: 0 to infinity

        HAMMING: Hamming distance for binary vectors.
            - Operator: <~>
            - Best for: Binary embeddings, fingerprints
            - Range: 0 to vector dimension

        JACCARD: Jaccard distance for binary vectors.
            - Operator: <%>
            - Best for: Set similarity, binary features
            - Range: 0 (identical) to 1 (no overlap)

    Example:
        >>> from pgvectordb.base import DistanceMetric
        >>> await rag.build_index(metric=DistanceMetric.COSINE)
    """

    COSINE = "cosine"  # <=> operator
    L2 = "l2"  # <-> operator (Euclidean)
    INNER_PRODUCT = "inner_product"  # <#> operator
    L1 = "l1"  # <+> operator (Manhattan/Taxicab)
    HAMMING = "hamming"  # <~> operator (binary vectors)
    JACCARD = "jaccard"  # <%> operator (binary vectors)


class VectorPrecision(str, Enum):
    """
    Vector precision types for storage optimization.

    Lower precision reduces storage requirements but may affect accuracy.

    Attributes:
        FLOAT32: Default full precision (4 bytes per dimension).
            - Max dimensions: 2,000
            - Best for: Maximum accuracy

        FLOAT16: Half precision (2 bytes per dimension).
            - Max dimensions: 4,000
            - Storage savings: 50%
            - Best for: When storage is limited

        BINARY: Binary vectors (1 bit per dimension).
            - Max dimensions: 64,000
            - Storage savings: 97%
            - Best for: Binary embeddings, Hamming distance

    Example:
        >>> from pgvectordb.base import VectorPrecision
        >>> await rag.create_halfvec_table()  # Uses FLOAT16
    """

    FLOAT32 = "float32"  # Default: 4 bytes per dimension
    FLOAT16 = "float16"  # Half-precision: 2 bytes per dimension (halfvec)
    BINARY = "binary"  # Binary: 1 bit per dimension


class IterativeScanMode(str, Enum):
    """
    Iterative scan modes for filtered searches (pgvector 0.8+).

    When using metadata filters with vector search, iterative scans can
    improve recall by examining more candidates.

    Attributes:
        OFF: Disable iterative scanning.
            - Behavior: Standard index scan
            - Recall: Lower with selective filters

        STRICT_ORDER: Exact distance ordering.
            - Behavior: Guarantees results in exact distance order
            - Performance: Slower but precise

        RELAXED_ORDER: Better recall with slight order variance.
            - Behavior: May return results in slightly different order
            - Performance: Better recall, faster than strict
            - Recommended: For most filtered search use cases

    Example:
        >>> from pgvectordb.base import IterativeScanMode
        >>> await rag.set_iterative_scan(IterativeScanMode.RELAXED_ORDER)
    """

    OFF = "off"
    STRICT_ORDER = "strict_order"  # Exact distance ordering
    RELAXED_ORDER = "relaxed_order"  # Better recall, slight order variance


# ==================== Security Constants ====================

# Allowlist for BM25 text search configurations (PostgreSQL text search configs)
ALLOWED_TEXT_CONFIGS = frozenset(
    [
        "simple",
        "arabic",
        "armenian",
        "basque",
        "catalan",
        "danish",
        "dutch",
        "english",
        "finnish",
        "french",
        "german",
        "greek",
        "hindi",
        "hungarian",
        "indonesian",
        "irish",
        "italian",
        "lithuanian",
        "nepali",
        "norwegian",
        "portuguese",
        "romanian",
        "russian",
        "serbian",
        "spanish",
        "swedish",
        "tamil",
        "turkish",
        "yiddish",
    ]
)

# Allowlist for query parameters (GUC parameters)
VALID_QUERY_PARAMS = frozenset(
    [
        "ivfflat.probes",
        "hnsw.ef_search",
        "diskann.query_search_list_size",
        "diskann.query_rescore",
        "hnsw.iterative_scan",
        "ivfflat.iterative_scan",
        "hnsw.max_scan_tuples",
        "hnsw.scan_mem_multiplier",
        "ivfflat.max_probes",
        # DiskANN Build Params
        "diskann.force_parallel_workers",
        "diskann.min_vectors_for_parallel_build",
        "diskann.parallel_flush_interval",
        "diskann.parallel_initial_start_nodes_count",
    ]
)


# ==================== Custom Exceptions ====================


class RetrievalSystemError(Exception):
    """
    Base exception for all pgVectorDB retrieval system errors.

    All custom exceptions in this module inherit from this class,
    allowing you to catch any pgVectorDB error with a single except clause.

    Example:
        >>> try:
        ...     await rag.semantic_search("query")
        ... except RetrievalSystemError as e:
        ...     print(f"pgVectorDB error: {e}")
    """

    pass


class InitializationError(RetrievalSystemError):
    """
    Raised when the system is not properly initialized.

    This error occurs when you try to perform operations before calling
    `initialize()`, or when required extensions are not available.

    Common causes:
        - Calling search methods before `initialize()`
        - Missing required PostgreSQL extensions (pgvector, vectorscale, pg_textsearch)
        - Database connection issues during initialization

    Example:
        >>> rag = pgVectorDB(...)
        >>> await rag.semantic_search("query")  # Raises InitializationError
        InitializationError: System not initialized. Call initialize() first.

        >>> await rag.initialize()
        >>> await rag.semantic_search("query")  # Works now
    """

    pass


class ValidationError(RetrievalSystemError):
    """
    Raised when input validation fails.

    This error occurs when invalid parameters are passed to methods.

    Common causes:
        - Empty document or query strings
        - Invalid collection names (special characters)
        - Negative k values in search
        - Label values outside smallint range
        - Invalid filter operator syntax

    Example:
        >>> await rag.semantic_search("", k=5)
        ValidationError: query must be a non-empty string

        >>> await rag.semantic_search("query", k=-1)
        ValidationError: k must be a positive integer
    """

    pass


class DatabaseError(RetrievalSystemError):
    """
    Raised when database operations fail.

    This error wraps underlying database exceptions and provides
    context about what operation failed.

    Common causes:
        - Connection failures
        - Permission denied
        - Table not found
        - Index creation failures
        - SQL syntax errors (internal)

    Example:
        >>> await rag.add_documents(docs)
        DatabaseError: Failed to add documents: connection refused
    """

    pass


class RateLimitError(RetrievalSystemError):
    """
    Raised when embedding API rate limit is hit.

    This error should NOT be retried immediately. Wait before retrying.

    Common causes:
        - Too many embedding requests to external API
        - API quota exceeded
        - Concurrent batch operations hitting rate limits

    Handling:
        - Implement exponential backoff
        - Reduce batch sizes
        - Use local embedding model if rate limits are frequent

    Example:
        >>> try:
        ...     await rag.add_documents_batch(large_batch)
        ... except RateLimitError:
        ...     await asyncio.sleep(60)  # Wait before retry
        ...     await rag.add_documents_batch(large_batch, batch_size=50)  # Smaller batches
    """

    pass


# ==================== Type Definitions ====================


class QueryResult(TypedDict):
    """
    Structured result from search operations.

    This TypedDict provides type hints for search results, making it easier
    to work with results in type-checked code.

    Attributes:
        id: Unique document identifier (langchain_id)
        content: Document text content
        metadata: Document metadata dictionary
        score: Relevance/similarity score (interpretation varies by search method)

    Note on scores:
        - Semantic search: Lower is more similar (distance)
        - Keyword search (FTS): Higher is more relevant
        - Keyword search (BM25): Higher is more relevant (negated from raw scores)
        - Hybrid search: Normalized/fused scores

    Example:
        >>> results: List[QueryResult] = await rag.semantic_search("query")
        >>> for result in results:
        ...     print(f"ID: {result['id']}")
        ...     print(f"Score: {result['score']:.4f}")
        ...     print(f"Content: {result['content'][:100]}...")
    """

    id: str
    content: str
    metadata: Dict[str, Any]
    score: float


# ==================== Extension Requirements ====================

# Map of features to their required extensions
EXTENSION_REQUIREMENTS = {
    "DiskANN index": "vectorscale",
    "BM25 search": "pg_textsearch",
    "Label filtering": "vectorscale",
    "DiskANN build parameters": "vectorscale",
    "BM25 index": "pg_textsearch",
    "BM25 index stats": "pg_textsearch",
    "dump_bm25_index": "pg_textsearch",
    "spill_bm25_index": "pg_textsearch",
}


__all__ = [
    # Enums
    "IndexType",
    "KeywordSearchType",
    "StorageLayout",
    "DistanceMetric",
    "VectorPrecision",
    "IterativeScanMode",
    # Exceptions
    "RetrievalSystemError",
    "InitializationError",
    "ValidationError",
    "DatabaseError",
    "RateLimitError",
    # Constants
    "ALLOWED_TEXT_CONFIGS",
    "VALID_QUERY_PARAMS",
    "EXTENSION_REQUIREMENTS",
    # Types
    "QueryResult",
]
