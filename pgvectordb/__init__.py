"""
pgVectorDB - Production PostgreSQL Vector Database
===================================================

**Version:** 0.0.5.post1
**Status:** Production-Ready with Multi-Embedding Support

A comprehensive PostgreSQL-based RAG (Retrieval-Augmented Generation) system with
advanced vector indexing, multiple search methods, and production utilities.

Module Structure
----------------
- **core.py**: Main pgVectorDB class and all functionality
- **base.py**: Enums, exceptions, constants, type definitions
- **extensions.py**: PostgreSQL extension management with graceful degradation
- **config.py**: Configuration defaults and helpers
- **metrics.py**: RAG evaluation metrics
- **schema.py**: SQLAlchemy table definitions
- **spaces.py**: Vector space abstractions for multi-embedding search

Extension Requirements
----------------------
- **pgvector** (REQUIRED): Core vector operations
- **vectorscale** (OPTIONAL): DiskANN index support
- **pg_textsearch** (OPTIONAL): BM25 keyword search

Quick Start
-----------
    >>> from pgvectordb import pgVectorDB, IndexType
    >>> pgvdb = pgVectorDB(
    ...     collection_name="docs",
    ...     embedding_model=embeddings,
    ...     connection_string="postgresql+asyncpg://user:pass@localhost/db"
    ... )
    >>> await pgvdb.initialize()
    >>> await pgvdb.add_documents(documents)
    >>> results = await pgvdb.semantic_search("query", k=5)
"""

# Import from new modular base
from .base import (
    # Constants
    ALLOWED_TEXT_CONFIGS,
    EXTENSION_REQUIREMENTS,
    VALID_QUERY_PARAMS,
    DatabaseError,
    DistanceMetric,
    IndexType,
    InitializationError,
    IterativeScanMode,
    KeywordSearchType,
    # Types
    QueryResult,
    RateLimitError,
    # Exceptions
    RetrievalSystemError,
    # Enums
    SearchMethod,
    StorageLayout,
    ValidationError,
    VectorPrecision,
)

# Import main class from core (backward compatible)
from .core import pgVectorDB

# Import extension manager
from .extensions import ExtensionManager

# Import query builders (v0.0.6)
try:
    from .query.unified import SearchConfig, UnifiedQueryBuilder
except ImportError:
    UnifiedQueryBuilder = None
    SearchConfig = None

# Import metrics
# Import config
from .config import Config, get_production_config, get_test_config
from .metrics import (
    EvaluationDataset,
    EvaluationResult,
    KValueAnalysis,
    RAGEvaluator,
    create_sample_evaluation_dataset,
)

# Import schema helpers
from .schema import (
    build_qualified_name,
    get_distance_operator,
    get_index_ops,
    get_label_definitions_table,
    get_vector_table,
    quote_identifier,
)

# Import spaces module (v0.0.3)
try:
    from .spaces import (
        CategorySpace,
        NumberMode,
        NumberSpace,
        RecencySpace,
        TextSpace,
        TimeUnit,
        VectorSpace,
        encode_document_spaces,
        encode_query_spaces,
        validate_spaces,
    )
except ImportError:
    VectorSpace = None
    TextSpace = None
    NumberSpace = None
    CategorySpace = None
    RecencySpace = None
    NumberMode = None
    TimeUnit = None

# Import rerankers module (v0.0.3)
try:
    from .rerankers import (
        AWSBedrockReranker,
        BaseReranker,
        CohereReranker,
        CrossEncoderReranker,
        HuggingFaceReranker,
        create_reranker,
    )
except ImportError:
    BaseReranker = None
    CrossEncoderReranker = None
    CohereReranker = None
    AWSBedrockReranker = None
    HuggingFaceReranker = None
    create_reranker = None

__version__ = "0.0.5.post1"

__all__ = [
    # Core class
    "pgVectorDB",
    # Extension manager
    "ExtensionManager",
    # Enums
    "SearchMethod",
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
    # Type definitions
    "QueryResult",
    # Metrics
    "RAGEvaluator",
    "EvaluationDataset",
    "EvaluationResult",
    "KValueAnalysis",
    "create_sample_evaluation_dataset",
    # Config
    "Config",
    "get_test_config",
    "get_production_config",
    # Schema helpers
    "get_vector_table",
    "get_label_definitions_table",
    "quote_identifier",
    "build_qualified_name",
    "get_distance_operator",
    "get_index_ops",
    # Spaces (v0.0.3)
    "VectorSpace",
    "TextSpace",
    "NumberSpace",
    "CategorySpace",
    "RecencySpace",
    "NumberMode",
    "TimeUnit",
    "validate_spaces",
    "encode_document_spaces",
    "encode_query_spaces",
    # Rerankers (v0.0.3)
    "BaseReranker",
    "CrossEncoderReranker",
    "CohereReranker",
    "AWSBedrockReranker",
    "HuggingFaceReranker",
    "create_reranker",
    # Query builders (v0.0.6)
    "UnifiedQueryBuilder",
    "SearchConfig",
]
