"""
pgVectorDB - Production PostgreSQL Vector Database
===================================================

**Version:** 0.0.2
**Status:** Production-Ready with Modular Architecture

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

Extension Requirements
----------------------
- **pgvector** (REQUIRED): Core vector operations
- **vectorscale** (OPTIONAL): DiskANN index support
- **pg_textsearch** (OPTIONAL): BM25 keyword search

Quick Start
-----------
    >>> from src import pgVectorDB, IndexType
    >>> rag = pgVectorDB(
    ...     collection_name="docs",
    ...     embedding_model=embeddings,
    ...     connection_string="postgresql+asyncpg://user:pass@localhost/db"
    ... )
    >>> await rag.initialize()
    >>> await rag.add_documents(documents)
    >>> results = await rag.semantic_search("query", k=5)
"""

# Import from new modular base
from .base import (
    # Enums
    IndexType,
    KeywordSearchType,
    StorageLayout,
    DistanceMetric,
    VectorPrecision,
    IterativeScanMode,
    # Exceptions
    RetrievalSystemError,
    InitializationError,
    ValidationError,
    DatabaseError,
    RateLimitError,
    # Constants
    ALLOWED_TEXT_CONFIGS,
    VALID_QUERY_PARAMS,
    EXTENSION_REQUIREMENTS,
    # Types
    QueryResult,
)

# Import extension manager
from .extensions import ExtensionManager

# Import main class from core (backward compatible)
from .core import pgVectorDB

# Import metrics
from .metrics import (
    RAGEvaluator,
    EvaluationDataset,
    EvaluationResult,
    KValueAnalysis,
    create_sample_evaluation_dataset,
)

# Import config
from .config import Config, get_test_config, get_production_config

# Import schema helpers
from .schema import (
    get_vector_table,
    get_label_definitions_table,
    quote_identifier,
    build_qualified_name,
    get_distance_operator,
    get_index_ops,
)

__version__ = "0.0.2"

__all__ = [
    # Core class
    "pgVectorDB",
    # Extension manager
    "ExtensionManager",
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
]

