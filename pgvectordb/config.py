import os
from typing import Any, Dict

from dotenv import load_dotenv
from langchain_core.embeddings import Embeddings

# Load environment variables
load_dotenv(os.path.join(os.path.dirname(__file__), "..", "config", ".env"))


class Config:
    """Default configuration for pgVectorDB.

    All defaults are carefully chosen based on official documentation:
    - pgvector v0.8.0: https://github.com/pgvector/pgvector
    - pgvectorscale: https://github.com/timescale/pgvectorscale
    - pg_textsearch: https://github.com/timescale/pg_textsearch
    """

    # ==================== Environment Defaults ====================
    ENVIRONMENT = os.getenv("ENVIRONMENT", "development")

    # Database Connection
    DB_HOST = os.getenv("DB_HOST", "localhost")
    DB_PORT = int(os.getenv("DB_PORT", "5432"))
    DB_NAME = os.getenv("DB_NAME", "postgres")
    DB_USER = os.getenv("DB_USER", "postgres")
    DB_PASSWORD = os.getenv("DB_PASSWORD", "password")

    # Embedding Model
    EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "huggingface")
    HUGGINGFACE_MODEL = os.getenv(
        "HUGGINGFACE_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
    )
    BEDROCK_MODEL_ID = os.getenv("BEDROCK_MODEL_ID", "amazon.titan-embed-text-v1")
    BEDROCK_REGION = os.getenv("BEDROCK_REGION", "us-east-1")

    # ==================== Vector Index Defaults ====================

    # HNSW defaults (pgvector)
    DEFAULT_HNSW_M = 16  # Max connections per layer
    DEFAULT_HNSW_EF_CONSTRUCTION = 64  # Construction candidate list size
    DEFAULT_HNSW_EF_SEARCH = 40  # Query candidate list size

    # IVFFlat defaults (pgvector)
    DEFAULT_IVFFLAT_LISTS = 100  # Will be auto-calculated if None
    DEFAULT_IVFFLAT_PROBES = 10  # Number of lists to search

    # DiskANN defaults (pgvectorscale)
    DEFAULT_DISKANN_NUM_NEIGHBORS = 50  # Connections per node
    DEFAULT_DISKANN_SEARCH_LIST_SIZE = 100  # Search candidate list size
    DEFAULT_DISKANN_MAX_ALPHA = 1.2  # Graph diversity factor
    DEFAULT_DISKANN_QUERY_RESCORE = 50  # Candidates to rescore
    DEFAULT_DISKANN_STORAGE_LAYOUT = "memory_optimized"  # SBQ compression

    # DiskANN parallel build defaults
    DEFAULT_DISKANN_FORCE_PARALLEL_WORKERS = None  # Use PG default
    DEFAULT_DISKANN_MIN_VECTORS_FOR_PARALLEL = 100000  # Min vectors for parallel
    DEFAULT_DISKANN_PARALLEL_FLUSH_INTERVAL = 0.1  # 10% of vectors

    # ==================== Iterative Scan Defaults (pgvector 0.8+) ====================

    DEFAULT_ITERATIVE_SCAN_MODE = "relaxed_order"  # Better recall
    DEFAULT_MAX_SCAN_TUPLES = 20000  # HNSW max tuples to visit
    DEFAULT_SCAN_MEM_MULTIPLIER = 2  # HNSW memory multiplier
    DEFAULT_IVFFLAT_MAX_PROBES = 100  # IVFFlat max probes

    # ==================== BM25 Defaults (pg_textsearch) ====================

    DEFAULT_BM25_K1 = 1.2  # Term frequency saturation (0.1-10.0)
    DEFAULT_BM25_B = 0.75  # Length normalization (0.0-1.0)
    DEFAULT_BM25_TEXT_CONFIG = "english"  # PostgreSQL text search config

    # ==================== Batch Processing Defaults ====================

    DEFAULT_BATCH_SIZE = 100  # Documents per batch
    DEFAULT_BULK_LOAD_THRESHOLD = 10000  # Use COPY above this threshold

    # ==================== Connection Pool Defaults ====================

    DEFAULT_POOL_SIZE = 5
    DEFAULT_MAX_OVERFLOW = 10

    # ==================== Extension Version Minimums ====================

    MIN_VECTOR_VERSION = "0.5.0"  # Required for basic features
    MIN_VECTOR_VERSION_ITERATIVE = "0.8.0"  # Required for iterative scans
    MIN_VECTORSCALE_VERSION = "0.2.0"  # Required for DiskANN
    MIN_PG_TEXTSEARCH_VERSION = "0.4.0"  # Updated: latest pg_textsearch release

    # ==================== Quality Thresholds ====================

    DEFAULT_RECALL_THRESHOLD = 0.95  # Minimum acceptable recall
    DEFAULT_TRIGRAM_THRESHOLD = 0.3  # Minimum trigram similarity

    # ==================== RRF Defaults ====================

    DEFAULT_RRF_K = 60  # RRF constant
    DEFAULT_HYBRID_WEIGHTS = (0.5, 0.5)  # (semantic, keyword) weights

    # ==================== Multimodal Defaults (v0.0.3) ====================

    DEFAULT_NUMBER_SPACE_DIMS = 1  # Dimensions for NumberSpace
    DEFAULT_MULTIMODAL_METRIC = "cosine"  # Default metric for multimodal search
    DEFAULT_SPACE_WEIGHT = 1.0  # Default weight per space
    DEFAULT_MULTIMODAL_KEYWORD_WEIGHT = (
        0.0  # BM25 weight in multimodal hybrid (0 = off)
    )

    # ==================== Reranker Defaults (v0.0.3) ====================

    DEFAULT_RERANKER_TOP_K = 5  # Results to return after reranking
    DEFAULT_RERANKER_CANDIDATE_K = (
        100  # Initial candidates to retrieve before reranking
    )
    DEFAULT_CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    DEFAULT_COHERE_RERANK_MODEL = "rerank-english-v3.0"
    DEFAULT_BEDROCK_RERANK_MODEL = "amazon.rerank-v1:0"
    DEFAULT_BEDROCK_REGION = "us-east-1"
    DEFAULT_HF_RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"
    DEFAULT_RERANKER_BATCH_SIZE = 32  # Docs per batch for local rerankers
    DEFAULT_RERANKER_MAX_LENGTH = 512  # Max token length per pair

    # ==================== BM25 Performance ====================

    DEFAULT_BM25_PARALLEL_WORKERS = None  # None = use PostgreSQL default

    @classmethod
    def get_connection_string(cls) -> str:
        """Get database connection string from environment variables."""
        # Check for explicit connection string
        conn_str = os.getenv("DB_CONNECTION_STRING")
        if conn_str:
            return conn_str

        # Construct from components
        return f"postgresql+asyncpg://{cls.DB_USER}:{cls.DB_PASSWORD}@{cls.DB_HOST}:{cls.DB_PORT}/{cls.DB_NAME}"

    @classmethod
    def get_embeddings(cls) -> Embeddings:
        """Get embedding model based on configuration."""
        if cls.EMBEDDING_PROVIDER == "bedrock":
            try:
                from langchain_aws import BedrockEmbeddings

                return BedrockEmbeddings(
                    model_id=cls.BEDROCK_MODEL_ID, region_name=cls.BEDROCK_REGION
                )
            except ImportError as e:
                raise ImportError(
                    "Please install langchain-aws to use Bedrock embeddings"
                ) from e

        elif cls.EMBEDDING_PROVIDER == "huggingface":
            try:
                from langchain_huggingface import HuggingFaceEmbeddings

                return HuggingFaceEmbeddings(model_name=cls.HUGGINGFACE_MODEL)
            except ImportError as e:
                raise ImportError(
                    "Please install langchain-huggingface to use HuggingFace embeddings"
                ) from e

        else:
            raise ValueError(
                f"Unsupported embedding provider: {cls.EMBEDDING_PROVIDER}"
            )


def get_test_config() -> Dict[str, Any]:
    """Get configuration for testing."""
    return {
        "db_host": "localhost",
        "db_port": 5432,
        "db_name": "test_pgvectordb",
        "db_user": "postgres",
        "db_password": "postgres",
        "pool_size": 2,
        "max_overflow": 2,
    }


def get_production_config() -> Dict[str, Any]:
    """Get recommended production configuration."""
    return {
        "pool_size": 10,
        "max_overflow": 20,
        "hnsw_ef_search": 100,
        "ivfflat_probes": 20,
        "diskann_query_rescore": 200,
        "maintenance_work_mem": "4GB",
        "max_parallel_maintenance_workers": 4,
    }
