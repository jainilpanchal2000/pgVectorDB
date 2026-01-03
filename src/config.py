from typing import Dict, Any, Optional

class Config:
    """Default configuration for pgVectorDB.
    
    All defaults are carefully chosen based on official documentation:
    - pgvector v0.8.0: https://github.com/pgvector/pgvector
    - pgvectorscale: https://github.com/timescale/pgvectorscale  
    - pg_textsearch: https://github.com/timescale/pg_textsearch
    """
    
    # ==================== Vector Index Defaults ====================
    
    # HNSW defaults (pgvector)
    DEFAULT_HNSW_M = 16                    # Max connections per layer
    DEFAULT_HNSW_EF_CONSTRUCTION = 64      # Construction candidate list size
    DEFAULT_HNSW_EF_SEARCH = 40            # Query candidate list size
    
    # IVFFlat defaults (pgvector)
    DEFAULT_IVFFLAT_LISTS = 100            # Will be auto-calculated if None
    DEFAULT_IVFFLAT_PROBES = 10            # Number of lists to search
    
    # DiskANN defaults (pgvectorscale)
    DEFAULT_DISKANN_NUM_NEIGHBORS = 50     # Connections per node
    DEFAULT_DISKANN_SEARCH_LIST_SIZE = 100 # Search candidate list size
    DEFAULT_DISKANN_MAX_ALPHA = 1.2        # Graph diversity factor
    DEFAULT_DISKANN_QUERY_RESCORE = 50     # Candidates to rescore
    DEFAULT_DISKANN_STORAGE_LAYOUT = "memory_optimized"  # SBQ compression
    
    # DiskANN parallel build defaults
    DEFAULT_DISKANN_FORCE_PARALLEL_WORKERS = None       # Use PG default
    DEFAULT_DISKANN_MIN_VECTORS_FOR_PARALLEL = 100000   # Min vectors for parallel
    DEFAULT_DISKANN_PARALLEL_FLUSH_INTERVAL = 0.1       # 10% of vectors
    
    # ==================== Iterative Scan Defaults (pgvector 0.8+) ====================
    
    DEFAULT_ITERATIVE_SCAN_MODE = "relaxed_order"  # Better recall
    DEFAULT_MAX_SCAN_TUPLES = 20000        # HNSW max tuples to visit
    DEFAULT_SCAN_MEM_MULTIPLIER = 2        # HNSW memory multiplier
    DEFAULT_IVFFLAT_MAX_PROBES = 100       # IVFFlat max probes
    
    # ==================== BM25 Defaults (pg_textsearch) ====================
    
    DEFAULT_BM25_K1 = 1.2                  # Term frequency saturation (0.1-10.0)
    DEFAULT_BM25_B = 0.75                  # Length normalization (0.0-1.0)
    DEFAULT_BM25_TEXT_CONFIG = "english"   # PostgreSQL text search config
    
    # ==================== Batch Processing Defaults ====================
    
    DEFAULT_BATCH_SIZE = 100               # Documents per batch
    DEFAULT_BULK_LOAD_THRESHOLD = 10000    # Use COPY above this threshold
    
    # ==================== Connection Pool Defaults ====================
    
    DEFAULT_POOL_SIZE = 5
    DEFAULT_MAX_OVERFLOW = 10
    
    # ==================== Extension Version Minimums ====================
    
    MIN_VECTOR_VERSION = "0.5.0"           # Required for basic features
    MIN_VECTOR_VERSION_ITERATIVE = "0.8.0" # Required for iterative scans
    MIN_VECTORSCALE_VERSION = "0.2.0"      # Required for DiskANN
    MIN_PG_TEXTSEARCH_VERSION = "0.3.0"    # Required for BM25
    
    # ==================== Quality Thresholds ====================
    
    DEFAULT_RECALL_THRESHOLD = 0.95        # Minimum acceptable recall
    DEFAULT_TRIGRAM_THRESHOLD = 0.3        # Minimum trigram similarity
    
    # ==================== RRF Defaults ====================
    
    DEFAULT_RRF_K = 60                     # RRF constant
    DEFAULT_HYBRID_WEIGHTS = (0.5, 0.5)    # (semantic, keyword) weights


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
