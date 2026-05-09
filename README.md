# pgVectorDB - Production PostgreSQL Vector Database

Production-ready Retrieval-Augmented Generation (RAG) system built on PostgreSQL with pgvector. Features advanced vector search, comprehensive evaluation metrics, and optimization tools.

**Version:** 0.0.5.post1
**Status:** Production-Ready (Security & Robustness Hardened)

📖 **[Full Configuration Guide](docs/CONFIGURATION.md)** | 🛠️ **[Refactoring Summary](docs/REFACTORING_SUMMARY.md)**

---

## 🌟 Features

### 🚀 **Advanced Vector Support (New in v0.0.2)**
- **Half-Precision (halfvec)**: Store vectors with 16-bit floats (50% storage savings).
- **Binary Quantization**: Store vectors as bits (87.5% storage savings, ultra-fast Hamming search).
- **Sparse Vectors**: Support for high-dimensional sparse data (TF-IDF, one-hot).
- **Subvector Indexing**: Index only the first N dimensions (Matryoshka embeddings) for faster search.
- **Matryoshka Support**: Two-stage search with subvectors + full re-ranking.

### 🤖 **2 Embedding Providers**
- **HuggingFace** - Free, local, offline embeddings (default).
- **AWS Bedrock** - Managed embedding service (Titan, Cohere).

### 🔍 **3 Vector Index Types**
- **HNSW** - Fast approximate nearest neighbor search (best for <1M vectors).
- **IVFFlat** - Inverted file index for large datasets (100K-10M vectors).
- **DiskANN** - Disk-based vector search with memory optimization (>10M vectors).

### 🔤 **2 Keyword Search Types**
- **FTS (Full-Text Search)** - PostgreSQL's native ts_rank.
- **BM25** - Industry-standard ranking via `pg_textsearch` (configurable k1, b).

### 🎯 **10 Search Methods**
1. **keyword_search** - Pure keyword search (FTS or BM25).
2. **universal_keyword_search** - Keyword search across content + metadata.
3. **semantic_search** - Vector similarity search.
4. **metadata_filter** - Pure metadata filtering.
5. **metadata_keyword_search** - Filtered keyword search.
6. **metadata_semantic_search** - Filtered vector search.
7. **hybrid_search** - Keyword + Semantic combined (weighted or RRF).
8. **ensemble_search** - Metadata + Keyword + Semantic.
9. **trigram_search** - Fuzzy text matching (typo-tolerant).
10. **metadata_trigram_search** - Filtered fuzzy search.

### 🛡️ **Robustness & Production Readiness**
- **Error Isolation**: `add_documents_batch_isolated` commits valid batches while logging errors.
- **Intelligent Fallback**: Automatic fallback on embedding rate limits.
- **Deduplication**: Content-hash based deduplication in `upsert_documents`.
- **Concurrent Indexing**: Non-blocking `build_index_concurrent` for zero downtime.
- **Recall Monitoring**: Measure exact vs approximate search recall.

---

## 🆕 v0.0.5 — PyPI Release

The codebase uses a flat modular layout with focused mixin files:

```
pgvectordb/
├── __init__.py          # Main exports & version
├── core.py              # Main pgVectorDB class
├── search.py            # 10 search method implementations
├── base.py              # Enums, exceptions, constants
├── extensions.py        # Extension checking & graceful degradation
├── config.py            # Configuration defaults
├── metrics.py           # RAG evaluation metrics
├── schema.py            # Centralized SQLAlchemy table definitions
├── spaces.py            # Multimodal vector space abstractions
├── rerankers.py         # CrossEncoder, Cohere, Bedrock, HuggingFace rerankers
└── mixins/
    ├── documents.py     # Document CRUD operations
    ├── indexing.py      # Index build, tune, and maintenance
    ├── analytics.py     # Stats, benchmarking, diagnostics
    ├── storage.py       # Export/import and specialized tables
    └── multimodal.py   # Multi-space search
```

### Extension Requirements & Graceful Degradation

| Extension | Required | Purpose | Status |
|-----------|----------|---------|--------|
| **pgvector** | ✅ Yes | Core vector operations | Required |
| **vectorscale** | ❌ Optional | DiskANN index, label filtering | Gracefully degrades if missing |
| **pg_textsearch** | ❌ Optional | BM25 text ranking | Fallback to FTS if missing |

---

## 📦 Installation

### Option 1: Install from PyPI (Recommended)

```bash
pip install pgvectordb

# With HuggingFace embedding support
pip install pgvectordb[huggingface]

# With AWS Bedrock support
pip install pgvectordb[aws]

# Everything (all optional extras)
pip install pgvectordb[all]

# For Jupyter notebooks
pip install pgvectordb[jupyter]
```

### Option 2: Using Docker (for the PostgreSQL database)

We provide a custom PostgreSQL 17 image with `pgvector`, `pgvectorscale`, and `pg_textsearch` pre-installed.

```bash
cd docker
docker compose up -d
```

### Option 3: Manual PostgreSQL Setup

1. **Install PostgreSQL Extensions:**
```sql
CREATE EXTENSION vector;
CREATE EXTENSION pg_trgm;
-- Optional:
CREATE EXTENSION vectorscale CASCADE;
CREATE EXTENSION pg_textsearch;
```

2. **Install Python Package:**
```bash
pip install pgvectordb
```

---

## 🚀 Advanced Usage Patterns

### 1. Storage Optimization (Halfvec & Binary)

Reduce storage costs and improve performance with specialized vector types.

**Half-Precision (50% Savings):**
```python
# Create a specialized table for half-precision vectors
# All vectors will be stored as FLOAT16 (2 bytes per dim)
await rag.create_halfvec_table(table_name="my_docs_halfvec")
```

**Binary Quantization (87.5% Savings):**
This creates an index on the *bit* representation of your vectors. Perfect for initial retrieval followed by re-ranking.
```python
# 1. Build binary index (indexes 1-bit per dimension)
await rag.build_index_binary_quantized(m=16, ef_construction=64)

# 2. Search using two-stage process:
#    - Stage 1: Fast Hamming search on binary index (fetch 50 candidates)
#    - Stage 2: Re-rank top 50 using actual float vectors
results = await rag.search_with_binary_rerank(
    query="optimization techniques",
    k=10,
    rerank_top=50
)
```

### 2. High-Dimensional Sparse Vectors

Efficiently store and search sparse data (like TF-IDF or one-hot encodings) with up to 10,000+ dimensions.

```python
# Create table for sparse vectors
await rag.create_sparsevec_table(max_dimensions=10000)

# Add documents (embedding field should be sparse format)
# Note: Embedding model must output sparse vectors
```

### 3. Matryoshka / Subvector Search

Speed up search by indexing only the first `N` dimensions of your embeddings (e.g., first 256 of 1536).

```python
# Index only the first 256 dimensions
await rag.build_index_with_subvectors(
    subvector_dims=256,
    index_type=IndexType.HNSW
)

# Search using:
# 1. Fast search on 256 dims
# 2. Re-rank with full 1536 dims
results = await rag.search_with_subvector_rerank(
    query="matryoshka embedding",
    k=10,
    subvector_dims=256
)
# Note: Requires an embedding model trained for this (e.g., OpenAI text-embedding-3)
```

### 4. Robust Bulk Loading

For importing large datasets without crashing on individual errors.

```python
docs = [...] # Large list of documents

# 1. Isolated Batches: Failures in one batch don't affect others
failed_batches = await rag.add_documents_batch_isolated(
    documents=docs,
    batch_size=100,
    continue_on_error=True
)

# 2. Optimized Bulk Load (Faster):
#    - Pre-computes embeddings
#    - Uses COPY protocol
#    - Rebuilds indexes at the end
await rag.bulk_load_documents(docs)
```

### 5. Concurrent Indexing (Zero Downtime)

Build or rebuild indexes without locking the table for reads/writes.

```python
# Uses CREATE INDEX CONCURRENTLY
await rag.build_index_concurrent(
    index_type=IndexType.HNSW,
    m=16,
    ef_construction=64
)
```

---

## 🛠️ Utility Methods (Full List)

The system includes **60+ methods** across `pgvectordb/core.py` and its mixins:

**Document Management**
- `add_documents`, `add_documents_batch`, `add_documents_batch_isolated`
- `aupdate_documents`, `upsert_documents` (deduplication)
- `bulk_load_documents` (fast copy)
- `adelete`, `aget_by_ids`

**Index Operations**
- `build_index`, `build_index_concurrent`
- `build_bm25_index`, `build_index_binary_quantized`
- `build_index_with_subvectors`, `create_metadata_index`
- `areindex`, `adrop_vector_index`
- `get_index_build_progress`

**Advanced Search**
- `asimilarity_search_by_vector`, `asimilarity_search_with_score`
- `semantic_search_with_reranker` (cross-encoder)
- `search_with_binary_rerank`, `search_with_subvector_rerank`
- `compute_centroid` (vector aggregation)

**Table Management**
- `create_halfvec_table`, `create_sparsevec_table`
- `create_label_definitions`, `get_label_ids_by_names`

**Analytics & Monitoring**
- `get_stats`, `get_index_stats`, `get_bm25_index_stats`
- `get_slow_queries`, `compute_recall`
- `dump_bm25_index`, `spill_bm25_index`
- `explain_query`, `validate_collection`

---

## 📝 Configuration

Copy `config/.env.example` to `config/.env`.

```dotenv
# Environment
ENVIRONMENT=local

# Embedding Provider (huggingface | bedrock)
EMBEDDING_PROVIDER=huggingface
HUGGINGFACE_MODEL=sentence-transformers/all-MiniLM-L6-v2

# Database
LOCAL_DB_HOST=localhost
LOCAL_DB_PORT=9002
DB_NAME=postgres
DB_USER=user
DB_PASSWORD=root
```

See [docs/CONFIGURATION.md](docs/CONFIGURATION.md) for AWS Bedrock setup.

---

## 🤝 Contributing

We welcome contributions! Please check `TODO.md` for upcoming features.

1. Fork the repo
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request
