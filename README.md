# pgVectorDB - Production PostgreSQL Vector Database

Production-ready Retrieval-Augmented Generation (RAG) system built on PostgreSQL with pgvector. Features advanced vector search, comprehensive evaluation metrics, and optimization tools.

**NEW:** Support for AWS Bedrock embeddings and flexible database configurations (local/remote)!

📖 **[Full Configuration Guide](docs/CONFIGURATION.md)**

---

## 🌟 Features

### 🤖 **2 Embedding Providers**
- **HuggingFace** - Free, local, offline embeddings (default)
  - Models: sentence-transformers, instructor, etc.
  - Runs on CPU or GPU
  - No API costs or rate limits
- **AWS Bedrock** - Managed embedding service
  - Models: Amazon Titan, Cohere Embed, etc.
  - Supports model IDs and ARNs (cross-region/cross-account)
  - Enterprise-grade performance
  - Pay-per-use pricing

### 💾 **Flexible Database Connections**
- **Local Database** - For development and testing
- **Remote Database** - For production deployments
- **Environment-Aware** - Automatic configuration based on environment
- **Test-Safe** - Tests always use local database

### 🔍 **3 Vector Index Types**
- **HNSW** - Fast approximate nearest neighbor search (best for <1M vectors)
- **IVFFlat** - Inverted file index for large datasets (100K-10M vectors)  
- **DiskANN** - Disk-based vector search with memory optimization (>10M vectors)

### 🔤 **2 Keyword Search Types**
- **FTS (Full-Text Search)** - PostgreSQL's native ts_rank (fast, simple)
- **BM25** - Industry-standard ranking (Elasticsearch, Lucene) via pg_textsearch
  - Configurable k1 (term frequency saturation)
  - Configurable b (length normalization)
  - 29+ language support (english, french, german, spanish, etc.)

### 🎯 **10 Search Methods**
1. **keyword_search** - Pure keyword search (FTS or BM25)
2. **universal_keyword_search** - Keyword search across content + metadata fields
3. **semantic_search** - Vector similarity search
4. **metadata_filter** - Pure metadata filtering (no query)
5. **metadata_keyword_search** - Filtered keyword search (FTS or BM25)
6. **metadata_semantic_search** - Filtered vector search
7. **hybrid_search** - Keyword (FTS/BM25) + Semantic combined (weighted or RRF)
8. **ensemble_search** - Metadata + Keyword (FTS/BM25) + Semantic (most comprehensive)
9. **trigram_search** - Fuzzy text matching (typo-tolerant)
10. **metadata_trigram_search** - Filtered fuzzy search

### 📊 **7 Evaluation Metrics**
- **Precision@K** - Quality of top K results
- **Recall@K** - Coverage of relevant documents
- **F1@K** - Harmonic mean of precision/recall
- **MAP** - Mean Average Precision (rank-aware)
- **MRR** - Mean Reciprocal Rank (first relevant result)
- **NDCG@K** - Normalized Discounted Cumulative Gain
- **Hit Rate@K** - Queries with ≥1 relevant result

### ⚙️ **13 Filter Operators**
- **Comparison:** `$eq`, `$ne`, `$lt`, `$lte`, `$gt`, `$gte`
- **Set:** `$in`, `$nin`, `$between`
- **Pattern:** `$like`, `$ilike`, `$exists`
- **Logical:** `$and`, `$or`

### 🛠️ **33 Utility Methods**
- **Document Management (6):** add_documents, add_documents_batch, aupdate_documents, update_metadata, adelete, aget_by_ids
- **Index Operations (6):** build_index, build_bm25_index, create_metadata_index, areindex, adrop_vector_index, set_query_params
- **Advanced Search (2):** asimilarity_search_by_vector, asimilarity_search_with_score
- **Analytics & Monitoring (5):** get_stats, get_index_stats, count_by_metadata, explain_query, validate_collection
- **Data Export/Import (2):** export_to_json, import_from_json
- **Database Operations (1):** vacuum_analyze
- **Benchmarking (1):** benchmark_search_methods
- **LangChain Integration (1):** as_retriever

### ✨ **Production Features**
- Connection pooling with configurable pool size
- Comprehensive error handling and validation
- Automatic extension installation (vector, pg_trgm, vectorscale, pg_textsearch)
- Batch operations with progress tracking
- Query parameter tuning for all index types
- BM25 native support via pg_textsearch
- Label-based filtering for DiskANN
- Multiple distance metrics (cosine, L2, inner product)
- Schema isolation support

---

## � Docker Setup

### PostgreSQL Extensions Used

This project uses three powerful PostgreSQL extensions for advanced vector and text search:

#### 1. **pgvector** ([github.com/pgvector/pgvector](https://github.com/pgvector/pgvector))
- **Purpose:** Open-source vector similarity search for PostgreSQL
- **Version:** v0.8.0
- **Features:**
  - Store vector embeddings directly in PostgreSQL
  - Three index types: HNSW, IVFFlat, and basic vector support
  - Supports L2 distance, inner product, and cosine similarity
  - Compatible with PostgreSQL 12+
  - Production-ready with billions of vectors in use

#### 2. **pgvectorscale** ([github.com/timescale/pgvectorscale](https://github.com/timescale/pgvectorscale))
- **Purpose:** High-performance vector search with DiskANN algorithm
- **Built by:** Timescale (PostgreSQL time-series database experts)
- **Features:**
  - DiskANN index for 10M+ vectors with minimal memory
  - StreamingDiskANN for fast incremental index building
  - 28x faster index creation than HNSW
  - 16x less memory usage than HNSW
  - Built with Rust for maximum performance
  - Requires pgvector to be installed first

#### 3. **pg_textsearch** ([github.com/timescale/pg_textsearch](https://github.com/timescale/pg_textsearch))
- **Purpose:** BM25 full-text search ranking for PostgreSQL
- **Built by:** Timescale
- **Features:**
  - Industry-standard BM25 algorithm (used by Elasticsearch, Lucene)
  - Configurable parameters: k1 (term frequency), b (length normalization)
  - 29+ language support with stemming and stop words
  - Drop-in replacement for PostgreSQL's ts_rank
  - Better relevance than default FTS ranking

### Our Custom Dockerfile

We built a custom PostgreSQL 17 image with all three extensions pre-installed:

**Base Image:** `pgvector/pgvector:pg17` (PostgreSQL 17 + pgvector)

**Additional Components:**
- **Rust toolchain** - Required for building pgvectorscale
- **cargo-pgrx 0.16.1** - PostgreSQL extension framework for Rust
- **Build tools** - clang-16, libclang-16-dev, postgresql-server-dev-17

**Extensions Installed:**
1. `pgvector v0.8.0` - Vector similarity search
2. `pgvectorscale` (latest) - DiskANN high-performance indexing
3. `pg_textsearch` (latest) - BM25 text search

**File:** [docker/Dockerfile](docker/Dockerfile)

### Docker Commands

#### Build the Image
```bash
# Using Docker
cd docker
docker build -t pg17-vectorscale-textsearch:latest .

# Using Podman
cd docker
podman build -t pg17-vectorscale-textsearch:latest .
```

#### Start the Container
```bash
# Using Docker Compose
cd docker
docker compose up -d

# Using Docker
docker run -d \
  --name pg17-vectorscale-textsearch \
  -p 9002:5432 \
  -e POSTGRES_USER=user \
  -e POSTGRES_PASSWORD=root \
  -e POSTGRES_DB=postgres \
  -v pgLocalData:/var/lib/postgresql/data \
  --shm-size=4gb \
  pg17-vectorscale-textsearch:latest

# Using Podman Compose
cd docker
podman-compose up -d

# Using Podman
podman run -d \
  --name pg17-vectorscale-textsearch \
  -p 9002:5432 \
  -e POSTGRES_USER=user \
  -e POSTGRES_PASSWORD=root \
  -e POSTGRES_DB=postgres \
  -v pgLocalData:/var/lib/postgresql/data \
  --shm-size=4gb \
  pg17-vectorscale-textsearch:latest
```

#### Stop and Remove
```bash
# Using Docker Compose
docker compose down              # Stop containers
docker compose down -v           # Stop and remove volumes

# Using Docker
docker stop pg17-vectorscale-textsearch
docker rm pg17-vectorscale-textsearch
docker volume rm pgLocalData     # Remove data volume

# Using Podman
podman stop pg17-vectorscale-textsearch
podman rm pg17-vectorscale-textsearch
podman volume rm pgLocalData
```

#### View Logs
```bash
# Docker
docker logs pg17-vectorscale-textsearch
docker logs -f pg17-vectorscale-textsearch  # Follow logs

# Podman
podman logs pg17-vectorscale-textsearch
podman logs -f pg17-vectorscale-textsearch
```

#### Access PostgreSQL Shell
```bash
# Docker
docker exec -it pg17-vectorscale-textsearch psql -U user -d postgres

# Podman
podman exec -it pg17-vectorscale-textsearch psql -U user -d postgres
```

#### Verify Extensions
```bash
# Check installed extensions
docker exec -it pg17-vectorscale-textsearch psql -U user -d postgres -c \
  "SELECT extname, extversion FROM pg_extension WHERE extname IN ('vector', 'vectorscale', 'pg_textsearch');"

# Expected output:
#    extname     | extversion
# ---------------+------------
#  vector        | 0.8.0
#  vectorscale   | 0.x.x
#  pg_textsearch | 0.x.x
```

### PostgreSQL Compatibility

| Extension      | Min Version | Recommended | Notes                                    |
|----------------|-------------|-------------|------------------------------------------|
| **pgvector**   | PostgreSQL 13+ | PostgreSQL 17+ | Supports PG 13-18, latest recommended |
| **pgvectorscale** | No specific minimum stated | PostgreSQL 17+ | Built with PGRX, tested on latest versions |
| **pg_textsearch** | PostgreSQL 17+ | PostgreSQL 17+ | Currently supports PG 17 and 18 only |

**Our Setup:** PostgreSQL 17 (latest stable) for full compatibility with all three extensions

**Note:** While pgvector supports PostgreSQL 13+, we recommend PostgreSQL 17+ for optimal performance and to ensure compatibility with all three extensions, especially pg_textsearch which requires PG 17+.

---

## 🎯 When to Use Which Extension

### pgvector - Core Vector Operations
**Use when:**
- You need basic vector similarity search
- Working with <1M vectors (HNSW) or 100K-10M vectors (IVFFlat)
- You want a mature, production-proven solution
- You need multi-language client support (30+ languages)

**Key Features:**
- 3 index types: HNSW (fast), IVFFlat (balanced), basic vector
- Multiple distance metrics: L2, cosine, inner product, L1, Hamming, Jaccard
- Up to 2,000 dimensions (vector), 4,000 (halfvec), 64,000 (bit)
- Sparse vectors support
- Binary quantization for compression

**Best for:** General-purpose vector search, RAG applications, semantic search

---

### pgvectorscale - High-Performance Large-Scale Vectors
**Use when:**
- You have >10M vectors (DiskANN algorithm)
- You need low latency at scale (28x faster than alternatives)
- Memory is limited (16x less memory than HNSW)
- You need label-based filtering for vector search
- Cost efficiency is critical (75% less cost than managed solutions)

**Key Features:**
- StreamingDiskANN index (inspired by Microsoft DiskANN)
- Statistical Binary Quantization (SBQ) for compression
- Parallel index building (28x faster than HNSW)
- Label-based filtered vector search (uses `&&` operator)
- Built with Rust for maximum performance

**Best for:** Production vector workloads at scale, large embedding datasets (millions+), cost-sensitive deployments

**Requires:** pgvector must be installed first (installed via CASCADE)

---

### pg_textsearch - BM25 Full-Text Search
**Use when:**
- You need keyword-based text search with relevance ranking
- BM25 algorithm is required (industry standard used by Elasticsearch, Lucene)
- You want better ranking than PostgreSQL's default ts_rank
- You need configurable ranking parameters (k1, b)
- Multi-language text search with stemming (29+ languages)

**Key Features:**
- Native BM25 ranking function
- Configurable parameters: k1 (term frequency), b (length normalization)
- Simple syntax: `ORDER BY content <@> 'search terms'`
- 29+ language support with stemming
- Partitioned table support
- Memtable architecture for efficient writes

**Best for:** Full-text search, document retrieval, keyword search, hybrid search (combining with vector search)

**Note:** Prerelease status (v0.1.1-dev) - feature-complete but not yet fully optimized

---

## 🔄 Using All Three Together (Our Setup)

### The Complete Stack:
1. **pgvector** - Vector similarity search for semantic queries
2. **pgvectorscale** - High-performance DiskANN for large-scale embeddings
3. **pg_textsearch** - BM25 keyword search for exact term matching

### Hybrid Search Strategy:
```python
# Our implementation supports:
# 1. Keyword Search (FTS or BM25) - exact term matching
# 2. Semantic Search (pgvector/pgvectorscale) - meaning-based
# 3. Hybrid Search - combines both with weighted fusion or RRF
# 4. Filtered Search - metadata filters + any search type

# Example: Best of all worlds
results = await rag.ensemble_search(
    query="machine learning algorithms",
    filter={"category": "ai", "year": {"$gte": 2020}},
    keyword_type=KeywordSearchType.BM25,  # pg_textsearch
    use_rrf=True  # Reciprocal Rank Fusion
)
```

### When to Use Each in Hybrid Mode:
- **BM25 (pg_textsearch):** Better for technical terms, product names, exact phrases
- **Semantic (pgvector/pgvectorscale):** Better for understanding intent, synonyms, context
- **Hybrid:** Combines both for optimal recall and precision

---

## 📦 Installation

### Option 1: Using Docker (Recommended)

1. **Start the PostgreSQL container:**
```bash
cd docker
docker compose up -d
```

2. **Verify extensions are loaded:**
```bash
docker exec -it pg17-vectorscale-textsearch psql -U user -d postgres -c \
  "CREATE EXTENSION IF NOT EXISTS vector; \
   CREATE EXTENSION IF NOT EXISTS vectorscale CASCADE; \
   CREATE EXTENSION IF NOT EXISTS pg_textsearch;"
```

All extensions are automatically initialized via [docker/init.sql](docker/init.sql)

### Option 2: Manual Installation

1. **PostgreSQL 12+ with extensions:**
```bash
# Core vector extension (required)
psql -c "CREATE EXTENSION IF NOT EXISTS vector;"

# Full-text search and trigram (auto-created by pgVectorDB)
psql -c "CREATE EXTENSION IF NOT EXISTS pg_trgm;"

# Optional: For native BM25 support
# Install from: https://github.com/timescale/pg_textsearch
psql -c "CREATE EXTENSION IF NOT EXISTS pg_textsearch;"

# Optional: For DiskANN support
psql -c "CREATE EXTENSION IF NOT EXISTS vectorscale CASCADE;"
```

2. **Python 3.9+ with required packages:**
```bash
pip install -r requirements.txt
```

### Configuration

1. **Copy and configure environment file:**
```bash
cp config/.env.example config/.env
# Edit config/.env with your database credentials
```

2. **Example config/.env:**
```env
DB_CONNECTION_STRING=postgresql+asyncpg://user:pass@localhost:9002/postgres
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
```

3. **Test your configuration:**
```bash
python scripts/test_connection.py
```

This will verify:
- All Python packages are installed
- Configuration file exists
- Database connection works
- PostgreSQL extensions are installed (vector, pg_trgm, vectorscale)
- Embedding model loads successfully
- pgVectorDB imports correctly

---

## Quick Start

### 0. Configure Environment (NEW!)

Copy the example configuration and customize:

```bash
# Copy configuration template
cp config/.env.example config/.env

# Edit with your settings (use notepad, nano, vim, etc.)
```

**For HuggingFace (Local, Free):**
```dotenv
ENVIRONMENT=local
EMBEDDING_PROVIDER=huggingface
HUGGINGFACE_MODEL=sentence-transformers/all-MiniLM-L6-v2
LOCAL_DB_HOST=localhost
LOCAL_DB_PORT=9002
```

**For AWS Bedrock (Managed):**
```dotenv
ENVIRONMENT=remote
EMBEDDING_PROVIDER=bedrock
BEDROCK_MODEL_ID=amazon.titan-embed-text-v1
# Or use ARN: arn:aws:bedrock:us-east-1::foundation-model/amazon.titan-embed-text-v1
AWS_REGION=us-east-1
REMOTE_DB_HOST=your-db-server.example.com
```

📖 See **[docs/CONFIGURATION.md](docs/CONFIGURATION.md)** for complete setup guide.

### 1. Test Requirements

Before starting, verify all requirements are met:

```bash
python scripts/test_connection.py
```

If any tests fail, install missing packages:
```bash
# For HuggingFace
pip install langchain-huggingface sentence-transformers torch

# For AWS Bedrock (additional)
pip install langchain-aws boto3
```

### 2. Basic Usage

**New way (recommended - uses config/.env):**

```python
import asyncio
from langchain_core.documents import Document
from src.config import Config
from src.core import pgVectorDB, IndexType

async def main():
    # Automatically uses settings from config/.env
    embeddings = Config.get_embeddings()
    connection_string = Config.get_connection_string()
    
    rag = pgVectorDB(
        collection_name="my_docs",
        embedding_model=embeddings,
        connection_string=connection_string,
        index_type=IndexType.HNSW
    )
    
    await rag.initialize()
    
    # Add documents
    docs = [
        Document(
            page_content="Python is a programming language.",
            metadata={"category": "programming", "year": 2024}
        ),
    ]
    await rag.add_documents(docs)
    
    # Build vector index
    await rag.build_index(m=16, ef_construction=64)
    
    # Build BM25 index for keyword search
    await rag.build_bm25_index(text_config="english", k1=1.2, b=0.75)
    
    # Semantic search
    results = await rag.semantic_search("What is Python?", k=5)
    for doc in results:
        print(f"{doc.page_content}")

asyncio.run(main())
```

---

## 🗂️ Multi-Table Architecture (One Schema, Multiple Collections)

### Understanding Schema and Tables

pgVectorDB uses the `collection_name` parameter as the **table name** within a PostgreSQL schema. This allows you to organize multiple independent document collections in the same database schema.

**Architecture:**
```
PostgreSQL Database
└── Schema: "public" (default)
    ├── Table: "technical_docs"     ← pgVectorDB collection 1
    ├── Table: "product_manuals"    ← pgVectorDB collection 2
    ├── Table: "customer_support"   ← pgVectorDB collection 3
    └── Table: "knowledge_base"     ← pgVectorDB collection 4
```

Each table is completely independent with its own:
- Vector embeddings
- Metadata
- Indexes (HNSW, IVFFlat, or DiskANN)
- BM25 indexes
- Full-text search indexes

---

### Example: Multiple Collections in One Schema

```python
import asyncio
from langchain_huggingface import HuggingFaceEmbeddings
from src.core import pgVectorDB, IndexType

async def setup_multi_collection():
    # Shared connection and embedding model
    conn_str = "postgresql+asyncpg://user:pass@localhost:9002/postgres"
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    
    # Collection 1: Technical Documentation
    tech_docs = pgVectorDB(
        collection_name="technical_docs",  # Creates table "technical_docs"
        embedding_model=embeddings,
        connection_string=conn_str,
        schema_name="public",  # Default schema
        index_type=IndexType.HNSW
    )
    await tech_docs.initialize()
    
    # Collection 2: Product Manuals
    product_manuals = pgVectorDB(
        collection_name="product_manuals",  # Creates table "product_manuals"
        embedding_model=embeddings,
        connection_string=conn_str,
        schema_name="public",
        index_type=IndexType.HNSW
    )
    await product_manuals.initialize()
    
    # Collection 3: Customer Support FAQs
    support_faqs = pgVectorDB(
        collection_name="customer_support",  # Creates table "customer_support"
        embedding_model=embeddings,
        connection_string=conn_str,
        schema_name="public",
        index_type=IndexType.DISKANN  # Different index type
    )
    await support_faqs.initialize()
    
    return tech_docs, product_manuals, support_faqs

# Use each collection independently
async def main():
    tech_docs, product_manuals, support_faqs = await setup_multi_collection()
    
    # Add documents to each collection
    await tech_docs.add_documents(technical_documents)
    await product_manuals.add_documents(manual_documents)
    await support_faqs.add_documents(faq_documents)
    
    # Build indexes for each collection
    await tech_docs.build_index(m=16, ef_construction=64)
    await product_manuals.build_index(m=16, ef_construction=64)
    await support_faqs.build_index(num_neighbors=50)
    
    # Search in specific collections
    tech_results = await tech_docs.semantic_search("API documentation", k=5)
    manual_results = await product_manuals.semantic_search("installation guide", k=5)
    faq_results = await support_faqs.semantic_search("how to reset password", k=5)
    
    print(f"Found {len(tech_results)} technical docs")
    print(f"Found {len(manual_results)} product manuals")
    print(f"Found {len(faq_results)} support FAQs")

asyncio.run(main())
```

---

### Use Cases for Multiple Tables

#### 1. **Department/Team Separation**
```python
# Marketing team
marketing_rag = pgVectorDB(collection_name="marketing_content", ...)

# Engineering team
engineering_rag = pgVectorDB(collection_name="engineering_docs", ...)

# Sales team
sales_rag = pgVectorDB(collection_name="sales_materials", ...)
```

#### 2. **Document Type Separation**
```python
# Different document types
contracts_rag = pgVectorDB(collection_name="legal_contracts", ...)
emails_rag = pgVectorDB(collection_name="email_archive", ...)
reports_rag = pgVectorDB(collection_name="financial_reports", ...)
```

#### 3. **Multi-Language Content**
```python
# Language-specific collections with different BM25 configs
english_rag = pgVectorDB(collection_name="docs_english", ...)
await english_rag.build_bm25_index(text_config="english")

french_rag = pgVectorDB(collection_name="docs_french", ...)
await french_rag.build_bm25_index(text_config="french")

german_rag = pgVectorDB(collection_name="docs_german", ...)
await german_rag.build_bm25_index(text_config="german")
```

#### 4. **Data Lifecycle Management**
```python
# Hot data (recent, frequently accessed)
hot_data = pgVectorDB(
    collection_name="docs_2024",
    index_type=IndexType.HNSW  # Fast queries
)

# Warm data (older, less accessed)
warm_data = pgVectorDB(
    collection_name="docs_2023",
    index_type=IndexType.IVFFLAT  # Balanced
)

# Cold data (archive)
cold_data = pgVectorDB(
    collection_name="docs_archive",
    index_type=IndexType.DISKANN  # Memory efficient
)
```

---

### Cross-Collection Search

Search across multiple collections and combine results:

```python
async def search_all_collections(query: str, k: int = 5):
    """Search across multiple collections and combine results."""
    
    # Search each collection
    tech_results = await tech_docs.semantic_search(query, k=k)
    manual_results = await product_manuals.semantic_search(query, k=k)
    faq_results = await support_faqs.semantic_search(query, k=k)
    
    # Combine and sort by score
    all_results = []
    
    for result in tech_results:
        result.metadata['source_collection'] = 'technical_docs'
        all_results.append(result)
    
    for result in manual_results:
        result.metadata['source_collection'] = 'product_manuals'
        all_results.append(result)
    
    for result in faq_results:
        result.metadata['source_collection'] = 'customer_support'
        all_results.append(result)
    
    # Sort by score and return top K
    all_results.sort(key=lambda x: x.score)
    return all_results[:k]

# Usage
combined_results = await search_all_collections("installation steps", k=10)
for doc in combined_results:
    print(f"[{doc.metadata['source_collection']}] {doc.content}")
```

---

### Using Custom Schemas

Organize collections into different PostgreSQL schemas:

```python
# Production data
prod_rag = pgVectorDB(
    collection_name="documents",
    schema_name="production",  # Schema: production
    ...
)

# Staging/testing data
staging_rag = pgVectorDB(
    collection_name="documents",
    schema_name="staging",  # Schema: staging
    ...
)

# Development data
dev_rag = pgVectorDB(
    collection_name="documents",
    schema_name="development",  # Schema: development
    ...
)
```

**Database Structure:**
```
PostgreSQL Database
├── Schema: "production"
│   └── Table: "documents"
├── Schema: "staging"
│   └── Table: "documents"
└── Schema: "development"
    └── Table: "documents"
```

---

### Best Practices for Multi-Table Setup

#### ✅ **DO:**

1. **Use Descriptive Collection Names**
   ```python
   # Good
   pgVectorDB(collection_name="customer_support_tickets_2024", ...)
   
   # Avoid
   pgVectorDB(collection_name="data", ...)
   ```

2. **Separate by Access Patterns**
   ```python
   # High-frequency searches
   active_docs = pgVectorDB(collection_name="active_docs", index_type=IndexType.HNSW)
   
   # Archive searches (less frequent)
   archive_docs = pgVectorDB(collection_name="archive_docs", index_type=IndexType.DISKANN)
   ```

3. **Use Connection Pooling for Multiple Collections**
   ```python
   # Share connection string with pooling
   conn_str = "postgresql+asyncpg://user:pass@localhost/db?pool_size=10&max_overflow=20"
   
   rag1 = pgVectorDB(collection_name="collection1", connection_string=conn_str, ...)
   rag2 = pgVectorDB(collection_name="collection2", connection_string=conn_str, ...)
   rag3 = pgVectorDB(collection_name="collection3", connection_string=conn_str, ...)
   ```

4. **Document Source in Metadata**
   ```python
   doc = Document(
       page_content="content...",
       metadata={
           "collection": "technical_docs",
           "department": "engineering",
           "created_at": "2024-12-26"
       }
   )
   ```

#### ❌ **DON'T:**

1. **Don't Mix Unrelated Data in One Table**
   ```python
   # Bad: mixing customer data, products, and support tickets
   everything = pgVectorDB(collection_name="all_data", ...)
   
   # Good: separate collections
   customers = pgVectorDB(collection_name="customers", ...)
   products = pgVectorDB(collection_name="products", ...)
   tickets = pgVectorDB(collection_name="support_tickets", ...)
   ```

2. **Don't Create Too Many Small Collections**
   ```python
   # Bad: one collection per user
   user_1_docs = pgVectorDB(collection_name="user_1_docs", ...)
   user_2_docs = pgVectorDB(collection_name="user_2_docs", ...)
   # ... 10,000 collections
   
   # Good: one collection with user_id in metadata
   all_docs = pgVectorDB(collection_name="user_documents", ...)
   # Filter by: {"user_id": {"$eq": "user_1"}}
   ```

---

### Monitoring Multiple Collections

```python
async def get_collection_stats():
    """Get statistics for all collections."""
    collections = [tech_docs, product_manuals, support_faqs]
    
    for rag in collections:
        stats = await rag.get_stats()
        print(f"\n{stats['table_name']}:")
        print(f"  Documents: {stats['total_documents']}")
        print(f"  Index Type: {stats['index_type']}")
        print(f"  Table Size: {stats['table_size']}")
        print(f"  Index Size: {stats['index_size']}")

# Check which tables exist in your schema
async def list_all_collections():
    """List all pgVectorDB collections in the database."""
    async with tech_docs.sqlalchemy_engine.connect() as conn:
        result = await conn.execute(text("""
            SELECT tablename 
            FROM pg_tables 
            WHERE schemaname = 'public'
            AND tablename NOT LIKE 'pg_%'
            ORDER BY tablename;
        """))
        tables = [row[0] for row in result.fetchall()]
        print(f"Found {len(tables)} collections:")
        for table in tables:
            print(f"  - {table}")
```

---

### Migration Example: Moving Data Between Collections

```python
async def migrate_collection(source_rag, target_rag, filter_condition=None):
    """Migrate documents from one collection to another."""
    
    # Get documents from source
    if filter_condition:
        docs = await source_rag.metadata_filter(filter=filter_condition, k=10000)
    else:
        # Get all documents (adjust k as needed)
        stats = await source_rag.get_stats()
        total_docs = stats['total_documents']
        docs = await source_rag.metadata_filter(filter={}, k=total_docs)
    
    # Convert to Document objects
    documents = [
        Document(page_content=doc.content, metadata=doc.metadata)
        for doc in docs
    ]
    
    # Add to target collection
    await target_rag.add_documents(documents)
    
    print(f"Migrated {len(documents)} documents")

# Example: Archive old documents
async def archive_old_documents():
    active_docs = pgVectorDB(collection_name="active_docs", ...)
    archive_docs = pgVectorDB(collection_name="archive_docs", ...)
    
    # Move documents older than 2023
    await migrate_collection(
        source_rag=active_docs,
        target_rag=archive_docs,
        filter_condition={"year": {"$lt": 2023}}
    )
```

---

## Search Methods

### 1. Keyword Search
Choose between **FTS** (Full-Text Search) or **BM25** ranking:

```python
from src.core import KeywordSearchType

# BM25 (recommended - better ranking)
results = await rag.keyword_search(
    "machine learning", 
    k=5, 
    search_type=KeywordSearchType.BM25
)

# Full-Text Search (classic PostgreSQL ts_rank)
results = await rag.keyword_search(
    "machine learning", 
    k=5, 
    search_type=KeywordSearchType.FTS
)
```

### 2. Semantic Search
```python
results = await rag.semantic_search("What is AI?", k=5)
```

### 3. Hybrid Search
Combines keyword + semantic with weighted scoring:
```python
# BM25 + semantic
results = await rag.hybrid_search(
    "neural networks", 
    k=5, 
    weights=(0.5, 0.5),  # (keyword_weight, semantic_weight)
    keyword_type=KeywordSearchType.BM25
)

# FTS + semantic
results = await rag.hybrid_search(
    "neural networks", 
    k=5, 
    weights=(0.5, 0.5),
    keyword_type=KeywordSearchType.FTS
)

# Or use Reciprocal Rank Fusion (RRF):
results = await rag.hybrid_search(
    "neural networks", 
    k=5, 
    use_rrf=True,
    keyword_type=KeywordSearchType.BM25
)
```

### 4. Metadata Filtering
```python
# Filter only
results = await rag.metadata_filter(
    filter={"category": {"$eq": "programming"}},
    k=10
)

# Filter + semantic search
results = await rag.metadata_semantic_search(
    query="Python tutorial",
    filter={"category": {"$eq": "programming"}, "year": {"$gte": 2020}},
    k=5
)
```

### 5. Ensemble Search
All-in-one: metadata filtering + hybrid search:
```python
results = await rag.ensemble_search(
    query="machine learning",
    filter={"year": {"$gte": 2020}},
    k=5,
    weights=(0.4, 0.6),
    keyword_type=KeywordSearchType.BM25  # Choose FTS or BM25
)
```

### 6. Trigram Search
Fuzzy text matching for typo tolerance:
```python
results = await rag.trigram_search("machin lerning", k=5, threshold=0.3)
```

---

## Filter Operators

### Comparison
```python
{"year": {"$eq": 2024}}                    # Equal
{"year": {"$ne": 2020}}                    # Not equal
{"year": {"$gt": 2020}}                    # Greater than
{"year": {"$gte": 2020}}                   # Greater or equal
{"year": {"$lt": 2025}}                    # Less than
{"year": {"$lte": 2024}}                   # Less or equal
{"year": {"$between": [2020, 2024]}}       # Between range
{"category": {"$in": ["ai", "ml"]}}        # In list
{"category": {"$nin": ["spam"]}}           # Not in list
```

### Pattern Matching
```python
{"title": {"$like": "%Python%"}}           # Case-sensitive
{"title": {"$ilike": "%python%"}}          # Case-insensitive
{"description": {"$exists": True}}         # Field exists
```

### Logical Operators
```python
# AND (implicit)
{"category": {"$eq": "ai"}, "year": {"$gte": 2020}}

# OR
{"$or": [
    {"category": {"$eq": "ai"}},
    {"category": {"$eq": "ml"}}
]}

# Complex
{"$and": [
    {"year": {"$gte": 2020}},
    {"$or": [{"category": {"$eq": "ai"}}, {"priority": {"$eq": "high"}}]}
]}
```

---

## Index Configuration

### HNSW (Fast, Recommended)
```python
rag = pgVectorDB(
    collection_name="docs",
    embedding_model=embeddings,
    connection_string=conn_str,
    index_type=IndexType.HNSW
)

await rag.build_index(
    m=16,              # Connections per layer (8-48)
    ef_construction=64 # Build quality (32-256)
)

# Query with custom quality
results = await rag.semantic_search(query, k=5, ef_search=40)
```

**Tuning:**
- Higher `m` = better recall, more memory
- Higher `ef_construction` = better quality, slower build
- Higher `ef_search` = better recall, slower queries

### IVFFlat (Large Datasets)
```python
rag = pgVectorDB(
    collection_name="docs",
    embedding_model=embeddings,
    connection_string=conn_str,
    index_type=IndexType.IVFFLAT
)

await rag.build_index(
    lists=100  # Number of clusters (sqrt of docs)
)

results = await rag.semantic_search(query, k=5, probes=10)
```

**Tuning:**
- `lists` ≈ sqrt(num_documents)
- `probes` = lists/10 to lists/2 (more = better recall, slower)

### DiskANN (Memory Efficient)
```python
rag = pgVectorDB(
    collection_name="docs",
    embedding_model=embeddings,
    connection_string=conn_str,
    index_type=IndexType.DISKANN
)

await rag.build_index(
    num_neighbors=50,
    search_list_size=100,
    storage_layout=StorageLayout.MEMORY_OPTIMIZED  # 4x compression
)
```

---

## Testing

### Comprehensive Test Suite

The project includes a comprehensive test suite in [test/test_suite.py](test/test_suite.py) with 10+ test functions covering all aspects of the system.

**Run all tests:**
```bash
python test/test_suite.py
```

### Test Coverage

#### 1. **Initialization Tests**
Tests all 3 index types (HNSW, IVFFlat, DiskANN):
- Database connection and extension setup
- Schema creation
- Vector store initialization
- Index type validation

#### 2. **Document Operations Tests**
- Add documents (single and batch)
- Update documents and metadata
- Delete documents
- Retrieve documents by IDs
- Document counting and statistics

#### 3. **Search Methods Tests**
All 10 search methods are tested:
- `keyword_search` (FTS and BM25)
- `universal_keyword_search`
- `semantic_search`
- `metadata_filter`
- `metadata_keyword_search`
- `metadata_semantic_search`
- `hybrid_search` (with RRF and weighted)
- `ensemble_search`
- `trigram_search`
- `metadata_trigram_search`

#### 4. **Filter Operators Tests**
All 13 filter operators validated:
- **Comparison:** `$eq`, `$ne`, `$lt`, `$lte`, `$gt`, `$gte`
- **Range:** `$between`
- **Set:** `$in`, `$nin`
- **Pattern:** `$like`, `$ilike`
- **Existence:** `$exists`

#### 5. **Index Operations Tests**
- Index building (HNSW, IVFFlat, DiskANN)
- BM25 index creation
- Reindexing operations
- Index statistics retrieval
- Vacuum and analyze

#### 6. **Analytics & Monitoring Tests**
- Collection statistics
- Index performance metrics
- Query explanation (EXPLAIN ANALYZE)
- Document counting by metadata
- Validation checks

#### 7. **Data Export/Import Tests**
- Export documents to JSON
- Import documents from JSON
- Data integrity verification

#### 8. **LangChain Integration Tests**
- Retriever creation with different search methods
- Search type configuration
- Result formatting

#### 9. **Error Handling Tests**
- Invalid initialization parameters
- Missing required extensions
- Invalid filter syntax
- Schema validation
- Constraint violations

#### 10. **Performance Tests**
- Batch processing (100+ documents)
- Large-scale search operations
- Index build time measurement
- Query response time tracking

### Test Structure

The test suite uses a custom tracking system:

```python
class TestResults:
    """Track test results."""
    def add_pass(self, test_name: str)
    def add_fail(self, test_name: str, error: str)
    def print_summary()
```

**Example output:**
```
✅ [PASS] HNSW Initialization
✅ [PASS] IVFFlat Initialization
✅ [PASS] DiskANN Initialization
✅ [PASS] Add Documents
✅ [PASS] Semantic Search
...

================================================================================
TEST SUMMARY
================================================================================
✅ PASSED: 45
❌ FAILED: 0
📊 TOTAL:  45
📈 SUCCESS RATE: 100.0%
================================================================================
```

### Running Specific Tests

While the test suite is designed to run all tests together, you can modify it to run specific test categories by commenting out tests in the `main()` function.

### Test Database Configuration

Tests use a dedicated schema to avoid conflicts:

```python
DB_HOST = "localhost"
DB_PORT = "9002"
DB_NAME = "postgres"
DB_USER = "user"
DB_PASSWORD = "root"
SCHEMA_NAME = "test"  # Isolated test schema
```

The test suite automatically:
1. Creates the test schema before running
2. Runs all tests
3. Drops the test schema after completion (cleanup)

### Test Data Generation

The test suite includes a sophisticated test data generator:

```python
def generate_test_documents(num_docs: int = 100) -> tuple[List[Document], List[List[int]]]:
    """Generate diverse test documents with metadata and labels."""
```

**Generated test data includes:**
- Diverse content across 8 categories (programming, AI, database, web, DevOps, security, cloud, mobile)
- Rich metadata (category, language, author, year, priority, status, tags)
- Label assignments for DiskANN testing
- Realistic content using templates and variations

---

## Evaluation

### Benchmark All Search Methods

Compare the performance of all 10 search methods using the benchmarking tool:

```bash
python eval/benchmark_all_methods.py
```

This tool:
- Tests all 10 search methods with the same queries
- Measures response times
- Compares result quality
- Exports results to CSV and JSON formats

**Output files:**
- `eval/benchmark_results.csv` - Tabular results
- `eval/benchmark_results.json` - Detailed JSON results

**Benchmark includes:**
- Query response time (ms)
- Results per method
- Score distributions
- Method comparison metrics

### Evaluate Search Quality

```python
from src.evaluation import RAGEvaluator, EvaluationDataset

# Create evaluation dataset
dataset = EvaluationDataset()
dataset.add_query(
    query="What is Python?",
    relevant_doc_ids=["1", "5", "10"]
)

# Evaluate
evaluator = RAGEvaluator(k=5)
result = evaluator.evaluate(
    queries=dataset.queries,
    retrieved_doc_ids=retrieved_results,
    ground_truth=dataset.ground_truth
)

print(f"Precision@5: {result.precision:.3f}")
print(f"Recall@5: {result.recall:.3f}")
print(f"F1@5: {result.f1_score:.3f}")
print(f"MAP: {result.map_score:.3f}")
print(f"MRR: {result.mrr_score:.3f}")
print(f"NDCG@5: {result.ndcg_score:.3f}")
print(f"Hit Rate: {result.hit_rate:.3f}")
```

### K-Value Optimization

Find the optimal K for your use case:

```python
from src.evaluation import KValueAnalysis

# Test multiple K values
k_values = [1, 3, 5, 10, 20, 50]
retrieved_by_k = {}

for k in k_values:
    results = []
    for query in queries:
        docs = await rag.semantic_search(query, k=k)
        doc_ids = [d.metadata['doc_id'] for d in docs]
        results.append(doc_ids)
    retrieved_by_k[k] = results

# Analyze
analyzer = KValueAnalysis()
analyzer.analyze(queries, retrieved_by_k, ground_truth)
analyzer.print_analysis()

# Get recommendation
rec = analyzer.get_recommendation()
print(f"Optimal K for balanced performance: {rec['optimal_balanced']}")
print(f"Best K for precision: {rec['max_precision'][0]}")
print(f"Best K for recall: {rec['max_recall'][0]}")
```

---

## Metrics Explained

### Precision@K
Percentage of retrieved documents that are relevant:
```
Precision@K = (Relevant docs in top K) / K
```
Use when: You want high-quality results, false positives are costly

### Recall@K
Percentage of relevant documents that were retrieved:
```
Recall@K = (Relevant docs in top K) / (Total relevant docs)
```
Use when: You want comprehensive coverage, missing results is costly

### F1@K
Harmonic mean of precision and recall:
```
F1@K = 2 × (Precision × Recall) / (Precision + Recall)
```
Use when: You want balanced performance

### MAP (Mean Average Precision)
Rank-aware precision metric, rewards relevant docs appearing earlier:
```
AP = Σ(Precision@k × relevance_k) / (Total relevant docs)
MAP = Average of AP across queries
```
Use when: Ranking order matters

### MRR (Mean Reciprocal Rank)
Focuses on position of first relevant result:
```
RR = 1 / (Rank of first relevant doc)
MRR = Average of RR across queries
```
Use when: Users typically click the first relevant result

### NDCG@K (Normalized DCG)
Position-discounted ranking quality:
```
DCG@K = Σ(relevance_i / log2(i + 1))
NDCG@K = DCG@K / Ideal_DCG@K
```
Use when: Position matters and you have graded relevance

### Hit Rate@K
Percentage of queries with at least one relevant result:
```
Hit Rate@K = (Queries with ≥1 relevant doc) / (Total queries)
```
Use when: You want to know if any relevant results are found

---

## Performance Tuning

### Database Configuration

Edit `postgresql.conf`:
```ini
shared_buffers = 4GB              # 25% of RAM
effective_cache_size = 12GB       # 75% of RAM
work_mem = 64MB
maintenance_work_mem = 2GB
```

### Connection Pooling

```python
connection_string = (
    "postgresql+asyncpg://user:pass@localhost/db"
    "?min_size=2&max_size=10&timeout=30"
)
```

### Batch Processing

```python
# Batch embed documents
contents = [doc.page_content for doc in documents]
embeddings_list = embeddings.embed_documents(contents)
```

### Index Selection Guide

| Dataset Size | Index | Build Time | Query Speed | Memory |
|--------------|-------|------------|-------------|--------|
| < 100K | HNSW | Medium | Fastest | High |
| 100K - 1M | HNSW | Slow | Fastest | High |
| > 1M | IVFFlat | Fast | Medium | Medium |
| > 10M | DiskANN | Slow | Medium | Low |

---

## Project Structure

```
Prod_RAG/
├── src/                         # Core source code
│   ├── __init__.py             # Package exports
│   ├── core.py                 # pgVectorDB class (~3100 lines)
│   └── evaluation.py           # Evaluation metrics
├── test/                        # Comprehensive test suite
│   └── test_suite.py           # 10+ test functions (~929 lines)
├── eval/                        # Optimization & benchmarking tools
│   ├── benchmark_all_methods.py  # Search method comparisons
│   ├── optimize_k.py            # K-value optimization
│   ├── benchmark_results.csv    # Benchmark data (CSV)
│   └── benchmark_results.json   # Benchmark data (JSON)
├── notebooks/                   # Interactive demos
│   ├── demo.ipynb              # Complete walkthrough
│   └── eval_demo.ipynb         # Evaluation examples
├── scripts/                     # Utility scripts
│   └── test_connection.py      # Connection & requirements tester
├── docker/                      # Docker deployment
│   ├── Dockerfile              # Python container
│   ├── docker-compose.yml      # Multi-container setup
│   ├── init.sql                # Database initialization
│   └── README.md               # Docker guide
├── config/                      # Configuration files
│   ├── .env                    # Environment variables
│   └── .env.example            # Template
├── examples/                    # Example scripts
│   └── README.md               # Examples guide
├── scripts/                     # Utility scripts
│   └── README.md               # Scripts guide
├── docs/                        # Extended documentation
│   └── README.md               # Documentation index
├── requirements.txt             # Python dependencies
├── README.md                    # Main documentation (this file)
└── STRUCTURE.md                 # Detailed folder structure
```

See [STRUCTURE.md](STRUCTURE.md) for detailed folder descriptions.

---

## Examples

### Complete Walkthrough
See [notebooks/demo.ipynb](notebooks/demo.ipynb) for a complete feature demonstration.

### Evaluation Metrics
See [notebooks/eval_demo.ipynb](notebooks/eval_demo.ipynb) for evaluation examples.

### K-Value Optimization
Run [eval/optimize_k.py](eval/optimize_k.py) to find optimal K values.

### Docker Deployment
See [docker/README.md](docker/README.md) for containerized deployment guide.

### Additional Examples
Check [examples/](examples/) folder for more usage examples.

---

## Benchmarks

Performance on 10K documents with `sentence-transformers/all-MiniLM-L6-v2`:

| Search Method | Precision@5 | Recall@5 | F1@5 | NDCG@5 | Latency |
|--------------|-------------|----------|------|---------|---------|
| Semantic | 0.842 | 0.756 | 0.797 | 0.889 | 45ms |
| Keyword | 0.785 | 0.698 | 0.739 | 0.824 | 12ms |
| Hybrid | 0.891 | 0.812 | 0.850 | 0.921 | 52ms |
| Ensemble | 0.903 | 0.834 | 0.867 | 0.935 | 58ms |

---

## Troubleshooting

### Connection Issues
```python
# Check database connection
psql -h localhost -p 9002 -U user -d postgres

# Verify extensions
psql -c "SELECT * FROM pg_extension WHERE extname IN ('vector', 'vectorscale');"
```

### Memory Issues
- Use IVFFlat or DiskANN instead of HNSW
- Reduce `m` parameter for HNSW
- Enable connection pooling

### Slow Queries
- Increase `ef_search` for HNSW
- Increase `probes` for IVFFlat
- Add metadata filters to reduce search space
- Use hybrid search instead of pure semantic

### Poor Recall
- Increase K value
- Use ensemble search
- Tune index parameters (higher `ef_construction`, more `probes`)
- Try different embedding models

---

## API Reference

### pgVectorDB Class

#### Constructor
```python
pgVectorDB(
    collection_name: str,
    embedding_model: Embeddings,
    connection_string: str,
    schema_name: str = "public",
    index_type: IndexType = IndexType.HNSW,
    pool_size: int = 5,
    max_overflow: int = 10
)
```

#### Initialization & Setup
- `initialize(overwrite_existing=False)` - Initialize vector store and extensions
- `close()` - Close database connections

#### Document Management (6 methods)
- `add_documents(documents, labels=None)` - Add documents with optional DiskANN labels
- `add_documents_batch(documents, batch_size=100, labels=None)` - Batch add with progress
- `aupdate_documents(ids, documents)` - Update existing documents
- `update_metadata(ids, metadata_updates)` - Update document metadata
- `adelete(ids)` - Delete documents by IDs
- `aget_by_ids(ids)` - Retrieve documents by IDs

#### Index Operations (6 methods)
- `build_index(**kwargs)` - Build vector index (HNSW/IVFFlat/DiskANN)
  - HNSW: `m`, `ef_construction`
  - IVFFlat: `lists`
  - DiskANN: `num_neighbors`, `search_list_size`, `storage_layout`, `metric`
- `build_bm25_index(text_config="english", k1=1.2, b=0.75)` - Build BM25 index
- `create_metadata_index(columns)` - Create B-tree indexes on metadata
- `areindex(index_type, **kwargs)` - Rebuild existing index
- `adrop_vector_index(index_name=None)` - Drop vector index
- `set_query_params(**params)` - Set runtime query parameters
  - HNSW: `ef_search`
  - IVFFlat: `probes`
  - DiskANN: `search_list_size`

#### Search Methods (10 methods)
- `keyword_search(query, k, search_type=KeywordSearchType.BM25)` - FTS or BM25 search
- `universal_keyword_search(query, k, search_type=KeywordSearchType.BM25)` - Search across all text fields
- `semantic_search(query, k, label_filter=None, **kwargs)` - Vector similarity search
- `metadata_filter(filter, k=100)` - Filter by metadata only
- `metadata_keyword_search(query, filter, k, search_type)` - Filtered keyword search
- `metadata_semantic_search(query, filter, k, label_filter=None)` - Filtered semantic search
- `hybrid_search(query, k, weights=(0.5, 0.5), use_rrf=False, keyword_type=BM25)` - Combined keyword + semantic
- `ensemble_search(query, filter, k, weights, use_rrf, keyword_type)` - Filtered hybrid search
- `trigram_search(query, k, threshold=0.3)` - Fuzzy text matching
- `metadata_trigram_search(query, filter, k, threshold)` - Filtered fuzzy search

#### Advanced Search (2 methods)
- `asimilarity_search_by_vector(embedding, k, filter=None, label_filter=None)` - Search by vector
- `asimilarity_search_with_score(query, k, filter=None, label_filter=None)` - Search with scores

#### Analytics & Monitoring (5 methods)
- `get_stats()` - Collection statistics (docs, size, index type)
- `get_index_stats()` - Detailed index metrics
- `count_by_metadata(filter)` - Count documents matching filter
- `explain_query(query, method="semantic")` - PostgreSQL EXPLAIN ANALYZE
- `validate_collection()` - Validate collection integrity
- `benchmark_search_methods(queries, k)` - Compare all search methods

#### Data Export/Import (2 methods)
- `export_to_json(output_path, filter=None, batch_size=1000)` - Export to JSON
- `import_from_json(input_path, batch_size=100)` - Import from JSON

#### Database Operations (1 method)
- `vacuum_analyze(full=False, analyze=True)` - Optimize database

#### LangChain Integration (1 method)
- `as_retriever(search_method="semantic_search", search_kwargs={})` - Create LangChain retriever

**Total: 33 public methods**

---

### RAGEvaluator

**Constructor:**
```python
RAGEvaluator(k: int = 5)
```

**Methods:**
- `evaluate(queries, retrieved_doc_ids, ground_truth)` - Calculate all 7 metrics
- `precision_at_k(retrieved, relevant, k)` - Precision@K
- `recall_at_k(retrieved, relevant, k)` - Recall@K
- `f1_score_at_k(precision, recall)` - F1@K
- `mean_average_precision(retrieved, relevant, k)` - MAP
- `mean_reciprocal_rank(retrieved, relevant)` - MRR
- `ndcg_at_k(retrieved, relevant, k)` - NDCG@K
- `hit_rate_at_k(retrieved, relevant, k)` - Hit Rate

### KValueAnalysis

**Methods:**
- `analyze(queries, retrieved_by_k, ground_truth)` - Analyze K values
- `print_analysis()` - Print results table
- `get_recommendation()` - Get optimal K recommendations
- `plot_metrics()` - Visualize metrics

---

### Enums & Types

#### IndexType
```python
class IndexType(str, Enum):
    HNSW = "hnsw"
    IVFFLAT = "ivfflat"
    DISKANN = "diskann"
```

#### KeywordSearchType
```python
class KeywordSearchType(str, Enum):
    FTS = "fts"      # PostgreSQL ts_rank
    BM25 = "bm25"    # pg_textsearch BM25
```

#### DistanceMetric
```python
class DistanceMetric(str, Enum):
    COSINE = "cosine"
    L2 = "l2"
    INNER_PRODUCT = "inner_product"
```

#### StorageLayout
```python
class StorageLayout(str, Enum):
    MEMORY_OPTIMIZED = "memory_optimized"  # SBQ compression
    PLAIN = "plain"  # Uncompressed
```

#### Custom Exceptions
```python
class RetrievalSystemError(Exception)  # Base exception
class InitializationError(RetrievalSystemError)
class ValidationError(RetrievalSystemError)
class DatabaseError(RetrievalSystemError)
```

---

## License

This project is open source and available for use.

---

## BM25 Configuration Guide

### Language Support
BM25 supports 29+ text configurations for different languages:

```python
# English (default)
await rag.build_bm25_index(text_config="english")

# Other languages
await rag.build_bm25_index(text_config="french")
await rag.build_bm25_index(text_config="german")
await rag.build_bm25_index(text_config="spanish")
await rag.build_bm25_index(text_config="portuguese")
await rag.build_bm25_index(text_config="italian")
await rag.build_bm25_index(text_config="dutch")
await rag.build_bm25_index(text_config="russian")
await rag.build_bm25_index(text_config="simple")  # Basic stemming
```

Full list: arabic, basque, catalan, danish, dutch, english, finnish, french, german, greek, hindi, hungarian, indonesian, irish, italian, lithuanian, nepali, norwegian, portuguese, romanian, russian, serbian, spanish, swedish, tamil, turkish, yiddish

### BM25 Parameter Tuning

**k1 (term frequency saturation, default=1.2):**
- Lower (0.5-1.0): Less emphasis on term frequency, better for short documents
- Higher (1.5-2.0): More emphasis on term frequency, better for long documents
- Typical range: 1.2-2.0

**b (length normalization, default=0.75):**
- Lower (0.0-0.5): Less penalty for long documents
- Higher (0.8-1.0): More penalty for long documents
- 0.0 = no normalization, 1.0 = full normalization
- Typical range: 0.5-0.9

```python
# Short documents (tweets, product titles)
await rag.build_bm25_index(k1=0.8, b=0.5)

# Medium documents (articles, blogs) - default
await rag.build_bm25_index(k1=1.2, b=0.75)

# Long documents (papers, books)
await rag.build_bm25_index(k1=1.5, b=0.9)
```

### FTS vs BM25 Comparison

| Feature | FTS (ts_rank) | BM25 (pg_textsearch) |
|---------|---------------|----------------------|
| Speed | Faster | Slightly slower |
| Ranking | Basic frequency | Advanced TF-IDF |
| Length normalization | Limited | Full control (b) |
| Term saturation | No | Yes (k1) |
| Language support | 29+ configs | 29+ configs |
| Setup | Built-in | Requires extension |
| Use case | Fast lookups | Better ranking |

**When to use FTS:**
- Speed is critical
- Simple keyword matching is sufficient
- No extension installation required

**When to use BM25:**
- Ranking quality matters
- Working with varied document lengths
- Need industry-standard results (Elasticsearch-like)
- Fine-tuning control needed

---

**Version:** 2.1.0  
**Last Updated:** December 26, 2025



# Project Structure

## Overview
Production-ready PostgreSQL Vector Database (pgVectorDB) system with organized folders for development, deployment, and documentation.

```
Prod_RAG/
│
├── src/                          # Core source code
│   ├── __init__.py              # Package initialization & exports
│   ├── core.py                  # Main pgVectorDB class (1552 lines)
│   └── evaluation.py            # RAG evaluation metrics (942 lines)
│
├── test/                         # Comprehensive test suite
│   └── test.py                  # 33 test functions (1241 lines)
│
├── eval/                         # Evaluation & optimization tools
│   └── optimize_k.py            # K-value optimization script
│
├── notebooks/                    # Jupyter notebooks for demos
│   ├── demo.ipynb               # Complete feature demonstration (74 cells)
│   └── eval_demo.ipynb          # Evaluation metrics examples
│
├── docker/                       # Docker deployment files
│   ├── Dockerfile               # Python app container
│   ├── docker-compose.yml       # Multi-container setup
│   ├── init.sql                 # Database initialization
│   └── README.md                # Docker setup guide
│
├── config/                       # Configuration files
│   └── .env                     # Database credentials & settings
│
├── examples/                     # Example scripts & usage
│   └── (add your example files here)
│
├── scripts/                      # Utility scripts
│   └── (add utility scripts here)
│
├── docs/                         # Additional documentation
│   └── (add documentation files here)
│
├── .vscode/                      # VS Code settings
│   └── settings.json
│
├── .gitignore                    # Git ignore rules
├── requirements.txt              # Python dependencies
├── README.md                     # Main documentation
└── STRUCTURE.md                  # This file
```

---

## Folder Descriptions

### `/src` - Source Code
**Purpose:** Core system implementation  
**Key Files:**
- `core.py` - pgVectorDB class with 10 search methods
- `evaluation.py` - 7 evaluation metrics (Precision, Recall, F1, MAP, MRR, NDCG, Hit Rate)
- `__init__.py` - Package exports (pgVectorDB, IndexType, etc.)

### `/test` - Tests
**Purpose:** Comprehensive test suite  
**File:** `test_suite.py` (~929 lines)
**Coverage:**
- Initialization tests (3 index types: HNSW, IVFFlat, DiskANN)
- Document operations (CRUD, batch, metadata updates)
- All 10 search methods (keyword, semantic, hybrid, etc.)
- All 13 filter operators ($eq, $ne, $in, $between, etc.)
- Index operations (build, reindex, vacuum, stats)
- Analytics and monitoring
- Data export/import (JSON format)
- LangChain integration (as_retriever)
- Error handling and validation
- Performance benchmarks

### `/eval` - Evaluation Tools
**Purpose:** RAG system optimization & benchmarking  
**Files:**
- `optimize_k.py` - K-value optimization for search quality
- `benchmark_all_methods.py` - Compare all 10 search methods
- `benchmark_results.csv` - Benchmark results (CSV format)
- `benchmark_results.json` - Benchmark results (JSON format)
**Features:**
- Search method performance comparison
- K-value optimization for precision/recall
- Result export in multiple formats

### `/notebooks` - Interactive Demos
**Purpose:** Jupyter notebook demonstrations  
**Notebooks:**
- `demo.ipynb` - Complete walkthrough (all 10 search methods)
- `eval_demo.ipynb` - Evaluation metrics examples

### `/docker` - Deployment
**Purpose:** Containerized deployment  
**Files:**
- `Dockerfile` - Python application container
- `docker-compose.yml` - PostgreSQL + pgvector + app
- `init.sql` - Database initialization script
- `README.md` - Docker setup instructions

### `/config` - Configuration
**Purpose:** Environment settings  
**Files:**
- `.env` - Database connection, embedding model settings

### `/examples` - Usage Examples
**Purpose:** Standalone example scripts  
**Suggested files:**
- `basic_search.py` - Simple search example
- `advanced_filtering.py` - Complex filter examples
- `batch_processing.py` - Large dataset handling
- `custom_evaluation.py` - Custom metric evaluation

### `/scripts` - Utilities
**Purpose:** Development & maintenance scripts  
**Files:**
- `test_connection.py` - Comprehensive system validation
  - Python packages verification
  - Database connection testing
  - PostgreSQL extensions check (vector, pg_trgm, vectorscale)
  - Embedding model loading test
  - pgVectorDB import validation

### `/docs` - Documentation
**Purpose:** Extended documentation  
**Suggested files:**
- `API_REFERENCE.md` - Complete API documentation
- `DEPLOYMENT.md` - Deployment guide
- `TROUBLESHOOTING.md` - Common issues & solutions
- `ARCHITECTURE.md` - System architecture diagrams

---

## Quick Navigation

### For Users
1. **Getting Started:** [README.md](README.md)
2. **Interactive Demo:** [notebooks/demo.ipynb](notebooks/demo.ipynb)
3. **Docker Setup:** [docker/README.md](docker/README.md)

### For Developers
1. **Source Code:** [src/core.py](src/core.py)
2. **Tests:** [test/test.py](test/test.py)
3. **Evaluation:** [src/evaluation.py](src/evaluation.py)

### For DevOps
1. **Docker Setup:** [docker/](docker/)
2. **Configuration:** [config/.env](config/.env)
3. **Scripts:** [scripts/](scripts/)

---

## File Count Summary

| Folder | Files | Lines of Code | Purpose |
|--------|-------|---------------|---------|
| src/ | 3 | ~3,100+ | Core implementation |
| test/ | 1 | ~929 | Test suite |
| eval/ | 4 | ~200+ | Benchmarking & optimization |
| notebooks/ | 2 | - | Interactive demos |
| scripts/ | 1 | - | Utilities |
| docker/ | 4 | - | Deployment |
| config/ | 1 | - | Settings |

**Total:** ~4,200+ lines of production code

### Python Packages
- **LangChain:** Core RAG framework
- **pgvector:** PostgreSQL vector extension
- **asyncpg:** Async PostgreSQL driver
- **HuggingFace:** Embedding models
- **SQLAlchemy:** Database ORM

### System Requirements
- **PostgreSQL 12+** with pgvector extension
- **Python 3.9+**
- **Docker** (optional, for containerized deployment)

---

## Development Workflow

### Setup
```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Configure database
cp config/.env.example config/.env
# Edit config/.env with your credentials

# 3. Initialize database (if not using Docker)
psql -c "CREATE EXTENSION vector;"
psql -c "CREATE EXTENSION vectorscale CASCADE;"
```

### Testing
```bash
# Run comprehensive test suite
python test/test_suite.py

# Or with more verbose output
python -m pytest test/test_suite.py -v
```

### Docker Deployment
```bash
cd docker/
docker-compose up -d
```

---

## Contributing

When adding new features:
1. Add source code to `src/`
2. Add tests to `test/`
3. Add examples to `examples/`
4. Update documentation in `docs/`
5. Update README.md

---

## Version

**Current Version:** 0.0.2  
**Last Updated:** December 26, 2025  
**Main Class:** `pgVectorDB`


