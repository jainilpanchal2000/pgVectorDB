# Installation & Setup

pgVectorDB requires a PostgreSQL database with specific extensions for vector search. The fastest way to get a fully configured database is with the provided Docker image.

---

## Option 1: Docker (Recommended)

The `docker/` folder in the repo contains a custom PostgreSQL 17 image pre-installed with all four required extensions: `pgvector`, `pg_trgm`, `pgvectorscale` (DiskANN), and `pg_textsearch` (BM25).

### Build & Start

```bash
# Clone the repo
git clone https://github.com/jainilpanchal2000/pgVectorDB.git
cd pgVectorDB/docker

# Build the image (takes 5–10 min — compiles Rust extensions from source)
docker compose build

# Start the database in the background
docker compose up -d

# Verify all 4 extensions are installed
docker compose exec db psql -U user -d postgres -c \
  "SELECT extname, extversion FROM pg_extension WHERE extname IN ('vector','pg_trgm','vectorscale','pg_textsearch');"
```

### What's Inside the Image

| Layer | Details |
|-------|---------|
| **Base** | `pgvector/pgvector:pg17` (PostgreSQL 17 + pgvector 0.8.2) |
| **pgvector** | v0.8.2 — includes parallel HNSW buffer overflow fix |
| **pgvectorscale** | v0.9.0 — DiskANN index + label filtering (via Rust/pgrx) |
| **pg_textsearch** | v1.0.0 — BM25 ranking (production-ready: pg_dump, VACUUM support) |
| **pg_trgm** | Built-in PostgreSQL module — trigram fuzzy search |

### Default Connection

```
Host:     localhost
Port:     9002
Database: postgres
User:     user
Password: root
```

```bash
# Connection string for pgVectorDB
postgresql+asyncpg://user:root@localhost:9002/postgres
```

Override any value via `.env` in the `docker/` folder:

```bash
# docker/.env
POSTGRES_USER=myuser
POSTGRES_PASSWORD=mysecret
POSTGRES_DB=mydb
DB_PORT=5433
```

!!! note "shm_size"
    The compose file sets `shm_size: 4gb`. This is required for HNSW index builds on large datasets. Reduce to `1gb` for development.

### Stop / Reset

```bash
docker compose down          # Stop (keeps data volume)
docker compose down -v       # Stop + wipe data volume (clean slate)
```

---

## Option 2: Existing PostgreSQL

If you already have PostgreSQL 17+, install the extensions manually.

### Required (must have)

```sql
CREATE EXTENSION IF NOT EXISTS vector;    -- pgvector: vector storage + HNSW/IVFFlat
CREATE EXTENSION IF NOT EXISTS pg_trgm;   -- trigram fuzzy search
```

### Optional (unlocks more features)

```sql
-- DiskANN index + label filtering (requires vectorscale)
CREATE EXTENSION IF NOT EXISTS vectorscale CASCADE;

-- BM25 keyword search (requires pg_textsearch)
CREATE EXTENSION IF NOT EXISTS pg_textsearch;
```

!!! tip "Timescale Cloud"
    [Timescale Cloud](https://www.timescale.com/cloud) provides hosted PostgreSQL with `pgvector`, `vectorscale`, and `pg_textsearch` pre-installed — no compilation needed.

### Extension Availability

| Extension | Required | Installed by | Missing = |
|-----------|----------|--------------|-----------|
| `vector` | ✅ Yes | `pgvector/pgvector` Docker image | Hard error on init |
| `pg_trgm` | ✅ Yes | Built into PostgreSQL | Hard error on init |
| `vectorscale` | ⬜ Optional | Timescale / compile from source | DiskANN unavailable |
| `pg_textsearch` | ⬜ Optional | Timescale / compile from source | BM25 falls back to FTS |

---

## Python Package

```bash
# Core (works with any embedding model you bring)
pip install pgvectordb

# With local HuggingFace embeddings
pip install "pgvectordb[huggingface]"

# With Cohere reranking
pip install "pgvectordb[cohere]"

# With AWS Bedrock reranking
pip install "pgvectordb[aws]"

# With local cross-encoder reranking (PyTorch)
pip install "pgvectordb[rerankers]"

# Everything
pip install "pgvectordb[all]"
```

---

## Quick Connection Test

```python
import asyncio
from pgvectordb import pgVectorDB, IndexType
from langchain_huggingface import HuggingFaceEmbeddings

async def test_connection():
    db = pgVectorDB(
        collection_name="test",
        embedding_model=HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2"),
        connection_string="postgresql+asyncpg://user:root@localhost:9002/postgres",
        index_type=IndexType.HNSW
    )
    await db.initialize()
    stats = await db.get_stats()
    print(f"✓ Connected — {stats['document_count']} documents")
    await db.close()

asyncio.run(test_connection())
```

!!! tip
    `initialize()` is idempotent — safe to call on every startup. It creates tables, triggers, and indexes only if they don't already exist.
