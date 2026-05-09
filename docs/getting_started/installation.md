# Installation and Setup

pgVectorDB connects to PostgreSQL with specific extensions for high-performance vector search.

## Prerequisites & Required Extensions

| Extension | Required | Purpose |
|-----------|----------|---------|
| `vector` | Yes | Core vector operations (pgvector) |
| `pg_trgm` | Yes | Trigram fuzzy matching |
| `vectorscale` | Optional | DiskANN index support |
| `pg_textsearch` | Optional | BM25 keyword search |

!!! warning
    `vector` and `pg_trgm` are **required**. Without them, pgVectorDB cannot work.

---

## Docker Setup (Recommended)

Use our pre-built image with all extensions:

```yaml
# docker-compose.yml
services:
  db:
    image: pg17-vectorscale-textsearch:latest
    container_name: pgvectordb
    shm_size: 4gb  # Required for vector operations
    ports:
      - "${DB_PORT:-9002}:5432"
    environment:
      POSTGRES_USER: ${POSTGRES_USER:-user}
      POSTGRES_PASSWORD: ${POSTGRES_PASSWORD:-root}
      POSTGRES_DB: ${POSTGRES_DB:-postgres}
    volumes:
      - pgvector_data:/var/lib/postgresql/data
    restart: unless-stopped

volumes:
  pgvector_data:
    driver: local
```

```bash
# Start the database
docker-compose up -d
```

---

## Python Package Installation

```bash
# Core package
pip install pgvectordb

# With optional dependencies
pip install "pgvectordb[huggingface]"   # Local embeddings
pip install "pgvectordb[cohere]"         # Cohere reranking
pip install "pgvectordb[aws]"            # AWS Bedrock
pip install "pgvectordb[rerankers]"      # Local cross-encoders
pip install "pgvectordb[all]"           # Everything
```

---

## Environment Variables

```bash
# Set database URL
export PGVECTORDB_URL="postgresql+asyncpg://user:root@localhost:9002/postgres"
```

---

## Quick Initialization

```python
import asyncio
from pgvectordb import pgVectorDB, IndexType

async def setup():
    db = pgVectorDB(
        collection_name="my_documents",
        embedding_model=my_embedding_model,
        connection_string="postgresql+asyncpg://user:root@localhost:9002/postgres",
        index_type=IndexType.DISKANN
    )
    
    # This creates extensions, tables, indexes - no SQL needed!
    await db.initialize()

asyncio.run(setup())
```

!!! tip
    Run `initialize()` once per collection. It creates all required PostgreSQL objects automatically.
