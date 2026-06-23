# Quickstart Guide

Get started with pgVectorDB in 5 minutes using the LanceDB-style fluent API.

---

## Installation

```bash
pip install pgvectordb[huggingface]
```

For the database, use our Docker image with all extensions pre-installed:

```bash
docker run -d \
  --name pgvectordb \
  -e POSTGRES_PASSWORD=postgres \
  -p 5432:5432 \
  ankane/pgvector:latest
```

---

## Basic Usage

### 1. Initialize the Database

```python
import asyncio
from langchain_huggingface import HuggingFaceEmbeddings
from pgvectordb import pgVectorDB

async def main():
    # Initialize embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    
    # Create database instance
    db = pgVectorDB(
        collection_name="my_docs",
        embedding_model=embeddings,
        connection_string="postgresql+asyncpg://postgres:postgres@localhost/postgres"
    )
    
    # Create tables and indexes
    await db.initialize()
    print("✓ Database initialized")
```

### 2. Add Documents

```python
    # Add documents with metadata
    documents = [
        "Machine learning is a subset of AI that enables computers to learn from data.",
        "PostgreSQL is a powerful open-source relational database system.",
        "Vector databases enable efficient similarity search for AI embeddings.",
        "Python is the most popular language for data science and ML.",
    ]
    
    metadatas = [
        {"category": "ai", "year": 2024},
        {"category": "database", "year": 2024},
        {"category": "database", "year": 2024},
        {"category": "programming", "year": 2024},
    ]
    
    ids = await db.add_texts(texts=documents, metadatas=metadatas)
    print(f"✓ Added {len(ids)} documents")
```

### 3. Search with Fluent API

The new LanceDB-style fluent API provides intuitive method chaining:

```python
    # Basic semantic search
    results = await (
        db.search("machine learning AI")
        .limit(3)
        .to_list()
    )
    
    for r in results:
        print(f"  [{r['score']:.4f}] {r['content'][:60]}...")
```

### 4. Filtered Search

```python
    # Search with metadata filter
    results = await (
        db.search("database systems")
        .where({"category": "database"})
        .limit(3)
        .to_list()
    )
```

### 5. Hybrid Search (Vector + Text)

```python
    # Hybrid search combining vector and full-text search
    results = await (
        db.search("machine learning frameworks")
        .where({"category": "ai"})
        .limit(5)
        .nearest_to_text("Python ML libraries")  # Add FTS component
        .to_list()
    )
```

---

## Complete Example

```python
import asyncio
from langchain_huggingface import HuggingFaceEmbeddings
from pgvectordb import pgVectorDB

async def main():
    # Setup
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    
    db = pgVectorDB(
        collection_name="quickstart",
        embedding_model=embeddings,
        connection_string="postgresql+asyncpg://postgres:postgres@localhost/postgres"
    )
    
    await db.initialize()
    
    # Add documents
    await db.add_texts(
        texts=[
            "Transformers revolutionized NLP with attention mechanisms.",
            "BERT is a bidirectional encoder for language understanding.",
            "GPT models use autoregressive language modeling.",
            "PostgreSQL supports JSONB for flexible document storage.",
        ],
        metadatas=[
            {"topic": "nlp", "model": "transformers"},
            {"topic": "nlp", "model": "bert"},
            {"topic": "nlp", "model": "gpt"},
            {"topic": "database", "model": "postgres"},
        ]
    )
    
    # Build index for faster search
    await db.build_index()
    
    # Search examples
    print("\n=== Semantic Search ===")
    results = await db.search("language models").limit(2).to_list()
    for r in results:
        print(f"  [{r['score']:.4f}] {r['content']}")
    
    print("\n=== Filtered Search ===")
    results = await (
        db.search("models")
        .where({"topic": "nlp"})
        .limit(2)
        .to_list()
    )
    for r in results:
        print(f"  [{r['score']:.4f}] {r['content'][:50]}...")
    
    # Cleanup
    await db.close()

if __name__ == "__main__":
    asyncio.run(main())
```

---

## Query Builder Methods

The fluent API supports these chainable methods:

| Method | Description | Example |
|--------|-------------|---------|
| `search(query)` | Start vector search | `db.search("query")` |
| `search_text(query)` | Start text search | `db.search_text("query")` |
| `limit(n)` | Set max results | `.limit(10)` |
| `offset(n)` | Skip first n results | `.offset(5)` |
| `where(filter)` | Apply metadata filter | `.where({"category": "ai"})` |
| `select(cols)` | Choose columns | `.select(["content", "metadata"])` |
| `distance_type(m)` | Set metric | `.distance_type("cosine")` |
| `nprobes(n)` | IVF probes | `.nprobes(20)` |
| `ef(n)` | HNSW ef_search | `.ef(100)` |
| `refine_factor(n)` | Oversampling | `.refine_factor(2)` |
| `distance_range(l,u)` | Distance bounds | `.distance_range(0, 0.5)` |
| `bypass_vector_index()` | Exact search | `.bypass_vector_index()` |
| `to_list()` | Execute → list | `.to_list()` |
| `to_pandas()` | Execute → DataFrame | `.to_pandas()` |
| `to_arrow()` | Execute → Arrow | `.to_arrow()` |

---

## Next Steps

- **[Search & Retrieval](search_and_retrieval.md)** — Deep dive into all search methods
- **[Metadata Filtering](filtering.md)** — All filter operators ($eq, $in, $between, etc.)
- **[Indexing & Performance](../advanced/indexing.md)** — HNSW, IVFFlat, DiskANN tuning
- **[Analytics](../user_guide/analytics_and_diagnostics.md)** — Query plans, benchmarks, recall
