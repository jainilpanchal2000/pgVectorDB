# Quickstart

This guide walks you through building a complete Retrieval-Augmented Generation (RAG) pipeline with pgVectorDB in under 5 minutes.

---

## 1. Setup

Make sure your PostgreSQL database is running (see [Installation](installation.md)) and pgVectorDB is installed.

For this guide we use HuggingFace sentence-transformers to generate embeddings locally (no API key needed):

```bash
pip install "pgvectordb[huggingface]"
```

---

## 2. Initialize the Database

Define your embedding model and initialize the `pgVectorDB` instance. The `initialize()` call creates all required PostgreSQL objects — extensions, table, triggers, and indexes.

```python
import asyncio
from langchain_huggingface import HuggingFaceEmbeddings
from pgvectordb import pgVectorDB

async def main():
    # 1. Initialize the embedding model
    #    all-MiniLM-L6-v2 produces 384-dimensional vectors
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    # 2. Connect to the database
    conn_string = "postgresql+asyncpg://myuser:mypassword@localhost/mydb"

    db = pgVectorDB(
        collection_name="my_documents",
        embedding_model=embeddings,
        connection_string=conn_string
    )

    # 3. Create the tables, triggers, and indexes (idempotent)
    await db.initialize()
    print("Database initialized!")

    # ... (code continues below)
```

---

## 3. Add Documents

Pass raw text strings plus optional metadata. pgVectorDB automatically computes embeddings and stores everything in PostgreSQL.

```python
    # ... (inside main function)

    documents = [
        "The quick brown fox jumps over the lazy dog.",
        "PostgreSQL is a powerful, open source object-relational database system.",
        "pgvector is an open-source vector similarity search extension for Postgres.",
        "Machine learning models map text to dense vector embeddings.",
        "The weather today is sunny with a high of 75 degrees."
    ]

    # Attach metadata for later filtering
    metadata = [
        {"source": "story",   "id": 1},
        {"source": "wiki",    "id": 2},
        {"source": "wiki",    "id": 3},
        {"source": "science", "id": 4},
        {"source": "news",    "id": 5}
    ]

    print("Adding documents...")
    doc_ids = await db.add_texts(texts=documents, metadatas=metadata)
    print(f"Added {len(doc_ids)} documents.")
```

---

## 4. Perform a Semantic Search

Query the vector index for the most semantically relevant documents:

```python
    # ... (inside main function)

    query = "What is a good database for storing embeddings?"

    print(f"\nSearching for: '{query}'")
    results = await db.semantic_search(query, k=2)

    for i, res in enumerate(results):
        print(f"\nResult {i+1}:")
        print(f"  Content: {res['content']}")
        print(f"  Score:   {res['score']:.4f}")
        print(f"  Metadata: {res['metadata']}")

    # Close the connection pool when done
    await db.close()

if __name__ == "__main__":
    asyncio.run(main())
```

### Expected Output

```
Database initialized!
Adding documents...
Added 5 documents.

Searching for: 'What is a good database for storing embeddings?'

Result 1:
  Content: pgvector is an open-source vector similarity search extension for Postgres.
  Score:   0.5432
  Metadata: {'source': 'wiki', 'id': 3}

Result 2:
  Content: PostgreSQL is a powerful, open source object-relational database system.
  Score:   0.6123
  Metadata: {'source': 'wiki', 'id': 2}
```

!!! note "Score Interpretation"
    Score semantics depend on the search method:

    | Search Method | Score Direction | Meaning |
    |--------------|----------------|---------|
    | `semantic_search` | **Lower = more similar** | Cosine distance (0 = identical) |
    | `keyword_search` (FTS) | **Higher = more relevant** | `ts_rank` score |
    | `keyword_search` (BM25) | **Higher = more relevant** | BM25 score |
    | `hybrid_search` | **Higher = more relevant** | Fused RRF or weighted score |

---

## 5. Filter by Metadata

Combine vector search with metadata filtering to build multi-tenant systems or category-scoped search:

```python
# Only search within 'wiki' documents
results = await db.semantic_search(
    query="database extensions",
    filter={"source": "wiki"},
    k=5
)
```

See [Metadata Filtering](../user_guide/filtering.md) for all 13 available filter operators.

---

## Next Steps

You've built a basic vector search pipeline. From here, explore:

- **[Search & Retrieval](../user_guide/search_and_retrieval.md)** — All 10 search methods including Hybrid (BM25 + Vector) and trigram fuzzy search
- **[Metadata Filtering](../user_guide/filtering.md)** — `$eq`, `$in`, `$between`, `$and`, `$or`, and 8 more operators
- **[Multimodal Search](../user_guide/multimodal_search.md)** — Multiple embeddings per document (text + price + category)
- **[Reranking](../user_guide/reranking.md)** — Cross-encoder and Cohere reranking for precision
- **[Indexing & Performance](../advanced/indexing.md)** — HNSW, IVFFlat, and DiskANN tuning
