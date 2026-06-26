# Quickstart

This guide creates a small collection, adds documents with metadata, runs fluent searches, and inspects a query in one pass.

## 1. Install the package

```bash
pip install "pgvectordb[huggingface]"
```

For PostgreSQL setup, use the [installation guide](installation.md). The project Docker Compose setup includes `pgvector`, `pg_trgm`, `vectorscale`, and `pg_textsearch` so you can try semantic search, fuzzy search, DiskANN, and BM25 from the same database.

## 2. Start from one async script

```python
import asyncio

from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

from pgvectordb import IndexType, pgVectorDB


async def main():
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    db = pgVectorDB(
        collection_name="quickstart_docs",
        embedding_model=embeddings,
        connection_string="postgresql+asyncpg://user:root@localhost:9002/postgres",
        index_type=IndexType.HNSW,
    )

    await db.initialize()

    documents = [
        Document(
            page_content="pgVectorDB stores vectors, text, and metadata in PostgreSQL.",
            metadata={"topic": "database", "year": 2026, "tier": "core"},
        ),
        Document(
            page_content="Hybrid retrieval combines semantic similarity with keyword ranking.",
            metadata={"topic": "search", "year": 2026, "tier": "core"},
        ),
        Document(
            page_content="RAG evaluation measures Hit Rate, MRR, NDCG, precision, and recall.",
            metadata={"topic": "evaluation", "year": 2026, "tier": "advanced"},
        ),
        Document(
            page_content="Scalar indexes speed up JSONB metadata filters like category and year.",
            metadata={"topic": "optimization", "year": 2026, "tier": "advanced"},
        ),
    ]

    ids = await db.add_documents(documents)
    print(f"Added {len(ids)} documents")

    await db.build_index()

    results = await (
        db.query("how do I improve retrieval quality?")
        .semantic()
        .where({"year": {"$gte": 2026}})
        .limit(3)
        .to_list()
    )

    for result in results:
        print(f"{result['score']:.4f}  {result['content']}")

    await db.close()


if __name__ == "__main__":
    asyncio.run(main())
```

## 3. Use the fluent API for search modes

`db.query(...)` is the recommended entry point. You choose the retrieval mode with chainable methods, add filters and tuning parameters, then execute with an output method.

=== "Semantic"

    ```python
    results = await (
        db.query("retrieval evaluation metrics")
        .semantic()
        .limit(5)
        .to_list()
    )
    ```

=== "Keyword / BM25"

    ```python
    results = await (
        db.query("PostgreSQL JSONB filters")
        .keyword()
        .bm25_params(k1=1.2, b=0.75)
        .limit(5)
        .to_list()
    )
    ```

=== "Hybrid"

    ```python
    results = await (
        db.query("fast filtered vector search")
        .hybrid()
        .weights(semantic=0.7, keyword=0.3)
        .where({"tier": "advanced"})
        .limit(5)
        .to_list()
    )
    ```

=== "Fuzzy"

    ```python
    results = await (
        db.query("Postgres vectro serch")
        .trigram()
        .threshold(0.2)
        .limit(5)
        .to_list()
    )
    ```

## 4. Add filters, analysis, and output formats

Filters use MongoDB-style JSON syntax and compile to PostgreSQL JSONB predicates.

```python
results = await (
    db.query("production diagnostics")
    .semantic()
    .where({"topic": {"$in": ["optimization", "evaluation"]}})
    .select(["content", "metadata", "score"])
    .ef(100)
    .limit(10)
    .to_list()
)
```

Use query analysis before tuning a slow or low-recall search.

```python
plan = (
    db.query("production diagnostics")
    .semantic()
    .where({"topic": "optimization"})
    .explain_plan()
)

metrics = await (
    db.query("production diagnostics")
    .semantic()
    .where({"topic": "optimization"})
    .analyze_plan()
)
```

For data science workflows, install the dataframe extras and use tabular outputs.

```python
frame = await db.query("search quality").semantic().limit(20).to_pandas()
arrow_table = await db.query("search quality").semantic().limit(20).to_arrow()
```

## 5. What to learn next

| If you want to... | Read this |
| --- | --- |
| Pick the right search mode | [Search & Retrieval](../user_guide/search_and_retrieval.md) |
| Add structured metadata filters | [Metadata Filtering](../user_guide/filtering.md) |
| Combine text, numbers, categories, and freshness | [Multimodal Search](../user_guide/multimodal_search.md) |
| Evaluate retrieval quality | [Metrics & Evaluation](../user_guide/metrics_and_evaluation.md) |
| Inspect SQL plans and benchmark queries | [Analytics & Diagnostics](../user_guide/analytics_and_diagnostics.md) |
| Tune HNSW, IVFFlat, DiskANN, scalar indexes, and quantization | [Indexing & Performance](../advanced/indexing.md) |

## Fluent API cheat sheet

| Method | Use it for |
| --- | --- |
| `db.query("text")` | Start a fluent query. |
| `.semantic()` | Vector similarity search. |
| `.keyword()` | PostgreSQL FTS or BM25 keyword search. |
| `.hybrid()` | Semantic plus keyword fusion. |
| `.trigram()` | Typo-tolerant fuzzy text search. |
| `.where({...})` | Metadata filtering with JSONB operators. |
| `.limit(n)` / `.offset(n)` | Result size and pagination. |
| `.select([...])` | Return only selected fields. |
| `.ef(n)` / `.nprobes(n)` / `.refine_factor(n)` | Recall and latency tuning. |
| `.rerank(reranker)` | Apply a second-stage reranker. |
| `.explain_plan()` / `.analyze_plan()` | Inspect query planning and runtime metrics. |
| `.to_list()` / `.to_pandas()` / `.to_arrow()` | Execute and choose the result format. |
