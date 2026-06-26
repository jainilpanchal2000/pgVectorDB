# Migration Guide

This guide maps older method-based pgVectorDB usage to the current fluent query style. Legacy search methods still exist for compatibility, but new docs and examples use `db.query(...)`.

## Quick Mapping

| Legacy method | Fluent API |
| --- | --- |
| `semantic_search(query, k=10)` | `await db.query(query).semantic().limit(10).to_list()` |
| `keyword_search(query, k=10)` | `await db.query(query).keyword().limit(10).to_list()` |
| `keyword_search(..., search_type=KeywordSearchType.BM25)` | `await db.query(query).keyword().bm25().limit(10).to_list()` |
| `hybrid_search(query, k=10)` | `await db.query(query).hybrid().limit(10).to_list()` |
| `hybrid_search(..., use_rrf=True)` | `await db.query(query).hybrid().rrf(k=60).limit(10).to_list()` |
| `metadata_semantic_search(query, filter, k=10)` | `await db.query(query).semantic().where(filter).limit(10).to_list()` |
| `metadata_keyword_search(query, filter, k=10)` | `await db.query(query).keyword().where(filter).limit(10).to_list()` |
| `trigram_search(query, k=10)` | `await db.query(query).trigram().limit(10).to_list()` |
| `semantic_search(..., use_exact_search=True)` | `await db.query(query).semantic().bypass_vector_index().limit(10).to_list()` |

## Semantic Search

Before:

```python
results = await db.semantic_search("machine learning", k=10)
```

After:

```python
results = await db.query("machine learning").semantic().limit(10).to_list()
```

## Keyword and BM25 Search

Before:

```python
results = await db.keyword_search("database optimization", k=10)
```

After:

```python
results = await db.query("database optimization").keyword().limit(10).to_list()
```

For BM25:

```python
results = await (
    db.query("database optimization")
    .keyword()
    .bm25_params(k1=1.2, b=0.75)
    .limit(10)
    .to_list()
)
```

## Filtered Search

Before:

```python
results = await db.metadata_semantic_search(
    query="AI frameworks",
    filter={"category": "ai", "year": {"$gte": 2024}},
    k=10,
)
```

After:

```python
results = await (
    db.query("AI frameworks")
    .semantic()
    .where({"category": "ai", "year": {"$gte": 2024}})
    .limit(10)
    .to_list()
)
```

## Hybrid Search

Before:

```python
results = await db.hybrid_search(
    query="machine learning",
    k=10,
    use_rrf=True,
    rrf_k=60,
)
```

After:

```python
results = await (
    db.query("machine learning")
    .hybrid()
    .rrf(k=60)
    .limit(10)
    .to_list()
)
```

For weighted hybrid search:

```python
results = await (
    db.query("machine learning")
    .hybrid()
    .weights(semantic=0.7, keyword=0.3)
    .limit(10)
    .to_list()
)
```

## Fuzzy Search

Before:

```python
results = await db.trigram_search("vectro databse", k=10)
```

After:

```python
results = await (
    db.query("vectro databse")
    .trigram()
    .threshold(0.2)
    .limit(10)
    .to_list()
)
```

## Query Tuning

Before:

```python
await db.set_query_params({"hnsw.ef_search": 100})
results = await db.semantic_search("query", k=10)
```

After, prefer per-query tuning while experimenting:

```python
results = await (
    db.query("query")
    .semantic()
    .ef(100)
    .limit(10)
    .to_list()
)
```

Keep `set_query_params(...)` for runtime defaults that should apply broadly.

## Query Analysis

Before:

```python
plan_lines = await db.explain_query(
    query="machine learning",
    search_method="semantic_search",
    k=10,
)
```

After, use the fluent analyzer for the query you are composing:

```python
plan = db.query("machine learning").semantic().limit(10).explain_plan()

metrics = await (
    db.query("machine learning")
    .semantic()
    .limit(10)
    .analyze_plan()
)
```

Use `explain_query(...)` when you need raw PostgreSQL `EXPLAIN (ANALYZE, BUFFERS, VERBOSE)` output.

## Output Formats

Legacy methods return lists. Fluent queries let you choose output shape:

```python
rows = await db.query("retrieval quality").semantic().limit(10).to_list()
frame = await db.query("retrieval quality").semantic().limit(10).to_pandas()
arrow_table = await db.query("retrieval quality").semantic().limit(10).to_arrow()
```

## Ingestion and Indexing Names

The current docs use `add_documents(...)` for LangChain `Document` objects and `build_index(...)` for vector index creation.

```python
from langchain_core.documents import Document
from pgvectordb import DistanceMetric

await db.add_documents([
    Document(page_content="Index tuning guide", metadata={"topic": "optimization"})
])

await db.build_index(metric=DistanceMetric.COSINE)
```

## When to Keep Legacy Methods

Keep legacy methods when you already have stable code or need explicit low-level behavior. Prefer the fluent API for new application code because it keeps filters, mode selection, output conversion, analysis, and query tuning in one readable chain.

## Related Guides

- [Quickstart](../getting_started/quickstart.md)
- [Search & Retrieval](search_and_retrieval.md)
- [Metadata Filtering](filtering.md)
- [Analytics & Diagnostics](analytics_and_diagnostics.md)