# Analytics & Diagnostics

pgVectorDB includes SQL analysis, search benchmarks, recall checks, collection validation, index statistics, and PostgreSQL maintenance helpers. Use this guide when search is slow, recall is low, filtered queries under-return, or production data health is uncertain.

## Fast Query Analysis

Use fluent query analysis while iterating on application queries.

```python
plan = (
    db.query("machine learning")
    .semantic()
    .where({"category": "ai"})
    .explain_plan()
)

metrics = await (
    db.query("machine learning")
    .semantic()
    .where({"category": "ai"})
    .ef(100)
    .limit(10)
    .analyze_plan()
)

print(metrics["execution_time_ms"])
print(metrics["rows_returned"])
print(metrics["search_method"])
```

`explain_plan()` is useful before execution. It returns the query builder's planned search shape, not raw PostgreSQL `EXPLAIN` output.

```python
{
    "search_method": "semantic",
    "query": "machine learning",
    "filter": {"category": "ai"},
    "limit": 2,
    "index_type": "hnsw",
}
```

`analyze_plan()` executes the configured query and returns runtime metrics plus the active query configuration.

```python
{
    "execution_time_ms": 6.86,
    "rows_returned": 2,
    "search_method": "semantic",
    "config": {
        "limit": 2,
        "offset": 0,
        "filter": {"category": "ai"},
        "ef": 100,
        "keyword_type": "bm25",
        "hybrid_mode": "weighted",
        "semantic_weight": 0.5,
        "keyword_weight": 0.5,
        "text_config": "english",
        "bypass_vector_index": False,
    },
}
```

Timings vary by hardware, database cache state, table size, and index state. Use `execution_time_ms` for local comparisons between query settings, not as a universal benchmark.

## Raw PostgreSQL EXPLAIN

Use `explain_query()` when you need raw PostgreSQL `EXPLAIN (ANALYZE, BUFFERS, VERBOSE)` output. For most application code, prefer `db.query(...).explain_plan()` and `await db.query(...).analyze_plan()` because they follow the fluent query you are composing. Use `explain_query()` when you want PostgreSQL's physical plan details.

```python
plan_lines = await db.explain_query(
    query="PostgreSQL search",
    search_method="keyword_search",
    k=2,
)

for line in plan_lines:
    print(line)
```

Supported methods are `semantic_search`, `keyword_search`, and `hybrid_search`.

Actual output from a small local `keyword_search` demo looks like this:

```text
Limit  (cost=1.06..1.07 rows=1 width=84) (actual time=0.119..0.121 rows=1 loops=1)
  Output: langchain_id, content, langchain_metadata, (ts_rank(content_tsvector, '''postgresql'' & ''search'''::tsquery))
  Buffers: shared hit=4
  ->  Sort  (cost=1.06..1.07 rows=1 width=84) (actual time=0.118..0.119 rows=1 loops=1)
      Output: langchain_id, content, langchain_metadata, (ts_rank(content_tsvector, '''postgresql'' & ''search'''::tsquery))
      Sort Key: (ts_rank(docs_explain_actual_demo.content_tsvector, '''postgresql'' & ''search'''::tsquery)) DESC
        Sort Method: quicksort  Memory: 25kB
        Buffers: shared hit=4
      ->  Seq Scan on public.docs_explain_actual_demo  (cost=0.00..1.05 rows=1 width=84) (actual time=0.095..0.097 rows=1 loops=1)
          Output: langchain_id, content, langchain_metadata, ts_rank(content_tsvector, '''postgresql'' & ''search'''::tsquery)
          Filter: (docs_explain_actual_demo.content_tsvector @@ '''postgresql'' & ''search'''::tsquery)
          Rows Removed by Filter: 3
          Buffers: shared hit=1
```

That plan is expected for a tiny demo table. On production-sized tables, use this output to verify whether PostgreSQL selects the intended index path and whether buffers show disk reads.

Look for:

| Signal | Meaning | Action |
| --- | --- | --- |
| Index scan | PostgreSQL is using an index. | Continue tuning query params. |
| Sequential scan | No useful index was selected. | Build/check vector or scalar indexes. |
| High shared reads | Query is reading from disk. | Check cache, index size, and memory. |
| High execution time with low rows | Filter/index mismatch. | Add scalar indexes or iterative scan settings. |

## Collection Statistics

`get_stats()` returns a lightweight overview.

```python
stats = await db.get_stats()

print(stats["table_name"])
print(stats["document_count"])
print(stats["table_size"])
print(stats["index_type"])
print(stats["index_built"])
```

`get_index_stats()` gives deeper table, index, and bloat information.

```python
stats = await db.get_index_stats()

table_stats = stats.get("table_stats", {})
print(table_stats.get("live_tuples"))
print(table_stats.get("dead_tuples"))
print(table_stats.get("bloat_ratio"))

size = stats.get("size", {})
print(size.get("total"))
print(size.get("table"))
print(size.get("indexes"))
```

If the bloat ratio is high after heavy updates or deletes, run maintenance:

```python
await db.vacuum_analyze()
```

## Data Integrity Checks

Use `validate_collection()` after migrations, bulk loads, and incident recovery.

```python
validation = await db.validate_collection()

if validation["healthy"]:
    print("collection is healthy")
else:
    for issue in validation["issues"]:
        print(issue)

print(validation["stats"])
```

It checks for null embeddings, empty content, null IDs, duplicate IDs, and expected embedding dimensions.

## Benchmark Search Methods

`benchmark_search_methods()` compares semantic, keyword, hybrid, and trigram methods on representative queries.

```python
queries = [
    "machine learning",
    "PostgreSQL vector indexing",
    "real-time recommendations",
]

benchmarks = await db.benchmark_search_methods(queries, k=10)

for method, metrics in benchmarks.items():
    print(method)
    print(metrics["avg_time_ms"])
    print(metrics["qps"])
    print(metrics["min_time_ms"], metrics["max_time_ms"])
```

Use benchmarks to decide whether a search mode is viable for production latency before running relevance evaluation.

## Recall Measurement

`compute_recall()` compares approximate vector search against exact search.

```python
recall = await db.compute_recall(
    test_queries=["AI", "neural networks", "vector databases"],
    k=10,
)

print(recall["recall@k"])
print(recall["queries_tested"])
```

Tune query parameters and repeat:

```python
for ef in [40, 80, 120, 200]:
    await db.set_query_params({"hnsw.ef_search": ef})
    recall = await db.compute_recall(test_queries, k=10)
    print(ef, recall["recall@k"])
```

If exact search is strong but ANN recall is weak, increase `.ef(...)`, `.nprobes(...)`, or index build quality. If exact search is also weak, evaluate embeddings, chunking, filters, hybrid search, or reranking.

## Scalar Index Diagnostics

Create scalar indexes for frequent filters, then compare filtered timings.

```python
await db.create_scalar_index("category", index_type="bitmap")
await db.create_scalar_index("price", index_type="btree")

metrics = await (
    db.query("portable computer")
    .semantic()
    .where({"category": "electronics", "price": {"$between": [400, 900]}})
    .ef(100)
    .limit(10)
    .analyze_plan()
)

print(metrics["execution_time_ms"])
```

For raw PostgreSQL plan details, use `explain_query()` or run `EXPLAIN` directly against the generated SQL in a database session.

## BM25 Monitoring

If you use `pg_textsearch`, inspect BM25 index stats.

```python
stats = await db.get_bm25_index_stats()

for index in stats["indexes"]:
    print(index["name"])
    print(index["scans"])
    print(index["tuples_read"])
```

Related BM25 helpers include `dump_bm25_index()`, `spill_bm25_index()`, and `merge_bm25_segments()` for deeper operational debugging.

## Slow Queries

`get_slow_queries()` reads from `pg_stat_statements` when it is enabled.

```python
slow_queries = await db.get_slow_queries(limit=10)

for query in slow_queries:
    print(query["calls"])
    print(query["avg_time_ms"])
    print(query["query"])
```

`pg_stat_statements` must be loaded by PostgreSQL. The project Docker setup includes initialization support for it.

## Iterative Scan for Filtered Recall

ANN indexes can under-return when metadata filters remove many candidates. pgvector iterative scan continues scanning until enough filtered candidates are found.

```python
from pgvectordb import IterativeScanMode

db.set_iterative_scan(
    mode=IterativeScanMode.RELAXED_ORDER,
    max_scan_tuples=50000,
    scan_mem_multiplier=2.0,
)
```

For IVFFlat, set a probe ceiling:

```python
db.set_iterative_scan(
    mode=IterativeScanMode.RELAXED_ORDER,
    max_probes=50,
)
```

Use `STRICT_ORDER` when exact distance ordering matters more than speed. Use `RELAXED_ORDER` for better recall with less latency pressure.

## Centroid Analysis

`compute_centroid()` returns the average embedding for a collection or filtered subset.

```python
all_centroid = await db.compute_centroid()
ai_centroid = await db.compute_centroid(filter={"category": "ai"})
```

Use centroids for exploratory analysis, drift detection, or representative-vector workflows.

## Production Runbook

```python
async def run_diagnostics(db, test_queries):
    stats = await db.get_stats()
    print("collection", stats["table_name"])
    print("documents", stats["document_count"])
    print("index type", stats["index_type"])

    validation = await db.validate_collection()
    print("healthy", validation["healthy"])
    for issue in validation.get("issues", []):
        print("issue", issue)

    index_stats = await db.get_index_stats()
    table_stats = index_stats.get("table_stats", {})
    if table_stats.get("bloat_ratio", 0) > 0.2:
        await db.vacuum_analyze()

    benchmarks = await db.benchmark_search_methods(test_queries, k=10)
    for method, metrics in benchmarks.items():
        print(method, metrics["avg_time_ms"], metrics["qps"])

    recall = await db.compute_recall(test_queries, k=10)
    print("recall@10", recall["recall@k"])

    slow_queries = await db.get_slow_queries(limit=5)
    for query in slow_queries:
        print(query["avg_time_ms"], query["query"])
```

## Triage Guide

| Symptom | First checks | Likely fix |
| --- | --- | --- |
| Search is slow | `analyze_plan()`, `benchmark_search_methods()`, `get_slow_queries()` | Build/tune vector index, reduce K, add scalar indexes, improve cache. |
| Filtered search returns too few results | `compute_recall()`, iterative scan settings | Increase `.ef(...)` / `.nprobes(...)`, enable iterative scan, review filters. |
| Keyword search is weak | BM25 extension/index stats, text config | Build BM25 index, tune `k1`/`b`, use hybrid search. |
| ANN recall is weak | `compute_recall()` vs exact search | Increase query params or rebuild with stronger index settings. |
| Ranking quality is weak | `RAGEvaluator`, MRR, NDCG | Add hybrid search, reranking, or multimodal signals. |
| Storage grows after updates | `get_index_stats()`, bloat ratio | Run `vacuum_analyze()`, review ingestion/upsert strategy. |

## Related Guides

- [Search & Retrieval](search_and_retrieval.md)
- [Metadata Filtering](filtering.md)
- [Metrics & Evaluation](metrics_and_evaluation.md)
- [Indexing & Performance](../advanced/indexing.md)