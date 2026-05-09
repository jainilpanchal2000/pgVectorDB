# Analytics & Diagnostics

pgVectorDB includes a production-grade diagnostics toolkit via the `AnalyticsMixin`. These methods let you monitor index health, profile query performance, validate data integrity, and benchmark search configurations — all without leaving Python.

---

## Collection Statistics

### `get_stats()`

A lightweight summary of the collection — document count, table size, and current indexes:

```python
stats = await db.get_stats()

print(f"Index type:     {stats['index_type']}")
print(f"Documents:      {stats['document_count']:,}")
print(f"Table size:     {stats['table_size']}")
print(f"Index built:    {stats['index_built']}")

for idx in stats["indexes"]:
    print(f"  Index: {idx['name']}")
```

### `get_index_stats()`

Deep-dive into index health, PostgreSQL table statistics, and size breakdown:

```python
stats = await db.get_index_stats()

# Table operation counts
ts = stats["table_stats"]
print(f"Live tuples:    {ts['live_tuples']:,}")
print(f"Dead tuples:    {ts['dead_tuples']:,}")
print(f"Bloat ratio:    {ts['bloat_ratio']:.1%}")
print(f"Last vacuum:    {ts['last_vacuum']}")
print(f"Last analyze:   {ts['last_analyze']}")

# Size breakdown
sz = stats["size"]
print(f"Total size:     {sz['total']}")
print(f"Table size:     {sz['table']}")
print(f"Indexes size:   {sz['indexes']}")
```

!!! tip "When to vacuum"
    If `bloat_ratio` exceeds 20%, run `await db.vacuum_analyze()` to reclaim dead tuple space and refresh planner statistics.

---

## Data Integrity

### `validate_collection()`

Comprehensive integrity check — catches common data quality issues before they affect search quality:

```python
validation = await db.validate_collection()

if validation["healthy"]:
    print("✓ Collection is healthy")
else:
    print(f"⚠ Found {validation['issues_found']} issue(s):")
    for issue in validation["issues"]:
        print(f"  - {issue}")

# Stats regardless of health
s = validation["stats"]
print(f"\nTotal documents:  {s['total_documents']:,}")
print(f"Null embeddings:  {s['null_embeddings']}")
print(f"Empty content:    {s['empty_content']}")
print(f"Duplicate IDs:    {s['duplicate_ids']}")
```

Validates:

| Check | What it detects |
|-------|----------------|
| Null embeddings | Documents inserted without a vector |
| Empty content | Documents with no text |
| Null IDs | Documents missing `langchain_id` |
| Duplicate IDs | Data corruption / failed upserts |

---

## Performance Profiling

### `explain_query()`

Run `EXPLAIN ANALYZE BUFFERS` on any search method to see the PostgreSQL query plan:

```python
plan = await db.explain_query(
    query="machine learning for NLP",
    search_method="semantic_search",  # or "keyword_search", "hybrid_search"
    k=10
)

for line in plan:
    print(line)
```

**What to look for:**

- `Index Scan using hnsw_...` — good, the vector index is being used
- `Seq Scan` — the index is not being used (check if it was built)
- `Buffers: shared hit=...` — how many pages were served from cache vs. disk

### `benchmark_search_methods()`

Time all 4 main search methods on the same set of test queries and get QPS metrics:

```python
test_queries = [
    "machine learning for NLP",
    "PostgreSQL vector indexing",
    "real-time recommendation systems",
    "transformer models attention mechanism",
    "distributed database sharding strategies",
]

results = await db.benchmark_search_methods(test_queries, k=10)

print(f"\n{'Method':<30} {'Avg (ms)':>10} {'QPS':>8} {'Min (ms)':>10} {'Max (ms)':>10}")
print("-" * 70)
for method, metrics in results.items():
    print(
        f"{method:<30} "
        f"{metrics['avg_time_ms']:>10.1f} "
        f"{metrics['qps']:>8.1f} "
        f"{metrics['min_time_ms']:>10.1f} "
        f"{metrics['max_time_ms']:>10.1f}"
    )
```

Example output:

```
Method                         Avg (ms)      QPS   Min (ms)   Max (ms)
----------------------------------------------------------------------
semantic_search                    12.3     81.3        9.1       18.7
keyword_search                      4.1    243.9        3.2        6.8
hybrid_search                      15.7     63.7       11.4       24.2
trigram_search                      8.9    112.4        7.1       12.3
```

---

## Recall Measurement

### `compute_recall()`

Measure how accurately the ANN index approximates exact search results — the key metric for tuning `ef_search` (HNSW) or `probes` (IVFFlat):

```python
recall = await db.compute_recall(
    test_queries=["AI applications", "neural network training", "vector databases"],
    k=10,
    sample_size=100   # Limit queries if dataset is large
)

print(f"Recall@{recall['k']}: {recall['recall@k']:.2%}")
print(f"Queries tested:  {recall['queries_tested']}")
```

**Workflow for tuning `ef_search`:**

```python
# Baseline
await db.set_query_params({"hnsw.ef_search": 40})
r1 = await db.compute_recall(test_queries, k=10)

# Higher ef_search
await db.set_query_params({"hnsw.ef_search": 100})
r2 = await db.compute_recall(test_queries, k=10)

print(f"ef_search=40:  Recall@10 = {r1['recall@k']:.2%}")
print(f"ef_search=100: Recall@10 = {r2['recall@k']:.2%}")
```

---

## Centroid Analysis

### `compute_centroid()`

Compute the average (centroid) vector of all documents, or a filtered subset. Useful for cluster analysis, topic modeling, and building representative query vectors.

```python
# Centroid of all documents
all_centroid = await db.compute_centroid()

# Centroid of a specific category
ai_centroid = await db.compute_centroid(filter={"category": "ai"})

print(f"Centroid dimensions: {len(ai_centroid)}")
print(f"First 5 values: {ai_centroid[:5]}")
```

---

## BM25 Index Monitoring

### `get_bm25_index_stats()`

Monitor BM25 index usage patterns from `pg_stat_user_indexes`:

```python
bm25_stats = await db.get_bm25_index_stats()

for idx in bm25_stats["indexes"]:
    print(f"Index: {idx['name']}")
    print(f"  Scans:         {idx['scans']:,}")
    print(f"  Tuples read:   {idx['tuples_read']:,}")
    print(f"  Tuples fetched:{idx['tuples_fetched']:,}")
```

---

## Slow Query Monitoring

### `get_slow_queries()`

Pull the top slow queries from `pg_stat_statements` (requires the `pg_stat_statements` extension):

```python
slow_queries = await db.get_slow_queries(limit=10)

for q in slow_queries:
    print(f"Calls:      {q['calls']}")
    print(f"Mean time:  {q['mean_exec_time']:.1f}ms")
    print(f"Query:      {q['query'][:120]}...")
    print()
```

!!! note
    `pg_stat_statements` must be enabled in `postgresql.conf`:
    ```
    shared_preload_libraries = 'pg_stat_statements'
    ```

---

## Iterative Scan Configuration

### `set_iterative_scan()`

Configure iterative scanning for better recall on filtered queries (pgvector 0.8+). This is a **synchronous** call — no `await`:

```python
from pgvectordb import IterativeScanMode

# Enable relaxed-order iterative scan for HNSW
db.set_iterative_scan(
    mode=IterativeScanMode.RELAXED_ORDER,
    max_scan_tuples=50000,        # Max nodes to visit
    scan_mem_multiplier=2.0       # Memory budget multiplier
)

# For IVFFlat, configure max probes
db.set_iterative_scan(
    mode=IterativeScanMode.RELAXED_ORDER,
    max_probes=50
)
```

---

## Full Diagnostics Workflow

Here's a production-ready diagnostics runbook combining all tools:

```python
async def run_diagnostics(db):
    print("=" * 60)
    print("pgVectorDB Diagnostics Report")
    print("=" * 60)

    # 1. Basic stats
    stats = await db.get_stats()
    print(f"\n📊 Collection: {stats['table_name']}")
    print(f"   Documents:  {stats['document_count']:,}")
    print(f"   Size:       {stats['table_size']}")
    print(f"   Index type: {stats['index_type']}")

    # 2. Validate integrity
    validation = await db.validate_collection()
    status = "✅ Healthy" if validation["healthy"] else f"⚠️ {validation['issues_found']} issues"
    print(f"\n🔍 Data Integrity: {status}")
    for issue in validation["issues"]:
        print(f"   - {issue}")

    # 3. Benchmark performance
    test_queries = ["test query 1", "test query 2", "test query 3"]
    bench = await db.benchmark_search_methods(test_queries, k=10)
    print("\n⚡ Search Performance:")
    for method, m in bench.items():
        print(f"   {method:<30} {m['avg_time_ms']:.1f}ms avg, {m['qps']:.0f} QPS")

    # 4. Check recall
    recall = await db.compute_recall(test_queries, k=10)
    print(f"\n🎯 ANN Recall@10: {recall['recall@k']:.2%}")

    # 5. Index bloat
    idx_stats = await db.get_index_stats()
    if "table_stats" in idx_stats:
        bloat = idx_stats["table_stats"]["bloat_ratio"]
        if bloat > 0.2:
            print(f"\n⚠️  High bloat ({bloat:.1%}) — consider running vacuum_analyze()")
        else:
            print(f"\n✅ Bloat ratio OK ({bloat:.1%})")
```
