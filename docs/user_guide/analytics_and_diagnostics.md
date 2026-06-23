# Analytics & Diagnostics

pgVectorDB includes a production-grade diagnostics toolkit via the `AnalyticsMixin` and the new fluent API's query analysis methods.

---

## Query Plan Analysis (New in v0.0.6)

The fluent API provides built-in query plan inspection:

### Explain Plan

Get the query execution plan without executing:

```python
plan = (
    db.search("machine learning")
    .where({"category": "ai"})
    .explain_plan()
)

print(plan)
# {
#   'plan_type': 'Index Scan',
#   'index_name': 'idx_hnsw_cosine',
#   'estimated_cost': 12.34,
#   'estimated_rows': 10,
#   'index_cond': 'embedding <=> $1',
#   'filter': "(langchain_metadata->>'category') = 'ai'",
#   'using_index': True
# }
```

### Analyze Plan

Execute with `EXPLAIN ANALYZE` for real metrics:

```python
metrics = await (
    db.search("machine learning")
    .where({"category": "ai"})
    .limit(10)
    .analyze_plan()
)

print(metrics)
# {
#   'execution_time_ms': 2.34,
#   'planning_time_ms': 0.12,
#   'actual_rows': 10,
#   'actual_loops': 1,
#   'shared_hit_blocks': 45,
#   'shared_read_blocks': 0,
#   'using_index': True,
#   'index_name': 'idx_hnsw_cosine'
# }
```

**What to look for:**

- `using_index: True` — Vector index is being used ✅
- `Seq Scan` — Index not used, check if index was built ⚠️
- `shared_read_blocks` — Disk reads (lower is better)
- `shared_hit_blocks` — Cache hits (higher is better)

---

## Collection Statistics

### `get_stats()`

Lightweight summary of the collection:

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

Deep-dive into index health and table statistics:

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
    If `bloat_ratio` exceeds 20%, run `await db.vacuum_analyze()` to reclaim space.

---

## Data Integrity

### `validate_collection()`

Comprehensive integrity check:

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

| Check | What it detects |
|-------|-----------------|
| Null embeddings | Documents without vectors |
| Empty content | Documents with no text |
| Null IDs | Missing `langchain_id` |
| Duplicate IDs | Data corruption |

---

## Performance Profiling

### Legacy `explain_query()`

For non-fluent API usage:

```python
plan = await db.explain_query(
    query="machine learning",
    k=10,
    search_method="semantic_search"
)

for line in plan:
    print(line)
```

### `benchmark_search_methods()`

Time all search methods:

```python
test_queries = [
    "machine learning",
    "PostgreSQL vector indexing",
    "real-time recommendations",
]

results = await db.benchmark_search_methods(test_queries, k=10)

print(f"\n{'Method':<30} {'Avg (ms)':>10} {'QPS':>8}")
print("-" * 50)
for method, m in results.items():
    print(f"{method:<30} {m['avg_time_ms']:>10.1f} {m['qps']:>8.1f}")
```

Example output:

```
Method                              Avg (ms)      QPS
--------------------------------------------------
semantic_search                         12.3     81.3
keyword_search                           4.1    243.9
hybrid_search                           15.7     63.7
trigram_search                           8.9    112.4
```

---

## Recall Measurement

### `compute_recall()`

Measure ANN index accuracy vs exact search:

```python
recall = await db.compute_recall(
    test_queries=["AI", "neural networks", "vector databases"],
    k=10
)

print(f"Recall@{recall['k']}: {recall['recall@k']:.2%}")
print(f"Queries tested: {recall['queries_tested']}")
```

**Tuning workflow:**

```python
# Test with different ef_search values
test_queries = ["query 1", "query 2", "query 3"]

for ef in [40, 80, 120, 200]:
    await db.set_query_params({"hnsw.ef_search": ef})
    r = await db.compute_recall(test_queries, k=10)
    print(f"ef_search={ef:3d}: Recall@10 = {r['recall@k']:.2%}")
```

---

## Scalar Index Diagnostics

### Verify Scalar Index Usage

Check if scalar indexes are being used for filtered queries:

```python
# Create scalar indexes
await db.create_scalar_index("category", index_type="btree")
await db.create_scalar_index("tags", index_type="bitmap")

# Check if index is used in filtered query
plan = (
    db.search("query")
    .where({"category": "ai"})
    .explain_plan()
)

print(f"Using index: {plan['using_index']}")
print(f"Index name: {plan.get('index_name', 'N/A')}")
```

---

## Centroid Analysis

### `compute_centroid()`

Compute average vector of documents:

```python
# Centroid of all documents
all_centroid = await db.compute_centroid()

# Centroid of a category
ai_centroid = await db.compute_centroid(filter={"category": "ai"})

print(f"Centroid dimensions: {len(ai_centroid)}")
print(f"First 5 values: {ai_centroid[:5]}")
```

---

## BM25 Index Monitoring

### `get_bm25_index_stats()`

Monitor BM25 index usage:

```python
bm25_stats = await db.get_bm25_index_stats()

for idx in bm25_stats["indexes"]:
    print(f"Index: {idx['name']}")
    print(f"  Scans:         {idx['scans']:,}")
    print(f"  Tuples read:   {idx['tuples_read']:,}")
```

---

## Slow Query Monitoring

### `get_slow_queries()`

Pull slow queries from `pg_stat_statements`:

```python
slow_queries = await db.get_slow_queries(limit=10)

for q in slow_queries:
    print(f"Calls:      {q['calls']}")
    print(f"Mean time:  {q['mean_exec_time']:.1f}ms")
    print(f"Query:      {q['query'][:120]}...")
```

!!! note
    Requires `pg_stat_statements` in `postgresql.conf`:
    ```
    shared_preload_libraries = 'pg_stat_statements'
    ```

---

## Iterative Scan Configuration

### `set_iterative_scan()`

Configure iterative scanning for better filtered recall (pgvector 0.8+):

```python
from pgvectordb import IterativeScanMode

# HNSW iterative scan
db.set_iterative_scan(
    mode=IterativeScanMode.RELAXED_ORDER,
    max_scan_tuples=50000,
    scan_mem_multiplier=2.0
)

# IVFFlat iterative scan
db.set_iterative_scan(
    mode=IterativeScanMode.RELAXED_ORDER,
    max_probes=50
)
```

---

## Full Diagnostics Workflow

Production-ready diagnostics runbook:

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
    for issue in validation.get("issues", []):
        print(f"   - {issue}")

    # 3. Check index health
    idx_stats = await db.get_index_stats()
    if "table_stats" in idx_stats:
        bloat = idx_stats["table_stats"]["bloat_ratio"]
        print(f"\n💾 Index Bloat: {bloat:.1%}")
        if bloat > 0.2:
            print("   ⚠️  Run vacuum_analyze()")

    # 4. Benchmark with fluent API
    test_queries = ["machine learning", "database indexing"]
    print("\n⚡ Query Performance:")
    for query in test_queries:
        metrics = await db.search(query).limit(10).analyze_plan()
        print(f"   '{query[:20]}...': {metrics['execution_time_ms']:.2f}ms")

    # 5. Check recall
    recall = await db.compute_recall(test_queries, k=10)
    print(f"\n🎯 ANN Recall@10: {recall['recall@k']:.2%}")
```
