# Vector Indexing & Performance Tuning

PostgreSQL sequential scans are accurate but slow for production-scale vector retrieval. pgVectorDB provides unified APIs for vector indexes (HNSW, IVFFlat, DiskANN) and scalar indexes (BTree, GIN) for metadata filtering.

---

## Index Types Overview

| Index Type | Best For | Memory Model | Requires |
|-----------|---------|--------------|----------|
| **HNSW** | <1M vectors | In-memory graph | pgvector |
| **IVFFlat** | 100K–10M vectors | In-memory clusters | pgvector |
| **DiskANN** | >10M vectors | Disk-spilled graph | vectorscale |
| **BTree (scalar)** | Range filters | Standard | pgvector |
| **GIN (scalar)** | Low-cardinality equality | Standard | pgvector |

!!! note
    All distance metrics supported: Cosine (`<=>`), L2 (`<->`), Inner Product (`<#>`), L1 (`<+>`), Hamming (`<~>`), Jaccard (`<%>`).

---

## Vector Indexes

### HNSW (Hierarchical Navigable Small World)

Default and recommended for <10M vectors.

**Characteristics:**
- In-memory graph structure
- Sub-millisecond queries at <1M vectors
- Builds on empty tables, updates dynamically
- No training step

```python
from pgvectordb import IndexType

await db.create_index(
    index_type=IndexType.HNSW,
    m=24,                # Default: 16. Graph width
    ef_construction=100  # Default: 64. Search quality during build
)
```

| Parameter | Range | Effect |
|-----------|-------|--------|
| `m` | 8–64 | Higher = better recall, more memory |
| `ef_construction` | 64–400 | Higher = better recall, slower build |

**Query-time tuning:**

```python
# Via fluent API
results = await (
    db.search("query")
    .ef(100)  # Uses SET LOCAL hnsw.ef_search = 100
    .limit(10)
    .to_list()
)

# Or set globally
await db.set_query_params({"hnsw.ef_search": 100})
```

---

### IVFFlat (Inverted File)

For 100K–10M vectors. Uses less memory than HNSW.

**⚠️ Requires existing data** to compute centroids.

```python
await db.create_index(
    index_type=IndexType.IVFFLAT,
    lists=200   # K-Means clusters
)
```

| Data Size | Recommended `lists` |
|-----------|---------------------|
| <1M rows | `rows / 1000` |
| >1M rows | `sqrt(rows)` |

**Query-time tuning:**

```python
# Via fluent API
results = await (
    db.search("query")
    .nprobes(20)  # More partitions = better recall
    .limit(10)
    .to_list()
)

# Or set globally
await db.set_query_params({"ivfflat.probes": 20})
```

---

### DiskANN (via vectorscale)

For datasets too large for RAM (10M+ vectors).

```python
from pgvectordb import StorageLayout

await db.create_index(
    index_type=IndexType.DISKANN,
    storage_layout=StorageLayout.MEMORY_OPTIMIZED
)
```

| Layout | Compression | Memory Savings |
|--------|-------------|----------------|
| `MEMORY_OPTIMIZED` | Statistical Binary Quantization | ~75% |
| `PLAIN` | None | 0% (fastest) |

**Query-time tuning:**

```python
await db.set_query_params({
    "diskann.query_search_list_size": 100,
    "diskann.query_rescore": 50,
})
```

---

## Scalar Indexes for Metadata (New in v0.0.6)

Create indexes on frequently filtered metadata fields for 10-100x faster filtered queries.

### B-Tree Index

For range queries (`$gt`, `$lt`, `$between`, `$eq`):

```python
# Create B-Tree index on numeric field
await db.create_scalar_index("price", index_type="btree")

# Multiple fields
await db.create_scalar_index(["price", "year", "rating"], index_type="btree")
```

**Best for:**
- Numeric comparisons: `{"price": {"$lt": 100}}`
- Range queries: `{"year": {"$between": [2020, 2024]}}`
- Equality on high-cardinality fields: `{"product_id": "abc123"}`

### GIN Index ("Bitmap")

For low-cardinality equality (`$eq`, `$in`):

```python
# For text/array fields
await db.create_scalar_index("category", index_type="bitmap")  # Creates GIN
await db.create_scalar_index("tags", index_type="bitmap")
```

**Best for:**
- Low-cardinality categories: `{"category": "ai"}`
- Array containment
- Text matching with `$like`

!!! note "Bitmap vs GIN"
    PostgreSQL doesn't have native bitmap indexes. We're using GIN indexes which excel for low-cardinality fields and array operations — similar use case to what bitmap indexes serve in other databases.

### How Scalar Indexes Work

**Auto-detection:** pgVectorDB automatically detects metadata types and creates appropriate expressions:

```python
# For numeric fields
((langchain_metadata->>'price')::numeric)

# For boolean fields  
((langchain_metadata->>'is_active')::boolean)

# For text fields
(langchain_metadata->>'category')
```

### When to Create Scalar Indexes

Create indexes when:

1. **You have 100K+ documents** — Overhead starts to pay off
2. **You filter by the field** — 10-100x faster filtered queries
3. **The field has good selectivity**
   - BTree: Good for range queries, high-cardinality equality
   - GIN: Good for low-cardinality equality

### Verify Index Usage

Check if indexes are being used:

```python
# Check query plan
plan = (
    db.search("query")
    .where({"price": {"$lt": 100}})
    .explain_plan()
)

print(f"Using index: {plan['using_index']}")
print(f"Index name: {plan.get('index_name', 'Sequential Scan')}")

# Analyze actual execution
metrics = await (
    db.search("query")
    .where({"category": "ai"})
    .analyze_plan()
)

print(f"Execution time: {metrics['execution_time_ms']:.2f}ms")
print(f"Using index: {metrics['using_index']}")
```

### Drop Scalar Indexes

```python
# Drop specific index
await db.drop_scalar_index("idx_price_btree")

# Drop all scalar indexes for a column
await db.drop_scalar_index(column="price")
```

---

## BM25 Full-Text Index

For faster keyword search with the `pg_textsearch` extension:

```python
await db.build_bm25_index(text_config="english", k1=1.2, b=0.75)
```

---

## Index Management

### Drop Vector Index

```python
await db.adrop_vector_index()
```

### Rebuild Index

After heavy writes:

```python
await db.build_index()
```

### Vacuum & Analyze

Reclaim dead tuples and update statistics:

```python
await db.vacuum_analyze()
```

---

## Production Tuning

### Memory for Index Build

```python
# Before building large indexes
await db.set_maintenance_work_mem("8GB")
await db.create_index(index_type=IndexType.HNSW, m=32, ef_construction=200)
```

| Dataset Size | Recommended |
|-------------|-------------|
| <1M vectors | 2GB |
| 1M–5M vectors | 4–8GB |
| >5M vectors | 16GB+ |

### Parallel Workers

```python
await db.set_parallel_workers(gather=4, maintenance=7)
```

| Parameter | Controls |
|-----------|----------|
| `gather` | Parallel sequential scans |
| `maintenance` | Parallel index build |

### Iterative Scan (pgvector 0.8+)

Better recall for filtered searches:

```python
from pgvectordb import IterativeScanMode

db.set_iterative_scan(
    mode=IterativeScanMode.RELAXED_ORDER,
    max_scan_tuples=50000
)
```

| Mode | Behavior |
|------|----------|
| `OFF` | Standard scan |
| `STRICT_ORDER` | Exact ordering, slower |
| `RELAXED_ORDER` | Better recall, recommended |

---

## Complete Example

```python
import asyncio
from pgvectordb import pgVectorDB, IndexType
from langchain_huggingface import HuggingFaceEmbeddings

async def main():
    db = pgVectorDB(
        collection_name="products",
        embedding_model=HuggingFaceEmbeddings(),
        connection_string="postgresql+asyncpg://..."
    )
    await db.initialize()
    
    # Add documents with varied metadata
    await db.add_texts(
        texts=["Laptop", "Phone", "Tablet", "Watch"],
        metadatas=[
            {"category": "electronics", "price": 999, "rating": 4.5},
            {"category": "electronics", "price": 699, "rating": 4.2},
            {"category": "electronics", "price": 499, "rating": 4.0},
            {"category": "wearables", "price": 299, "rating": 4.3},
        ]
    )
    
    # Build vector index
    await db.create_index(index_type=IndexType.HNSW)
    
    # Create scalar indexes for filtering
    await db.create_scalar_index("price", index_type="btree")
    await db.create_scalar_index("category", index_type="bitmap")
    
    # Filtered search with tuned parameters
    results = await (
        db.search("portable computer")
        .where({
            "$and": [
                {"category": "electronics"},
                {"price": {"$between": [400, 800]}},
                {"rating": {"$gte": 4.0}}
            ]
        })
        .ef(100)  # Higher recall for filtered search
        .limit(5)
        .to_list()
    )
    
    # Verify index usage
    plan = db.search("test").where({"price": {"$lt": 500}}).explain_plan()
    print(f"Index used: {plan['using_index']}")
    
    await db.close()

if __name__ == "__main__":
    asyncio.run(main())
```

---

## Query Parameter Reference

| Parameter | Index | Effect |
|-----------|-------|--------|
| `hnsw.ef_search` | HNSW | Graph traversal depth |
| `ivfflat.probes` | IVFFlat | Clusters to search |
| `diskann.query_search_list_size` | DiskANN | Candidates to visit |
| `diskann.query_rescore` | DiskANN | Exact rescoring |

Set via fluent API:

```python
results = await (
    db.search("query")
    .ef(100)
    .nprobes(20)
    .refine_factor(2)
    .to_list()
)
```
