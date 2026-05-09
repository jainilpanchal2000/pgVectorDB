# Vector Indexing & Performance Tuning

PostgreSQL sequential scans are 100% accurate but too slow for production-scale vector retrieval. pgVectorDB provides a unified API via `create_index()` to manage three state-of-the-art Approximate Nearest Neighbor (ANN) index types, plus production-grade tuning tools.

---

## Index Types Overview

| Index Type | Best For | Memory Model | Requires |
|-----------|---------|--------|----------|
| HNSW | <1M vectors | In-memory graph | `pgvector` |
| IVFFlat | 100K–10M vectors | In-memory clusters | `pgvector` |
| DiskANN | >10M vectors | Disk-spilled graph | `vectorscale` |

!!! note
    All three indexes support all 6 distance metrics: Cosine (`<=>`), L2 (`<->`), Inner Product (`<#>`), L1 (`<+>`), Hamming (`<~>`), Jaccard (`<%>`).

---

## 1. HNSW (Hierarchical Navigable Small World)

`IndexType.HNSW` is the default and recommended index for most use cases under 10 million vectors.

**Characteristics:**

- In-memory index structure
- Extremely fast query times (sub-millisecond at <1M vectors)
- Can be built on an empty table and updates dynamically as documents are added
- No training step required

### Build Parameters

```python
from pgvectordb import IndexType

await db.create_index(
    index_type=IndexType.HNSW,
    m=24,                # Default: 16. Max connections per node (graph width)
    ef_construction=100  # Default: 64. Search quality during build
)
```

| Parameter | Range | Effect |
|-----------|-------|--------|
| `m` | 8–64 | Higher = better recall, more memory per node |
| `ef_construction` | 64–400 | Higher = better recall, slower build time |

### Query-Time Tuning

`ef_search` controls graph traversal depth during search. Increase it for better recall at the cost of latency:

```python
# Set ef_search for current session
await db.set_query_params({"hnsw.ef_search": 100})
```

---

## 2. IVFFlat (Inverted File with Flat Compression)

`IndexType.IVFFLAT` divides the vector space into distinct clusters using K-Means.

**Characteristics:**

- Uses significantly less memory than HNSW
- Builds faster than HNSW
- **⚠️ MUST have data before building** — requires existing vectors to compute cluster centroids

### Build Parameters

```python
await db.create_index(
    index_type=IndexType.IVFFLAT,
    lists=200   # Number of K-Means clusters
)
```

| Data Size | Recommended `lists` |
|----------|---------------------|
| <1M rows | `rows / 1000` |
| >1M rows | `sqrt(rows)` |

### Query-Time Tuning

Control how many clusters to search — more probes = better recall, higher latency:

```python
await db.set_query_params({"ivfflat.probes": 20})
```

---

## 3. DiskANN (via `vectorscale`)

`IndexType.DISKANN` is designed for datasets too large to fit in RAM. It spills the ANN graph to disk while maintaining fast search via SSD-optimized I/O.

**Characteristics:**

- Requires Timescale `vectorscale` extension
- Massive scalability (10M+ vectors without RAM constraint)
- Supports label-based partition filtering inside the graph

```python
await db.create_index(index_type=IndexType.DISKANN)
```

### DiskANN Storage Layouts

```python
from pgvectordb import StorageLayout

await db.create_index(
    index_type=IndexType.DISKANN,
    storage_layout=StorageLayout.MEMORY_OPTIMIZED,  # Uses SBQ compression
)
```

| Layout | Compression | Memory Savings | Use Case |
|--------|-------------|----------------|----------|
| `MEMORY_OPTIMIZED` | Statistical Binary Quantization (SBQ) | ~75% | Large datasets, RAM constrained |
| `PLAIN` | None (full precision) | 0% | Fastest queries, RAM available |

### DiskANN Query-Time Parameters

```python
# Number of candidates to visit during graph search
await db.set_query_params({"diskann.query_search_list_size": 100})

# Number of candidates to rescore with exact distances
await db.set_query_params({"diskann.query_rescore": 50})
```

---

## BM25 Index (Full-Text Search)

For faster BM25 keyword search, build a dedicated BM25 index using the `pg_textsearch` extension:

```python
await db.build_bm25_index(text_config="english")
```

---

## Metadata Indexes

For large collections (10M+), create indexes on frequently filtered metadata fields to speed up the `WHERE` clause that precedes vector search:

```python
# GIN index on multiple metadata fields (equality filters)
await db.create_metadata_index(columns=["tenant_id", "category", "status"])
```

For high-cardinality numeric filters, create a B-Tree index manually:

```sql
-- Faster numeric range queries ($gt, $lt, $between)
CREATE INDEX idx_price ON my_table (((langchain_metadata->>'price')::numeric));
```

---

## Index Management

### Drop a Vector Index

```python
await db.adrop_vector_index()
```

### Rebuild After Heavy Writes

After a large number of inserts/updates/deletes, rebuild the index to restore optimal graph structure:

```python
await db.build_index()
```

### Vacuum & Analyze

After deletes, reclaim dead tuple storage and update query planner statistics:

```python
await db.vacuum_analyze()
```

---

## Production Performance Tuning

### `set_maintenance_work_mem`

Allocate more RAM to index build operations. Higher values allow the HNSW graph to be constructed in memory, dramatically reducing build time for large indexes.

```python
# Set before building a large index
await db.set_maintenance_work_mem("8GB")
await db.create_index(index_type=IndexType.HNSW, m=32, ef_construction=200)
```

!!! warning
    Don't set higher than available server RAM minus what other processes need. A typical safe value is 50–60% of total RAM.

| Dataset Size | Recommended `maintenance_work_mem` |
|-------------|-------------------------------------|
| <1M vectors | 2GB |
| 1M–5M vectors | 4–8GB |
| >5M vectors | 16GB+ |

### `set_parallel_workers`

Configure parallel workers for both queries and index builds:

```python
# 4 workers for parallel sequential scans (use_exact_search=True)
# 7 workers for parallel index build (HNSW, IVFFlat)
await db.set_parallel_workers(gather=4, maintenance=7)
```

| Parameter | Controls | Effect |
|-----------|----------|--------|
| `gather` | `max_parallel_workers_per_gather` | Speeds up exact (sequential scan) search |
| `maintenance` | `max_parallel_maintenance_workers` | Speeds up index build |

### `set_iterative_scan` (pgvector 0.8+)

For filtered searches, iterative scan improves recall by continuing graph traversal until `k` matching documents are found:

```python
from pgvectordb import IterativeScanMode

# Configure — this is a synchronous call, no await needed
db.set_iterative_scan(
    mode=IterativeScanMode.RELAXED_ORDER,
    max_scan_tuples=50000    # HNSW: max nodes to visit
)
```

| Mode | Behavior | Recommended |
|------|----------|-------------|
| `OFF` | Standard index scan | Default |
| `STRICT_ORDER` | Exact distance ordering, slower | When ordering matters |
| `RELAXED_ORDER` | Better recall, slight order variance | **Most filtered searches** |

---

## DiskANN Parallel Build Hints

For very large DiskANN builds, configure parallel build parameters:

```python
await db.set_query_params({
    "diskann.force_parallel_workers": 8,
    "diskann.min_vectors_for_parallel_build": 10000,
})
```

---

## Query Parameter Reference

All valid parameters for `set_query_params()`:

| Parameter | Index | Effect |
|-----------|-------|--------|
| `hnsw.ef_search` | HNSW | Graph traversal depth (recall vs latency) |
| `hnsw.iterative_scan` | HNSW | Enable iterative scan for filtered search |
| `hnsw.max_scan_tuples` | HNSW | Max nodes to visit in iterative scan |
| `hnsw.scan_mem_multiplier` | HNSW | Memory multiplier for iterative scan |
| `ivfflat.probes` | IVFFlat | Clusters to search (recall vs latency) |
| `ivfflat.iterative_scan` | IVFFlat | Enable iterative scan |
| `ivfflat.max_probes` | IVFFlat | Max probes in iterative scan |
| `diskann.query_search_list_size` | DiskANN | Candidates to visit |
| `diskann.query_rescore` | DiskANN | Candidates to rescore with exact distances |
| `diskann.force_parallel_workers` | DiskANN | Parallel build workers |
| `diskann.min_vectors_for_parallel_build` | DiskANN | Minimum vectors before parallelizing |
