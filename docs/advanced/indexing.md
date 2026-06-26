# Indexing & Performance

pgVectorDB gives you PostgreSQL-native controls for recall, latency, storage, and filtered search performance. The main workflow is: choose an index family, build it with `build_index()`, add scalar indexes for common filters, measure recall and latency, then tune query parameters.

## Choose an Index

| Index | Best for | Requires | Query-time tuning |
| --- | --- | --- | --- |
| HNSW | Default choice for strong recall and dynamic updates. | `pgvector` | `.ef(n)` / `hnsw.ef_search` |
| IVFFlat | Large collections with predictable memory use. | `pgvector`; data must exist before build | `.nprobes(n)` / `ivfflat.probes` |
| DiskANN | Very large datasets and memory-sensitive workloads. | `vectorscale` | `diskann.query_search_list_size`, rescore settings |
| Scalar B-tree | Numeric ranges and high-cardinality equality. | PostgreSQL | Metadata filters via `.where(...)` |
| Scalar bitmap/GIN | Low-cardinality equality and membership-style filters. | PostgreSQL | Metadata filters via `.where(...)` |

Configure the vector index family when creating the database instance:

```python
from pgvectordb import IndexType, pgVectorDB

db = pgVectorDB(
    collection_name="products",
    embedding_model=embeddings,
    connection_string="postgresql+asyncpg://user:root@localhost:9002/postgres",
    index_type=IndexType.HNSW,
)

await db.initialize()
```

## Build the Vector Index

Use `build_index()` for HNSW, IVFFlat, and DiskANN. The active `index_type` controls which index builder is used.

```python
from pgvectordb import DistanceMetric

await db.build_index(
    metric=DistanceMetric.COSINE,
    m=16,
    ef_construction=64,
)
```

You can also pass typed options for longer configurations:

```python
from pgvectordb import IndexBuildOptions

await db.build_index(options=IndexBuildOptions(metric=DistanceMetric.COSINE, m=24))
```

## HNSW

HNSW is the best default for most RAG workloads. It builds a graph over vectors and searches by graph traversal.

```python
await db.build_index(
    metric=DistanceMetric.COSINE,
    m=24,
    ef_construction=100,
)
```

| Parameter | Effect | Starting point |
| --- | --- | --- |
| `m` | Graph connectivity. Higher improves recall and memory use. | `16` to `24` |
| `ef_construction` | Build-time candidate pool. Higher improves recall and build time. | `64` to `200` |
| `.ef(n)` | Query-time candidate pool. Higher improves recall and latency. | `80` to `200` |

```python
results = await (
    db.query("portable computer")
    .semantic()
    .ef(120)
    .limit(10)
    .to_list()
)
```

## IVFFlat

IVFFlat clusters vectors into lists. It uses less memory than HNSW but must be trained on existing data.

```python
await db.build_index(
    metric=DistanceMetric.COSINE,
    lists=200,
)
```

| Collection size | Starting `lists` |
| --- | --- |
| 100K rows | 100 |
| 1M rows | 200 to 1,000 |
| 10M rows | 1,000 to 3,000 |

Increase probes to search more clusters:

```python
results = await (
    db.query("database optimization")
    .semantic()
    .nprobes(20)
    .limit(10)
    .to_list()
)
```

## DiskANN

DiskANN is for very large datasets where memory is the bottleneck. It requires the `vectorscale` extension.

```python
from pgvectordb import StorageLayout

await db.build_index(
    metric=DistanceMetric.COSINE,
    num_neighbors=50,
    search_list_size=100,
    max_alpha=1.2,
    storage_layout=StorageLayout.MEMORY_OPTIMIZED,
    include_labels=True,
)
```

| Setting | Effect |
| --- | --- |
| `num_neighbors` | Graph degree. Higher can improve recall with more storage. |
| `search_list_size` | Build candidate list. Higher improves quality and build time. |
| `storage_layout` | `MEMORY_OPTIMIZED` uses compression; `PLAIN` favors speed. |
| `include_labels` | Adds label-aware filtering for DiskANN workloads. |

Tune DiskANN query settings with `set_query_params()`:

```python
await db.set_query_params({
    "diskann.query_search_list_size": 100,
    "diskann.query_rescore": 50,
})
```

## Scalar Indexes for Filters

Scalar indexes are essential when vector search is combined with frequent metadata filters.

```python
await db.create_scalar_index("price", index_type="btree")
await db.create_scalar_index("category", index_type="bitmap")
await db.create_scalar_index("tags", index_type="gin")
```

| Index type | Best for | Example filter |
| --- | --- | --- |
| `btree` | Numeric ranges and high-cardinality equality. | `{"price": {"$between": [100, 500]}}` |
| `bitmap` | Low-cardinality categorical filters. | `{"category": "electronics"}` |
| `gin` | Array or membership-style filters. | `{"tags": {"$in": ["sale", "new"]}}` |
| `labellist` | DiskANN label arrays. | Label-filtered ANN traversal. |

Create one scalar index per field. If you filter by `price`, `year`, and `rating`, create each separately:

```python
await db.create_scalar_index("price", index_type="btree")
await db.create_scalar_index("year", index_type="btree")
await db.create_scalar_index("rating", index_type="btree")
```

## BM25 Keyword Index

When `pg_textsearch` is available, build a BM25 index for stronger keyword ranking.

```python
await db.build_bm25_index(text_config="english", k1=1.2, b=0.75)
```

Use BM25 through the fluent API:

```python
results = await (
    db.query("PostgreSQL vector indexing")
    .keyword()
    .bm25_params(k1=1.2, b=0.75)
    .limit(10)
    .to_list()
)
```

## Query-Time Tuning

Prefer per-query fluent settings while experimenting:

```python
results = await (
    db.query("filtered vector search")
    .semantic()
    .where({"topic": "optimization"})
    .ef(100)
    .refine_factor(2)
    .limit(10)
    .to_list()
)
```

Use global query params for broader runtime defaults:

```python
await db.set_query_params({
    "hnsw.ef_search": 100,
    "ivfflat.probes": 20,
})
```

Use exact search as ground truth:

```python
exact = await db.query("index tuning").semantic().bypass_vector_index().limit(10).to_list()
```

## Iterative Scan for Filtered Recall

Filtered ANN search may return fewer than `k` results if the graph finds candidates that the metadata filter removes. Iterative scan lets pgvector continue scanning for more matching rows.

```python
from pgvectordb import IterativeScanMode

db.set_iterative_scan(
    mode=IterativeScanMode.RELAXED_ORDER,
    max_scan_tuples=50000,
    scan_mem_multiplier=2.0,
)
```

Use `STRICT_ORDER` if exact distance ordering matters more than latency.

## Quantization and Storage Optimizations

pgVectorDB includes storage and search patterns for large vector workloads.

| Technique | Best for | Tradeoff |
| --- | --- | --- |
| Half precision / `halfvec` | Reducing vector storage roughly in half. | Small accuracy changes; validate recall. |
| Binary quantization | Very compact candidate search. | Usually needs full-vector reranking. |
| Sparse vectors | High-dimensional sparse features. | Requires sparse-compatible data and distances. |
| Subvector / Matryoshka indexing | Fast first-stage search with full-vector rerank. | Works best with embeddings trained for prefix dimensions. |

Subvector indexing is useful for Matryoshka-style embeddings:

```python
await db.build_index_with_subvectors(
    subvector_dims=256,
    metric=DistanceMetric.COSINE,
)

results = await db.search_with_subvector_rerank(
    query="high recall search",
    k=10,
    rerank_top=100,
)
```

Binary quantized search uses a compact first-stage candidate set and reranks with full vectors:

```python
await db.build_index_binary_quantized(metric=DistanceMetric.HAMMING)
```

Always validate these optimizations with [Metrics & Evaluation](../user_guide/metrics_and_evaluation.md) and `compute_recall()`.

## Build and Maintenance Settings

Large index builds benefit from more PostgreSQL maintenance memory and parallel workers.

```python
await db.set_maintenance_work_mem("8GB")
await db.set_parallel_workers(gather=4, maintenance=7)
await db.build_index(metric=DistanceMetric.COSINE, m=32, ef_construction=200)
```

After heavy writes, updates, or deletes:

```python
await db.vacuum_analyze()
```

## Measure Before and After

Use diagnostics after every meaningful index or query-parameter change.

```python
metrics = await (
    db.query("portable computer")
    .semantic()
    .where({"category": "electronics", "price": {"$between": [400, 900]}})
    .ef(100)
    .limit(10)
    .analyze_plan()
)

recall = await db.compute_recall(
    test_queries=["portable computer", "budget laptop", "workstation"],
    k=10,
)

print(metrics["execution_time_ms"])
print(recall["recall@k"])
```

## End-to-End Optimization Example

```python
import asyncio

from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

from pgvectordb import DistanceMetric, IndexType, IterativeScanMode, pgVectorDB


async def main():
    db = pgVectorDB(
        collection_name="products",
        embedding_model=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2"),
        connection_string="postgresql+asyncpg://user:root@localhost:9002/postgres",
        index_type=IndexType.HNSW,
    )
    await db.initialize()

    await db.add_documents([
        Document(page_content="Lightweight laptop for travel", metadata={"category": "electronics", "price": 899, "rating": 4.5}),
        Document(page_content="Noise cancelling headphones", metadata={"category": "electronics", "price": 199, "rating": 4.7}),
        Document(page_content="Ergonomic office chair", metadata={"category": "furniture", "price": 349, "rating": 4.4}),
    ])

    await db.set_maintenance_work_mem("2GB")
    await db.build_index(metric=DistanceMetric.COSINE, m=24, ef_construction=100)
    await db.create_scalar_index("category", index_type="bitmap")
    await db.create_scalar_index("price", index_type="btree")

    db.set_iterative_scan(mode=IterativeScanMode.RELAXED_ORDER, max_scan_tuples=50000)

    results = await (
        db.query("portable computer")
        .semantic()
        .where({"category": "electronics", "price": {"$between": [400, 1000]}})
        .ef(120)
        .limit(5)
        .to_list()
    )

    for row in results:
        print(row["score"], row["content"])

    await db.close()


asyncio.run(main())
```

## Related Guides

- [Search & Retrieval](../user_guide/search_and_retrieval.md)
- [Metadata Filtering](../user_guide/filtering.md)
- [Analytics & Diagnostics](../user_guide/analytics_and_diagnostics.md)
- [Metrics & Evaluation](../user_guide/metrics_and_evaluation.md)