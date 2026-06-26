# Search & Retrieval

pgVectorDB’s search API is centered on `db.query(...)`. Start with a text query, choose a mode, add filters and tuning, then execute.

```python
results = await (
    db.query("how to tune filtered vector search")
    .hybrid()
    .where({"topic": "optimization"})
    .weights(semantic=0.7, keyword=0.3)
    .limit(10)
    .to_list()
)
```

## Fluent Query Pattern

The builder is lazy. Chained methods only update configuration; execution happens when you call an output or analysis method.

| Stage | Methods |
| --- | --- |
| Choose retrieval mode | `.semantic()`, `.keyword()`, `.hybrid()`, `.trigram()`, `.search_mode(...)` |
| Add constraints | `.where(...)`, `.select(...)`, `.limit(...)`, `.offset(...)` |
| Tune search | `.ef(...)`, `.nprobes(...)`, `.refine_factor(...)`, `.distance_range(...)`, `.bypass_vector_index()` |
| Tune keyword/hybrid | `.bm25()`, `.fts(...)`, `.bm25_params(...)`, `.weights(...)`, `.rrf(...)`, `.phrase(...)`, `.universal(...)` |
| Improve ranking | `.rerank(...)` |
| Execute | `.to_list()`, `.to_pandas()`, `.to_arrow()`, `.analyze_plan()` |

## Semantic Search

Use semantic search when meaning matters more than exact wording.

```python
results = await (
    db.query("customer cannot access account")
    .semantic()
    .limit(5)
    .to_list()
)
```

Add vector-index tuning for recall-sensitive workloads:

```python
results = await (
    db.query("database backup retention policy")
    .semantic()
    .ef(100)
    .refine_factor(2)
    .limit(10)
    .to_list()
)
```

| Parameter | Index family | Effect |
| --- | --- | --- |
| `.ef(n)` | HNSW | Visits more graph candidates. Higher recall, higher latency. |
| `.nprobes(n)` | IVFFlat | Searches more clusters. Higher recall, higher latency. |
| `.refine_factor(n)` | ANN refinement | Fetches more candidates before returning final top-k. |
| `.bypass_vector_index()` | Any | Exact search for ground truth and recall measurement. |
| `.distance_range(low, high)` | Vector search | Keeps only results in a distance window. |

## Keyword Search

Use keyword search when exact terms, names, codes, and domain vocabulary matter.

```python
results = await (
    db.query("SOC 2 encryption retention")
    .keyword()
    .fts(text_config="english")
    .limit(10)
    .to_list()
)
```

Use BM25 when `pg_textsearch` is available and you want stronger keyword ranking.

```python
results = await (
    db.query("database optimization guide")
    .keyword()
    .bm25_params(k1=1.2, b=0.75)
    .limit(10)
    .to_list()
)
```

| BM25 setting | Increase when... | Decrease when... |
| --- | --- | --- |
| `k1` | Repeated terms should matter more. | You want less term-frequency saturation. |
| `b` | You want stronger document-length normalization. | Short and long documents should be treated similarly. |

Phrase and universal keyword search are useful for stricter matching or metadata-aware keyword search.

```python
phrase_results = await db.query("zero downtime index").keyword().phrase().limit(5).to_list()

universal_results = await (
    db.query("premium support")
    .keyword()
    .universal(metadata_fields=["title", "tags", "plan"])
    .limit(5)
    .to_list()
)
```

## Hybrid Search

Hybrid search combines vector similarity and keyword ranking. It is the default choice for RAG systems that need both semantic recall and exact term precision.

```python
results = await (
    db.query("HNSW filtered recall")
    .hybrid()
    .weights(semantic=0.75, keyword=0.25)
    .where({"topic": "optimization"})
    .limit(10)
    .to_list()
)
```

Use Reciprocal Rank Fusion when the score scales between semantic and keyword search are difficult to compare.

```python
results = await (
    db.query("PostgreSQL vector index tuning")
    .hybrid()
    .rrf(k=60)
    .limit(10)
    .to_list()
)
```

| Fusion | Use when | Tradeoff |
| --- | --- | --- |
| `.weights(semantic, keyword)` | You know which signal should dominate. | Needs tuning and evaluation. |
| `.rrf(k=60)` | You want robust rank fusion without score calibration. | Less direct control over signal strength. |

## Trigram Fuzzy Search

Trigram search uses PostgreSQL `pg_trgm` for typo-tolerant text matching.

```python
results = await (
    db.query("vectro databse")
    .trigram()
    .threshold(0.2)
    .limit(10)
    .to_list()
)
```

Use it for misspellings, product names, titles, people names, and query autosuggest-style workflows. Lower thresholds admit fuzzier matches; higher thresholds improve precision.

## Metadata Filters

Filters apply to every search mode through `.where(...)`.

```python
results = await (
    db.query("wireless headphones")
    .hybrid()
    .where({
        "category": "electronics",
        "price": {"$between": [50, 200]},
        "rating": {"$gte": 4.0},
    })
    .limit(10)
    .to_list()
)
```

Use scalar indexes for frequently filtered fields:

```python
await db.create_scalar_index("price", index_type="btree")
await db.create_scalar_index("category", index_type="bitmap")
```

See [Metadata Filtering](filtering.md) for all operators and indexing guidance.

## Reranking

Reranking is a two-stage pattern: retrieve more candidates than you need, score them with a stronger model, then return fewer final results.

```python
from pgvectordb.rerankers import CrossEncoderReranker

reranker = CrossEncoderReranker(model="cross-encoder/ms-marco-MiniLM-L-6-v2")

results = await (
    db.query("how to configure diskann")
    .hybrid()
    .limit(100)
    .rerank(reranker)
    .to_list()
)

top_results = results[:10]
```

Use reranking when top result order matters more than raw retrieval latency, especially for answer generation, support search, legal search, and high-value product search. In fluent queries, `.limit(...)` controls the candidate set that reaches the reranker; slice the returned list for the final result count. See [Reranking](reranking.md) for backend setup and candidate-count guidance.

## SQL Analysis

Use `explain_plan()` before executing when you want to understand the planned search shape.

```python
plan = (
    db.query("filtered vector search")
    .semantic()
    .where({"topic": "optimization"})
    .explain_plan()
)
```

Use `analyze_plan()` to execute the configured query and return timing plus configuration metadata.

```python
metrics = await (
    db.query("filtered vector search")
    .semantic()
    .where({"topic": "optimization"})
    .ef(100)
    .limit(10)
    .analyze_plan()
)

print(metrics["execution_time_ms"])
print(metrics["rows_returned"])
print(metrics["search_method"])
```

For deeper PostgreSQL `EXPLAIN (ANALYZE, BUFFERS, VERBOSE)` output, use `explain_query()` from the diagnostics guide.

## Output Formats

```python
results = await db.query("search quality").semantic().limit(10).to_list()
frame = await db.query("search quality").semantic().limit(10).to_pandas()
arrow_table = await db.query("search quality").semantic().limit(10).to_arrow()
```

Use `to_list()` for application code, `to_pandas()` for analysis notebooks, and `to_arrow()` for larger tabular pipelines.

## Common Recipes

### Multi-tenant RAG

```python
results = await (
    db.query("API documentation")
    .hybrid()
    .where({"tenant_id": tenant_id, "visibility": "published"})
    .limit(10)
    .to_list()
)
```

### Date-bounded semantic search

```python
results = await (
    db.query("quarterly revenue risks")
    .semantic()
    .where({"year": {"$between": [2024, 2026]}})
    .ef(100)
    .limit(20)
    .to_list()
)
```

### Recall check with exact search

```python
exact = await db.query("index tuning").semantic().bypass_vector_index().limit(10).to_list()
ann = await db.query("index tuning").semantic().ef(80).limit(10).to_list()

exact_ids = {row["id"] for row in exact}
ann_ids = {row["id"] for row in ann}
recall_at_10 = len(exact_ids & ann_ids) / len(exact_ids)
```

## Score Direction

| Search type | Score direction | Notes |
| --- | --- | --- |
| Semantic | Lower distance is better in raw vector distance; normalized query results may expose relevance-style scores depending on method. |
| Keyword / BM25 | Higher is better. | PostgreSQL FTS or BM25 rank. |
| Hybrid | Higher is better. | Fused score from weighted or RRF ranking. |
| Trigram | Higher is better. | Similarity score from trigram matching. |

Always validate ranking with your own ground truth using [Metrics & Evaluation](metrics_and_evaluation.md) before committing to production tuning.