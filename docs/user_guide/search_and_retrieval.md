# Search & Retrieval

pgVectorDB provides a LanceDB-style fluent API for all search operations. This guide covers the complete query builder interface.

---

## The Fluent API Pattern

All searches follow a consistent pattern:

```python
# Chain methods to build query, execute with to_list()
results = await (
    db.search("query text")           # Start search
    .where({"category": "ai"})       # Add filters
    .limit(10)                       # Set result count
    .to_list()                       # Execute
)
```

**Key principle**: Methods return `self` for chaining. Query only executes when you call `to_list()`, `to_pandas()`, or `to_arrow()`.

---

## Search Types

### 1. Semantic Search (Vector)

The default search type uses vector similarity:

```python
# Start with a text query — embeddings computed automatically
results = await db.search("machine learning").limit(5).to_list()

# Or use pre-computed vector
vector = [0.1, 0.2, ...]  # Your embedding
results = await db.search(vector).limit(5).to_list()
```

**Under the hood**: Computes embedding → ANN search via `<=>` operator → Returns top-k by cosine distance.

### 2. Full-Text Search

Keyword search using PostgreSQL's FTS or BM25:

```python
# Standard FTS
results = await db.search_text("database optimization").limit(5).to_list()

# Phrase search
results = await (
    db.search_text("machine learning")
    .phrase_query()
    .limit(5)
    .to_list()
)
```

### 3. Hybrid Search (Vector + Text)

Combine both for better results:

```python
# Start with vector, add text search
results = await (
    db.search("machine learning")
    .nearest_to_text("neural networks")  # Add FTS component
    .limit(5)
    .to_list()
)

# Or start with text, add vector
results = await (
    db.search_text("deep learning")
    .nearest_to(query_vector)  # Add vector component
    .limit(5)
    .to_list()
)
```

**Fusion methods** (automatic):
- **RRF** (default): Reciprocal Rank Fusion — best for combining rankings
- **Weighted**: Score-based fusion — use when you know relative importance

---

## Filtering

Apply metadata filters with `.where()`:

```python
# Simple equality
results = await (
    db.search("query")
    .where({"category": "ai"})
    .to_list()
)

# Operators
results = await (
    db.search("query")
    .where({"priority": {"$gt": 5}, "status": "active"})
    .to_list()
)

# Logical operators
results = await (
    db.search("query")
    .where({
        "$and": [
            {"category": "ai"},
            {"$or": [{"year": 2024}, {"priority": {"$gte": 8}}]}
        ]
    })
    .to_list()
)
```

See [Metadata Filtering](filtering.md) for all 13 operators.

---

## Query Parameters

Fine-tune search behavior:

### IVF Parameters (IVFFlat index)

```python
results = await (
    db.search("query")
    .nprobes(20)  # Search more partitions for better recall
    .limit(10)
    .to_list()
)
```

### HNSW Parameters

```python
results = await (
    db.search("query")
    .ef(100)  # Larger candidate pool for better recall
    .limit(10)
    .to_list()
)
```

### Refinement

```python
results = await (
    db.search("query")
    .refine_factor(2)  # Fetch 2x results, rerank, return top k
    .limit(10)
    .to_list()
)
```

### Distance Range

```python
results = await (
    db.search("query")
    .distance_range(lower=0, upper=0.5)  # Only results within distance
    .limit(10)
    .to_list()
)
```

### Exact Search

```python
# For ground truth or recall measurement
results = await (
    db.search("query")
    .bypass_vector_index()  # Forces sequential scan
    .limit(10)
    .to_list()
)
```

---

## Execution Methods

Choose your output format:

```python
# List of dictionaries (default)
results = await db.search("query").to_list()
# [{'id': '...', 'content': '...', 'metadata': {...}, 'score': 0.42}, ...]

# Pandas DataFrame
df = await db.search("query").to_pandas()

# PyArrow Table (for large results)
table = await db.search("query").to_arrow()
```

---

## Query Analysis

### Explain Plan

Get the query execution plan without running:

```python
plan = db.search("query").where({"category": "ai"}).explain_plan()
print(plan)
# {
#   'plan_type': 'Index Scan',
#   'index_name': 'idx_embedding_cosine',
#   'estimated_cost': 12.34,
#   'estimated_rows': 10,
#   'index_cond': 'embedding <=> query_vector',
#   'filter': "(langchain_metadata ->> 'category') = 'ai'",
#   'using_index': True
# }
```

### Analyze Plan

Execute with `EXPLAIN ANALYZE` for real metrics:

```python
metrics = await db.search("query").analyze_plan()
print(metrics)
# {
#   'execution_time_ms': 2.34,
#   'planning_time_ms': 0.12,
#   'actual_rows': 10,
#   'actual_loops': 1,
#   'shared_hit_blocks': 45,
#   'shared_read_blocks': 0,
#   'using_index': True
# }
```

---

## Complete Examples

### Example 1: Multi-tenant Search

```python
# Isolate by tenant_id
results = await (
    db.search("API documentation")
    .where({"tenant_id": "org_123"})
    .limit(10)
    .to_list()
)
```

### Example 2: Date Range with Vector Search

```python
results = await (
    db.search("quarterly report")
    .where({
        "year": {"$gte": 2023},
        "status": "published"
    })
    .limit(20)
    .to_list()
)
```

### Example 3: Hybrid Search with Reranking

```python
results = await (
    db.search("machine learning")
    .nearest_to_text("neural networks deep learning")
    .where({"category": "research"})
    .limit(50)  # Fetch more for reranking
    .rerank(cross_encoder_reranker)  # Apply cross-encoder
    .limit(10)  # Return top 10 after rerank
    .to_list()
)
```

### Example 4: Exact Search for Recall Measurement

```python
# Ground truth
exact_results = await (
    db.search("query")
    .bypass_vector_index()
    .limit(10)
    .to_list()
)

# Approximate
ann_results = await (
    db.search("query")
    .limit(10)
    .to_list()
)

# Calculate recall
exact_ids = {r['id'] for r in exact_results}
ann_ids = {r['id'] for r in ann_results}
recall = len(exact_ids & ann_ids) / len(exact_ids)
```

---

## Score Interpretation

| Search Type | Score Direction | Description |
|-------------|-----------------|-------------|
| Semantic (vector) | Lower = better | Cosine distance (0 = identical) |
| Text (FTS) | Higher = better | ts_rank score |
| Text (BM25) | Higher = better | BM25 relevance score |
| Hybrid | Higher = better | Fused score (RRF or weighted) |
