# Migration Guide: Old API to Fluent API

This guide shows how to migrate from the legacy method-based API to the new LanceDB-style fluent API in v0.0.6+.

!!! note "Backward Compatibility"
    All legacy methods continue to work. The fluent API is additive — you can use both styles interchangeably.

---

## Quick Reference

| Old Method | New Fluent API | Notes |
|------------|----------------|-------|
| `semantic_search(query, k=10)` | `db.search(query).limit(10).to_list()` | Now supports method chaining |
| `keyword_search(query, k=10)` | `db.search(query).to_list()` | Both FTS and BM25 |
| `hybrid_search(query, k=10)` | `db.search(query).nearest_to_text(...).to_list()` | Vector + FTS fusion |
| `metadata_semantic_search(query, filter, k=10)` | `db.search(query).where(filter).limit(10).to_list()` | Built-in filtering |
| `semantic_search(..., use_exact_search=True)` | `db.search(query).bypass_vector_index().to_list()` | Force exact search |

---

## Examples

### 1. Basic Semantic Search

**Old API:**
```python
results = await db.semantic_search(
    query="machine learning",
    k=10
)
```

**New Fluent API:**
```python
results = await db.search("machine learning").limit(10).to_list()
```

---

### 2. Keyword Search (FTS)

**Old API:**
```python
results = await db.keyword_search(
    query="machine learning",
    k=10,
    search_type=KeywordSearchType.FTS
)
```

**New Fluent API:**
```python
# Currently returns VectorQueryBuilder which auto-embeds
# For pure FTS, use:
results = await db.search_text("machine learning").limit(10).to_list()
# Note: search_text method needs separate implementation
```

---

### 3. Filtered Search

**Old API:**
```python
results = await db.metadata_semantic_search(
    query="AI frameworks",
    filter={"category": "ai"},
    k=10
)
```

**New Fluent API:**
```python
results = await (
    db.search("AI frameworks")
    .where({"category": "ai"})
    .limit(10)
    .to_list()
)
```

---

### 4. Complex Filter

**Old API:**
```python
results = await db.metadata_semantic_search(
    query="database optimization",
    filter={
        "$and": [
            {"category": "database"},
            {"year": {"$gte": 2023}},
        ]
    },
    k=10
)
```

**New Fluent API:**
```python
results = await (
    db.search("database optimization")
    .where({
        "$and": [
            {"category": "database"},
            {"year": {"$gte": 2023}},
        ]
    })
    .limit(10)
    .to_list()
)
```

---

### 5. Hybrid Search (Vector + Keyword)

**Old API:**
```python
results = await db.hybrid_search(
    query="machine learning",
    k=10,
    use_rrf=True,
    rrf_k=60
)
```

**New Fluent API:**
```python
results = await (
    db.search("machine learning")  # Vector search base
    .nearest_to_text("neural networks")  # Add FTS component
    .limit(10)
    .to_list()
)
```

---

### 6. Exact Search (Bypass Index)

**Old API:**
```python
results = await db.semantic_search(
    query="test",
    k=10,
    use_exact_search=True
)
```

**New Fluent API:**
```python
results = await (
    db.search("test")
    .bypass_vector_index()  # Forces sequential scan
    .limit(10)
    .to_list()
)
```

---

### 7. Query Parameter Tuning

**Old API:**
```python
await db.set_query_params({"hnsw.ef_search": 100})
results = await db.semantic_search("query", k=10)
```

**New Fluent API:**
```python
results = await (
    db.search("query")
    .ef(100)  # Per-query parameter
    .limit(10)
    .to_list()
)
```

---

### 8. Explain Query Plan

**Old API:**
```python
plan = await db.explain_query(
    query="machine learning",
    search_method="semantic_search",
    k=10
)
```

**New Fluent API:**
```python
plan = db.search("machine learning").explain_plan()
# or
plan = (
    db.search("query")
    .where({"category": "ai"})
    .explain_plan(verbose=True)
)
```

---

### 9. Analyze Query Performance

**Old API:**
```python
details = await db.explain_query(
    query="test",
    search_method="semantic_search",
    k=10,
    analyze=True
)
```

**New Fluent API:**
```python
metrics = await db.search("test").analyze_plan()
print(f"Execution time: {metrics['execution_time_ms']}ms")
```

---

### 10. Search with Reranking

**Old API:**
```python
results = await db.semantic_search_with_reranker(
    query="machine learning",
    k=10,
)
```

**New Fluent API:**
```python
results = await (
    db.search("machine learning")
    .limit(50)  # Fetch more
    .rerank(cross_encoder_reranker)  # Apply reranker
    .limit(10)  # Return top 10
    .to_list()
)
```

---

### 11. Multiple Output Formats

**Old API:**
```python
results = await db.semantic_search("query", k=10)
# results is List[QueryResult]
```

**New Fluent API:**
```python
# As list
results = await db.search("query").limit(10).to_list()

# As pandas DataFrame
df = await db.search("query").limit(10).to_pandas()

# As PyArrow Table
table = await db.search("query").limit(10).to_arrow()
```

---

## Which API Should I Use?

Use the **Fluent API** when:
- You want cleaner, chainable query composition
- You need to build queries dynamically
- You want per-query parameter tuning
- You need query analysis (explain/analyze)

Keep using the **Legacy API** when:
- You have existing code that works
- You need specific methods not yet mapped (e.g., `trigram_search`)
- You prefer explicit method calls
- You're building wrappers or abstractions

---

## Internal Implementation

The fluent API delegates to existing methods:

```python
# This fluent call:
results = await db.search("query").where({"category": "ai"}).limit(10).to_list()

# Internally calls:
results = await self.metadata_semantic_search(
    query="query",
    filter={"category": "ai"},
    k=10
)
```

This ensures:
- **Backward compatibility** — old code still works
- **Consistent behavior** — same underlying implementation
- **No duplication** — fluent API wraps existing methods
