# Query Controls - ANN Tuning and Advanced Retrieval

**Version:** 0.0.7  
**Focus:** Fine-grained control over approximate nearest neighbor (ANN) search behavior

This guide covers advanced query controls for optimizing the recall/latency trade-off and implementing specialized retrieval patterns like near-duplicate detection.

---

## Distance Range Filtering

Filter results by distance from the query vector. Useful for thresholded retrieval and near-duplicate detection.

### Within Distance

Find all results within a maximum distance:

```python
results = await db.query("machine learning")
    .semantic()
    .within_distance(0.1)  # Cosine distance ≤ 0.1
    .limit(10)
    .to_list()
```

### Distance Range

Find results between minimum and maximum distances:

```python
results = await db.query("machine learning")
    .semantic()
    .distance_range(0.05, 0.3)  # 0.05 ≤ distance ≤ 0.3
    .limit(10)
    .to_list()
```

### Use Cases

**Near-Duplicate Detection:**
```python
duplicates = await db.query(document.embedding)
    .semantic()
    .distance_range(0.0, 0.05)  # Very similar documents
    .where({"id": {"$ne": document.id}})  # Exclude self
    .limit(10)
    .to_list()
```

**Quality Threshold:**
```python
# Only return results with confidence ≥ 0.9 (cosine distance ≤ 0.1)
results = await db.query("technology")
    .semantic()
    .within_distance(0.1)
    .limit(10)
    .to_list()
```

---

## Exact Search Bypass

Force exact (brute force) search for maximum recall at the cost of latency.

```python
results = await db.query("important query")
    .semantic()
    .exact_search()  # Bypass ANN approximation
    .limit(10)
    .to_list()
```

### When to Use

- **Small collections** (< 10,000 vectors): Exact search is often faster
- **Critical queries**: When missing relevant results is unacceptable
- **Evaluation**: Compare ANN recall vs. ground truth
- **Debugging**: Validate index behavior

### Implementation Details

| Index Type | Exact Search Behavior |
|------------|----------------------|
| **HNSW** | Sets `hnsw.ef_search = 100000` |
| **IVFFlat** | Skips list probes, scans all lists |
| **DiskANN** | Uses exact mode or max search list |

### Disable Exact Search

```python
results = await db.query("normal query")
    .semantic()
    .exact_search(False)  # Use ANN approximation
    .limit(10)
    .to_list()
```

---

## Filter Strategies

Control when metadata filters are applied relative to vector search.

### Pre-Filter (Default for selective filters)

Apply metadata filter **before** vector search:

```python
results = await db.query("AI applications")
    .semantic()
    .pre_filter()
    .where({"category": "ai", "status": "published"})
    .limit(10)
    .to_list()
```

**Best for:** Highly selective filters (few matching documents)

**Trade-offs:**
- ✅ Lower latency (smaller search space)
- ⚠️ May reduce recall if filter excludes relevant vectors lost to ANN approximation

### Post-Filter

Apply metadata filter **after** vector search:

```python
results = await db.query("technology")
    .semantic()
    .post_filter()
    .where({"status": "active"})  # Many active docs
    .limit(10)
    .to_list()
```

**Best for:** Non-selective filters (many matching documents)

**Trade-offs:**
- ✅ Perfect recall within returned set
- ⚠️ Higher latency (fetches extra results)

**Implementation:** Fetches `limit * fetch_multiplier` (default 2x) results, then filters.

### Auto Mode (Default)

Automatically choose strategy based on filter selectivity:

```python
results = await db.query("some query")
    .semantic()
    .where({"category": "ai"})
    # .pre_filter() or .post_filter() chosen automatically
    .limit(10)
    .to_list()
```

**Auto Strategy:**
- If estimated results < 100: Use pre-filter
- If estimated results ≥ 100: Use post-filter

---

## ANN Parameter Tuning

Fine-tune ANN parameters for your recall/latency requirements.

### HNSW - ef_search

```python
results = await db.query("machine learning")
    .semantic()
    .ef(200)  # HNSW ef_search parameter
    .limit(10)
    .to_list()
```

| ef_search | Recall | Latency | Use Case |
|-----------|--------|---------|----------|
| 10-20 | Lower | Fastest | Exploration, speed critical |
| 40-80 | Balanced | Moderate | Default workloads |
| 100-200 | Higher | Slower | Quality critical |
| 100000 | Exact | Slowest | Maximum recall |

### IVFFlat - nprobes

```python
results = await db.query("machine learning")
    .semantic()
    .nprobes(20)  # Number of lists to probe
    .limit(10)
    .to_list()
```

| nprobes | Recall | Latency | Use Case |
|---------|--------|---------|----------|
| 1 | Lowest | Fastest | Rough filtering |
| 5-10 | Balanced | Moderate | Default workloads |
| 20-50 | Higher | Slower | Quality critical |
| lists_count | Exact | Slowest | Brute force |

### Refine Factor (DiskANN)

```python
results = await db.query("machine learning")
    .semantic()
    .refine_factor(4)  # Oversampling factor
    .limit(10)
    .to_list()
```

**Refine factor** multiples the search list size before reranking. Higher values improve recall.

---

## Complete Example

```python
# Maximum recall search with complex filtering
results = await db.query("machine learning infrastructure")
    .semantic()
    .exact_search()  # Force exact search
    .pre_filter()  # Apply filter first
    .where({
        "category": "ai",
        "status": "published",
        "date": {"$gte": "2024-01-01"}
    })
    .within_distance(0.2)  # Only high-quality matches
    .limit(20)
    .to_list()
```

---

## Performance Guidelines

### Choosing Parameters

1. **Start with defaults** (ef=40, nprobes=5)
2. **Measure recall** on representative queries
3. **Increase parameters** if recall < target
4. **Use exact_search** for critical queries

### Common Patterns

| Scenario | Recommended Settings |
|----------|---------------------|
| Web search | `ef(40)`, `post_filter()` |
| Duplicate detection | `distance_range(0, 0.05)`, `exact_search()` |
| High-quality retrieval | `ef(100)`, `within_distance(0.1)` |
| Large-scale exploration | `ef(20)`, `pre_filter()` |
| Evaluation/benchmarking | `exact_search()` |

---

## API Reference

### Distance Filtering

```python
.within_distance(radius: float) -> UnifiedQueryBuilder
.distance_range(min_dist: float, max_dist: float) -> UnifiedQueryBuilder
```

### Exact Search

```python
.exact_search(exact: bool = True) -> UnifiedQueryBuilder
```

### Filter Strategy

```python
.pre_filter() -> UnifiedQueryBuilder
.post_filter() -> UnifiedQueryBuilder
```

### ANN Parameters

```python
.ef(n: int) -> UnifiedQueryBuilder
.nprobes(n: int) -> UnifiedQueryBuilder
.refine_factor(factor: int) -> UnifiedQueryBuilder
```

---

## Troubleshooting

### Low Recall

1. Increase `ef` (HNSW) or `nprobes` (IVFFlat)
2. Use `exact_search()` to verify index is working
3. Check index build parameters (m, ef_construction for HNSW)

### High Latency

1. Reduce `ef` or `nprobes`
2. Use `pre_filter()` for selective filters
3. Consider downsampling for initial exploration

### Distance Filter Not Working

- Distance filtering requires pgvector ≥ 0.5.0
- Check your index type supports pre-filtering (HNSW best, IVFFlat requires caution)

---

## Next Steps

- See [Filtering Strategies](./filtering_strategies.md) for detailed pre/post filter guidance
- See [Indexing & Performance](../advanced/indexing.md) for index tuning and benchmarking
