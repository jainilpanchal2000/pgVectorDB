# Filtering Strategies - Pre-Filter vs Post-Filter

**Version:** 0.0.7  
**Focus:** Understanding and optimizing metadata filter behavior

This guide explains when and why to use different metadata filtering strategies.

---

## Overview

pgvector supports two main approaches to combining vector similarity with metadata filtering:

| Strategy | When Applied | Best For | Trade-off |
|----------|-------------|----------|-----------|
| **Pre-Filter** | Before vector search | Selective filters (few matches) | May reduce recall |
| **Post-Filter** | After vector search | Non-selective filters (many matches) | Higher latency |

---

## Pre-Filter

**Pre-filtering** applies metadata constraints before running vector search.

### How It Works

1. Filter documents by metadata criteria
2. Build temporary index or use index with WHERE clause
3. Run vector search on filtered subset
4. Return top-k results

### When to Use

**Best for highly selective filters:**

```python
# Highly selective: few AI docs in 2024
results = await db.query("ml")
    .semantic()
    .pre_filter()  # Explicit strategy
    .where({
        "category": "ai",
        "year": 2024,
        "priority": {"$gt": 8}
    })
    .limit(10)
    .to_list()
```

**Common scenarios:**
- Filtering by specific category (low cardinality)
- Date range with small result set
- Status = "published" (if most docs are drafts)
- Multi-field AND conditions

### SQL Implementation

Pre-filtering generates SQL like:

```sql
SELECT id, embedding, metadata
FROM documents
WHERE metadata @> '{"category": "ai", "year": 2024}'
ORDER BY embedding <=> query_embedding
LIMIT 10
```

**Index Used:** Combined vector index + GIN index on metadata

### Trade-offs

| Pros | Cons |
|------|------|
| ✅ Lower latency (smaller search space) | ⚠️ May miss ANN-approximated vectors outside filter |
| ✅ Fewer disk reads | ⚠️ Vector index accuracy depends on filter selectivity |
| ✅ Better for selective queries | ⚠️ Can return fewer results than limit |

### Recall Considerations

The pre-filter may reduce recall because:

1. **IVFFlat:** Probes only a subset of lists based on query vector
2. **HNSW:** Navigates graph with filter constraints
3. **Filtered vectors never candidates:** Relevant vectors filtered by metadata are excluded

**When recall is critical:**
```python
results = await db.query("ml")
    .semantic()
    .pre_filter()
    .exact_search()  # Add exact search for maximum recall
    .where({"category": "ai"})
    .limit(10)
    .to_list()
```

---

## Post-Filter

**Post-filtering** runs vector search first, then filters results.

### How It Works

1. Run vector search requesting `limit * multiplier` results
2. Apply metadata filter to returned results
3. Return up to `limit` matching results

### When to Use

**Best for non-selective filters:**

```python
# Non-selective: many active documents
results = await db.query("technology")
    .semantic()
    .post_filter()  # Explicit strategy
    .where({"status": "active"})  # Many docs are active
    .limit(10)
    .to_list()
```

**Common scenarios:**
- Status = "active" (if majority of docs are active)
- Category filter for large categories
- Date range with many results
- Simple presence checks

### Fetch Multiplier

Post-filter uses `fetch_multiplier` (default: 2.0) to ensure enough results:

```python
# Default: fetches 20 results, returns up to 10 matching
results = await db.query("test")
    .semantic()
    .post_filter()
    .where({"status": "active"})
    .limit(10)  # Fetches 10 * 2.0 = 20
```

**Adjust for selectivity:**
```python
# Script 0: Low selectivity, more need
config = SearchConfig(
    limit=10,
    filter_strategy="post",
    fetch_multiplier=3.0  # Fetch 30 to get 10
)

# Script 1: Medium selectivity
config = SearchConfig(
    limit=10,
    filter_strategy="post",
    fetch_multiplier=1.5  # Fetch 15 to get 10
)
```

### SQL Implementation

Post-filtering generates SQL like:

```sql
WITH vector_results AS (
    SELECT id, embedding, metadata, (embedding <=> query_embedding) as distance
    FROM documents
    ORDER BY embedding <=> query_embedding
    LIMIT 20  -- limit * fetch_multiplier
)
SELECT * FROM vector_results
WHERE metadata @> '{"status": "active"}'
LIMIT 10
```

**Index Used:** Vector index only (fast), filter applied in memory

### Trade-offs

| Pros | Cons |
|------|------|
| ✅ Perfect recall on returned set | ⚠️ Higher latency (fetches extra) |
| ✅ Consistent behavior across index types | ⚠️ More memory usage |
| ✅ Simple implementation | ⚠️ May need high multiplier |

---

## Auto Strategy

The default strategy automatically chooses based on filter selectivity.

### Selection Logic

```python
def choose_strategy(filter, table_size):
    # Estimate result count
    estimated_results = estimate_selectivity(filter)
    
    if estimated_results < 100:  # Threshold
        return "pre_filter"  # Selective
    else:
        return "post_filter"  # Non-selective
```

### Override When Needed

Auto works well most of the time, but override for:

1. **Known selectivity:**
   ```python
   # You know category "rare" is tiny
   results = await db.query("test")
       .semantic()
       .pre_filter()  # Force even if auto would guess wrong
       .where({"category": "rare"})
       .limit(10)
       .to_list()
   ```

2. **Critical recall:**
   ```python
   # Must not miss any results
   results = await db.query("test")
       .semantic()
       .post_filter()  # Never miss due to ANN approximation
       .exact_search()  # BRute force for maximum recall
       .limit(10)
       .to_list()
   ```

---

## Decision Flowchart

```
                    ┌─────────────────────────────────────┐
                    │     New query with metadata filter? │
                    └───────────────┬─────────────────────┘
                                    │
                    ┌───────────────▼─────────────────────┐
                    │     How selective is the filter?    │
                    └───────┬───────────────────┬───────┘
                            │                   │
              ┌─────────────▼────┐   ┌──────────▼────────┐
              │ Few results (<100)│   │ Many results (≥100)│
              │ Selective         │   │ Non-selective      │
              └────────┬─────────┘   └─────────┬──────────┘
                       │                     │
         ┌─────────────▼────┐   ┌─────────────▼─────────┐
         │   .pre_filter()   │   │    .post_filter()      │
         │   + faster         │   │    + perfect recall    │
         │   - recall risk    │   │    - higher latency    │
         └────────────────────┘   └────────────────────────┘
```

---

## Performance Comparison

### Scenario 1: Selective Filter

**Setup:** 100,000 documents, 50 match filter

| Strategy | Latency | Recall | Details |
|----------|---------|--------|---------|
| Pre-Filter | 5ms | 92% | 50 candidates, fast ANN |
| Post-Filter | 25ms | 100% | Fetch 200, filter 50, return 10 |

**Winner:** Pre-filter (4x faster, recall acceptable)

### Scenario 2: Non-Selective Filter

**Setup:** 100,000 documents, 80,000 match filter

| Strategy | Latency | Recall | Details |
|----------|---------|--------|---------|
| Pre-Filter | 200ms | 85% | 80k candidates, slow ANN |
| Post-Filter | 30ms | 100% | Fetch 20, filter 16, return 10 |

**Winner:** Post-filter (6x faster, perfect recall)

---

## Implementation Guide

### Pattern 1: Category Filter (Selective)

```python
def search_by_category(db, query, category):
    # Categories are usually selective
    return await db.query(query)
        .semantic()
        .pre_filter()
        .where({"category": category})
        .limit(10)
        .to_list()
```

### Pattern 2: Status Filter (Non-Selective)

```python
def search_active_docs(db, query):
    # "active" usually matches most docs
    return await db.query(query)
        .semantic()
        .post_filter()
        .where({"status": "active"})
        .limit(10)
        .to_list()
```

### Pattern 3: Dynamic Strategy Selection

```python
async def optimized_search(db, query, filter_dict):
    """Choose strategy based on estimated selectivity."""
    # Estimate selectivity (simplified)
    if await is_selective(db, filter_dict):
        return await db.query(query)
            .semantic()
            .pre_filter()
            .where(filter_dict)
            .limit(10)
            .to_list()
    else:
        return await db.query(query)
            .semantic()
            .post_filter()
            .where(filter_dict)
            .limit(10)
            .to_list()

async def is_selective(db, filter_dict):
    """Estimate if filter is selective."""
    # Fast metadata-only query to estimate
    count = await db.query("")
        .metadata_only()
        .where(filter_dict)
        .count()
    return count < 100
```

---

## Index Recommendations

### GIN Index for Metadata

Pre-filtering benefits from GIN indexes on metadata:

```python
# Create GIN index for faster pre-filter
await db.gin.ensure_gin_index("metadata", "jsonb")

# Now pre-filter queries are faster
results = await db.query("test")
    .semantic()
    .pre_filter()
    .where({"category": "ai"})  # Faster with GIN
    .limit(10)
    .to_list()
```

See [Index Management](../advanced/indexing.md) for details.

---

## Troubleshooting

### Post-Filter Returns Few Results

```python
# Problem: Only getting 3 results when limit=10
results = await db.query("test")
    .semantic()
    .post_filter()
    .where({"rare_field": "rare_value"})
    .limit(10)
    .to_list()  # Returns only 3!

# Solution: Increase fetch_multiplier
config = SearchConfig(fetch_multiplier=10.0)  # Fetch 100
```

### Pre-Filter Missing Expected Results

```python
# Problem: Expected document not returned
results = await db.query("test")
    .semantic()
    .pre_filter()
    .where({"category": "ai"})
    .limit(10)
    .to_list()

# Solution: Use post_filter or exact_search
results = await db.query("test")
    .semantic()
    .post_filter()  # Try post-filter
    .exact_search()  # Or force exact
    .where({"category": "ai"})
    .limit(10)
    .to_list()
```

---

## Summary

| Filter Characteristic | Recommended Strategy |
|------------------------|---------------------|
| < 100 results expected | Pre-filter (fast) |
| ≥ 100 results expected | Post-filter (accurate) |
| Critical recall needed | Post-filter + exact_search |
| Speed priority | Pre-filter (monitor recall) |
| Unknown selectivity | Auto (default) |
