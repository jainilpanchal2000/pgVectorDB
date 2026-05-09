# Metadata Filtering

pgVectorDB implements MongoDB-style JSON queries that translate to secure PostgreSQL `JSONB` clauses. Supports **13 filter operators**.

!!! tip "Use with Any Search Method"
    The filter parameter works with ALL search methods: `semantic_search`, `hybrid_search`, `ensemble_search`, etc.

---

## Filter Operators Reference

### Comparison Operators

#### `$eq` - Equality (Implicit)
```python
filter = {"department": "HR"}
# SQL: langchain_metadata->>'department' = 'HR'
```

#### `$ne` - Not Equal
```python
filter = {"status": {"$ne": "archived"}}
# SQL: langchain_metadata->>'status' != 'archived'
```

#### `$lt`, `$lte`, `$gt`, `$gte` - Numeric Comparisons
```python
filter = {"price": {"$lt": 100}}
filter = {"rating": {"$gte": 4.5}}
# SQL: (langchain_metadata->>'price')::numeric < 100
```

#### `$between` - Range (Inclusive)
```python
filter = {"year": {"$between": [2020, 2023]}}
# SQL: (langchain_metadata->>'year')::numeric BETWEEN 2020 AND 2023
```

### Set Operators

#### `$in` - In Array
```python
filter = {"category": {"$in": ["tech", "science"]}}
# SQL: langchain_metadata->>'category' = ANY(ARRAY['tech', 'science'])
```

#### `$nin` - Not In Array
```python
filter = {"tag": {"$nin": ["draft", "deleted"]}}
# SQL: langchain_metadata->>'tag' != ALL(ARRAY['draft', 'deleted'])
```

### Existence Operators

#### `$exists` - Key Existence
```python
filter = {
    "deleted_at": {"$exists": False},  # Must NOT exist
    "premium_tier": {"$exists": True}   # Must exist
}
# SQL: langchain_metadata->>'deleted_at' IS NULL AND langchain_metadata->>'premium_tier' IS NOT NULL
```

### Pattern Matching

#### `$like`, `$ilike` - SQL LIKE Patterns
```python
filter = {
    "email": {"$like": "%@company.com"},
    "title": {"$ilike": "%database%"}  # Case insensitive
}
# SQL: langchain_metadata->>'email' LIKE '%@company.com'
```

### Logical Operators

#### `$and` - AND Grouping
```python
filter = {
    "$and": [
        {"department": "finance"},
        {"level": {"$gte": 3}}
    ]
}
```

#### `$or` - OR Grouping
```python
filter = {
    "$or": [
        {"status": "published"},
        {"author": {"$eq": "admin"}}
    ]
}
```

#### Nested Logic
```python
filter = {
    "$and": [
        {"year": {"$between": [2020, 2023]}},
        {"$or": [
            {"category": "tech"},
            {"views": {"$gt": 1000}}
        ]}
    ]
}
```

---

## Usage Examples

### With Semantic Search
```python
results = await db.semantic_search(
    query="database optimization",
    filter={"tenant_id": "org_123", "status": "published"},
    k=5
)
```

### With Hybrid Search
```python
results = await db.hybrid_search(
    query="machine learning",
    k=5,
    filter={"department": "ai"},
    weights=(0.7, 0.3)
)
```

### With Count
```python
count = await db.count_by_metadata(
    filter={"status": "active", "department": "HR"}
)
```

---

## Production Indexing

For large collections (10M+), create B-Tree indexes on frequently filtered fields:

```sql
-- For equality ($eq)
CREATE INDEX idx_tenant_id ON vector_docs ((langchain_metadata->>'tenant_id'));

-- For numeric range ($gt, $between)
CREATE INDEX idx_price ON vector_docs (((langchain_metadata->>'price')::numeric));
```
