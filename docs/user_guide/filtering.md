# Metadata Filtering

pgVectorDB implements MongoDB-style JSON queries that translate to secure PostgreSQL `JSONB` clauses. Use with the fluent API via `.where()`.

---

## The `.where()` Method

Add filters to any search:

```python
results = await (
    db.search("machine learning")
    .where({"category": "ai", "status": "published"})
    .limit(10)
    .to_list()
)
```

**Key point**: Filters are applied **before** the vector search (pre-filter) by default, ensuring accurate results.

---

## Filter Operators Reference

### Comparison Operators

#### `$eq` - Equality (Implicit)
```python
# Implicit equality (preferred)
results = await db.search("query").where({"department": "HR"}).to_list()

# SQL: langchain_metadata->>'department' = 'HR'
```

#### `$ne` - Not Equal
```python
results = await (
    db.search("query")
    .where({"status": {"$ne": "archived"}})
    .to_list()
)
# SQL: langchain_metadata->>'status' != 'archived'
```

#### `$lt`, `$lte`, `$gt`, `$gte` - Numeric Comparisons
```python
results = await (
    db.search("query")
    .where({"price": {"$lt": 100}})
    .to_list()
)
results = await (
    db.search("query")
    .where({"rating": {"$gte": 4.5}})
    .to_list()
)
# SQL: (langchain_metadata->>'price')::numeric < 100
```

#### `$between` - Range (Inclusive)
```python
results = await (
    db.search("quarterly report")
    .where({"year": {"$between": [2020, 2023]}})
    .to_list()
)
# SQL: (langchain_metadata->>'year')::numeric BETWEEN 2020 AND 2023
```

### Set Operators

#### `$in` - In Array
```python
results = await (
    db.search("technology")
    .where({"category": {"$in": ["tech", "science", "engineering"]}})
    .to_list()
)
# SQL: langchain_metadata->>'category' = ANY(ARRAY['tech', 'science', 'engineering'])
```

#### `$nin` - Not In Array
```python
results = await (
    db.search("query")
    .where({"tag": {"$nin": ["draft", "deleted", "archived"]}})
    .to_list()
)
# SQL: langchain_metadata->>'tag' != ALL(ARRAY['draft', 'deleted', 'archived'])
```

### Existence Operators

#### `$exists` - Key Existence
```python
results = await (
    db.search("premium content")
    .where({
        "deleted_at": {"$exists": False},  # Must NOT exist
        "premium_tier": {"$exists": True}   # Must exist
    })
    .to_list()
)
# SQL: langchain_metadata->>'deleted_at' IS NULL AND langchain_metadata->>'premium_tier' IS NOT NULL
```

### Pattern Matching

#### `$like`, `$ilike` - SQL LIKE Patterns
```python
results = await (
    db.search("corporate email")
    .where({
        "email": {"$like": "%@company.com"},
        "title": {"$ilike": "%database%"}  # Case insensitive
    })
    .to_list()
)
# SQL: langchain_metadata->>'email' LIKE '%@company.com'
```

### Logical Operators

#### `$and` - AND Grouping
```python
results = await (
    db.search("finance report")
    .where({
        "$and": [
            {"department": "finance"},
            {"level": {"$gte": 3}},
            {"status": "active"}
        ]
    })
    .to_list()
)
```

#### `$or` - OR Grouping
```python
results = await (
    db.search("urgent items")
    .where({
        "$or": [
            {"priority": "critical"},
            {"created_at": {"$gte": "2024-01-01"}, "status": "unread"}
        ]
    })
    .to_list()
)
```

#### Nested Logic
```python
results = await (
    db.search("research papers")
    .where({
        "$and": [
            {"year": {"$between": [2020, 2023]}},
            {"$or": [
                {"category": "ai"},
                {"citations": {"$gt": 1000}}
            ]}
        ]
    })
    .to_list()
)
```

---

## Usage Examples

### Multi-tenant Isolation

```python
# Essential for SaaS applications
results = await (
    db.search("project documentation")
    .where({"tenant_id": current_user.tenant_id})  # Always filter by tenant
    .limit(10)
    .to_list()
)
```

### Category + Date Range

```python
results = await (
    db.search("breaking news")
    .where({
        "category": "news",
        "published_at": {"$gte": "2024-01-01"},
        "status": "published"
    })
    .limit(20)
    .to_list()
)
```

### Permission-based Search

```python
results = await (
    db.search("confidential documents")
    .where({
        "$or": [
            {"owner_id": current_user.id},
            {"shared_with": {"$in": current_user.teams}},
            {"visibility": "public"}
        ]
    })
    .limit(50)
    .to_list()
)
```

### Product Catalog Search

```python
results = await (
    db.search("wireless headphones")
    .where({
        "category": "electronics",
        "price": {"$between": [50, 200]},
        "in_stock": True,
        "rating": {"$gte": 4.0}
    })
    .limit(10)
    .to_list()
)
```

---

## Scalar Index Creation

For large collections (100K+ documents), create B-Tree or GIN indexes on frequently filtered fields:

```python
# B-Tree index for range queries ($gt, $lt, $between)
await db.create_scalar_index("price", index_type="btree")

# GIN index for low-cardinality fields ($eq, $in)
await db.create_scalar_index("category", index_type="bitmap")  # Uses GIN
```

**Benefits:**
- 10-100x faster filtered queries
- Reduced PostgreSQL CPU usage
- Better multi-tenant performance

See [Indexing & Performance](../advanced/indexing.md) for details.

---

## Complete Example

```python
import asyncio
from pgvectordb import pgVectorDB
from langchain_huggingface import HuggingFaceEmbeddings

async def main():
    db = pgVectorDB(
        collection_name="documents",
        embedding_model=HuggingFaceEmbeddings(),
        connection_string="postgresql+asyncpg://..."
    )
    await db.initialize()
    
    # Add sample documents
    await db.add_texts(
        texts=["AI research paper", "Database guide", "ML tutorial"],
        metadatas=[
            {"category": "ai", "year": 2024, "citations": 150},
            {"category": "tech", "year": 2023, "citations": 80},
            {"category": "ai", "year": 2024, "citations": 230},
        ]
    )
    
    # Search with complex filter
    results = await (
        db.search("machine learning")
        .where({
            "$and": [
                {"category": "ai"},
                {"year": {"$gte": 2024}},
                {"$or": [
                    {"citations": {"$gt": 100}},
                    {"featured": True}
                ]}
            ]
        })
        .limit(5)
        .to_list()
    )
    
    for r in results:
        print(f"[{r['score']:.4f}] {r['content']}")
        print(f"    Metadata: {r['metadata']}")
    
    await db.close()

if __name__ == "__main__":
    asyncio.run(main())
```
