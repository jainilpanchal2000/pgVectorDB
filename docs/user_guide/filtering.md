# Metadata Filtering

pgVectorDB filters are MongoDB-style dictionaries that compile to parameterized PostgreSQL JSONB predicates. Use them through `.where(...)` on the fluent query builder.

```python
results = await (
    db.query("machine learning")
    .semantic()
    .where({"category": "ai", "status": "published"})
    .limit(10)
    .to_list()
)
```

Filters are applied before result ranking where the underlying search method supports it. For large collections, pair frequent filters with scalar indexes.

## Operator Reference

### Equality

Implicit equality is the most common form.

```python
results = await (
    db.query("onboarding guide")
    .semantic()
    .where({"department": "HR"})
    .to_list()
)
```

Equivalent explicit form:

```python
results = await db.query("onboarding guide").semantic().where({"department": {"$eq": "HR"}}).to_list()
```

### Numeric Comparisons

```python
results = await (
    db.query("budget laptops")
    .hybrid()
    .where({"price": {"$lt": 1000}, "rating": {"$gte": 4.2}})
    .limit(10)
    .to_list()
)
```

Supported comparison operators:

| Operator | Meaning |
| --- | --- |
| `$ne` | Not equal. |
| `$lt` / `$lte` | Less than / less than or equal. |
| `$gt` / `$gte` | Greater than / greater than or equal. |
| `$between` | Inclusive range: `[low, high]`. |

### Ranges

```python
results = await (
    db.query("quarterly report")
    .semantic()
    .where({"year": {"$between": [2024, 2026]}})
    .limit(20)
    .to_list()
)
```

### Lists

```python
results = await (
    db.query("database tutorials")
    .keyword()
    .where({"category": {"$in": ["postgres", "vector", "rag"]}})
    .limit(10)
    .to_list()
)
```

```python
results = await (
    db.query("public docs")
    .semantic()
    .where({"status": {"$nin": ["draft", "archived", "deleted"]}})
    .to_list()
)
```

### Existence Checks

```python
results = await (
    db.query("premium content")
    .semantic()
    .where({
        "deleted_at": {"$exists": False},
        "premium_tier": {"$exists": True},
    })
    .to_list()
)
```

### Pattern Matching

```python
results = await (
    db.query("corporate contact")
    .keyword()
    .where({
        "email": {"$like": "%@company.com"},
        "title": {"$ilike": "%database%"},
    })
    .to_list()
)
```

`$like` is case-sensitive. `$ilike` is case-insensitive.

### Logical Operators

Use `$and` and `$or` to build nested filters.

```python
results = await (
    db.query("research papers")
    .hybrid()
    .where({
        "$and": [
            {"year": {"$between": [2020, 2026]}},
            {
                "$or": [
                    {"category": "ai"},
                    {"citations": {"$gt": 1000}},
                ]
            },
        ]
    })
    .limit(10)
    .to_list()
)
```

## Common Patterns

### Multi-tenant Isolation

```python
results = await (
    db.query("project documentation")
    .hybrid()
    .where({"tenant_id": current_user.tenant_id})
    .limit(10)
    .to_list()
)
```

Always include the tenant filter in application-level retrieval paths for SaaS workloads.

### Permission-aware Search

```python
results = await (
    db.query("confidential roadmap")
    .semantic()
    .where({
        "$or": [
            {"owner_id": current_user.id},
            {"team_id": {"$in": current_user.team_ids}},
            {"visibility": "public"},
        ]
    })
    .limit(20)
    .to_list()
)
```

### Product Catalog Search

```python
results = await (
    db.query("wireless noise cancelling headphones")
    .hybrid()
    .where({
        "category": "electronics",
        "price": {"$between": [50, 200]},
        "in_stock": True,
        "rating": {"$gte": 4.0},
    })
    .weights(semantic=0.65, keyword=0.35)
    .limit(10)
    .to_list()
)
```

### Fresh Content

Use metadata filters when freshness is a hard constraint.

```python
results = await (
    db.query("security advisory")
    .semantic()
    .where({"published_at": {"$gte": "2026-01-01"}})
    .limit(10)
    .to_list()
)
```

Use `RecencySpace` when freshness should be a ranking signal instead of a strict cutoff. See [Multimodal Search](multimodal_search.md).

## Scalar Indexes

Create scalar indexes for fields that appear in frequent filters. This is one of the main optimization differentiators in pgVectorDB because it keeps metadata filtering fast inside PostgreSQL.

```python
await db.create_scalar_index("price", index_type="btree")
await db.create_scalar_index("category", index_type="bitmap")
await db.create_scalar_index("tags", index_type="gin")
```

| Index type | Best for | Example filter |
| --- | --- | --- |
| `btree` | Numeric ranges and high-cardinality equality. | `{"price": {"$between": [50, 200]}}` |
| `bitmap` | Low-cardinality categories. | `{"category": "electronics"}` |
| `gin` | Array or containment-style metadata. | `{"tags": {"$in": ["rag", "postgres"]}}` |
| `labellist` | DiskANN label arrays. | Label-filtered ANN search. |

The scalar index builder auto-detects metadata value types and creates expression indexes such as numeric casts for range fields.

## Verify Filter Performance

Use the fluent analyzer for a quick timing check:

```python
metrics = await (
    db.query("portable computer")
    .semantic()
    .where({"price": {"$between": [400, 900]}})
    .ef(100)
    .limit(10)
    .analyze_plan()
)

print(metrics["execution_time_ms"])
print(metrics["rows_returned"])
```

Use `explain_query()` from [Analytics & Diagnostics](analytics_and_diagnostics.md) when you need raw PostgreSQL `EXPLAIN` output.

## Complete Example

```python
import asyncio

from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

from pgvectordb import pgVectorDB


async def main():
    db = pgVectorDB(
        collection_name="products",
        embedding_model=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2"),
        connection_string="postgresql+asyncpg://user:root@localhost:9002/postgres",
    )
    await db.initialize()

    await db.add_documents([
        Document(page_content="Wireless headphones with active noise cancellation", metadata={"category": "electronics", "price": 149, "rating": 4.6}),
        Document(page_content="USB-C studio microphone for podcasting", metadata={"category": "electronics", "price": 89, "rating": 4.4}),
        Document(page_content="Cotton travel backpack", metadata={"category": "travel", "price": 72, "rating": 4.1}),
    ])

    await db.create_scalar_index("price", index_type="btree")
    await db.create_scalar_index("category", index_type="bitmap")

    results = await (
        db.query("portable audio gear")
        .hybrid()
        .where({
            "$and": [
                {"category": "electronics"},
                {"price": {"$between": [50, 200]}},
                {"rating": {"$gte": 4.0}},
            ]
        })
        .limit(5)
        .to_list()
    )

    for row in results:
        print(row["score"], row["content"], row["metadata"])

    await db.close()


asyncio.run(main())
```

## Next Steps

- [Search & Retrieval](search_and_retrieval.md)
- [Multimodal Search](multimodal_search.md)
- [Analytics & Diagnostics](analytics_and_diagnostics.md)
- [Indexing & Performance](../advanced/indexing.md)