# Multimodal Search

pgVectorDB's multimodal system lets you attach **multiple independent embeddings to a single document** — one for text, one for price, one for category, one for recency, and so on. During search, scores from all spaces are weighted and fused into a single ranked list.

This eliminates the need for post-retrieval re-ranking on structured data and enables semantic search over mixed data types (text + numbers + categories) natively in PostgreSQL.

---

## Concept: What is a Space?

A **Space** defines one embedding channel on a document. Each space has:

- A **name** (used as a column name: `embedding_{name}`)
- A **field** that maps to a key in `document.metadata` or `document.page_content`
- A **weight** that controls its relative influence during search
- An **encoder** that converts the field value into a dense vector

```python
from pgvectordb.spaces import TextSpace, NumberSpace, CategorySpace, RecencySpace
```

### All Available Spaces

| Space | Input Type | Best For |
|-------|-----------|---------|
| `TextSpace` | `str` (text content) | Semantic text similarity |
| `NumberSpace` | `float` / `int` | Numeric fields like price, rating, views |
| `CategorySpace` | `str` (category label) | Categorical fields like city, product type |
| `RecencySpace` | `datetime` / Unix timestamp | Time-based relevance (newer = better) |

---

## Step 1: Define Your Spaces

### `TextSpace`

Encodes text using your configured embedding model. If `dimensions=0`, it auto-detects from the model at `register_spaces()` time.

```python
from pgvectordb.spaces import TextSpace

text_space = TextSpace(
    name="description",   # Column: embedding_description
    field="content",      # Maps to document.page_content
    dimensions=384,       # Or 0 to auto-detect
    weight=1.0
)
```

### `NumberSpace`

Encodes numeric values into a vector using a configurable mode:

```python
from pgvectordb.spaces import NumberSpace, NumberMode

price_space = NumberSpace(
    name="price",
    field="price",           # Key in document.metadata
    min_value=0,
    max_value=1_000_000,
    mode=NumberMode.NORMALIZED,   # See table below
    weight=0.3
)
```

| `NumberMode` | Behavior |
|-------------|----------|
| `NORMALIZED` | Linearly scales value to [0, 1] range |
| `MINIMUM` | Encodes "smaller is better" preference |
| `MAXIMUM` | Encodes "larger is better" preference |

### `CategorySpace`

Encodes categorical labels as one-hot-style dense vectors:

```python
from pgvectordb.spaces import CategorySpace

city_space = CategorySpace(
    name="city",
    field="city",                    # Key in document.metadata
    categories=["NYC", "LA", "Chicago", "Houston"],
    weight=0.2
)
```

### `RecencySpace`

Encodes timestamps so more recent documents score higher:

```python
from pgvectordb.spaces import RecencySpace, TimeUnit

recency_space = RecencySpace(
    name="recency",
    field="published_at",    # Key in document.metadata (datetime or Unix timestamp)
    time_unit=TimeUnit.DAYS,
    weight=0.1
)
```

| `TimeUnit` | Granularity |
|-----------|-------------|
| `SECONDS` | Fine-grained |
| `MINUTES` | Medium |
| `HOURS` | Medium-coarse |
| `DAYS` | Coarse (most common) |

---

## Step 2: Register Spaces

Call `register_spaces()` on your initialized `pgVectorDB` instance:

```python
from pgvectordb import pgVectorDB
from pgvectordb.spaces import TextSpace, NumberSpace, CategorySpace

db = pgVectorDB(
    collection_name="real_estate",
    embedding_model=embeddings,
    connection_string="postgresql+asyncpg://user:pass@localhost/db"
)
await db.initialize()

spaces = [
    TextSpace(name="description", field="content", weight=0.5),
    NumberSpace(name="price",   field="price",   min_value=0, max_value=5_000_000, weight=0.3),
    CategorySpace(name="city",  field="city",    categories=["NYC", "LA", "Chicago"], weight=0.2),
]
db.register_spaces(spaces)
```

!!! note
    `register_spaces()` is synchronous. It validates the space list and auto-detects `TextSpace` dimensions from the embedding model.

---

## Step 3: Add Documents

Use `add_documents_multimodal()` instead of `add_documents()`. It automatically extracts the right field for each space and populates all embedding columns.

```python
from langchain_core.documents import Document

docs = [
    Document(
        page_content="Spacious 2BR with skyline views, modern kitchen",
        metadata={"price": 850_000, "city": "NYC", "bedrooms": 2}
    ),
    Document(
        page_content="Cozy studio in the heart of downtown",
        metadata={"price": 320_000, "city": "Chicago", "bedrooms": 0}
    ),
    Document(
        page_content="Luxury penthouse, private terrace, doorman building",
        metadata={"price": 3_200_000, "city": "NYC", "bedrooms": 4}
    ),
]

doc_ids = await db.add_documents_multimodal(docs, batch_size=100, show_progress=True)
print(f"Added {len(doc_ids)} documents")
```

Under the hood, this creates an `embedding_{name}` column for each space (if it doesn't exist) and populates all columns in a single `INSERT ... ON CONFLICT DO UPDATE`.

---

## Step 4: Build Multimodal Indexes

Create a vector index on each space's embedding column:

```python
from pgvectordb import DistanceMetric

index_map = await db.build_multimodal_index(
    metric=DistanceMetric.COSINE,
    m=16,
    ef_construction=64
)

for space_name, index_name in index_map.items():
    print(f"  {space_name}: {index_name}")
# description: idx_real_estate_description
# price:       idx_real_estate_price
# city:        idx_real_estate_city
```

---

## Step 5: Search Across All Spaces

### `multimodal_search`

The core method. Query each space independently and fuse results by weighted distance:

```python
results = await db.multimodal_search(
    query_params={
        "description": "modern downtown apartment with views",  # TextSpace
        "price": 500_000,                                        # NumberSpace
        "city": "NYC",                                           # CategorySpace
    },
    weights={
        "description": 0.5,
        "price":       0.3,
        "city":        0.2,
    },
    k=10,
    filter={"bedrooms": {"$gte": 1}},   # Optional pre-filter
)

for r in results:
    print(f"Score: {r['score']:.4f} | {r['content'][:60]}")
    print(f"  Price: ${r['metadata'].get('price'):,} | City: {r['metadata'].get('city')}")
```

!!! tip "Weight Strategy"
    Weights are automatically normalized to sum to 1.0. Start with equal weights, then adjust based on evaluation results. Use `RAGEvaluator` to measure the impact of weight changes.

### `multimodal_hybrid_search`

Fuses multimodal vector search with BM25/FTS keyword search for even richer relevance:

```python
results = await db.multimodal_hybrid_search(
    query_params={
        "description": "cozy apartment near park",
        "price": 350_000,
    },
    weights={"description": 0.7, "price": 0.3},
    keyword_weight=0.25,   # 25% keyword, 75% multimodal vector
    k=10,
)
```

---

## Full End-to-End Example

```python
import asyncio
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from pgvectordb import pgVectorDB, DistanceMetric
from pgvectordb.spaces import TextSpace, NumberSpace, CategorySpace

async def main():
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    db = pgVectorDB(
        collection_name="listings",
        embedding_model=embeddings,
        connection_string="postgresql+asyncpg://user:pass@localhost/db"
    )
    await db.initialize()

    # 1. Register spaces
    db.register_spaces([
        TextSpace(name="description", field="content",  weight=0.5),
        NumberSpace(name="price",     field="price",
                    min_value=0, max_value=5_000_000,   weight=0.3),
        CategorySpace(name="city",    field="city",
                      categories=["NYC", "LA", "Chicago"], weight=0.2),
    ])

    # 2. Add documents
    docs = [
        Document(page_content="Modern 2BR, open floor plan",
                 metadata={"price": 750_000, "city": "NYC"}),
        Document(page_content="Cozy studio, great light",
                 metadata={"price": 280_000, "city": "Chicago"}),
        Document(page_content="Luxury penthouse with terrace",
                 metadata={"price": 4_500_000, "city": "NYC"}),
    ]
    await db.add_documents_multimodal(docs)

    # 3. Build indexes
    await db.build_multimodal_index(metric=DistanceMetric.COSINE)

    # 4. Search
    results = await db.multimodal_search(
        query_params={"description": "bright open apartment", "price": 600_000, "city": "NYC"},
        weights={"description": 0.5, "price": 0.3, "city": 0.2},
        k=5
    )

    for r in results:
        print(f"{r['score']:.4f} | {r['content']}")

asyncio.run(main())
```

---

## Monitoring Multimodal Indexes

```python
stats = await db.get_multimodal_index_stats()

for space_name, info in stats.items():
    print(f"\n{space_name}:")
    print(f"  Column:     {info['column']}")
    print(f"  Dimensions: {info['dimensions']}")
    print(f"  Index:      {info['index_name']} ({'exists' if info['index_exists'] else 'missing'})")
    print(f"  Size:       {info['index_size']}")
```
