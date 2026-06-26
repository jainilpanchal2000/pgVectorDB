# Multimodal Search

Multimodal search lets one document have multiple embedding spaces: text for meaning, numbers for structured preferences, categories for exact business dimensions, and recency for freshness. pgVectorDB searches those spaces together and fuses the scores in PostgreSQL.

This is useful when pure semantic search is not enough. A real-estate query such as “fresh waterfront homes under 900k near transit” has text intent, price constraints, location preference, and freshness. A support-search query may need text meaning, product category, ticket severity, and recent updates.

## Space Types

| Space | Input | Typical fields | Ranking signal |
| --- | --- | --- | --- |
| `TextSpace` | Text | `content`, `title`, `description` | Semantic similarity from the embedding model. |
| `NumberSpace` | int/float | `price`, `rating`, `mileage`, `sqft` | Lower, higher, or nearest numeric preference. |
| `CategorySpace` | string | `city`, `product_type`, `department` | Category match/similarity. |
| `RecencySpace` | datetime, ISO string, Unix timestamp | `published_at`, `updated_at`, `created_at` | Exponential time decay where fresher values score higher. |

## Define Spaces

```python
from pgvectordb.spaces import (
    CategorySpace,
    NumberMode,
    NumberSpace,
    RecencySpace,
    TextSpace,
    TimeUnit,
)

spaces = [
    TextSpace(name="description", field="content"),
    NumberSpace(
        name="price",
        field="price",
        min_value=0,
        max_value=2_000_000,
        mode=NumberMode.MINIMUM,
    ),
    CategorySpace(
        name="city",
        field="city",
        categories=["Austin", "Denver", "Seattle", "Chicago"],
    ),
    RecencySpace(
        name="freshness",
        field="published_at",
        time_unit=TimeUnit.DAY,
        period_value=14,
    ),
]

db.register_spaces(spaces)
```

`register_spaces()` is synchronous. It validates names, detects dimensions for text spaces when needed, and prepares the collection for multimodal ingestion.

## NumberSpace Modes

| Mode | Use when | Example |
| --- | --- | --- |
| `NumberMode.MINIMUM` | Lower values should rank better. | Price, latency, distance, mileage. |
| `NumberMode.MAXIMUM` | Higher values should rank better. | Rating, popularity, margin, quality score. |
| `NumberMode.SIMILAR` | Values close to the query should rank better. | Bedrooms, temperature, target budget, square footage. |

## RecencySpace

`RecencySpace` turns freshness into a first-class ranking signal. It uses exponential decay:

$$
score = e^{-age / \tau}
$$

where $\tau$ is `period_value * time_unit`. If `period_value=14` and `time_unit=TimeUnit.DAY`, a document from today scores near 1.0, a document around 14 days old scores near 0.37, and a document around 42 days old scores near 0.05.

```python
freshness = RecencySpace(
    name="freshness",
    field="published_at",
    time_unit=TimeUnit.DAY,
    period_value=14,
)
```

Use it for news, listings, tickets, documentation, changelogs, events, promotions, and any domain where “recent” should matter but should not be the only ranking factor.

!!! note "Re-encoding freshness"
    Recency embeddings are computed relative to the wall-clock time when encoded. For long-lived collections, refresh recency embeddings periodically or choose a decay period that matches your update cadence.

## Add Multimodal Documents

Use `add_documents_multimodal()` so pgVectorDB extracts each field and writes one embedding column per space.

```python
from langchain_core.documents import Document

docs = [
    Document(
        page_content="Updated bungalow near the lake with renovated kitchen",
        metadata={
            "price": 685_000,
            "city": "Austin",
            "published_at": "2026-06-20T09:00:00Z",
            "bedrooms": 3,
        },
    ),
    Document(
        page_content="Downtown condo near transit and restaurants",
        metadata={
            "price": 520_000,
            "city": "Denver",
            "published_at": "2026-05-15T12:00:00Z",
            "bedrooms": 2,
        },
    ),
]

ids = await db.add_documents_multimodal(docs, batch_size=100)
```

## Build Multimodal Indexes

Each space gets its own vector column and index.

```python
from pgvectordb import DistanceMetric

index_map = await db.build_multimodal_index(
    metric=DistanceMetric.COSINE,
    m=16,
    ef_construction=64,
)

print(index_map)
```

## Search Across Spaces with the Fluent API

Use `.across_spaces(...)` when you want multimodal ranking to live beside the rest of your query code: filters, limits, output formats, analysis, and reranking.

```python
results = await (
    db.query("fresh waterfront home near transit")
    .across_spaces(
        spaces,
        weights={
            "description": 0.70,
            "freshness": 0.30,
        },
    )
    .where({"city": "Austin", "bedrooms": {"$gte": 2}})
    .limit(10)
    .to_list()
)
```

Use `.in_space(space)` when you want to target one registered space explicitly.

```python
results = await (
    db.query("lakefront bungalow")
    .in_space(spaces[0])
    .where({"city": "Austin"})
    .limit(10)
    .to_list()
)
```

The fluent path is best when the text query is the primary input and metadata filters carry the structured constraints.

## Direct Per-Space Query Values

Use `multimodal_search()` when you need to pass explicit query values for non-text spaces such as price, category, or freshness. Weights control how much each signal contributes to the fused ranking.

```python
results = await db.multimodal_search(
    query_params={
        "description": "fresh waterfront home near transit",
        "price": 750_000,
        "city": "Austin",
        "freshness": "2026-06-25T00:00:00Z",
    },
    weights={
        "description": 0.55,
        "price": 0.20,
        "city": 0.15,
        "freshness": 0.10,
    },
    filter={"bedrooms": {"$gte": 2}},
    k=10,
)
```

Weighting is a product decision, not a magic constant. Start with text-heavy weights, then use `RAGEvaluator` or `compute_recall()` with real queries to validate changes.

| Workload | Suggested starting weights |
| --- | --- |
| Product search | Description 0.60, category 0.20, price 0.15, recency 0.05. |
| Real estate | Description 0.50, price 0.20, city 0.15, recency 0.15. |
| Support tickets | Text 0.55, product/category 0.20, severity 0.15, recency 0.10. |
| News/search feeds | Text 0.45, category 0.15, recency 0.40. |

## Direct Multimodal Hybrid Search

`multimodal_hybrid_search()` fuses multimodal vector ranking with keyword search.

```python
results = await db.multimodal_hybrid_search(
    query_params={
        "description": "renovated home near light rail",
        "price": 800_000,
        "freshness": "2026-06-25T00:00:00Z",
    },
    weights={"description": 0.65, "price": 0.20, "freshness": 0.15},
    keyword_weight=0.25,
    filter={"city": "Austin"},
    k=10,
)
```

Use this when text terms carry important exact meaning: model numbers, neighborhoods, SKUs, policy names, procedure names, and acronyms.

## Monitoring Spaces

```python
stats = await db.get_multimodal_index_stats()

for space_name, info in stats.items():
    print(space_name)
    print(info["column"])
    print(info["dimensions"])
    print(info["index_name"])
    print(info["index_exists"])
```

Run this after registering spaces, adding data, and building indexes to confirm that each space has a populated column and index.

## Complete Example

```python
import asyncio

from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

from pgvectordb import DistanceMetric, pgVectorDB
from pgvectordb.spaces import CategorySpace, NumberMode, NumberSpace, RecencySpace, TextSpace, TimeUnit


async def main():
    db = pgVectorDB(
        collection_name="listings",
        embedding_model=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2"),
        connection_string="postgresql+asyncpg://user:root@localhost:9002/postgres",
    )
    await db.initialize()

    spaces = [
        TextSpace(name="description", field="content"),
        NumberSpace(name="price", field="price", min_value=0, max_value=2_000_000, mode=NumberMode.MINIMUM),
        CategorySpace(name="city", field="city", categories=["Austin", "Denver", "Seattle"]),
        RecencySpace(name="freshness", field="published_at", time_unit=TimeUnit.DAY, period_value=14),
    ]
    db.register_spaces(spaces)

    await db.add_documents_multimodal([
        Document(
            page_content="Renovated bungalow near the lake",
            metadata={"price": 685_000, "city": "Austin", "published_at": "2026-06-20T09:00:00Z"},
        ),
        Document(
            page_content="Downtown condo near transit",
            metadata={"price": 520_000, "city": "Denver", "published_at": "2026-05-15T12:00:00Z"},
        ),
    ])

    await db.build_multimodal_index(metric=DistanceMetric.COSINE)

    results = await db.multimodal_search(
        query_params={
            "description": "fresh home near water",
            "price": 750_000,
            "city": "Austin",
            "freshness": "2026-06-25T00:00:00Z",
        },
        weights={"description": 0.55, "price": 0.20, "city": 0.15, "freshness": 0.10},
        k=5,
    )

    for row in results:
        print(row["score"], row["content"])

    await db.close()


asyncio.run(main())
```

## Related Guides

- [Embeddings & Spaces](embeddings_and_spaces.md)
- [Search & Retrieval](search_and_retrieval.md)
- [Metadata Filtering](filtering.md)
- [Metrics & Evaluation](metrics_and_evaluation.md)
- [Indexing & Performance](../advanced/indexing.md)