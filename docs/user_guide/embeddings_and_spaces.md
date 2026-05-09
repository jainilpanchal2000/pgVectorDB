# Embeddings & Spaces

pgVectorDB introduces the concept of **Spaces** to handle multi-embedding and multimodal RAG. Instead of being restricted to one vector per document, you can configure multiple Spaces — each encoding a different signal (text, price, category, recency) into its own vector column.

For the full multimodal search pipeline, see [Multimodal Search](multimodal_search.md).

---

## Space Types Overview

| Space | Import | Input | Best For |
|-------|--------|-------|---------|
| `TextSpace` | `pgvectordb.spaces` | Text string | Semantic text similarity |
| `VectorSpace` | `pgvectordb.spaces` | Raw float list | Pre-computed embeddings (images, audio) |
| `NumberSpace` | `pgvectordb.spaces` | `float` / `int` | Numeric fields (price, rating, views) |
| `CategorySpace` | `pgvectordb.spaces` | Category string | Categorical fields (city, product type) |
| `RecencySpace` | `pgvectordb.spaces` | `datetime` / timestamp | Time-based relevance |

---

## `TextSpace`

Encodes text using the `pgVectorDB` embedding model. If `dimensions=0`, the dimension count is auto-detected from the model at `register_spaces()` time.

```python
from pgvectordb.spaces import TextSpace

text_space = TextSpace(
    name="description",   # Creates column: embedding_description
    field="content",      # "content" maps to document.page_content; any other value maps to metadata
    dimensions=384,       # Set to 0 for auto-detection
    weight=1.0
)
```

---

## `VectorSpace`

Accepts arbitrary pre-computed dense vectors. Use for image embeddings (CLIP), audio embeddings, or any external embedding pipeline.

```python
from pgvectordb.spaces import VectorSpace

image_space = VectorSpace(
    name="image_clip",    # Creates column: embedding_image_clip
    dimensions=512,       # Must match your model's output size
    weight=1.5            # Higher weight = more influence in fusion
)
```

**Inserting with pre-computed vectors:**

```python
# Compute externally
image_embeddings = clip_model.embed_images(["shoe.jpg", "jacket.jpg"])

multi_embeddings = [
    {"text_desc": text_embeddings[0], "image_clip": image_embeddings[0]},
    {"text_desc": text_embeddings[1], "image_clip": image_embeddings[1]},
]

await db.add_embeddings(
    texts=["Red running shoe", "Blue jacket"],
    embeddings=multi_embeddings
)
```

---

## `NumberSpace`

Encodes numeric metadata fields (price, rating, age, views) into a vector representation using a configurable mode.

```python
from pgvectordb.spaces import NumberSpace, NumberMode

price_space = NumberSpace(
    name="price",
    field="price",            # Key in document.metadata
    min_value=0,
    max_value=1_000_000,
    mode=NumberMode.NORMALIZED,
    weight=0.3
)
```

### `NumberMode` Values

| Mode | Behavior | Use When |
|------|----------|----------|
| `NORMALIZED` | Linearly scales to [0, 1] | You want balanced numeric similarity |
| `MINIMUM` | Encodes "smaller is better" bias | Price (cheaper is better) |
| `MAXIMUM` | Encodes "larger is better" bias | Rating, views (higher is better) |

```python
# Price: cheaper is better
price_space = NumberSpace(
    name="price", field="price",
    min_value=0, max_value=5_000_000,
    mode=NumberMode.MINIMUM,   # Prefers lower prices
    weight=0.3
)

# Rating: higher is better
rating_space = NumberSpace(
    name="rating", field="rating",
    min_value=1, max_value=5,
    mode=NumberMode.MAXIMUM,   # Prefers higher ratings
    weight=0.2
)
```

---

## `CategorySpace`

Encodes categorical string labels as dense vectors using one-hot-style encoding. Documents with matching categories will have lower distance between them.

```python
from pgvectordb.spaces import CategorySpace

city_space = CategorySpace(
    name="city",
    field="city",                                 # Key in document.metadata
    categories=["NYC", "LA", "Chicago", "Houston"],
    weight=0.2
)
```

!!! note
    If a document's category value is not in the `categories` list, it is treated as an unknown category. Extend the list when you add new categories.

---

## `RecencySpace`

Encodes timestamps so that more recent documents receive higher similarity scores. Useful for news, feed ranking, or any freshness-sensitive search.

```python
from pgvectordb.spaces import RecencySpace, TimeUnit

recency_space = RecencySpace(
    name="recency",
    field="published_at",    # Key in document.metadata (datetime or Unix timestamp)
    time_unit=TimeUnit.DAYS,
    weight=0.15
)
```

### `TimeUnit` Values

| `TimeUnit` | Granularity | Best For |
|-----------|-------------|---------|
| `SECONDS` | Very fine | Real-time feeds |
| `MINUTES` | Fine | Intraday content |
| `HOURS` | Medium | Hourly updates |
| `DAYS` | Coarse | News, blog posts |

---

## Using Multiple Spaces Together

### Initialization with Spaces

```python
from pgvectordb import pgVectorDB
from pgvectordb.spaces import TextSpace, NumberSpace, CategorySpace

db = pgVectorDB(
    collection_name="products",
    embedding_model=my_embeddings,
    connection_string="postgresql+asyncpg://user:pass@localhost/db"
)
await db.initialize()

spaces = [
    TextSpace(name="description", field="content",   weight=0.5),
    NumberSpace(name="price",     field="price",
                min_value=0, max_value=1_000_000,    weight=0.3),
    CategorySpace(name="category", field="category",
                  categories=["Electronics", "Clothing", "Home"], weight=0.2),
]
db.register_spaces(spaces)
```

### Inserting Documents

```python
from langchain_core.documents import Document

docs = [
    Document(
        page_content="Wireless noise-cancelling headphones",
        metadata={"price": 299.99, "category": "Electronics"}
    ),
    Document(
        page_content="Slim-fit cotton dress shirt",
        metadata={"price": 79.99, "category": "Clothing"}
    ),
]

await db.add_documents_multimodal(docs)
```

### Searching

```python
results = await db.multimodal_search(
    query_params={
        "description": "headphones for working from home",
        "price": 200.0,
        "category": "Electronics",
    },
    weights={"description": 0.5, "price": 0.3, "category": 0.2},
    k=5
)

for r in results:
    print(f"{r['score']:.4f} | {r['content']}")
```

See [Multimodal Search](multimodal_search.md) for the complete pipeline including index creation and monitoring.

---

## Weight Tuning Tips

Weights control how much influence each space has on the final ranking. They are **normalized** internally so they don't need to sum to 1.0.

```python
# Text-dominant: semantic similarity drives ranking
weights = {"description": 0.7, "price": 0.2, "category": 0.1}

# Price-dominant: use for e-commerce filtered browsing
weights = {"description": 0.3, "price": 0.6, "category": 0.1}

# Balanced: equal contribution from all signals
weights = {"description": 0.33, "price": 0.33, "category": 0.34}
```

!!! tip "Use `RAGEvaluator` to find optimal weights"
    Define a ground-truth dataset and use `metrics.py`'s `RAGEvaluator` to compare different weight configurations by Hit Rate, MRR, and NDCG. See [Metrics & Evaluation](metrics_and_evaluation.md).
