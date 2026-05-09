# Search & Retrieval: The Developer's Deep Dive

pgVectorDB implements the `SearchMixin`, providing an unparalleled **10 distinct search methods**. This guide is an exhaustive, developer-grade deep dive into how each method functions, the underlying SQL generated, exact parameter definitions, and when to use them in production architectures.

!!! note
    **Score Interpretation**: Semantic distance (Cosine) means *lower is better*. BM25, FTS rank, and RRF fusion scores mean *higher is better*.

---

## The Return Type: `QueryResult`

All search methods (unless otherwise specified) return a `List[QueryResult]`. The `QueryResult` is a TypedDict containing:

| Field | Type | Description |
|-------|------|-------------|
| `id` | str | The internal LangChain UUID of the document |
| `content` | str | The raw text chunk |
| `metadata` | dict | The associated JSONB metadata |
| `score` | float | The relevance score |

```python
from typing import List
from pgvectordb import QueryResult

results: List[QueryResult] = await db.semantic_search("query", k=5)

for doc in results:
    print(f"ID: {doc['id']}")
    print(f"Content: {doc['content'][:100]}...")
    print(f"Score: {doc['score']:.4f}")
    print(f"Metadata: {doc['metadata']}")
```

---

## 1. `keyword_search`
Performs a pure text search on the `content` column without utilizing vector embeddings.

### Under the Hood
Depending on `search_type`:
- `KeywordSearchType.FTS`: Uses PostgreSQL's built-in `ts_rank` and `plainto_tsquery`. It leverages the `content_tsvector` column and the GIN index created during `initialize()`.
- `KeywordSearchType.BM25`: Uses the custom `<@>` operator provided by the `pg_textsearch` extension. This calculates true BM25 scores (returned as negative values in SQL, which pgVectorDB normalizes).

### Signature & Parameters

```python
async def keyword_search(
    query: str,
    k: int = 4,
    filter: Optional[Dict[str, Any]] = None,
    search_type: KeywordSearchType = KeywordSearchType.FTS,
    k1: float = 1.2,           # BM25 parameter (Term frequency saturation)
    b: float = 0.75,           # BM25 parameter (Length normalization)
    text_config: str = "english" # Dictionary for stemming
) -> List[QueryResult]

```

### Usage

```python
results = await db.keyword_search("database replication", k=10, search_type=KeywordSearchType.BM25)

```

---

## 2. `universal_keyword_search`
A highly specialized FTS/BM25 search that boosts the final score if the query terms are also found inside specific JSONB `metadata` fields.

### Under the Hood
It dynamically constructs a `CASE WHEN` SQL clause. For every field specified in `metadata_fields`, it performs an `ILIKE :like_query`. If true, it adds `1.0` to the base FTS/BM25 score.

### Usage

```python
# Finds documents about "scaling", but heavily favors documents where 
# "scaling" is in the 'title' or 'category' tags.
results = await db.universal_keyword_search(
    query="scaling",
    k=5,
    metadata_fields=["title", "category"] 
)

```

---

## 3. `semantic_search`
The core dense vector retrieval method. 

### Under the Hood
Calculates the embedding of the query using your `embedding_model`, then executes an Approximate Nearest Neighbor (ANN) search using the `<=>` operator (Cosine Distance) against the `embedding` column.

### Parameters
- `use_exact_search` (bool): If set to `True`, pgVectorDB executes `SET LOCAL enable_indexscan = off` immediately before the query. This forces PostgreSQL to perform a sequential scan, bypassing HNSW/IVFFlat for 100% accurate recall at the cost of high latency.

### Usage

```python
results = await db.semantic_search(
    query="How to migrate to the cloud?",
    k=5,
    use_exact_search=False
)

```

---

## 4. `metadata_filter`
A strict filtering operation returning documents without any text or vector search involved.

### Under the Hood
Translates your JSON dictionary into raw SQL `WHERE` clauses against the `langchain_metadata` JSONB column. All returned documents are assigned a dummy score of `1.0`.

### Parameters
- `order_by` (str): Specific metadata key to order the results by (e.g., "created_at").
- `ascending` (bool): Sort direction.

### Usage

```python
results = await db.metadata_filter(
    filter={"status": "published", "author": "admin"},
    k=10,
    order_by="published_date",
    ascending=False
)

```

---

## 5. `metadata_keyword_search`
Applies strict metadata filtering *before* executing keyword search. 

### Under the Hood
Constructs a Common Table Expression (CTE) to pre-filter the table using your `filter` dict. The `ts_rank` / BM25 search is then executed *only* on the smaller CTE result set.

```python
results = await db.metadata_keyword_search(
    query="connection pooling",
    filter={"component": "database", "severity": {"$gt": 2}},
    k=5
)

```

---

## 6. `metadata_semantic_search`
Applies strict metadata filtering *before* executing vector search. This is the foundation of multi-tenant RAG systems.

### Under the Hood
Appends the filter clauses directly into the `WHERE` clause alongside the vector `<=>` operator. The HNSW index handles this pre-filtering efficiently.

```python
results = await db.metadata_semantic_search(
    query="Reset password",
    filter={"tenant_id": "org_555"}, # Essential for data isolation
    k=3
)

```

---

## 7. `hybrid_search`
Combines `keyword_search` and `semantic_search` into a single ranked list.

### Under the Hood (Fusion Mechanisms)
pgVectorDB executes both queries asynchronously. It then merges the results using one of two algorithms:
1. **Reciprocal Rank Fusion (RRF)**: Used by default. Ranks are converted to scores via `1 / (rrf_k + rank)`. Excellent because it doesn't require vector distances and TF-IDF scores to be normalized to the same scale.
2. **Weighted Fusion**: Used if the `weights` parameter (tuple of 2 floats summing to 1.0) is provided. It normalizes both sets of scores to a `0-1` range (inverting semantic distances so 1 is best) and applies the weights.

### Parameters
- `rrf_k` (int): Smoothing constant for RRF. Default is `60`.
- `weights` (Tuple[float, float], optional): e.g., `(0.7, 0.3)` means 70% semantic, 30% keyword.

### Usage

```python
results = await db.hybrid_search(
    query="Machine learning performance",
    k=5,
    keyword_type=KeywordSearchType.BM25,
    rrf_k=60
)

```

---

## 8. `ensemble_search`
The ultimate search method. It performs `metadata_filter` to isolate a subset of documents, and then performs a `hybrid_search` (with RRF or weights) on that exact subset.

### Usage

```python
results = await db.ensemble_search(
    query="Financial reports Q3",
    filter={"department": "finance", "year": 2023},
    k=10,
    weights=(0.8, 0.2)
)

```

---

## 9. `trigram_search`
Fuzzy text search highly tolerant to spelling mistakes. 

### Under the Hood
Utilizes the `pg_trgm` extension. It uses the `gin_trgm_ops` index created during `initialize()`. It calculates the similarity between the query string and document content by counting matching 3-letter combinations (trigrams).

### Usage

```python
# Will match documents containing "PostgreSQL"
results = await db.trigram_search(
    query="PostgreSQLLLL", 
    k=5
)

```

---

## 10. `metadata_trigram_search`
Combines metadata filtering with fuzzy trigram matching. Uses a CTE for pre-filtering.

```python
results = await db.metadata_trigram_search(
    query="useer auth", # Typo "useer"
    filter={"category": "documentation"},
    k=5
)

```

---

## Additional Search Methods

### 11. `asimilarity_search_by_vector`
Search using pre-computed embeddings instead of generating from a text query. Useful when you've already embedded documents externally or want to compare embedding models.

```python
# Use the same embedding model to generate embeddings
query_embedding = embedding_model.embed_query("machine learning foundations")

results = await db.asimilarity_search_by_vector(
    embedding=query_embedding,
    k=5,
    label_filter=[1, 2]  # Optional: DiskANN label filtering
)
```

!!! tip
    Use this when comparing embedding models - generate embeddings once and reuse them.

### 12. `asimilarity_search_with_score`
Returns `(QueryResult, float)` tuples instead of a plain list. Useful when you need raw scores for custom processing.

```python
results = await db.asimilarity_search_with_score(
    query="database optimization",
    k=5
)

for doc, raw_score in results:
    print(f"Doc: {doc['content'][:50]}")
    print(f"Raw Score: {raw_score:.4f}")
```
