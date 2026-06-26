# Reranking

Reranking is the second stage of a high-precision retrieval pipeline. First, pgVectorDB uses PostgreSQL, `pgvector`, BM25/FTS, hybrid fusion, or multimodal spaces to retrieve candidates quickly. Then a stronger reranker scores each query/document pair together and reorders the candidate set.

Use it when the first page of results matters more than raw retrieval latency: answer generation, support search, legal search, product search, recommendations, and any workflow where a plausible-but-wrong top result is expensive.

```text
query -> first-stage retrieval -> candidate set -> reranker -> best ordered results
```

## Fluent Reranking

The fluent API keeps reranking inside the same query chain you use for semantic, keyword, hybrid, and filtered search.

```python
from pgvectordb.rerankers import CrossEncoderReranker

reranker = CrossEncoderReranker(model="cross-encoder/ms-marco-MiniLM-L-6-v2")

reranked_candidates = await (
    db.query("best noise-cancelling headphones under $200")
    .hybrid()
    .where({"category": "audio"})
    .rrf(k=60)
    .limit(100)
    .rerank(reranker)
    .to_list()
)

results = reranked_candidates[:5]
```

In fluent queries, `.limit(100)` controls how many first-stage candidates are retrieved and reranked. Slice the returned list when you want to show fewer final results.

| Value | Controls | Starting point |
| --- | --- | --- |
| `.limit(candidate_count)` | How many retrieved rows reach the reranker. | 50 for fast local reranking, 100 for balanced RAG, 200+ for recall-sensitive search. |
| `results[:final_count]` | How many reranked rows your app displays or sends to an LLM. | 5 to 10 for answer generation, 10 to 20 for search results. |

## Choose the First Stage

Reranking improves ordering; it does not replace recall. Pick the first-stage retriever that brings the right documents into the candidate set.

=== "Hybrid RAG"

    ```python
    reranked = await (
        db.query("database backup retention policy")
        .hybrid()
        .weights(semantic=0.7, keyword=0.3)
        .limit(100)
        .rerank(reranker)
        .to_list()
    )
    ```

=== "Semantic"

    ```python
    reranked = await (
        db.query("how to prevent postgres deadlocks")
        .semantic()
        .ef(100)
        .limit(100)
        .rerank(reranker)
        .to_list()
    )
    ```

=== "Keyword / BM25"

    ```python
    reranked = await (
        db.query("SOC 2 encryption retention")
        .keyword()
        .bm25_params(k1=1.2, b=0.75)
        .limit(100)
        .rerank(reranker)
        .to_list()
    )
    ```

=== "Multimodal"

    ```python
    reranked = await (
        db.query("fresh waterfront home near transit")
        .across_spaces(spaces, weights={"description": 0.7, "freshness": 0.3})
        .where({"city": "Austin"})
        .limit(100)
        .rerank(reranker)
        .to_list()
    )
    ```

## Reranker Backends

| Reranker | Use when | Requires |
| --- | --- | --- |
| `CrossEncoderReranker` | You want local reranking with no API calls. | `pip install "pgvectordb[rerankers]"` |
| `HuggingFaceReranker` | You want a local transformer reranker such as BGE or Jina. | `pip install "pgvectordb[rerankers]"` |
| `CohereReranker` | You want a managed low-latency reranking API. | `pip install "pgvectordb[cohere]"` and `COHERE_API_KEY` |
| `AWSBedrockReranker` | You want managed reranking through AWS credentials and region controls. | `pip install "pgvectordb[aws]"` |

### Local Cross-Encoder

```python
from pgvectordb.rerankers import CrossEncoderReranker

reranker = CrossEncoderReranker(
    model="cross-encoder/ms-marco-MiniLM-L-6-v2",
    batch_size=32,
)
```

This is a strong default for prototypes and private datasets. CPU is enough for small candidate sets; GPU helps when reranking many queries or large candidate windows.

### Local HuggingFace Model

```python
from pgvectordb.rerankers import HuggingFaceReranker

reranker = HuggingFaceReranker(
    model="BAAI/bge-reranker-v2-m3",
    device="cuda",
)
```

Use this when you want a specific open model, multilingual reranking, or local control over inference.

### Cohere

```python
from pgvectordb.rerankers import CohereReranker

reranker = CohereReranker(model="rerank-english-v3.0")
```

Set `COHERE_API_KEY` in the environment or pass `api_key=...` to the constructor.

### AWS Bedrock

```python
from pgvectordb.rerankers import AWSBedrockReranker

reranker = AWSBedrockReranker(
    model_id="cohere.rerank-v3-5:0",
    region_name="us-east-1",
)
```

Use Bedrock when AWS identity, network controls, or regional deployment matter.

## Factory Configuration

Use `create_reranker()` when the backend comes from configuration.

```python
from pgvectordb.rerankers import create_reranker

reranker = create_reranker(
    "cohere",
    model="rerank-english-v3.0",
)

reranked = await (
    db.query("postgres memory tuning")
    .hybrid()
    .limit(100)
    .rerank(reranker)
    .to_list()
)
```

## Compatibility Helper

`rerank_search()` remains available for code that wants one call with separate candidate and final-result counts.

```python
results = await db.rerank_search(
    query="best noise-cancelling headphones under $200",
    reranker=reranker,
    search_method="hybrid",
    k=100,
    rerank_top_k=5,
)
```

Use the fluent API when you are already composing search with `.where(...)`, `.rrf(...)`, `.ef(...)`, `.across_spaces(...)`, and output methods. Use `rerank_search()` when migrating older code or when you specifically want `k` and `rerank_top_k` in a single helper.

| `search_method` | First-stage retrieval |
| --- | --- |
| `"semantic"` | Vector similarity search. |
| `"hybrid"` | Semantic plus keyword fusion. |
| `"keyword"` | PostgreSQL full-text search. |
| `"bm25"` | BM25 ranking with `pg_textsearch`. |
| `"multimodal"` | Direct multimodal search with `query_params` and `weights`. |

## Multimodal + Reranking

For fluent multimodal reranking, pass registered spaces to `.across_spaces(...)` and rerank the resulting candidate set.

```python
reranked = await (
    db.query("modern 2BR apartment downtown with park views")
    .across_spaces(
        spaces,
        weights={"description": 0.55, "price": 0.20, "city": 0.15, "freshness": 0.10},
    )
    .where({"city": "NYC"})
    .limit(100)
    .rerank(reranker)
    .to_list()
)

results = reranked[:10]
```

When you need explicit per-space query values, use the direct helper and pass `query_params` through `rerank_search()`.

```python
results = await db.rerank_search(
    query="modern 2BR apartment downtown with park views",
    reranker=reranker,
    search_method="multimodal",
    k=100,
    rerank_top_k=10,
    query_params={
        "description": "modern 2BR apartment downtown with park views",
        "price": 800_000,
        "city": "NYC",
    },
    weights={"description": 0.55, "price": 0.30, "city": 0.15},
)
```

## Tuning Candidate Counts

| Candidate count | Best for | Tradeoff |
| --- | --- | --- |
| 20 to 50 | Interactive search and local CPU reranking. | Fast, but may miss relevant documents before reranking. |
| 100 | Balanced RAG and support search. | Good starting point for recall and latency. |
| 200 to 500 | High-value search, legal, compliance, or offline evaluation. | Better recall, higher model/API cost. |

Reranking can only reorder candidates it receives. If important documents never reach the candidate set, tune first-stage retrieval first: use hybrid search, increase `.ef(...)` or `.nprobes(...)`, add filters carefully, or measure recall with exact search.

## How It Works

```text
Query
  -> first-stage retrieval with semantic, keyword, hybrid, or multimodal search
  -> candidate QueryResult rows
  -> reranker scores each query/document pair
  -> rows are sorted by rerank score
  -> application uses the top rows
```

## Related Guides

- [Search & Retrieval](search_and_retrieval.md)
- [Multimodal Search](multimodal_search.md)
- [Metadata Filtering](filtering.md)
- [Metrics & Evaluation](metrics_and_evaluation.md)
- [Indexing & Performance](../advanced/indexing.md)
