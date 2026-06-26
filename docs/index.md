# pgVectorDB

Build production retrieval in PostgreSQL: vector search, keyword/BM25 search, metadata filters, multimodal ranking, reranking, evaluation, and SQL diagnostics behind one fluent Python API.

pgVectorDB is for teams that want RAG and application search without moving content, metadata, indexes, and query plans into a separate vector database service. It builds on PostgreSQL, `pgvector`, `pg_trgm`, `vectorscale`, `pg_textsearch`, SQLAlchemy async, and LangChain-compatible documents.

[Get started](getting_started/quickstart.md){ .md-button .md-button--primary }
[Choose a search mode](user_guide/search_and_retrieval.md){ .md-button }
[Tune production search](advanced/indexing.md){ .md-button }

```python
results = await (
    db.query("fresh waterfront homes near transit")
    .hybrid()
    .where({"city": "Austin", "price": {"$between": [400000, 900000]}})
    .weights(semantic=0.7, keyword=0.3)
    .ef(100)
    .limit(10)
    .to_list()
)
```

## Choose Your Path

<div class="grid cards" markdown>

-   **Build your first collection**

    Install the package, connect to PostgreSQL, add LangChain `Document` objects, build an index, and run your first fluent query.

    [Quickstart](getting_started/quickstart.md)

-   **Pick the right retrieval mode**

    Use semantic search for meaning, keyword/BM25 for exact terms, hybrid for RAG defaults, and trigram search for typo-tolerant lookup.

    [Search & Retrieval](user_guide/search_and_retrieval.md)

-   **Add business signals**

    Combine text, price, category, and freshness with `TextSpace`, `NumberSpace`, `CategorySpace`, and `RecencySpace`.

    [Multimodal Search](user_guide/multimodal_search.md)

-   **Improve result order**

    Retrieve candidates cheaply, then rerank the top set with CrossEncoder, Cohere, AWS Bedrock, or HuggingFace.

    [Reranking](user_guide/reranking.md)

-   **Prove quality before tuning**

    Measure Hit Rate, Precision, Recall, F1, MAP, MRR, and NDCG before changing weights, indexes, or rerankers.

    [Metrics & Evaluation](user_guide/metrics_and_evaluation.md)

-   **Debug PostgreSQL behavior**

    Inspect planned searches, execution timing, recall checks, table stats, bloat, and slow-query diagnostics.

    [Analytics & Diagnostics](user_guide/analytics_and_diagnostics.md)

</div>

## Why Teams Use It

| Differentiator | What it gives you | Start here |
| --- | --- | --- |
| Fluent API | One lazy `db.query(...)` builder for semantic, keyword, hybrid, trigram, filters, tuning, reranking, and output formats. | [Quickstart](getting_started/quickstart.md) |
| PostgreSQL-native storage | Keep vectors, content, metadata, indexes, full-text search, and query analysis in the database you already operate. | [Core Concepts](getting_started/core_concepts.md) |
| Multimodal ranking | Blend meaning, numeric preferences, category matches, and freshness signals instead of doing fragile post-processing. | [Multimodal Search](user_guide/multimodal_search.md) |
| Built-in evaluators | Compare retrieval settings with Hit Rate, Precision, Recall, F1, MAP, MRR, and NDCG before shipping changes. | [Metrics & Evaluation](user_guide/metrics_and_evaluation.md) |
| SQL diagnostics | Inspect planned queries, runtime metrics, recall checks, table stats, bloat, and slow-query behavior. | [Analytics & Diagnostics](user_guide/analytics_and_diagnostics.md) |
| Production indexing | Tune HNSW, IVFFlat, DiskANN, scalar indexes, BM25, query params, quantization, and subvector indexing. | [Indexing & Performance](advanced/indexing.md) |
| Reranking backends | Improve final ordering with local CrossEncoder models or managed Cohere, AWS Bedrock, and HuggingFace rerankers. | [Reranking](user_guide/reranking.md) |
| LangChain compatibility | Use pgVectorDB as a retriever or vector store while keeping lower-level PostgreSQL controls available. | [LangChain Integration](user_guide/langchain_integration.md) |

## Common Workflows

| Goal | Guide |
| --- | --- |
| Install PostgreSQL extensions and the Python package | [Installation](getting_started/installation.md) |
| Build a first searchable collection | [Quickstart](getting_started/quickstart.md) |
| Understand documents, metadata, embeddings, filters, indexes, and evaluation | [Core Concepts](getting_started/core_concepts.md) |
| Choose semantic, keyword, hybrid, or fuzzy search | [Search & Retrieval](user_guide/search_and_retrieval.md) |
| Add metadata filters and scalar indexes | [Metadata Filtering](user_guide/filtering.md) |
| Combine text, price, category, and freshness | [Multimodal Search](user_guide/multimodal_search.md) |
| Prove retrieval quality before shipping | [Metrics & Evaluation](user_guide/metrics_and_evaluation.md) |
| Tune latency, recall, and storage | [Indexing & Performance](advanced/indexing.md) |

## Feature Map

| Area | Capabilities |
| --- | --- |
| Search | Semantic, keyword FTS, BM25, hybrid weighted fusion, RRF, trigram fuzzy search, metadata search, ensemble search. |
| Filtering | MongoDB-style JSONB filters: equality, comparison, ranges, lists, existence checks, pattern matching, `$and`, `$or`, `$not`. |
| Indexing | HNSW, IVFFlat, DiskANN, concurrent builds, scalar B-tree/GIN indexes, BM25 indexes, trigram indexes. |
| Optimization | `ef`, `nprobes`, `refine_factor`, exact-search bypass, distance ranges, halfvec, binary quantization, sparse vectors, subvector indexing. |
| Multimodal | `TextSpace`, `NumberSpace`, `CategorySpace`, `RecencySpace`, weighted fusion, multimodal indexes, multimodal hybrid search. |
| Evaluation | `RAGEvaluator`, `EvaluationDataset`, K-value analysis, benchmark scripts, recall measurement. |
| Production | Batch ingestion, isolated batch errors, upserts, deduplication, import/export, table stats, bloat checks, slow-query diagnostics. |

## Examples

| Example | What it demonstrates |
| --- | --- |
| [examples/fluent_api_demo.py](https://github.com/jainilpanchal2000/pgVectorDB/blob/main/examples/fluent_api_demo.py) | Fluent API search modes, filters, and analysis. |
| [examples/06_unified_api_quickstart.ipynb](https://github.com/jainilpanchal2000/pgVectorDB/blob/main/examples/06_unified_api_quickstart.ipynb) | Notebook walkthrough for the unified query API. |
| [examples/03_multimodal_search.ipynb](https://github.com/jainilpanchal2000/pgVectorDB/blob/main/examples/03_multimodal_search.ipynb) | Text, numeric, category, and recency spaces. |
| [examples/05_rag_evaluation.ipynb](https://github.com/jainilpanchal2000/pgVectorDB/blob/main/examples/05_rag_evaluation.ipynb) | Retrieval metrics and RAG evaluation workflow. |
| [examples/product_search.py](https://github.com/jainilpanchal2000/pgVectorDB/blob/main/examples/product_search.py) | Product search patterns with metadata and ranking. |
| [examples/real_estate_nlq.py](https://github.com/jainilpanchal2000/pgVectorDB/blob/main/examples/real_estate_nlq.py) | Natural-language real-estate search with business signals. |

## API Reference

The API reference is generated from source and is best used after you know the workflow you want:

- [pgVectorDB Core](api_reference/pgvectordb.md)
- [Spaces](api_reference/spaces.md)
- [Rerankers](api_reference/rerankers.md)
- [Metrics](api_reference/metrics.md)
- [Configuration](api_reference/config.md)
- [Exceptions](api_reference/exceptions.md)

!!! tip "Start small"
    Begin with the [Quickstart](getting_started/quickstart.md), then move to [Search & Retrieval](user_guide/search_and_retrieval.md) and [Indexing & Performance](advanced/indexing.md) once you have real data and real latency/recall targets.
