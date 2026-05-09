# pgVectorDB - PostgreSQL Vector Database

Welcome to the official developer documentation for **pgVectorDB**.

pgVectorDB is a production-ready Retrieval-Augmented Generation (RAG) orchestration layer built on PostgreSQL, `pgvector`, and `langchain_postgres`. It offers **10 distinct search methods**, multi-embedding multimodal search, robust connection pooling, and multi-tenant metadata isolation — all without managing your own vector database infrastructure.

---

## Why pgVectorDB?

| Feature | Description |
|---------|-------------|
| **10 Search Methods** | Semantic, keyword (FTS/BM25), hybrid (RRF), trigram fuzzy, ensemble |
| **Multimodal Spaces** | Multiple embeddings per document — text + price + category + recency |
| **Infinite Scaling** | HNSW (<1M), IVFFlat (10M), DiskANN (10M+) with label partitioning |
| **13 Filter Operators** | MongoDB-style JSONB filtering (`$eq`, `$between`, `$in`, `$and`, `$or`, etc.) |
| **Reranking** | Cross-encoder, Cohere, AWS Bedrock, HuggingFace API |
| **Statistical RAG Evaluation** | Hit Rate, MRR, NDCG — benchmark any search pipeline |
| **Production Diagnostics** | Query plans, benchmarks, recall measurement, index health |

---

## Documentation Architecture

### Getting Started
- [Installation](getting_started/installation.md) — Docker & Python setup
- [Quickstart](getting_started/quickstart.md) — Build your first RAG in 5 minutes
- [Core Concepts](getting_started/core_concepts.md) — Architecture, mixin system, security design

### User Guide
- [Vector Store Operations](user_guide/vector_store.md) — CRUD, batch ingestion, upsert, DiskANN labels
- [Embeddings & Spaces](user_guide/embeddings_and_spaces.md) — TextSpace, NumberSpace, CategorySpace, RecencySpace
- [Multimodal Search](user_guide/multimodal_search.md) — Multi-embedding RAG with weighted space fusion
- [Search & Retrieval](user_guide/search_and_retrieval.md) — All 10 search methods in depth
- [Metadata Filtering](user_guide/filtering.md) — 13 filter operators with SQL translation
- [Reranking](user_guide/reranking.md) — `rerank_search()`, cross-encoder, Cohere, Bedrock
- [Metrics & Evaluation](user_guide/metrics_and_evaluation.md) — Hit Rate, MRR, NDCG, A/B testing
- [Analytics & Diagnostics](user_guide/analytics_and_diagnostics.md) — Stats, benchmarks, query plans, recall
- [LangChain Integration](user_guide/langchain_integration.md) — Native LangChain retriever & chains

### Advanced
- [Indexing & Performance](advanced/indexing.md) — HNSW, IVFFlat, DiskANN tuning + `maintenance_work_mem`
- [Configuration](advanced/configuration.md) — Connection pooling, schema isolation, environment configs

### API Reference
- [pgVectorDB Core](api_reference/pgvectordb.md) — Auto-generated class reference
- [Spaces](api_reference/spaces.md) — VectorSpace, TextSpace, NumberSpace, CategorySpace, RecencySpace
- [Rerankers](api_reference/rerankers.md) — CrossEncoderReranker, CohereReranker, AWSBedrockReranker
- [Metrics](api_reference/metrics.md) — RAGEvaluator, EvaluationDataset
- [Configuration](api_reference/config.md) — Config, get_production_config, get_test_config
- [Exceptions](api_reference/exceptions.md) — RetrievalSystemError, InitializationError, DatabaseError

---

## Examples & Notebooks

See the `examples/` folder for full working notebooks:

| Notebook | Description |
|----------|-------------|
| `01_quickstart.ipynb` | Basic RAG pipeline |
| `02_advanced_search.ipynb` | Hybrid search with RRF |
| `03_multimodal_search.ipynb` | Multi-embedding search with Spaces |
| `04_storage_optimization.ipynb` | DiskANN tuning and label filtering |
| `05_rag_evaluation.ipynb` | Metric evaluation with RAGEvaluator |

---

## Status

- **Version:** `0.0.5`
- **Status:** Production-Ready

!!! bug "Report Issues"
    Found a bug? Open an issue on [GitHub](https://github.com/jainilpanchal/pgvectordb)
