# Examples

Use these scripts and notebooks as starting points for real pgVectorDB workflows. The examples live in the repository and are grouped here by what they demonstrate.

## Learning Path

Start here and progress through in order:

| Step | Goal | Example | Covers |
| --- | --- | --- | --- |
| 1 | First working collection | [examples/01_quickstart.ipynb](https://github.com/jainilpanchal2000/pgVectorDB/blob/main/examples/01_quickstart.ipynb) | install, initialize, ingest, basic search |
| 2 | Fluent API basics | [examples/06_unified_api_quickstart.ipynb](https://github.com/jainilpanchal2000/pgVectorDB/blob/main/examples/06_unified_api_quickstart.ipynb) | `db.query(...)`, search modes, lazy execution, output formats |
| 3 | Advanced search | [examples/02_advanced_search.ipynb](https://github.com/jainilpanchal2000/pgVectorDB/blob/main/examples/02_advanced_search.ipynb) | semantic, keyword, hybrid, filters, reranking |
| 4 | Multimodal ranking | [examples/03_multimodal_search.ipynb](https://github.com/jainilpanchal2000/pgVectorDB/blob/main/examples/03_multimodal_search.ipynb) | `TextSpace`, `NumberSpace`, `CategorySpace`, `RecencySpace` |
| 5 | Optimization | [examples/04_storage_optimization.ipynb](https://github.com/jainilpanchal2000/pgVectorDB/blob/main/examples/04_storage_optimization.ipynb) | DiskANN, HNSW, IVFFlat, quantization |
| 6 | Evaluation | [examples/05_rag_evaluation.ipynb](https://github.com/jainilpanchal2000/pgVectorDB/blob/main/examples/05_rag_evaluation.ipynb) | `RAGEvaluator`, metrics, K-value analysis |

## Additional Examples

| Goal | Example | Covers |
| --- | --- | --- |
| Comprehensive demo | [notebooks/demo.ipynb](https://github.com/jainilpanchal2000/pgVectorDB/blob/main/notebooks/demo.ipynb) | All features with visualizations |
| Evaluation scripts | [notebooks/eval_demo.ipynb](https://github.com/jainilpanchal2000/pgVectorDB/blob/main/notebooks/eval_demo.ipynb) | Evaluation metrics and benchmarking |
| Fluent API walkthrough | [examples/fluent_api_demo.py](https://github.com/jainilpanchal2000/pgVectorDB/blob/main/examples/fluent_api_demo.py) | Python script demonstrating the Fluent API |
| Product search | [examples/product_search.py](https://github.com/jainilpanchal2000/pgVectorDB/blob/main/examples/product_search.py) | catalog metadata, filters, ranking |
| Real-estate search | [examples/real_estate_nlq.py](https://github.com/jainilpanchal2000/pgVectorDB/blob/main/examples/real_estate_nlq.py) | multimodal signals, recency, numbers |
| Minimal demo | [examples/demo.py](https://github.com/jainilpanchal2000/pgVectorDB/blob/main/examples/demo.py) | compact end-to-end usage |

## Coverage Matrix

| Differentiator | Where to learn | Example path |
| --- | --- | --- |
| Fluent API | [Quickstart](getting_started/quickstart.md), [Search & Retrieval](user_guide/search_and_retrieval.md) | [fluent_api_demo.py](https://github.com/jainilpanchal2000/pgVectorDB/blob/main/examples/fluent_api_demo.py), [06_unified_api_quickstart.ipynb](https://github.com/jainilpanchal2000/pgVectorDB/blob/main/examples/06_unified_api_quickstart.ipynb) |
| Unified Query | [Search & Retrieval](user_guide/search_and_retrieval.md) | [notebooks/demo.ipynb](https://github.com/jainilpanchal2000/pgVectorDB/blob/main/notebooks/demo.ipynb) |
| Metadata filtering | [Filtering](user_guide/filtering.md), [Migration Guide](user_guide/migration_guide.md) | [notebooks/demo.ipynb](https://github.com/jainilpanchal2000/pgVectorDB/blob/main/notebooks/demo.ipynb) (Metadata-Only Search section) |
| DiskANN Label Filtering | [Analytics & Diagnostics](user_guide/analytics_and_diagnostics.md) | [notebooks/demo.ipynb](https://github.com/jainilpanchal2000/pgVectorDB/blob/main/notebooks/demo.ipynb) (Label Filtering section) |
| EXPLAIN ANALYZE | [Analytics & Diagnostics](user_guide/analytics_and_diagnostics.md) | [06_unified_api_quickstart.ipynb](https://github.com/jainilpanchal2000/pgVectorDB/blob/main/examples/06_unified_api_quickstart.ipynb) |
| RecencySpace | [Core Concepts](getting_started/core_concepts.md), [Multimodal Search](user_guide/multimodal_search.md) | [03_multimodal_search.ipynb](https://github.com/jainilpanchal2000/pgVectorDB/blob/main/examples/03_multimodal_search.ipynb) |
| Built-in evaluators | [Metrics & Evaluation](user_guide/metrics_and_evaluation.md) | [05_rag_evaluation.ipynb](https://github.com/jainilpanchal2000/pgVectorDB/blob/main/examples/05_rag_evaluation.ipynb), [notebooks/eval_demo.ipynb](https://github.com/jainilpanchal2000/pgVectorDB/blob/main/notebooks/eval_demo.ipynb) |
| SQL analyzers | [Search & Retrieval](user_guide/search_and_retrieval.md), [Analytics & Diagnostics](user_guide/analytics_and_diagnostics.md) | [fluent_api_demo.py](https://github.com/jainilpanchal2000/pgVectorDB/blob/main/examples/fluent_api_demo.py), [notebooks/demo.ipynb](https://github.com/jainilpanchal2000/pgVectorDB/blob/main/notebooks/demo.ipynb) |
| Optimization | [Indexing & Performance](advanced/indexing.md), [Metadata Filtering](user_guide/filtering.md) | [04_storage_optimization.ipynb](https://github.com/jainilpanchal2000/pgVectorDB/blob/main/examples/04_storage_optimization.ipynb) |
| Reranking | [Rerankers](user_guide/reranking.md) | [02_advanced_search.ipynb](https://github.com/jainilpanchal2000/pgVectorDB/blob/main/examples/02_advanced_search.ipynb) |
| LangChain integration | [LangChain Integration](user_guide/langchain_integration.md) | [01_quickstart.ipynb](https://github.com/jainilpanchal2000/pgVectorDB/blob/main/examples/01_quickstart.ipynb) |

## Evaluation Scripts

The `eval/` folder contains benchmark and dataset utilities for deeper retrieval testing.

| Script | Use it for |
| --- | --- |
| [eval/scripts/benchmark_all_methods.py](https://github.com/jainilpanchal2000/pgVectorDB/blob/main/eval/scripts/benchmark_all_methods.py) | Compare retrieval methods across a benchmark set. |
| [eval/scripts/optimize_k.py](https://github.com/jainilpanchal2000/pgVectorDB/blob/main/eval/scripts/optimize_k.py) | Analyze K-value tradeoffs. |
| [eval/scripts/test_metrics_correctness.py](https://github.com/jainilpanchal2000/pgVectorDB/blob/main/eval/scripts/test_metrics_correctness.py) | Validate metric behavior. |
| [eval/scripts/generate_synthetic_dataset.py](https://github.com/jainilpanchal2000/pgVectorDB/blob/main/eval/scripts/generate_synthetic_dataset.py) | Generate synthetic evaluation data. |