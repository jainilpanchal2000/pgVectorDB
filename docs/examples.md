# Examples

Use these scripts and notebooks as starting points for real pgVectorDB workflows. The examples live in the repository and are grouped here by what they demonstrate.

| Goal | Example | Covers |
| --- | --- | --- |
| First working collection | [examples/01_quickstart.ipynb](https://github.com/jainilpanchal2000/pgVectorDB/blob/main/examples/01_quickstart.ipynb) | install, initialize, ingest, search |
| Fluent API walkthrough | [examples/fluent_api_demo.py](https://github.com/jainilpanchal2000/pgVectorDB/blob/main/examples/fluent_api_demo.py) | `db.query(...)`, search modes, filters, analysis |
| Unified query notebook | [examples/06_unified_api_quickstart.ipynb](https://github.com/jainilpanchal2000/pgVectorDB/blob/main/examples/06_unified_api_quickstart.ipynb) | fluent API, lazy execution, outputs |
| Advanced retrieval | [examples/02_advanced_search.ipynb](https://github.com/jainilpanchal2000/pgVectorDB/blob/main/examples/02_advanced_search.ipynb) | semantic, keyword, hybrid, filters |
| Multimodal ranking | [examples/03_multimodal_search.ipynb](https://github.com/jainilpanchal2000/pgVectorDB/blob/main/examples/03_multimodal_search.ipynb) | `TextSpace`, `NumberSpace`, `CategorySpace`, `RecencySpace` |
| Storage optimization | [examples/04_storage_optimization.ipynb](https://github.com/jainilpanchal2000/pgVectorDB/blob/main/examples/04_storage_optimization.ipynb) | DiskANN, quantization, storage choices |
| RAG evaluation | [examples/05_rag_evaluation.ipynb](https://github.com/jainilpanchal2000/pgVectorDB/blob/main/examples/05_rag_evaluation.ipynb) | `RAGEvaluator`, metrics, K-value analysis |
| Product search | [examples/product_search.py](https://github.com/jainilpanchal2000/pgVectorDB/blob/main/examples/product_search.py) | catalog metadata, filters, ranking |
| Real-estate natural language search | [examples/real_estate_nlq.py](https://github.com/jainilpanchal2000/pgVectorDB/blob/main/examples/real_estate_nlq.py) | multimodal business signals, recency, numeric preferences |
| Minimal demo | [examples/demo.py](https://github.com/jainilpanchal2000/pgVectorDB/blob/main/examples/demo.py) | compact end-to-end usage |

## Coverage Matrix

| Differentiator | Where to learn | Example path |
| --- | --- | --- |
| Fluent API | [Quickstart](getting_started/quickstart.md), [Search & Retrieval](user_guide/search_and_retrieval.md) | [fluent_api_demo.py](https://github.com/jainilpanchal2000/pgVectorDB/blob/main/examples/fluent_api_demo.py) |
| RecencySpace | [Core Concepts](getting_started/core_concepts.md), [Multimodal Search](user_guide/multimodal_search.md) | [03_multimodal_search.ipynb](https://github.com/jainilpanchal2000/pgVectorDB/blob/main/examples/03_multimodal_search.ipynb) |
| Built-in evaluators | [Metrics & Evaluation](user_guide/metrics_and_evaluation.md) | [05_rag_evaluation.ipynb](https://github.com/jainilpanchal2000/pgVectorDB/blob/main/examples/05_rag_evaluation.ipynb) |
| SQL analyzers | [Search & Retrieval](user_guide/search_and_retrieval.md), [Analytics & Diagnostics](user_guide/analytics_and_diagnostics.md) | [fluent_api_demo.py](https://github.com/jainilpanchal2000/pgVectorDB/blob/main/examples/fluent_api_demo.py) |
| Optimization | [Indexing & Performance](advanced/indexing.md), [Metadata Filtering](user_guide/filtering.md) | [04_storage_optimization.ipynb](https://github.com/jainilpanchal2000/pgVectorDB/blob/main/examples/04_storage_optimization.ipynb) |
| Reranking | [Reranking](user_guide/reranking.md) | [02_advanced_search.ipynb](https://github.com/jainilpanchal2000/pgVectorDB/blob/main/examples/02_advanced_search.ipynb) |
| LangChain integration | [LangChain Integration](user_guide/langchain_integration.md) | [01_quickstart.ipynb](https://github.com/jainilpanchal2000/pgVectorDB/blob/main/examples/01_quickstart.ipynb) |

## Evaluation Scripts

The `eval/` folder contains benchmark and dataset utilities for deeper retrieval testing.

| Script | Use it for |
| --- | --- |
| [eval/scripts/benchmark_all_methods.py](https://github.com/jainilpanchal2000/pgVectorDB/blob/main/eval/scripts/benchmark_all_methods.py) | Compare retrieval methods across a benchmark set. |
| [eval/scripts/optimize_k.py](https://github.com/jainilpanchal2000/pgVectorDB/blob/main/eval/scripts/optimize_k.py) | Analyze K-value tradeoffs. |
| [eval/scripts/test_metrics_correctness.py](https://github.com/jainilpanchal2000/pgVectorDB/blob/main/eval/scripts/test_metrics_correctness.py) | Validate metric behavior. |
| [eval/scripts/generate_synthetic_dataset.py](https://github.com/jainilpanchal2000/pgVectorDB/blob/main/eval/scripts/generate_synthetic_dataset.py) | Generate synthetic evaluation data. |