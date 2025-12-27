# Project Roadmap & TODOs

This document tracks future improvements, feature requests, and technical debt for the `pgVectorDB` project.

## 🚀 Features to Implement

### Core RAG Functionality
- [ ] **HyDE (Hypothetical Document Embeddings)**: Implement HyDE strategy for better zero-shot retrieval.
- [ ] **Query Expansion**: Add multi-query generation to improve recall for ambiguous queries.
- [ ] **Re-ranking**: Integrate Cross-Encoder re-ranking (e.g., `ms-marco-MiniLM`) for the final top-K results to improve precision.

### API & Integration
- [ ] **REST API**: Wrap `pgVectorDB` in a FastAPI service for external consumption.
- [ ] **Async Ingestion**: Implement a background worker (e.g., Celery/Redis) for processing large document batches.
- [ ] **UI Dashboard**: Simple frontend to test queries and visualize results (like the benchmark output).

## 🔒 Security Enhancements

- [ ] **Input Sanitization**: Although FTS is fixed, ensuring a strict validation layer for all user inputs before SQL construction is critical.
- [ ] **RBAC**: Implement application-level Role-Based Access Control if this becomes a multi-user system.
- [ ] **Secret Management**: Move away from plain `.env` files for production secrets (use AWS Secrets Manager / Vault).
- [ ] **SQL Injection Audit**: Regularly review `text()` constructions in SQLAlchemy logic to ensure all parameters are bound safely.

## ⚡ Performance Optimization

- [ ] **Index Tuning**: Expose HNSW parameters (`m`, `ef_construction`) in `Config` for tuning based on dataset size.
- [ ] **Connection Pooling**: Stress test `asyncpg` pool settings under high concurrency.
- [ ] **Caching**: Add Redis layer to cache frequent query embeddings and results.

## 🛠️ Refactoring & Maintenance

- [ ] **Type Safety**: Run `mypy` and resolve strict type checking errors.
- [ ] **Testing**: Increase unit test coverage, specifically for edge cases in `hybrid_search`.
- [ ] **Documentation**: Generate API documentation (Sphinx/MkDocs) for the `pgVectorDB` class.
- [ ] **Dependency Management**: Lock dependencies with `poetry` or `pip-tools` for reproducible builds.

## ✅ Completed
- [x] **High Recall Tuning**: Validated `k=20` strategy for >90% recall.
- [x] **FTS Fix**: Switched to `OR`-based logic for robust keyword search.
- [x] **Evaluation Framework**: Refactored `metrics.py` and `eval/` directory structure.
