# Changelog

All notable changes to pgVectorDB are documented here.
This project adheres to [Semantic Versioning](https://semver.org/).

---

## [0.0.5] — 2026-05-09 (PyPI Release Candidate)

### Security
- **Fixed** `set_maintenance_work_mem` — added regex allowlist validation
  (`^\d+\s*(kB|MB|GB|TB)?$`) to prevent SQL injection via the memory value string.
- **Fixed** `set_parallel_workers` — added `int()` coercion and non-negative
  bounds check on `gather` and `maintenance` parameters.

### Bug Fixes
- **Fixed** `vacuum_analyze` — VACUUM was being executed inside an implicit
  SQLAlchemy transaction, which PostgreSQL rejects. Now issues an explicit
  `COMMIT` before running `VACUUM ANALYZE` / `VACUUM FULL ANALYZE`.
- **Fixed** `MIN_PG_TEXTSEARCH_VERSION` mismatch — `extensions.py` had `1.0.0`
  while `config.py` had `0.4.0` (correct). Aligned both to `0.4.0` with a
  clarifying comment.

### Performance
- **Fixed** `update_metadata` — replaced N+1 SELECT + UPDATE loop with a single
  bulk `UPDATE … SET langchain_metadata = COALESCE(…, '{}') || :updates WHERE
  langchain_id = ANY(:ids)`. Reduces database round-trips from O(2N) to O(1).
- **Fixed** `upsert_documents` — hoisted the database connection outside the
  per-document loop. Previously a new connection was opened for every document
  and again for the content-hash update step, risking connection pool exhaustion
  on large batches.

### Packaging
- **Removed** `psycopg2-binary` from core dependencies — already transitively
  required by `langchain-postgres`; explicit listing caused "binary wheel in
  production" warnings.
- **Removed** `nest-asyncio` from core dependencies — Jupyter-only convenience
  library not appropriate for production. Added as a new `[jupyter]` optional
  extra.
- **Removed** `flake8` from dev dependencies — `ruff` covers all lint rules and
  is already configured.
- **Fixed** `ruff==0.15.2` — version does not exist on PyPI. Changed to
  `ruff>=0.4.0`.

### Documentation
- Bumped version to `0.0.5` in `__init__.py` docstring and `__version__`.
- Updated `README.md`: correct package layout (`pgvectordb/` not `src/`),
  added `pip install pgvectordb` PyPI quickstart, updated method count to 60+.
- Added this `CHANGELOG.md`.

---

## [0.0.4] — 2026-05-09

### Features
- **Binary quantization index** (`build_index_binary_quantized`) — creates a
  Hamming-distance HNSW index on `binary_quantize(embedding)` for 87.5%
  storage savings.
- **Two-stage binary search** (`search_with_binary_rerank`) — fast Hamming
  retrieval followed by full-vector cosine re-ranking.
- **Subvector indexing** (`build_index_with_subvectors`) and **subvector
  reranking** (`search_with_subvector_rerank`) — supports Matryoshka embeddings
  (OpenAI `text-embedding-3`, Nomic).
- **Concurrent index builds** (`build_index_concurrent`) — `CREATE INDEX
  CONCURRENTLY` for zero-downtime index creation or replacement.
- **Index build progress** (`get_index_build_progress`) — polls
  `pg_stat_progress_create_index` for live feedback.
- **Batch error isolation** (`add_documents_batch_isolated`) — each batch is
  committed independently; failures are reported, not raised by default.
- **Content-hash upsert** (`upsert_documents`) — deduplicates by MD5 of
  `page_content` using an optional `content_hash` column.
- **Slow query monitoring** (`get_slow_queries`) — queries
  `pg_stat_statements` for the slowest vector/embedding operations.
- **BM25 parallel build hint** — `build_bm25_index` accepts
  `max_parallel_maintenance_workers` for faster index creation.
- **Metadata GIN index** (`create_metadata_index`) — creates `gin_trgm_ops`
  indexes on JSONB metadata fields for fast text filtering.
- **SQLAlchemy inspector** (`_index_exists`) — uses `run_sync(inspect)` for
  robust index existence checks with a `pg_indexes` fallback.

### Bug Fixes
- Resolved `PostgresSyntaxError` in `search_with_binary_rerank` caused by
  asyncpg misinterpreting `::vector(N)` in named-parameter queries. Embedding
  is now interpolated as a validated float-list literal.

---

## [0.0.3] — 2026-02-20

### Features
- **Rerankers module** (`pgvectordb/rerankers.py`) with four backends:
  - `CrossEncoderReranker` (sentence-transformers, local)
  - `CohereReranker` (Cohere Rerank API)
  - `AWSBedrockReranker` (Amazon Bedrock `amazon.rerank-v1:0`)
  - `HuggingFaceReranker` (transformers text-classification pipeline)
  - `create_reranker` factory function
- **Vector spaces module** (`pgvectordb/spaces.py`) for multimodal search:
  - `TextSpace` — dense embeddings from any LangChain model
  - `NumberSpace` — min-max normalized numeric fields (min/max/similar modes)
  - `CategorySpace` — one-hot categorical encoding with optional negative filter
  - `RecencySpace` — exponential time-decay for timestamp fields

### Performance
- Modularized main class into focused mixins under `pgvectordb/mixins/`:
  `DocumentsMixin`, `IndexingMixin`, `AnalyticsMixin`, `StorageMixin`,
  `MultimodalMixin`.

---

## [0.0.2] — 2026-02-20

### Features
- **Half-precision storage** (`create_halfvec_table`) — `halfvec` type
  (2 bytes/dim, 50% savings).
- **Sparse vector storage** (`create_sparsevec_table`) — `sparsevec` type
  for TF-IDF / one-hot data.
- **Label filtering** for DiskANN — `add_documents` accepts `labels` for
  partition-based filtered search.
- **Export / import** (`export_to_json`, `import_from_json`).
- **Metadata update** (`update_metadata`) — bulk metadata patching.
- **Document update** (`aupdate_documents`) — in-place content + embedding
  update with optional `update_embeddings=False` for metadata-only changes.
- **Embedding fallback** (`_embed_documents_with_fallback`) — on batch
  embedding failure falls back to per-document embedding; rate-limit errors
  raise `RateLimitError` immediately.
- **Iterative scan** (`set_iterative_scan`) — configures HNSW/IVFFlat
  iterative scanning for better recall on filtered queries.
- **Centroid computation** (`compute_centroid`) — average embedding for a
  filtered or full collection.
- **Label definitions table** (`create_label_definitions`,
  `get_label_ids_by_names`).

---

## [0.0.1] — Initial Release

- `pgVectorDB` core class with HNSW, IVFFlat, and DiskANN index support.
- 10 search methods: `semantic_search`, `keyword_search` (FTS + BM25),
  `hybrid_search`, `ensemble_search`, `trigram_search`,
  `metadata_semantic_search`, `metadata_keyword_search`,
  `metadata_trigram_search`.
- `ExtensionManager` with graceful degradation for optional extensions.
- BM25 index (`build_bm25_index`) using `pg_textsearch`.
- DiskANN build parameter tuning (`set_diskann_build_params`).
- Query parameter tuning (`set_query_params`) — `hnsw.ef_search`,
  `ivfflat.probes`, `diskann.query_search_list_size`, etc.
- Analytics: `get_stats`, `get_index_stats`, `explain_query`,
  `benchmark_search_methods`, `validate_collection`, `compute_recall`.
- Maintenance: `vacuum_analyze`, `areindex`, `adrop_vector_index`.
- Configuration system (`pgvectordb/config.py`) with `.env` support.
