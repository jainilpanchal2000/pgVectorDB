# pgVectorDB TODO — One-Stop RAG Search Solution

**Last Updated:** 2026-02-19 23:36 IST
**Status:** ✅ v0.0.4 Structural Refactoring COMPLETE — v0.0.5 Test Suite + Code Quality in progress
**Goal:** Multiple embeddings per table, multimodal search, full reranker support

---

## v0.0.2 — ✅ COMPLETE (35/35 tasks)

All v0.0.2 tasks are done: SQLAlchemy ORM, identifier quoting, halfvec, binary, sparse, subvector,
BM25, DiskANN, batch isolation, embedding fallback, reranker, slow queries, and more.
See [git history](.) for full details. **All metrics at 10/10.**

---

## v0.0.3 — ✅ COMPLETE (11/11 tasks + Reranker Module)

> **Inspiration:** [Superlinked — Why You Don't Need Re-Ranking](https://superlinked.com/vectorhub/articles/why-do-not-need-re-ranking)
> and [Real Estate NLQ Agent](https://superlinked.com/vectorhub/articles/real-estate-nlq-agent)
> and [Optimizing RAG with Hybrid Search & Reranking](https://superlinked.com/vectorhub/articles/optimizing-rag-with-hybrid-search-reranking)

---

### ✅ Critical — Multi-Embedding Core (Tasks 36–39)

#### 36. Vector Space Definitions (`src/spaces.py`) — ✅ DONE
- [x] `VectorSpace` abstract base class (name, dimensions, encode method)
- [x] `TextSpace` — embed text fields using LangChain embedding model
- [x] `NumberSpace` — encode numeric fields with min-max normalization (minimum/maximum/similar modes)
- [x] `CategorySpace` — encode categorical fields as one-hot vectors
- [x] `RecencySpace` — encode timestamps via exponential time-decay (`exp(-age/τ)`) with configurable `TimeUnit`
- [x] `NumberMode` enum + `TimeUnit` enum + utility functions (`validate_spaces`, `encode_document_spaces`, `encode_query_spaces`)

#### 37. Multi-Embedding Table Schema — ✅ DONE
- [x] `get_multimodal_table()` in `schema.py` — creates table with N `embedding_*` columns
- [x] Auto-detect dimensions per space
- [x] Backward compatible — existing single-embedding tables unaffected

#### 38. Multimodal Document Ingestion — ✅ DONE
- [x] `register_spaces(spaces)` — define vector spaces for a collection
- [x] `add_documents_multimodal(docs)` — embed each field per space, insert all columns
- [x] `_ensure_multimodal_columns()` — auto-adds missing columns

#### 39. Weighted Multimodal Search — ✅ DONE
- [x] `multimodal_search(query_params, weights, k)` — weighted fusion across all spaces
- [x] Support all existing distance metrics (cosine, L2, inner_product)
- [x] Optional hard pre-filtering via `filter={}` param

---

### ✅ High Priority — Advanced Search (Tasks 40–43)

#### 40. Per-Space Indexing — ✅ DONE
- [x] `build_multimodal_index()` — builds separate HNSW index per space column
- [x] `get_multimodal_index_stats()` — stats per space

#### 41. Multimodal Hybrid Search — ✅ DONE
- [x] `multimodal_hybrid_search(query_params, weights, keyword_weight, k)`
- [x] Fuse multimodal vector scores + BM25 keyword scores via RRF

#### 42. Dynamic Query-Time Weighting — ✅ DONE
- [x] Accept weight dictionaries at search time: `{text_space: 0.5, price_space: 0.3, ...}`
- [x] No re-embedding or re-indexing needed
- [x] Default weight `Config.DEFAULT_SPACE_WEIGHT = 1.0`

#### 43. Hard Pre-Filtering Integration — ✅ DONE
- [x] `filter={}` param in `multimodal_search()` — applies metadata filters before vector fusion

---

### ✅ Medium Priority — Polish & Examples (Tasks 44–46)

#### 44. End-to-End Example: Product Search — ✅ DONE
- [x] `examples/product_search.py` — multimodal product search with 4 spaces
- [x] Fields: description (text), price (number), rating (number), category (categorical)
- [x] Demonstrates dynamic weight adjustment and CrossEncoder reranking

#### 45. End-to-End Example: Real Estate NLQ — ✅ DONE
- [x] `examples/real_estate_nlq.py` — real estate NLQ with 5 spaces
- [x] Fields: description (text), price, bedrooms, bathrooms numbers, city (categorical)
- [x] Shows weighted spaces replacing rigid SQL filters

#### 46. Test Suite Extension — ✅ DONE
- [x] `test_multimodal_features()` — 7 sub-tests: spaces, ingestion, search, weights, hybrid, index, stats
- [x] `test_reranking_features()` — 4 sub-tests: instantiation, rerank(), factory, rerank_search()
- [x] Wired into `run_all_tests()`

---

### ✅ NEW: Reranker Module (`src/rerankers.py`)

Inspired by Superlinked's hybrid search + reranking article:

| Class | Backend | Requires |
|-------|---------|----------|
| `CrossEncoderReranker` | Local sentence-transformers | `sentence-transformers` |
| `CohereReranker` | Cohere Rerank API | `cohere`, API key |
| `AWSBedrockReranker` | AWS Bedrock | `boto3`, AWS credentials |
| `HuggingFaceReranker` | Local transformers pipeline | `transformers`, `torch` |

**Core methods added:**
- `rerank_search(query, reranker, k, rerank_top_k, search_method)` — retrieve-then-rerank

**Config defaults added:**
- `DEFAULT_RERANKER_TOP_K = 5`
- `DEFAULT_RERANKER_CANDIDATE_K = 100`
- `DEFAULT_CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"`
- `DEFAULT_COHERE_RERANK_MODEL = "rerank-english-v3.0"`
- `DEFAULT_BEDROCK_RERANK_MODEL = "amazon.rerank-v1:0"`
- `DEFAULT_HF_RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"`

---

### 🔵 Future / Nice-to-Have

- [ ] **Late interaction / ColBERT-style** — store token-level embeddings for MaxSim scoring
- [ ] **Learned space weights** — auto-learn optimal weights from relevance feedback
- [ ] **NLQ agent integration** — LLM parses natural language queries into multimodal search params
- [ ] **Matryoshka embeddings** — truncate embeddings for faster coarse search, rerank with full

---

## pg_textsearch Upgrades (v0.0.5)

> **Reference:** https://github.com/timescale/pg_textsearch/blob/main/ROADMAP.md
> **Checked:** 2026-02-19  v0.3.0 through v0.5.0 are already released; v1.0.0 (Feb 2026) is production-ready

### Already Released  Verify We Leverage These

| Version | Feature | Notes |
|---------|---------|-------|
| v0.3.0 (Jan) | Block-Max WAND  4x faster BM25 queries | Automatic once pg_textsearch >= v0.3.0 installed |
| v0.4.0 (Jan) | Posting list compression  41% smaller BM25 indexes | Automatic; bump min version check |
| v0.5.0 (Jan) | Parallel index builds | Expose parallel hint in build_bm25_index() |
| v1.0.0 (Feb) | Production ready  pg_dump/restore, VACUUM, replication | Pin Docker image to >= v1.0.0 |

**Action items (no code changes yet  plan only):**
- [ ] Pin pg_textsearch >= v1.0.0 in Dockerfile and document minimum version in README
- [ ] Add version check in extensions.py  warn if pg_textsearch < v0.5.0
- [ ] Add parallel build hint to build_bm25_index()  suggest setting max_parallel_maintenance_workers
- [ ] Update eval/ benchmark to measure BM25 query speed and confirm Block-Max WAND speedup

### Post-v1.0 Features  Future TODOs

Definitely planned by Timescale; design for these now:

- [ ] **Boolean queries** (AND/OR/NOT via @@ operator)  add bm25_boolean_search() method when released
- [ ] **Background compaction**  expose set_bm25_compaction(enabled=True) to avoid write stalls on heavy update workloads
- [ ] **Expression index support**  allow BM25 index on computed columns (e.g. content || ' ' || title)
- [ ] **Multi-tenant BM25**  single index with tenant-id column scoping; useful for our multi-collection use cases
- [ ] **Positional queries**  phrase/exact search support; major improvement over current bag-of-words BM25

---
## Structural Refactoring — Codebase Health (v0.0.4) ✅ COMPLETE

> **Analysis Date:** 2026-02-19
> **Goal:** Improve maintainability, reduce tech debt, and prepare for scalable development.

---

### ✅ 1. Break Up the `core.py` Monolith — DONE

**Problem:** `src/core.py` is **4,726 lines / 196KB** — the single largest maintainability risk.
It contains 70+ methods spanning initialization, document CRUD, indexing, search (via mixin),
analytics, export/import, LangChain integration, multimodal, and reranking.

**Recommendation:** Extract into focused modules by domain:

| New Module | Methods to Extract | Est. Lines |
|---|---|---|
| `src/documents.py` | `add_documents`, `aupdate_documents`, `adelete`, `add_documents_batch`, `add_documents_batch_isolated`, `upsert_documents`, `bulk_load_documents`, `add_documents_orm`, `aget_by_ids`, `update_metadata` | ~700 |
| `src/indexing.py` | `build_index`, `_build_hnsw_index`, `_build_ivfflat_index`, `_build_diskann_index`, `areindex`, `adrop_vector_index`, `build_index_concurrent`, `build_bm25_index`, `build_index_with_subvectors`, `build_index_binary_quantized`, `set_query_params`, `set_diskann_build_params` | ~700 |
| `src/analytics.py` | `get_stats`, `get_index_stats`, `explain_query`, `benchmark_search_methods`, `validate_collection`, `compute_recall`, `compute_centroid`, `get_slow_queries`, `get_bm25_index_stats`, `get_index_build_progress`, `dump_bm25_index`, `spill_bm25_index` | ~600 |
| `src/storage.py` | `export_to_json`, `import_from_json`, `create_halfvec_table`, `create_sparsevec_table` | ~350 |
| `src/multimodal.py` | `register_spaces`, `_ensure_multimodal_columns`, `add_documents_multimodal`, `build_multimodal_index`, `multimodal_search`, `multimodal_hybrid_search`, `get_multimodal_index_stats`, `rerank_search` | ~750 |
| `src/integrations.py` | `as_retriever`, `VectorStoreRetriever` inner class | ~100 |

**After extraction, `core.py` should be ~500 lines** — just `__init__`, `initialize`, `close`,
validation helpers, and mixin composition.

- [x] Create document operations module (`pgvectordb/mixins/documents.py`)
- [x] Create indexing module (`pgvectordb/mixins/indexing.py`)
- [x] Create analytics module (`pgvectordb/mixins/analytics.py`)
- [x] Create storage module (`pgvectordb/mixins/storage.py`)
- [x] Create multimodal module (`pgvectordb/mixins/multimodal.py`)
- [x] Create integrations module (`pgvectordb/mixins/integrations.py`)
- [x] Slim `core.py` to ~500 lines with mixin composition

---

### ✅ 2. Remove Duplicate Definitions in `core.py` — DONE

**Problem:** `core.py` lines 198–304 **re-define** every enum (`IndexType`, `KeywordSearchType`,
`StorageLayout`, `DistanceMetric`, `VectorPrecision`, `IterativeScanMode`), every exception
(`RetrievalSystemError`, `InitializationError`, `ValidationError`, `DatabaseError`, `RateLimitError`),
every constant (`ALLOWED_TEXT_CONFIGS`, `VALID_QUERY_PARAMS`), and `QueryResult` — all of which
already exist in `base.py`.

The `try/except ImportError` fallback (lines 128–189) imports from `base.py`, but the unconditional
re-definitions on lines 198–304 **always overwrite** those imports. This means:
- `base.py` definitions are **never actually used** by `core.py`
- Any improvement to `base.py` docstrings/values is **silently ignored**
- Two sources of truth exist for every enum and exception

**Fix:** Delete lines 198–304 entirely. Keep only the `try/except` imports. If standalone
usage of `core.py` is needed, raise a clear `ImportError` instead of silently redefining.

- [x] Delete duplicate enum definitions from `core.py` (lines 198–240)
- [x] Delete duplicate exception definitions from `core.py` (lines 272–295)
- [x] Delete duplicate constant definitions from `core.py` (lines 243–269)
- [x] Delete duplicate `QueryResult` from `core.py` (lines 299–304)
- [x] Update fallback `try/except` to raise `ImportError` with clear message

---

### ✅ 3. Add Proper Python Packaging (`pyproject.toml`) — DONE

**Problem:** No `pyproject.toml` or `setup.py` exists. The project cannot be installed as a
package (`pip install -e .`), making imports fragile (requires `src.` prefix and `PYTHONPATH` hacks).

**Recommendation:**
- [x] Create `pyproject.toml` with `[build-system]`, `[project]` metadata, and `[project.optional-dependencies]` for `aws`, `rerankers`, `dev`
- [x] Rename `src/` to `pgvectordb/` (Python package naming convention)
- [x] Update all internal imports from `src.` to `pgvectordb.`
- [x] Add `pgvector` to `requirements.txt` (currently **missing** — `schema.py` imports `pgvector.sqlalchemy`)

---

### � 4. Reorganize Test Suite (v0.0.5 — PENDING)

**Problem:** `test/test_suite.py` is a single **1,282-line** file with 15 test functions,
a custom `TestResults` tracker, and hardcoded DB credentials (`user`/`root` on port `9002`).
Does not use `pytest` fixtures, markers, or parameterization despite `pytest` being in `requirements.txt`.

**Recommendation:**
- [ ] Split into test files mirroring source modules (`test_documents.py`, `test_search.py`, `test_indexing.py`, etc.)
- [ ] Replace custom `TestResults` class with native `pytest` assertions
- [ ] Use `pytest` fixtures for DB setup/teardown and embedding model initialization
- [ ] Move DB credentials to environment variables or `conftest.py`
- [ ] Add `pytest.ini` or `pyproject.toml [tool.pytest]` section
- [ ] Add `pytest` markers for slow/integration tests (`@pytest.mark.slow`)

---

### ✅ 5. Fix Dependency Management — DONE

**Problem in `requirements.txt`:**
- `pgvector` package is **missing** (needed by `schema.py` for `pgvector.sqlalchemy.Vector`)
- `cohere` package is **missing** (needed by `rerankers.py` for `CohereReranker`)
- Dev dependencies mixed with runtime dependencies in a single file
- All dependencies are pinned to exact versions (fragile across environments)

**Recommendation:**
- [x] Add `pgvector` to requirements
- [x] Add `cohere` to optional requirements
- [x] Split into `requirements.txt` (runtime) and `requirements-dev.txt` (dev tools -> `pyproject.toml`)
- [x] Use version ranges instead of exact pins (e.g., `sqlalchemy>=2.0,<3.0`)

---

### � 6. Improve File & Directory Organization (v0.0.5 — PARTIALLY DONE)

**Current issues:**
- `scripts/test_connection.py` (14KB) overlaps with test suite functionality
- `scripts/demo.py` (9.5KB) overlaps with `examples/` directory
- `eval/scripts/benchmark_all_methods.py` is 34KB — another large file
- `docs/` is gitignored but contains 3 markdown files tracked in the repo
- No `__init__.py` in `test/`, `scripts/`, or `examples/`

**Recommendation:**
- [ ] Move or merge `scripts/demo.py` into `examples/`
- [ ] Move `scripts/test_connection.py` into `test/` or remove if redundant
- [ ] Consider splitting `eval/scripts/benchmark_all_methods.py` into smaller benchmark scripts
- [ ] Fix `.gitignore` — `docs/` is listed as ignored but tracked; remove the ignore rule or untrack
- [ ] Add `__init__.py` to `test/` for proper pytest discovery

---

### � 7. Code Quality Improvements (v0.0.5 — PENDING)

**Across all files:**
- [ ] Replace f-string SQL interpolation with parameterized queries in remaining raw SQL (e.g., `_setup_full_text_search`, `_add_labels_column` use `f'ALTER TABLE "{self.schema_name}"...'`)
- [ ] Add type hints to all function return types (several methods lack `-> None` or `-> Dict`)
- [ ] Add `__all__` exports to new modules after refactoring
- [ ] Standardize import style — `core.py` uses `from src.X` while `__init__.py` uses `from .X` (relative)
- [ ] Remove `logging.basicConfig()` call from `core.py` line 192 (libraries should not configure root logger)
- [ ] Consider adding `py.typed` marker for type checking support

---

## Architecture Overview (v0.0.4)

```
pgvectordb/                         (pip install -e .)
+-- __init__.py                      public API exports
+-- core.py                          pgVectorDB class (~440 lines, composes mixins)
+-- base.py                          enums, exceptions, QueryResult, constants
+-- config.py                        Config defaults
+-- schema.py                        SQLAlchemy table definitions
+-- extensions.py                    PostgreSQL extension manager
+-- search.py                        SearchMixin (10 search methods)
+-- spaces.py                        TextSpace, NumberSpace, CategorySpace, RecencySpace
+-- rerankers.py                     CrossEncoder, Cohere, AWS, HuggingFace rerankers
+-- metrics.py                       RAG evaluation metrics
+-- mixins/
    +-- __init__.py                  re-exports all 6 mixin classes
    +-- documents.py   DocumentsMixin  (~995 lines)
    +-- indexing.py    IndexingMixin   (~1052 lines)
    +-- analytics.py   AnalyticsMixin  (~946 lines)
    +-- storage.py     StorageMixin    (~330 lines)
    +-- multimodal.py  MultimodalMixin (~813 lines)
    +-- integrations.py IntegrationsMixin (~122 lines)

MRO: pgVectorDB -> SearchMixin -> DocumentsMixin -> IndexingMixin
               -> AnalyticsMixin -> StorageMixin -> MultimodalMixin -> IntegrationsMixin

Retrieval: multimodal_search() | rerank_search() | multimodal_hybrid_search()
Rerankers: CrossEncoder | Cohere | AWSBedrock | HuggingFace
```

---

## Remaining for v0.0.5

| # | Task | Priority |
|---|------|----------|
| 4 | Reorganize test/test_suite.py into per-module pytest files | High |
| 6 | File/directory cleanup (scripts/demo.py -> examples/, eval split) | Medium |
| 7 | Code quality: remaining f-string SQL -> parameterized, type hints, py.typed marker | Medium |
| - | Late interaction / ColBERT-style embeddings | Future |
| - | Learned space weights from relevance feedback | Future |
| - | NLQ agent integration | Future |
| - | Matryoshka embeddings | Future |

