# pgVectorDB TODO — One-Stop RAG Search Solution

**Last Updated:** 2026-05-09
**Status:** ✅ v0.0.5 — PyPI Release Ready
**Goal:** Production PostgreSQL vector database with multi-embedding, multimodal search, and rerankers

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

#### 36. Vector Space Definitions (`pgvectordb/spaces.py`) — ✅ DONE
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

### ✅ NEW: Reranker Module (`pgvectordb/rerankers.py`)

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

## v0.0.4 — ✅ COMPLETE (Structural Refactoring)

> **Analysis Date:** 2026-02-19
> **Goal:** Improve maintainability, reduce tech debt, and prepare for scalable development.

### ✅ 1. Break Up the `core.py` Monolith — DONE

**Problem:** `core.py` was **4,726 lines / 196KB** — the single largest maintainability risk.

| New Module | Methods Extracted | Lines |
|---|---|---|
| `pgvectordb/mixins/documents.py` | `add_documents`, `aupdate_documents`, `adelete`, batch ops, `upsert_documents`, `bulk_load_documents` | ~1,000 |
| `pgvectordb/mixins/indexing.py` | `build_index`, HNSW/IVFFlat/DiskANN builders, `build_index_concurrent`, `build_bm25_index`, binary/subvector | ~1,150 |
| `pgvectordb/mixins/analytics.py` | `get_stats`, `explain_query`, `benchmark_search_methods`, `validate_collection`, `compute_recall` | ~970 |
| `pgvectordb/mixins/storage.py` | `export_to_json`, `import_from_json`, `create_halfvec_table`, `create_sparsevec_table` | ~380 |
| `pgvectordb/mixins/multimodal.py` | `register_spaces`, `multimodal_search`, `multimodal_hybrid_search`, `rerank_search` | ~1,040 |
| `pgvectordb/mixins/integrations.py` | `as_retriever`, `VectorStoreRetriever` | ~120 |

**After extraction:** `core.py` is ~485 lines — just `__init__`, `initialize`, `close`, and mixin composition.

### ✅ 2. Remove Duplicate Definitions — DONE
### ✅ 3. Add Proper Python Packaging — DONE
### ✅ 4. Reorganize Test Suite — DONE (13 pytest files + conftest.py)
### ✅ 5. Fix Dependency Management — DONE
### ✅ 6. File & Directory Organization — DONE
### ✅ 7. Code Quality Improvements — DONE

---

## v0.0.5 — ✅ COMPLETE (PyPI Release)

### Security Fixes
- [x] `set_maintenance_work_mem` — regex whitelist validation
- [x] `set_parallel_workers` — int coercion + bounds check
- [x] `vacuum_analyze` — VACUUM outside transaction (COMMIT before VACUUM)

### Performance Fixes
- [x] `update_metadata` — bulk JSONB `||` UPDATE (O(1) vs O(2N))
- [x] `upsert_documents` — single connection for all operations

### Packaging
- [x] Removed `psycopg2-binary` and `nest-asyncio` from core deps
- [x] Added `[jupyter]` optional extra
- [x] Fixed `ruff==0.15.2` → `ruff>=0.4.0`
- [x] Aligned `MIN_PG_TEXTSEARCH_VERSION` to `0.4.0` + warning for < 1.0.0
- [x] Version bump to `0.0.5`
- [x] `hatch build` produces valid `.whl` and `.tar.gz`
- [x] CHANGELOG.md created

### Docker
- [x] Pinned pgvector to v0.8.2, pgvectorscale to 0.9.0, pg_textsearch to v1.0.0
- [x] Replaced hardcoded volume with named Docker volume

### CI
- [x] `.github/workflows/ci.yml` — ruff check + syntax check + hatch build on 3.10/3.12/3.13
- [x] `.github/workflows/publish.yml` — trusted publishing on GitHub Release

### Documentation
- [x] README updated (version, layout, pip install, method count)
- [x] `requirements.txt` replaced with pyproject.toml redirect
- [x] TODO.md cleaned up

---

## pg_textsearch Upgrades (Verified)

> **Reference:** https://github.com/timescale/pg_textsearch/blob/main/ROADMAP.md
> **Verified:** 2026-05-09  Docker now pins v1.0.0

| Version | Feature | Status |
|---------|---------|--------|
| v0.3.0 | Block-Max WAND → 4x faster BM25 queries | ✅ Automatic when installed |
| v0.4.0 | Posting list compression → 41% smaller indexes | ✅ MIN version = 0.4.0 |
| v0.5.0 | Parallel index builds | ✅ Exposed via `max_parallel_maintenance_workers` hint |
| v1.0.0 | Production ready (pg_dump/restore, VACUUM) | ✅ Docker pin + warning for < 1.0.0 |

### Post-v1.0 Features — Future TODOs

- [ ] **Boolean queries** (AND/OR/NOT via @@ operator) — add `bm25_boolean_search()` when released
- [ ] **Background compaction** — expose `set_bm25_compaction(enabled=True)` for write-heavy workloads
- [ ] **Expression index support** — BM25 index on computed columns (e.g. `content || ' ' || title`)
- [ ] **Multi-tenant BM25** — single index with tenant-id column scoping
- [ ] **Positional queries** — phrase/exact search (major improvement over bag-of-words BM25)

---

## Architecture Overview (v0.0.5)

```
pgvectordb/                         (pip install pgvectordb)
├── __init__.py                      public API exports, __version__
├── core.py                          pgVectorDB class (~485 lines, composes mixins)
├── base.py                          enums, exceptions, QueryResult, constants
├── config.py                        Config defaults
├── schema.py                        SQLAlchemy table definitions
├── extensions.py                    PostgreSQL extension manager
├── search.py                        SearchMixin (10 search methods)
├── spaces.py                        TextSpace, NumberSpace, CategorySpace, RecencySpace
├── rerankers.py                     CrossEncoder, Cohere, AWS, HuggingFace rerankers
├── metrics.py                       RAG evaluation metrics
├── py.typed                         PEP 561 marker
└── mixins/
    ├── __init__.py                  re-exports all 6 mixin classes
    ├── documents.py   DocumentsMixin  (~1,020 lines)
    ├── indexing.py    IndexingMixin   (~1,155 lines)
    ├── analytics.py   AnalyticsMixin  (~985 lines)
    ├── storage.py     StorageMixin    (~380 lines)
    ├── multimodal.py  MultimodalMixin (~1,040 lines)
    └── integrations.py IntegrationsMixin (~122 lines)

MRO: pgVectorDB → SearchMixin → DocumentsMixin → IndexingMixin
               → AnalyticsMixin → StorageMixin → MultimodalMixin → IntegrationsMixin
```

---

## 🔵 Future / Nice-to-Have

- [ ] **Late interaction / ColBERT-style** — store token-level embeddings for MaxSim scoring
- [ ] **Learned space weights** — auto-learn optimal weights from relevance feedback
- [ ] **NLQ agent integration** — LLM parses natural language queries into multimodal search params
- [ ] **Matryoshka embeddings** — truncate embeddings for faster coarse search, rerank with full
