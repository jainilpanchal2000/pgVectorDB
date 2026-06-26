# pgVectorDB TODO - Roadmap

**Last Updated:** 2026-06-26  
**Current Release:** 0.0.6  
**Primary API:** `db.query(...)`  
**Docs/Examples Naming:** use `db` for the main `pgVectorDB` instance

---

## Current State

v0.0.6 is the fluent API and documentation release. The old public-facing examples have been moved toward the current `db.query(...)` style, while the lower-level search methods remain available internally and for advanced usage.

Implemented or refreshed:

- Fluent query API via `UnifiedQueryBuilder` in `pgvectordb/query/unified.py`.
- Search modes: semantic, keyword, BM25, hybrid, RRF hybrid, trigram, vector query, and multimodal space search.
- Query composition: `.where(...)`, `.limit(...)`, `.offset(...)`, `.select(...)`, `.ef(...)`, `.rrf(...)`, `.rerank(...)`, `.to_list()`, `.to_pandas()`, and `.to_arrow()`.
- Diagnostics: `explain_plan()`, `analyze_plan()`, and raw `explain_query()` examples verified against a real local PostgreSQL database.
- Reranking: fluent `.rerank(...)` accepts callable score functions and BaseReranker-like objects with `.rerank(query, documents, top_k)`.
- Docs refresh: README, homepage, user guides, examples page, reranking, search/retrieval, multimodal, embeddings/spaces, and analytics/diagnostics.
- Notebook refresh: example notebooks and repo notebooks have executed outputs; output audit passed after fixing empty-result demos.
- Quality gate: `ruff check .` and `ruff format --check .` pass as of 2026-06-26.

---

## v0.0.6 Release Checklist

### Must Finish

- [ ] Run full test suite against the local PostgreSQL test database.
- [ ] Run `uv run pyright` and fix or explicitly triage remaining type errors.
- [ ] Run strict docs build: `NO_MKDOCS_2_WARNING=true uv run mkdocs build --strict`.
- [ ] Re-run notebook execution and output audit after any notebook edits.
- [ ] Review README, docs, examples, and notebooks for stale API names.
- [ ] Confirm package metadata, optional dependency extras, and Python version classifiers before release.
- [ ] Review large notebook diffs and ensure saved outputs are intentional.

### Nice to Have Before Release

- [ ] Add a CI job for `ruff check .` and `ruff format --check .` if not already enforced.
- [ ] Add a docs build job with MkDocs strict mode.
- [ ] Add a lightweight notebook smoke job for the fastest quickstart notebook.
- [ ] Add a small script for the notebook output audit used during the docs refresh.
- [ ] Publish a short migration note from older direct examples to `db.query(...)`.

---

## Stale API Watchlist

When updating docs or examples, avoid reintroducing older public examples unless the section is explicitly about backward compatibility.

Prefer:

```python
results = await (
    db.query("machine learning")
    .hybrid()
    .where({"category": "ai"})
    .rrf(k=60)
    .limit(10)
    .to_list()
)
```

Avoid in new user-facing snippets:

- `db.search(...)`
- `search_text(...)`
- `add_texts(...)`
- `create_index(...)`
- `nearest_to_text(...)`
- `pgv_db` as the primary docs/example variable name

---

## Completed in v0.0.6

- [x] Add `SearchMethod` and `KeywordSearchType` enums.
- [x] Add fluent `db.query(...)` entry point.
- [x] Add semantic, keyword, hybrid, and trigram fluent builders.
- [x] Add fluent filters, pagination, projection, and output formats.
- [x] Add query diagnostics through `explain_plan()` and `analyze_plan()`.
- [x] Add scalar index helpers for metadata-heavy filtering.
- [x] Add BM25 index support and keyword search selection.
- [x] Add multimodal `.across_spaces(...)` examples.
- [x] Add `RecencySpace` documentation and examples.
- [x] Update reranking docs and object-reranker support.
- [x] Refresh public examples for fluent-first usage.
- [x] Execute notebooks and fix empty-output examples.
- [x] Run Ruff lint and formatting cleanup.

---

## v0.0.7 - Query Controls and Indexing Depth

Focus: make advanced retrieval behavior explicit, testable, and documented.

- [ ] Add or finalize query params for ANN tuning: `nprobes`, `ef_search`, `refine_factor`, and exact-search bypass.
- [ ] Add distance-range filtering for near-duplicate and thresholded retrieval workflows.
- [ ] Add pre-filter vs post-filter controls with clear recall/latency trade-off docs.
- [ ] Add GIN/array index helpers for tag and label-list style metadata.
- [ ] Add index statistics and readiness helpers such as `wait_for_index()`.
- [ ] Add benchmark coverage for filter timing, scalar indexes, and BM25 vs FTS behavior.

Example target API:

```python
results = await (
    db.query("machine learning")
    .semantic()
    .where({"status": "published"})
    .ef(100)
    .limit(10)
    .to_list()
)
```

---

## v0.0.8 - Compression and Storage Optimization

Focus: reduce storage and memory pressure while preserving acceptable recall.

- [ ] Product quantization planning and extension capability checks.
- [ ] Binary/RaBitQ-style quantization research and feasibility notes for PostgreSQL/pgvector.
- [ ] Subvector and Matryoshka retrieval examples with two-stage reranking.
- [ ] Storage savings benchmarks for float, halfvec, binary, and subvector layouts.
- [ ] Documentation for recall, latency, and storage trade-offs.

---

## v0.0.9+ - Advanced Retrieval Models

Focus: retrieval quality features that need deeper schema or embedding changes.

- [ ] Multi-vector or late-interaction search design, including ColBERT-style MaxSim scoring.
- [ ] Token-level embedding storage strategy.
- [ ] Batch search API for evaluating multiple queries efficiently.
- [ ] Table versioning or reproducibility design for experiments and rollback.
- [ ] Connection pool metrics and health reporting.

---

## Documentation Backlog

- [ ] Keep MkDocs as the default docs stack for v0.0.6.
- [ ] Add reusable snippets or generated examples only after the API stabilizes.
- [ ] Revisit VitePress or Docusaurus later if interactive docs become a release goal.
- [ ] Add a migration guide section for users coming from direct search methods.
- [ ] Add more real-output examples for `explain_plan()`, `analyze_plan()`, and BM25 diagnostics.

---

## Validation Commands

```bash
uv sync --group dev --group docs
uv run ruff check .
uv run ruff format --check .
uv run pytest
uv run pyright
NO_MKDOCS_2_WARNING=true uv run mkdocs build --strict
```

Notebook execution uses the local Docker database on port `9002`:

```bash
DB_HOST=localhost DB_PORT=9002 DB_USER=user DB_PASSWORD=root DB_NAME=postgres \
DB_CONNECTION_STRING=postgresql+asyncpg://user:root@localhost:9002/postgres \
uv run jupyter nbconvert --to notebook --execute --inplace examples/01_quickstart.ipynb
```

---

## References

- pgvector: https://github.com/pgvector/pgvector
- PostgreSQL EXPLAIN: https://www.postgresql.org/docs/current/sql-explain.html
- PostgreSQL indexes: https://www.postgresql.org/docs/current/indexes-types.html
- LanceDB query API inspiration: https://docs.lancedb.com/search/vector-search
- LanceDB filtering inspiration: https://docs.lancedb.com/search/filtering
- Product quantization: https://lear.inrialpes.fr/pubs/2011/JDS11/jegou_pq.pdf
- ColBERT: https://arxiv.org/abs/2004.12832
