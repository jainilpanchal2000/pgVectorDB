**Filtering Analysis & Enforcement Options**

This document analyzes how filtering is applied in the repository's vector search code and provides concrete, copyable options to guarantee pre-filtering for ANN (approximate nearest neighbor) searches.

Key code locations
- **`src/core.py`**: primary implementation for searches and indexes.
  - `set_query_params`: [src/core.py](src/core.py#L1061)
  - `_apply_query_params`: [src/core.py](src/core.py#L1102)
  - `semantic_search`: [src/core.py](src/core.py#L1551)
  - `asimilarity_search_by_vector`: [src/core.py](src/core.py#L1608)
  - `metadata_keyword_search` (CTE enforced): [src/core.py](src/core.py#L1826)
  - `metadata_semantic_search` (CTE enforced): [src/core.py](src/core.py#L1890)

Summary of current behaviors
- Plain `semantic_search` (no metadata filter): runs ANN across the whole table (no pre-filter). Planner may use a vector index for candidate retrieval; predicates are absent.
- `semantic_search` with `label_filter` (DiskANN + labels): code adds `WHERE labels && :labels`. If DiskANN index was built with `labels` (index created using `WITH (..., labels)`), pgvectorscale can restrict searches inside the ANN engine to those labels — true pre-filtering.
- `metadata_semantic_search`: uses a CTE `WITH filtered_docs AS (...)` and then runs ANN on the CTE. This encourages filtering-first, but without explicit `MATERIALIZED` it is not a strict guarantee because the planner may inline the CTE or choose a plan that results in post-filtering.

How to tell if a query pre-filters (diagnostics)
- Run `EXPLAIN ANALYZE` on the query and look for:
  - `Materialize` / `Subquery Scan on filtered_docs` — indicates the CTE was materialized (filter-first behavior).
  - `Index Scan ... Filter` — indicates index candidates were scanned and then WHERE was applied (post-filter on candidate set).
  - `Seq Scan` — full-table work (worst-case post-filtering).

Enforcement options (guaranteed pre-filtering)
1) DiskANN label-based search (pgvectorscale)
   - Build DiskANN index with labels included. In this codebase `build_index(..., include_labels=True)` will create a DiskANN index using labels.
   - Query using the `label_filter` argument. DiskANN will restrict the search inside the ANN engine to matching labels, guaranteeing pre-filtering.
   - Pros: Fastest for discrete label filtering at large scale. No planner ambiguity.
   - Cons: Requires pgvectorscale/DiskANN and labels stored in index-compatible format.

2) Partial (filtered) index
   - Create an index that only covers rows matching a predicate. This guarantees that queries with the same predicate will only search the index's subset.
   - Example (recommended: add a generated column for JSONB metadata keys to index easily):

     ALTER TABLE public.mytable ADD COLUMN category text GENERATED ALWAYS AS (langchain_metadata->>'category') STORED;

     CREATE INDEX idx_mytable_embedding_catX ON public.mytable
     USING hnsw (embedding vector_cosine_ops)
     WHERE category = 'X';

   - Query pattern:

     SELECT langchain_id, content, langchain_metadata,
            embedding <=> :embedding AS distance
     FROM public.mytable
     WHERE category = 'X'
     ORDER BY distance LIMIT :k;

   - Pros: Guaranteed index-only pre-filter for that predicate. Works with pgvector HNSW/IVFFlat as long as index supports predicate.
   - Cons: Partial indexes must match predicate exactly; not flexible for arbitrary predicates.

3) Partitioning (LIST or RANGE)
   - Partition the table by a discrete metadata key (e.g., category). Create vector indexes per-partition.
   - Example outline:

     CREATE TABLE public.docs_parent (
       langchain_id uuid PRIMARY KEY,
       content text,
       langchain_metadata jsonb,
       category text GENERATED ALWAYS AS (langchain_metadata->>'category') STORED,
       embedding vector(1536)
     ) PARTITION BY LIST (category);

     CREATE TABLE public.docs_catX PARTITION OF public.docs_parent FOR VALUES IN ('X');
     CREATE INDEX idx_docs_catX_embedding ON public.docs_catX USING hnsw (embedding vector_cosine_ops);

   - Query with `WHERE category = 'X'` will be partition-pruned; the planner will only search the matching partitions and their indexes (guaranteed pre-filtering).

4) Force materialization (CTE MATERIALIZED or temp table)
   - In `metadata_semantic_search` change the CTE to force materialization:

     WITH filtered_docs AS MATERIALIZED (
       SELECT langchain_id, content, langchain_metadata, embedding
       FROM schema.table
       WHERE {filter_clauses}
     )
     SELECT langchain_id, content, langchain_metadata,
            embedding <=> :embedding AS distance
     FROM filtered_docs
     ORDER BY distance LIMIT :k;

   - Alternatively, explicitly create a temp table per-query (more overhead but fully guaranteed):

     CREATE TEMP TABLE tmp_filtered AS
     SELECT langchain_id, content, langchain_metadata, embedding
     FROM schema.table
     WHERE {filter_clauses};
     ANALYZE tmp_filtered;
     SELECT ... FROM tmp_filtered ORDER BY embedding <=> :embedding LIMIT :k;

   - Pros: Guarantees filter-first behavior.
   - Cons: Materialization / temp table have per-query overhead; use when filtered set is much smaller than full dataset.

5) Partial index + generated columns recommended pattern
   - Use generated stored columns for frequently queried metadata keys to make predicates cheap and index-friendly:

     ALTER TABLE mytable ADD COLUMN category text GENERATED ALWAYS AS (langchain_metadata->>'category') STORED;
     CREATE INDEX idx_mytable_embedding_catX ON mytable USING hnsw (embedding vector_cosine_ops) WHERE category = 'X';

   - This gives predictable, fast, pre-filtered ANN operations for common predicates.

6) Diagnostics toggle (NOT for production)
   - For testing you can use `SET LOCAL enable_indexscan = off` or other planner toggles to observe alternative plans. The code's TODO suggests adding a `use_exact_search` flag to call `await conn.execute(text('SET LOCAL enable_indexscan = off'))` before queries for experiments. Do NOT rely on this in production.

Implementation suggestions for this repository
- Small, safe code changes I can make on request:
  1. Force CTE materialization in `metadata_semantic_search` and other metadata-* CTEs by using `MATERIALIZED` — small change, guarantees filter-first semantics.
  2. Call `await self._apply_query_params(conn)` at the start of search methods so `set_query_params()` actually affects runtime index tuning (the TODO already notes this).

Verification checklist
- After applying one or more enforcement options, verify with `EXPLAIN ANALYZE` (or call `await rag.explain_query(...)` from the code):
  - Confirm `Materialize` exists for CTE materialization or see `Index Scan` on a partial/partitioned index for pre-filtering.
  - Confirm `DiskANN` label-limited search uses labels in the plan (pgvectorscale should show label usage or internal notes).

References in repo
- TODO notes about DiskANN build params and label filtering: [TODO.md](TODO.md#L1)
- README mentions index types and `metadata_semantic_search`: [README.md](README.md#L45)

If you want, I can apply the small code edits now:
- Add `MATERIALIZED` to metadata CTEs and call `_apply_query_params(conn)` at the start of search methods so query params are applied correctly.

If you'd prefer docs only, this file provides the analysis and SQL examples you can copy into migrations or DBA scripts.
