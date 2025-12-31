# Project Checklist — Prioritized TODO

Purpose: single, checklist-style TODO for engineers and reviewers. Items are grouped by priority and include file references, subtasks, and acceptance criteria to make implementation straightforward.

How to use this file:
- Read the priority groups and pick the highest-priority item you will work on.
- Before implementing, mark the item as `in-progress` in the project's tracker (or use the todo tracker used by the team).
- After implementation, add tests and update the acceptance criteria in this file.

---

## High Priority — Parity & Query Controls

- [ ] **Iterative Index Scan Controls** — expose iterative scan modes and limits
	- Files: `src/core.py`, `src/config.py`, `test/test_suite.py`, `docs/CONFIGURATION.md`
	- Subtasks:
		- [ ] Add `hnsw.iterative_scan` and `ivfflat.iterative_scan` (values: `strict_order` | `relaxed_order`).
		- [ ] Add `hnsw.max_scan_tuples`, `hnsw.scan_mem_multiplier`, `ivfflat.max_probes`.
		- [ ] Update `VALID_QUERY_PARAMS` and `set_query_params()` with validation and logging.
		- [ ] Ensure `_apply_query_params()` issues `SET LOCAL <param> = <value>` inside query transactions.
	- Acceptance:
		- [ ] Tests confirm `SHOW hnsw.iterative_scan` returns the set value inside the transaction.
		- [ ] Benchmarks show expected recall/latency changes when tuning.

- [ ] **Exact-Search Toggle (`enable_indexscan`)** — programmatic exact-search for benchmarking
	- Files: `src/core.py`, `test/test_suite.py`, `docs/CONFIGURATION.md`, `eval/scripts/benchmark_all_methods.py`
	- Subtasks:
		- [ ] Add `use_exact_search: bool` parameter to `semantic_search()` and `asimilarity_search_by_vector()` (default False).
		- [ ] When True, apply `SET LOCAL enable_indexscan = off` in the transaction before the SELECT.
		- [ ] Add tests that compare exact vs approximate results on a deterministic dataset.
		- [ ] Add a benchmark mode to `eval` scripts to compare exact vs approximate (CSV/JSON output).
	- Acceptance:
		- [ ] Tests show `use_exact_search=True` matches the exact baseline.

- [ ] **DiskANN Parallel Build Parameters (pgvectorscale)** — pre-index tuning API
	- Files: `src/core.py`, `test/test_suite.py`, `docs/CONFIGURATION.md`
	- Subtasks:
		- [ ] Add `set_diskann_build_params()` to capture settings like `diskann.force_parallel_workers`, `diskann.min_vectors_for_parallel_build`, `diskann.parallel_flush_interval`, `diskann.parallel_initial_start_nodes_count`.
		- [ ] Run stored `SET` commands (session-level) before `CREATE INDEX` in `_build_diskann_index()` and clear afterwards.
		- [ ] Add docs and a smoke test that builds a DiskANN index using the settings.
	- Acceptance:
		- [ ] Smoke test demonstrates DiskANN build completes with the requested parallel settings.

---

## Medium Priority — Defaults, Tests, and Validation

- [ ] **Default Tuning Config**
	- Files: `src/config.py`, `src/core.py`, `docs/CONFIGURATION.md`
	- Subtasks:
		- [ ] Add `DEFAULT_IVFFLAT_PROBES`, `DEFAULT_HNSW_EF_SEARCH`, and default iterative-mode/limits to `src/config.py`.
		- [ ] Load defaults at `initialize()` or on first search unless overridden.
	- Acceptance:
		- [ ] Config exposes defaults and they are applied automatically.

- [ ] **Extension Version Check**
	- Files: `src/core.py`, `docker/README.md`, `scripts/test_connection.py`
	- Subtasks:
		- [ ] After `CREATE EXTENSION`, query `pg_extension` for `extversion` for `vector` and `vectorscale`.
		- [ ] Warn or fail if versions are older than the minimal supported version (configurable; e.g., >= 0.8.0).
	- Acceptance:
		- [ ] `initialize()` emits version warnings when appropriate; test harness can simulate older versions.

- [ ] **Tests: Settings & Behavior**
	- Files: `test/test_suite.py`, `eval/`
	- Subtasks:
		- [ ] Add tests that verify `_apply_query_params()` issues `SET LOCAL` for params.
		- [ ] Add exact vs approximate comparison tests for `use_exact_search`.
		- [ ] Add DiskANN null-handling and label-filtering tests.
	- Acceptance:
		- [ ] Tests added to `test/test_suite.py` and pass locally.

---

## Lower Priority — Docs, Benchmarking, and Ops

- [ ] **Docs & Examples**
	- Files: `README.md`, `docs/CONFIGURATION.md`, `docker/README.md`
	- Subtasks:
		- [ ] Add code snippets for `set_query_params(...)`, `use_exact_search`, and DiskANN build `SET` usage.
		- [ ] Add Docker notes recommending pgvector/pgvectorscale versions and memory/maintenance settings.
	- Acceptance:
		- [ ] Docs include 3–5 short, copy-paste examples.

- [ ] **Benchmarking Modes**
	- Files: `eval/scripts/benchmark_all_methods.py`, `eval/data/`
	- Subtasks:
		- [ ] Add a benchmark mode that toggles `enable_indexscan` and sweeps tuning parameters; write CSV/JSON outputs.
	- Acceptance:
		- [ ] Benchmarks reproducibly produce recall vs latency outputs.

---

## Cross-check & Misc (quick scan results)

- [ ] I searched the repo for TODO/FIXME markers and key hooks (`VALID_QUERY_PARAMS`, `set_query_params`, `pg_extension extversion` queries). No additional critical features were found that aren't captured above.
- [ ] `scripts/test_connection.py` already queries `pg_extension` for version info — we should centralize that logic into `src/core.py`'s `_ensure_extensions()`.

---

If you like this style, I will:
1. Mark the relevant todo items as in-progress in the todo tracker before I implement.
2. Start with the highest-priority item (#1 Iterative scan params + #2 Exact search toggle) and push changes to `src/core.py` and tests.

Which item should I start implementing now? (I recommend starting with #1 + #2.)

---

## Analysis & Gap Findings (Repo Verification)

Summary: I reviewed `TODO.md` and inspected `src/core.py`, `scripts/test_connection.py`, and relevant files to verify which tasks are implemented, partially implemented, or missing. The following is a concise, actionable summary you can copy into the repository as a reference for implementers and reviewers.

- **What I checked:** existence of `set_query_params()`, `_apply_query_params()`, query parameter allowlist (`VALID_QUERY_PARAMS`), semantic search paths (`semantic_search`, `asimilarity_search_by_vector`), DiskANN build path, BM25 builder, and extension/version checks.

- **High-level status per major TODO:**
	- Iterative Index Scan Controls: NOT IMPLEMENTED — the requested params (`hnsw.iterative_scan`, `ivfflat.iterative_scan`, `hnsw.max_scan_tuples`, `hnsw.scan_mem_multiplier`, `ivfflat.max_probes`) are not present in `VALID_QUERY_PARAMS` or `set_query_params()`.
	- Exact-Search Toggle (`enable_indexscan`): NOT IMPLEMENTED — `semantic_search()` and `asimilarity_search_by_vector()` do not accept `use_exact_search` and there is no `SET LOCAL enable_indexscan = off` usage in search transactions.
	- DiskANN Parallel Build Parameters: PARTIALLY IMPLEMENTED — DiskANN index creation is implemented (WITH clause options exist), but there is no `set_diskann_build_params()` to run pre-index `SET` session parameters for parallel build tuning.
	- Default Tuning Config: PARTIALLY IMPLEMENTED — `set_query_params()` exists for runtime params; repository lacks centralized default constants (e.g., `DEFAULT_IVFFLAT_PROBES`, `DEFAULT_HNSW_EF_SEARCH`) in `src/config.py`.
	- Extension Version Check: PARTIALLY IMPLEMENTED — `_ensure_extensions()` creates needed extensions but does not query `pg_extension` for `extversion`; `scripts/test_connection.py` already performs version checks and should be consolidated.
	- Tests (Settings & Behavior): NOT FULLY IMPLEMENTED — no tests currently verify `SET LOCAL` application or exact vs approximate comparisons; DiskANN label tests exist but not pre-index `SET` smoke tests.

- **Reference comparison (pgvector / pgvectorscale / pg_textsearch):**
	- pgvector: we use core vector features and HNSW/IVFFlat support; missing: exposing additional planner/scan tuning params referenced in TODO.
	- pgvectorscale (DiskANN): we implement DiskANN index creation and label filtering, but we are missing pre-index build parameter application (parallel build tuning) and a smoke test to validate them.
	- pg_textsearch: BM25 creation is implemented and validated against an allowlist; no immediate feature gap other than version checks and documentation examples.

### Prioritized Missing Features (recommended order)
1. Exact-search toggle (`use_exact_search`) in search methods (high value for benchmarking).  
2. Ensure `_apply_query_params()` is applied inside search transactions (call it right after opening a connection in each search method).  
3. Add iterative-scan query params to allowlist and `set_query_params()` (expose `hnsw.iterative_scan`, `hnsw.max_scan_tuples`, `hnsw.scan_mem_multiplier`, `ivfflat.iterative_scan`, `ivfflat.max_probes`).  
4. Add `set_diskann_build_params()` and apply pre-index `SET` commands before creating DiskANN index (smoke test).  
5. Centralize extension `extversion` checks in `_ensure_extensions()` and log/warn if below the minimum supported versions.  
6. Add tests that assert `SET LOCAL` behavior and exact vs approximate equality on deterministic datasets.  

### Concrete code pointers (where to change)
- `src/core.py`:
	- Call `await self._apply_query_params(conn)` immediately after `async with self.sqlalchemy_engine.connect() as conn:` in search methods (`semantic_search`, `asimilarity_search_by_vector`, and other search entrypoints) before executing the SELECT.  
	- Add `use_exact_search: bool = False` param to `semantic_search()` and `asimilarity_search_by_vector()`. When True, run `await conn.execute(text('SET LOCAL enable_indexscan = off'))` (or add special-case handling in `_apply_query_params()`).  
	- Extend `set_query_params()` to accept new iterative scan and DiskANN build parameters (and validate values).  
	- Implement `set_diskann_build_params()` and apply stored `SET` commands when building DiskANN index (transaction-scoped `SET LOCAL`).  
	- At the end of `_ensure_extensions()`, run `SELECT extname, extversion FROM pg_extension WHERE extname IN ('vector','vectorscale','pg_textsearch')` and compare versions to configurable minima.

- `src/config.py`:
	- Add constants `DEFAULT_IVFFLAT_PROBES`, `DEFAULT_HNSW_EF_SEARCH` and any iterative defaults. Load into `pgVectorDB` on `initialize()` if `_query_params` is empty.

- `test/test_suite.py` and `eval/scripts/benchmark_all_methods.py`:
	- Add tests for `SET LOCAL` behavior; add benchmark mode toggles for `use_exact_search` and ranges for `ef_search`/`probes`.

### Suggested small-step implementation plan (minimal, low-risk)
1. Add `await self._apply_query_params(conn)` in `semantic_search()` and `asimilarity_search_by_vector()` before running the SELECT. This ensures current `set_query_params()` values are applied during search.
2. Add `use_exact_search` boolean parameter to those methods and implement the simple `SET LOCAL enable_indexscan = off` when True.
3. Add extension version query in `_ensure_extensions()` and log results.  
4. Add tests for steps 1-2 (unit/integration tests that run a local DB or rely on the existing `scripts/test_connection.py`).
If you want, I can implement steps 1–3 now (small targeted patch). Tell me to proceed and I will mark the relevant TODO items in the tracker as in-progress and apply the code edits, then run a quick lint/syntax check.
