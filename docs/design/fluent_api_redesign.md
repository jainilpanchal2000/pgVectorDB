# Fluent API Design Notes

This page records the final direction of the pgVectorDB fluent API. It is design context for maintainers; user-facing examples live in [Quickstart](../getting_started/quickstart.md) and [Search & Retrieval](../user_guide/search_and_retrieval.md).

## Final Entry Point

The public fluent entry point is `db.query(...)`.

```python
results = await (
    db.query("machine learning")
    .semantic()
    .where({"category": "ai"})
    .limit(10)
    .to_list()
)
```

The builder is lazy. It stores the query configuration until an execution method is called.

| Concern | Fluent method family |
| --- | --- |
| Search mode | `.semantic()`, `.keyword()`, `.hybrid()`, `.trigram()`, `.search_mode(...)` |
| Filters | `.where(...)` |
| Pagination and projection | `.limit(...)`, `.offset(...)`, `.select(...)` |
| Vector tuning | `.ef(...)`, `.nprobes(...)`, `.refine_factor(...)`, `.distance_range(...)`, `.bypass_vector_index()` |
| Keyword tuning | `.fts(...)`, `.bm25()`, `.bm25_params(...)`, `.phrase(...)`, `.universal(...)` |
| Hybrid tuning | `.weights(...)`, `.rrf(...)` |
| Multimodal search | `.in_space(...)`, `.across_spaces(...)` |
| Reranking | `.rerank(...)` |
| Output | `.to_list()`, `.to_pandas()`, `.to_arrow()` |
| Analysis | `.explain_plan()`, `.analyze_plan()` |

## Why This Shape

The old public surface had many explicit methods: `semantic_search`, `keyword_search`, `hybrid_search`, `metadata_semantic_search`, `metadata_keyword_search`, `trigram_search`, and related variants. Those methods are still useful compatibility and implementation surfaces, but they force users to learn combinations instead of workflows.

The fluent API keeps the important choices in one readable chain:

```python
results = await (
    db.query("database optimization")
    .hybrid()
    .where({"topic": "postgres"})
    .rrf(k=60)
    .ef(100)
    .limit(10)
    .to_list()
)
```

This makes it easier to build dynamic application queries, expose search controls in products, and compare retrieval settings during evaluation.

## Legacy Mapping

| Legacy method | Fluent equivalent |
| --- | --- |
| `semantic_search(query, k=10)` | `await db.query(query).semantic().limit(10).to_list()` |
| `keyword_search(query, k=10)` | `await db.query(query).keyword().limit(10).to_list()` |
| `hybrid_search(query, k=10)` | `await db.query(query).hybrid().limit(10).to_list()` |
| `metadata_semantic_search(query, filter, k=10)` | `await db.query(query).semantic().where(filter).limit(10).to_list()` |
| `trigram_search(query, k=10)` | `await db.query(query).trigram().limit(10).to_list()` |

## Compatibility Rule

Do not remove the existing method-based search API without a formal deprecation plan. The fluent builder should remain a high-level composition layer over the stable lower-level methods.

## Documentation Rule

New user-facing docs should use `db.query(...)` first. Legacy methods should appear only in migration notes, compatibility sections, or API reference pages.