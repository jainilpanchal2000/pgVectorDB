# pgVectorDB TODO — Production PostgreSQL Vector Database

**Last Updated:** 2026-06-23
**Versioning:** Incremental releases 
**Goal:** Add LanceDB-inspired production features + community requests
**Naming Convention:** `pgv_db` (consistent across all docs/examples)

---

## 🎯 v0.0.6 — Fluent API & Query Optimization

### 0. Fluent/Chainable Query API (LanceDB-Style)
**Priority:** 🔴 HIGH | **Effort:** Medium

New chainable API for intuitive query building:

```python
# LanceDB-style chainable API
results = await pgv_db.search([0.1] * 384)                   # Vector search
    .where("category = 'armor'")                            # Metadata filter
    .limit(3)                                               # Result limit
    .select(["content", "metadata", "score"])               # Column projection
    .to_list()                                              # Execute & return

# Alternative: Async iterator
async for row in pgv_db.search("query").where({"status": "active"}).iter():
    print(row)

# With hybrid search
results = await pgv_db.search("machine learning")           # Text query
    .where({"category": "ai"})                              # Filter
    .rerank(reranker=cross_encoder)                        # Rerank
    .limit(10)
    .to_arrow()                                            # Return PyArrow table

# Full example
results = await pgv_db.search(
        query="neural networks",
        vector_column="embedding"                           # Multi-vector column support
    )
    .where("price < 1000 AND category = 'electronics'")    # SQL-like filter
    .limit(20)
    .offset(10)                                             # Pagination
    .select(["content", "price", "score"])                  # Project columns
    .explain()                                              # Show plan before execute
    .to_list()                                              # Execute

# With post-filtering
results = await pgv_db.search("query")
    .where({"tags": ["ml", "ai"]}, prefilter=False)        # Post-filter (default: prefilter=True)
    .limit(5)
    .to_pandas()
```

**Implementation:**
```python
# New SearchBuilder class in search.py
class SearchBuilder:
    def __init__(self, db, query, search_type="semantic"):
        self.db = db
        self.query = query
        self.search_type = search_type
        self._where = None
        self._limit = 10
        self._offset = 0
        self._select = None
        self._prefilter = True
        self._reranker = None
        self._explain = False
    
    def where(self, filter_expr) -> "SearchBuilder":
        self._where = filter_expr
        return self
    
    def limit(self, n: int) -> "SearchBuilder":
        self._limit = n
        return self
    
    def offset(self, n: int) -> "SearchBuilder":
        self._offset = n
        return self
    
    def select(self, columns: list) -> "SearchBuilder":
        self._select = columns
        return self
    
    def rerank(self, reranker) -> "SearchBuilder":
        self._reranker = reranker
        return self
    
    def explain(self) -> "SearchBuilder":
        self._explain = True
        return self
    
    async def to_list(self) -> list:
        # Build and execute query
        pass
    
    async def to_pandas(self) -> pd.DataFrame:
        # Return as DataFrame
        pass
    
    async def to_arrow(self) -> pa.Table:
        # Return as PyArrow table
        pass
    
    async def iter(self):
        # Async iterator for large results
        pass

# Add to pgVectorDB class
def search(self, query, search_type="semantic"):
    return SearchBuilder(self, query, search_type)
```

**Why this matters:**
- More intuitive than positional arguments
- Chainable method calls
- Lazy execution (build query, then execute)
- SQL-like readability

**Resources:**
- LanceDB query builder: https://docs.lancedb.com/search/vector-search
- Python method chaining patterns: https://en.wikipedia.org/wiki/Fluent_interface
- PyArrow integration: https://arrow.apache.org/docs/python/index.html

---

### 1. Query Explain & Analyze
**Priority:** 🔴 HIGH | **Effort:** Low

```python
# Show query plan without executing
plan = await pgv_db.explain_plan(
    query="machine learning",
    search_method="hybrid_search",
    k=10
)
# Returns: PostgreSQL EXPLAIN output with index usage

# Execute with runtime metrics
stats = await pgv_db.analyze_plan(
    query="AI applications",
    search_method="semantic_search"
)
# Returns: elapsed_time_ms, rows_scanned, index_hits, io_read_bytes
```

**Implementation:**
- Wrap PostgreSQL `EXPLAIN (ANALYZE, BUFFERS, COSTS)`
- Parse output into structured dict
- Add to `AnalyticsMixin`

---

### 1a. SearchMethod Enum (Replace strings)
**Priority:** 🔴 HIGH | **Effort:** Low

**Current:** Using strings for search methods
```python
# ❌ Current - string-based
search_method = "semantic_search"  # or "hybrid_search", "keyword_search"
```

**New:** Strongly-typed enum
```python
from pgvectordb import SearchMethod

# ✅ New - enum-based
search_method = SearchMethod.SEMANTIC  # or HYBRID, KEYWORD, etc.

# For fluent API - automatically handled
db = await pgv_db.search("query", search_method=SearchMethod.HYBRID)  # Optional parameter

# Supported methods:
SearchMethod.SEMANTIC           # Vector similarity
SearchMethod.KEYWORD            # FTS/BM25 keyword search
SearchMethod.HYBRID             # Vector + keyword fusion
SearchMethod.MULTIMODAL         # Multi-embedding spaces
SearchMethod.TRIGRAM            # Fuzzy text matching
SearchMethod.METADATA_FILTER    # Pure metadata filtering
```

**Implementation:**
```python
# In base.py - add to existing enums
class SearchMethod(str, Enum):
    """Search method types for query operations."""
    SEMANTIC = "semantic"
    KEYWORD = "keyword"
    HYBRID = "hybrid"
    MULTIMODAL = "multimodal"
    TRIGRAM = "trigram"
    METADATA_FILTER = "metadata_filter"

# Update existing methods
async def explain_plan(
    self,
    query: str,
    search_method: Union[str, SearchMethod] = SearchMethod.SEMANTIC,  # Accept both for backward compat
    **search_kwargs
):
    # Normalize to enum
    if isinstance(search_method, str):
        search_method = SearchMethod(search_method.lower().replace("_search", ""))
```

**Backward Compatibility:**
- Accept both strings and enums
- Convert strings to enums internally
- Warn on deprecated string usage

**Resources:**
- Python Enum: https://docs.python.org/3/library/enum.html
- StrEnum (Python 3.11+): https://docs.python.org/3/library/enum.html#enum.StrEnum

---

**Resources:**
- PostgreSQL EXPLAIN: https://www.postgresql.org/docs/current/sql-explain.html
- LanceDB explain_plan: https://docs.lancedb.com/search/optimize-queries
- pgvector performance: https://github.com/pgvector/pgvector#performance

---

### 2. Advanced Query Parameters
**Priority:** 🔴 HIGH | **Effort:** Low

```python
results = await pgv_db.semantic_search(
    query="neural networks",
    k=10,
    
    # New parameters
    nprobes=20,           # IVF: partitions to scan
    ef_search=100,        # HNSW: candidate pool size
    refine_factor=3,      # Fetch k*3, rerank with full vectors
    
    # Distance bounds (near-duplicate detection)
    distance_range=(0.0, 0.3),
    
    # Bypass ANN for exact search
    use_exact_search=False
)
```

**Implementation:**
- Extend `_apply_query_params()` in `IndexingMixin`
- Add `SET LOCAL` for per-query params
- Distance range as `WHERE embedding <=> query BETWEEN x AND y`

**Resources:**
- pgvector params: https://github.com/pgvector/pgvector#query-options
- LanceDB refine_factor: https://docs.lancedb.com/search/optimize-queries

---

### 3. BTree Scalar Index for Metadata
**Priority:** 🔴 HIGH | **Effort:** Low

```python
# Create BTree index for range queries
await pgv_db.create_scalar_index(
    column="price",
    index_type="btree"  # or "btree" by default
)

# Now fast range filtering
results = await pgv_db.metadata_semantic_search(
    query="laptop",
    filter={"price": {"$gte": 100, "$lte": 500}},
    k=10
)
```

**Implementation:**
- Wrap PostgreSQL `CREATE INDEX ... USING BTREE`
- Auto-detect numeric/temporal columns
- Add to `IndexingMixin`

**Resources:**
- PostgreSQL BTree: https://www.postgresql.org/docs/current/indexes-types.html
- LanceDB BTree: https://docs.lancedb.com/indexing/scalar-index

---

### 4. Bitmap Scalar Index
**Priority:** 🔴 HIGH | **Effort:** Medium

```python
# For low-cardinality categorical data
await pgv_db.create_scalar_index(
    column="category",
    index_type="bitmap"
)

# Fast multi-value filtering
results = await pgv_db.metadata_semantic_search(
    query="shoes",
    filter={"category": {"$in": ["sports", "casual"]}},
    k=10
)
```

**Implementation:**
- Use `CREATE INDEX ... USING BITMAP` (if available)
- Or GIN index fallback for PostgreSQL
- Store cardinality stats in metadata

**Resources:**
- Bitmap indexes: https://en.wikipedia.org/wiki/Bitmap_index
- LanceDB Bitmap: https://docs.lancedb.com/indexing/scalar-index

---

## 🎯 v0.0.7 — Advanced Quantization & Prefilter Control

### 5. Product Quantization (PQ) Support
**Priority:** 🟡 MEDIUM | **Effort:** Medium

```python
# IVF with Product Quantization
await pgv_db.build_index(
    index_type=IndexType.IVFFLAT,
    quantization=QuantizationType.PQ,
    num_subvectors=32,    # dimension // 8 default
    num_bits=8,           # bits per subvector
    num_partitions=128
)

# ~8-16x compression vs raw vectors
```

**Implementation:**
- Requires pgvector extension support check
- Graceful fallback if not available
- Add quantization enum to `base.py`

**Resources:**
- PQ explained: https://lear.inrialpes.fr/pubs/2011/JDS11/jegou_pq.pdf
- LanceDB PQ: https://docs.lancedb.com/indexing/quantization
- Faiss PQ: https://github.com/facebookresearch/faiss/wiki/Indexes-for-quantization

---

### 6. RaBitQ (Binary Quantization)
**Priority:** 🟡 MEDIUM | **Effort:** High

```python
# Extreme compression: 1 bit per dimension
await pgv_db.build_index(
    index_type=IndexType.IVFFLAT,
    quantization=QuantizationType.RABITQ,  # or RQ
    num_bits=1,           # 1 bit per dim = 128x compression
    num_partitions=256
)

# Requirements: dimensions must be divisible by 8
# 1024-dim float32 vector: 4KB → ~100 bytes with RaBitQ
```

**Implementation:**
- Check pgvector version for support
- Create with corrective scalars
- Significant recall/latency trade-offs

**Resources:**
- RaBitQ paper: https://arxiv.org/abs/2405.12451
- LanceDB RaBitQ: https://lancedb.com/blog/feature-rabitq-quantization/
- Binary quantization: https://github.com/pgvector/pgvector/issues/613

---

### 7. Pre-filter vs Post-filter Control
**Priority:** 🟡 MEDIUM | **Effort:** Low

```python
results = await pgv_db.semantic_search(
    query="machine learning",
    filter={"status": "published"},
    
    # Control filter timing
    filter_mode=FilterMode.PRE_FILTER,   # Default: filter then search
    # OR
    filter_mode=FilterMode.POST_FILTER   # Search then filter
)

# Prefilter: Better recall, potentially slower
# Postfilter: Faster, may return fewer results
```

**Implementation:**
- `PRE_FILTER`: CTE with filtered docs, then vector search
- `POST_FILTER`: Vector search, then filter results
- Document trade-offs clearly

**Resources:**
- LanceDB pre/post-filter: https://docs.lancedb.com/search/filtering
- pgvector iterative scan: https://github.com/pgvector/pgvector#iterative-scan

---

### 8. LabelList Index for Arrays
**Priority:** 🟢 MEDIUM | **Effort:** Low

```python
# Index array columns (tags, categories)
await pgv_db.create_scalar_index(
    column="tags",
    index_type="labellist"
)

# Fast array containment queries
results = await pgv_db.metadata_semantic_search(
    query="python",
    filter={"tags": {"$contains": ["ml", "ai"]}},  # All must match
    k=10
)

# Or ANY match
results = await pgv_db.metadata_filter(
    filter={"tags": {"$has_any": ["ml", "ai", "dl"]}}
)
```

**Implementation:**
- GIN index on array columns
- Use PostgreSQL array operators: `@>`, `&&`
- Extend filter operators in `search.py`

**Resources:**
- PostgreSQL arrays: https://www.postgresql.org/docs/current/arrays.html
- GIN index: https://www.postgresql.org/docs/current/gin.html

---

## 🎯 v0.0.8 — Multi-Vector Search & Table Versioning

### 9. Multi-Vector / Late Interaction (ColBERT)
**Priority:** 🔴 HIGH | **Effort:** High

```python
# Create table with multiple vectors per document
await pgv_db.create_multivector_table(
    dims_per_vector=128,   # Each token is 128-dim
    max_vectors_per_doc=256  # Max tokens per doc
)

# Add document with token-level embeddings
doc = Document(
    page_content="The quick brown fox",
    metadata={
        "token_embeddings": [     # List of vectors
            [0.1, 0.2, ...],      # "The" token
            [0.3, 0.4, ...],      # "quick" token
            # ... one per token
        ]
    }
)
await pgv_db.add_document_multivector(doc)

# Search with MaxSim scoring
query_tokens = embed_tokens("fast animal")  # Multiple query vectors
results = await pgv_db.multivector_search(
    query_vectors=query_tokens,
    column="token_embeddings",
    scoring="maxsim",  # Σ(max_sim(query_tok, doc_toks))
    k=10
)
```

**Implementation:**
- New table schema: `vector[]` column type
- MaxSim: For each doc, sum max similarity per query token
- CTE + lateral join for efficient computation

**Resources:**
- ColBERT paper: https://arxiv.org/abs/2004.12832
- LanceDB multivector: https://docs.lancedb.com/search/multivector-search
- MaxSim explanation: https://en.wikipedia.org/wiki/Max-similarity
- PyLate (ColBERT): https://github.com/lightonai/pylate

---

### 10. Table Versioning / Time Travel
**Priority:** 🟡 MEDIUM | **Effort:** High

```python
# Automatic versioning on writes
await pgv_db.add_documents(docs)  # Creates version automatically

# List all versions
versions = await pgv_db.list_versions()
# Returns: [{"version": 1, "timestamp": ..., "operation": "INSERT", "rows": 100}]

# Checkout previous version
await pgv_db.checkout_version(version=2)
results = await pgv_db.semantic_search("query")  # Searches v2 data

# Tag important versions
await pgv_db.create_tag(name="prod", version=5)
await pgv_db.create_tag(name="baseline", version=10)

# Rollback to tag
await pgv_db.rollback_to_tag("baseline")

# Cleanup old versions (keep last 30 days)
await pgv_db.set_retention_policy(days=30)
await pgv_db.prune_old_versions()
```

**Implementation:**
- Temporal tables or audit triggers
- `pgaudit` extension or custom triggers
- Version metadata table
- `_version` column in main table

**Resources:**
- PostgreSQL temporal tables: https://www.postgresql.org/docs/current/ddl-system-columns.html
- LanceDB versioning: https://docs.lancedb.com/tables/versioning
- System-versioned tables: https://en.wikipedia.org/wiki/Temporal_database
- Audit triggers: https://wiki.postgresql.org/wiki/Audit_trigger

---

### 11. Versioning for Reproducibility
**Priority:** 🟢 MEDIUM | **Effort:** Medium

```python
# ML experiment reproducibility
@contextmanager
async def experiment():
    version = await pgv_db.get_current_version()
    await pgv_db.create_tag(f"exp_{experiment_id}", version)
    try:
        yield
    finally:
        # Always restore original state
        await pgv_db.rollback_to_version(version)

# Use in ML pipeline
async with experiment():
    await pgv_db.add_documents(new_training_data)
    results = await evaluate()
    # Auto-rollback after experiment
```

---

## 📚 Phase 4+ — Future Considerations (Backlog)

These are NOT in the v0.0.9 scope but tracked for future:

### Possible Additions
| Feature | Notes | Resources |
|---------|-------|-----------|
| FM-Index (Substring) | Substring search in text | https://docs.lancedb.com/indexing/scalar-index |
| IVF-HNSW Hybrid | HNSW per partition | https://docs.lancedb.com/indexing/vector-index |
| Scalar Quantization (SQ) | Per-dim compression | https://docs.lancedb.com/indexing/quantization |
| Query Auto-Embedding | Text queries without manual embed | https://docs.lancedb.com/embedding |
| Batch Search API | Multiple queries at once | - |
| Connection Pool Metrics | Pool stats, health | - |

---

## 🔗 Essential Resources Reference

### Databases & Vector Search
| Resource | Link | Use For |
|----------|------|---------|
| **pgvector** | https://github.com/pgvector/pgvector | Core vector extension reference |
| **pgvector Performance** | https://github.com/pgvector/pgvector#performance | Tuning parameters |
| **PostgreSQL EXPLAIN** | https://www.postgresql.org/docs/current/sql-explain.html | Query optimization |
| **PostgreSQL Index Types** | https://www.postgresql.org/docs/current/indexes-types.html | BTree, Bitmap, GIN |
| **PostgreSQL Arrays** | https://www.postgresql.org/docs/current/arrays.html | Array operations |

### LanceDB (Feature Inspiration)
| Resource | Link | Use For |
|----------|------|---------|
| **Vector Index** | https://docs.lancedb.com/indexing/vector-index | Index types, parameters |
| **Scalar Index** | https://docs.lancedb.com/indexing/scalar-index | BTree, Bitmap, LabelList |
| **Optimization** | https://docs.lancedb.com/search/optimize-queries | Query tuning |
| **Quantization** | https://docs.lancedb.com/indexing/quantization | PQ, SQ, RaBitQ |
| **Multivector** | https://docs.lancedb.com/search/multivector-search | ColBERT-style |
| **Filtering** | https://docs.lancedb.com/search/filtering | Pre/post filter |
| **Versioning** | https://docs.lancedb.com/tables/versioning | Time travel |

### Papers & Theory
| Resource | Link | Use For |
|----------|------|---------|
| **Product Quantization** | https://lear.inrialpes.fr/pubs/2011/JDS11/jegou_pq.pdf | PQ algorithm |
| **RaBitQ** | https://arxiv.org/abs/2405.12451 | Binary quantization |
| **ColBERT** | https://arxiv.org/abs/2004.12832 | Late interaction |
| **MaxSim** | https://en.wikipedia.org/wiki/Max-similarity | Scoring |

### Tools & Libraries
| Resource | Link | Use For |
|----------|------|---------|
| **PyLate** | https://github.com/lightonai/pylate | ColBERT embeddings |
| **Faiss** | https://github.com/facebookresearch/faiss | Quantization patterns |
| **pgaudit** | https://www.pgaudit.org/ | Audit/vVersioning |

---

## 📚 Documentation Improvements — MDX Migration

### Current State
- **Format:** Markdown (.md) files
- **Engine:** MkDocs with Material theme
- **Good:** Simple, fast, GitHub-compatible
- **Limitations:** No React components, limited interactivity

### Proposed: MDX Support
**Goal:** Richer documentation with interactive components

#### Option 1: Use MkDocs-Macros (Easiest)
Add Python-based dynamic content without full MDX:

```python
# mkdocs.yml plugin: mkdocs-macros
plugins:
  - macros:
      include_dir: docs/_includes
```

```markdown
<!-- docs/index.md -->
{{ include_snippet("quickstart.py") }}

{{ feature_matrix() }}

{{ version_badge() }}
```

**Pros:** Minimal change, works with existing setup
**Cons:** Not true MDX, limited component ecosystem

#### Option 2: Migrate to VitePress (Vue-based)
LanceDB uses VitePress with custom components:

```javascript
// .vitepress/config.ts
import { defineConfig } from 'vitepress'

export default defineConfig({
  title: 'pgVectorDB',
  themeConfig: {
    // Similar to current MKDocs config
  }
})
```

```vue
<!-- docs/index.mdx -->
<script setup>
import FeatureMatrix from './components/FeatureMatrix.vue'
import CodeTabs from './components/CodeTabs.vue'
</script>

# pgVectorDB

<FeatureMatrix />

<CodeTabs :languages="['python', 'typescript']">
  <template #python>
    ```python
    await pgv_db.semantic_search("query")
    ```
  </template>
  <template #typescript>
    ```typescript
    await pgv_db.semanticSearch("query")
    ```
  </template>
</CodeTabs>
```

**Pros:** 
- True MDX with Vue components
- Interactive playgrounds (Vue Live)
- Better search (Algolia DocSearch)
- Dark mode, i18n out of box

**Cons:** 
- Migration effort (convert all .md → .mdx)
- New build system (Vite vs MkDocs)
- Hosting change (may need Netlify/Vercel)

#### Option 3: Use Docusaurus (React-based)
Meta's documentation framework:

```javascript
// docusaurus.config.js
module.exports = {
  title: 'pgVectorDB',
  themeConfig: {
    navbar: { items: [...] },
    prism: { theme: require('prism-react-renderer/themes/dracula') }
  }
}
```

```mdx
<!-- docs/index.mdx -->
import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';
import FeatureMatrix from '@site/src/components/FeatureMatrix';

# pgVectorDB

<FeatureMatrix />

<Tabs>
  <TabItem value="python" label="Python">
    ```python
    await pgv_db.semantic_search("query")
    ```
  </TabItem>
  <TabItem value="typescript" label="TypeScript">
    ```typescript
    await pgv_db.semanticSearch("query")
    ```
  </TabItem>
</Tabs>
```

**Pros:**
- Full React ecosystem
- Plugin ecosystem (docs, blog, search)
- Versioned docs built-in
- Easy deployment to Vercel/Netlify

**Cons:**
- Heavier than VitePress
- More complex configuration
- React knowledge required

### Recommendation
**Phase 1 (v0.0.6):** Stay with MkDocs + MkDocs-Macros
- Add dynamic snippets
- Add feature matrices
- Minimal disruption

**Phase 2 (v0.0.9):** Migrate to VitePress
- Better MDX support
- Matches LanceDB's quality
- Interactive components

### Quick Win: Enhance Current MkDocs
Add these to `mkdocs.yml` now:

```yaml
plugins:
  - macros:  # Add dynamic content
      include_dir: docs/_includes
  - minify:  # Minify HTML
  - git-revision-date-localized:  # Show last modified

extra:
  version: 0.0.5
  social:
    - icon: fontawesome/brands/github
      link: https://github.com/jainilpanchal2000/pgvectordb

hooks:
  - docs/hooks/feature_matrix.py  # Auto-generate feature table
```

**Resources:**
- MkDocs-Macros: https://mkdocs-macros-plugin.readthedocs.io/
- VitePress: https://vitepress.dev/
- Docusaurus: https://docusaurus.io/
- LanceDB docs as reference: https://github.com/lancedb/docs

---

## ✅ Completion Checklist

### v0.0.6 — Query Optimization
- [ ] `explain_plan()` method
- [ ] `analyze_plan()` method
- [ ] Query params: `nprobes`, `ef_search`, `refine_factor`
- [ ] Distance range filtering
- [ ] `use_exact_search` flag
- [ ] BTree scalar index
- [ ] Bitmap scalar index

### v0.0.7 — Quantization & Control
- [ ] Product Quantization (PQ) support
- [ ] RaBitQ quantization
- [ ] Pre-filter / Post-filter control
- [ ] LabelList index for arrays
- [ ] Index statistics with `wait_for_index()`

### v0.0.8 — Multi-Vector & Versioning
- [ ] Multi-vector table schema
- [ ] MaxSim scoring for late interaction
- [ ] `multivector_search()` method
- [ ] ColBERT integration example
- [ ] Table versioning with auto-tracking
- [ ] `checkout_version()` for time travel
- [ ] Tag-based versioning
- [ ] Rollback functionality
- [ ] Retention policy for cleanup

---

## 🎯 Success Metrics

For each release:
1. **All features tested** with unit tests
2. **Documentation updated** (docstrings + user guide)
3. **Example notebooks** for major features
4. **Performance benchmarks** showing improvement
5. **Backward compatibility** maintained

---

## 👥 Handoff Notes

When passing this TODO to another developer:

1. **Start with v0.0.6** — low risk, quick wins
   - Use `pgv_db` naming convention consistently
2. **pgvector version matters** — check extension support for quantization
3. **Test with real data** — synthetic benchmarks can be misleading
4. **Documentation is part of the feature** — don't skip docs/
5. **Ask questions early** — vector DB internals are complex

### Key Contacts / Communities
- pgvector Discord/GitHub discussions
- LanceDB Discord (for feature inspiration)
- PostgreSQL mailing lists

### Testing Requirements
- PostgreSQL 14+ with pgvector
- Test with 100k+ vectors for realistic benchmarks
- Memory monitoring for quantization features

---

**Last Updated:** 2026-06-23
**Next Review:** After v0.0.10 release
**Maintainer:** Jainil Panchal

---

*"Don't just copy — leverage PostgreSQL's strengths while adding vector-native features."*
