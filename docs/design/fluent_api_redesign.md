# Fluent API Design for All Search Methods

## Design Philosophy

The fluent API should support ALL search methods through a unified entry point with context-aware parameters. Each method has specific requirements:

## Proposed API Design

### Entry Point: `db.query()` or keep `db.search()`

```python
# Unified entry - type determined by context
results = await db.query("machine learning").semantic().limit(10).to_list()
results = await db.query("machine learning").keyword(type=KeywordSearchType.BM25).limit(10).to_list()
results = await db.query("machine learning").hybrid(weights=(0.7, 0.3)).limit(10).to_list()
```

### Alternative: Method-based selection

```python
# Option A: Chained method selector
results = await db.search("query").as_keyword(type=BM25).limit(10).to_list()
results = await db.search("query").as_semantic().limit(10).to_list()
results = await db.search("query").as_hybrid(weights=(0.7, 0.3)).limit(10).to_list()
results = await db.search("query").as_trigram(threshold=0.3).limit(10).to_list()

# Option B: Enum-based selection
results = await db.search("query").using(SearchMethod.KEYWORD).limit(10).to_list()

# Option C: Keep search() simple, add specific entry points
results = await db.semantic("query").limit(10).to_list()
results = await db.keyword("query").type(BM25).limit(10).to_list()
results = await db.hybrid("query").weights(0.7, 0.3).limit(10).to_list()
```

## Recommended Design: Hybrid Approach

```python
class pgVectorDB:
    # Entry points return specialized builders
    def search(self, query) -> SemanticQueryBuilder:
        """Default: semantic search"""
        return SemanticQueryBuilder(...)

    def keyword(self, query) -> KeywordQueryBuilder:
        """Dedicated keyword search entry"""
        return KeywordQueryBuilder(...)

    def hybrid(self, query) -> HybridQueryBuilder:
        """Dedicated hybrid search entry"""
        return HybridQueryBuilder(...)

    def trigram(self, query) -> TrigramQueryBuilder:
        """Dedicated trigram (fuzzy) search entry"""
        return TrigramQueryBuilder(...)

    def vector(self, embedding) -> VectorQueryBuilder:
        """Search with pre-computed embedding"""
        return VectorQueryBuilder(...)
```

## Builder Hierarchy

```python
# Base builder with common methods
class BaseQueryBuilder(ABC):
    def where(self, filter) -> Self
    def limit(self, n) -> Self
    def offset(self, n) -> Self
    def select(self, columns) -> Self
    def explain_plan(self) -> Dict
    async def analyze_plan(self) -> Dict
    async def to_list(self) -> List[QueryResult]
    async def to_pandas(self) -> DataFrame
    async def to_arrow(self) -> Table

# Semantic/Vector builder
class SemanticQueryBuilder(BaseQueryBuilder):
    def ef(self, n) -> Self          # HNSW
    def nprobes(self, n) -> Self     # IVF
    def refine_factor(self, n) -> Self
    def distance_range(self, lower, upper) -> Self
    def bypass_vector_index(self) -> Self
    def rerank(self, reranker) -> Self
    def nearest_to_text(self, text) -> HybridQueryBuilder  # Convert to hybrid

# Keyword builder
class KeywordQueryBuilder(BaseQueryBuilder):
    def type(self, search_type: KeywordSearchType) -> Self  # FTS or BM25
    def bm25_params(self, k1: float, b: float) -> Self
    def text_config(self, lang: str) -> Self
    def universal(self, metadata_fields: List[str]) -> Self  # Boost matches

# Hybrid builder
class HybridQueryBuilder(BaseQueryBuilder):
    def weights(self, semantic: float, keyword: float) -> Self
    def rrf_k(self, k: int) -> Self      # RRF parameter
    def use_rrf(self, enabled: bool = True) -> Self
    def keyword_type(self, type: KeywordSearchType) -> Self
    def bm25_params(self, k1: float, b: float) -> Self

# Trigram builder
class TrigramQueryBuilder(BaseQueryBuilder):
    def threshold(self, min_similarity: float) -> Self
    def case_sensitive(self, enabled: bool) -> Self
```

## Usage Examples

### 1. Semantic Search (Default)

```python
# Simple
results = await db.search("machine learning").limit(10).to_list()

# With tuning
results = await (
    db.search("machine learning")
    .ef(100)
    .refine_factor(2)
    .limit(10)
    .to_list()
)

# Filtered
results = await (
    db.search("machine learning")
    .where({"category": "ai", "year": {"$gte": 2024}})
    .limit(10)
    .to_list()
)
```

### 2. Keyword Search

```python
# BM25
results = await (
    db.keyword("machine learning")
    .type(KeywordSearchType.BM25)
    .bm25_params(k1=1.2, b=0.75)
    .limit(10)
    .to_list()
)

# FTS
results = await db.keyword("machine learning").type(KeywordSearchType.FTS).limit(10).to_list()

# Universal (boost metadata matches)
results = await (
    db.keyword("machine learning")
    .universal(metadata_fields=["title", "tags"])
    .limit(10)
    .to_list()
)
```

### 3. Hybrid Search

```python
# Weighted fusion
results = await (
    db.hybrid("machine learning")
    .weights(semantic=0.7, keyword=0.3)
    .limit(10)
    .to_list()
)

# RRF fusion
results = await (
    db.hybrid("machine learning")
    .use_rrf()
    .rrf_k(60)
    .keyword_type(KeywordSearchType.BM25)
    .limit(10)
    .to_list()
)
```

### 4. Trigram (Fuzzy) Search

```python
results = await (
    db.trigram("machin learnng")  # Typo-tolerant
    .threshold(0.3)
    .limit(10)
    .to_list()
)
```

### 5. Pre-computed Vector

```python
embedding = [0.1, 0.2, ...]  # Your embedding
results = await (
    db.vector(embedding)
    .where({"category": "ai"})
    .limit(10)
    .to_list()
)
```

### 6. Ensemble Search (Filter + Hybrid)

```python
results = await (
    db.hybrid("financial reports")
    .where({"department": "finance", "year": 2024})
    .weights(0.8, 0.2)
    .limit(10)
    .to_list()
)
```

## Multimodal / Spaces Integration

```python
class MultimodalQueryBuilder(BaseQueryBuilder):
    def in_space(self, space: VectorSpace) -> Self
    def across_spaces(self, spaces: List[VectorSpace], weights: List[float]) -> Self

# Usage
results = await (
    db.multimodal(query)
    .in_space(db.spaces.text)  # Text embedding space
    .limit(10)
    .to_list()
)

# Multi-space search
results = await (
    db.multimodal(query)
    .across_spaces(
        [db.spaces.text, db.spaces.price, db.spaces.category],
        weights=[0.7, 0.2, 0.1]
    )
    .limit(10)
    .to_list()
)
```

## Extension Requirements (Mandatory)

All features require these extensions:

```python
# Required for all pgVectorDB operations
REQUIRED_EXTENSIONS = {
    "vector": "Core vector operations (pgvector)",
    "pg_trgm": "Trigram fuzzy search",
    "vectorscale": "DiskANN index with SBQ compression",
    "pg_textsearch": "BM25 keyword ranking",
}
```

## Migration from Old API

| Old API | New Fluent API |
|---------|----------------|
| `db.semantic_search(query, k=10)` | `await db.search(query).limit(10).to_list()` |
| `db.keyword_search(query, k=10, search_type=BM25)` | `await db.keyword(query).type(BM25).limit(10).to_list()` |
| `db.hybrid_search(query, k=10, use_rrf=True)` | `await db.hybrid(query).use_rrf().limit(10).to_list()` |
| `db.trigram_search(query, k=10)` | `await db.trigram(query).limit(10).to_list()` |
| `db.metadata_semantic_search(query, filter, k=10)` | `await db.search(query).where(filter).limit(10).to_list()` |
| `db.asimilarity_search_by_vector(embedding, k=10)` | `await db.vector(embedding).limit(10).to_list()` |
