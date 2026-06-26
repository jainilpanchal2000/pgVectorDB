# pgVectorDB

pgVectorDB is a PostgreSQL-native vector search and RAG toolkit built on `pgvector`, SQLAlchemy async, and LangChain-compatible documents. It gives you a fluent query API, multimodal spaces, production indexing controls, SQL diagnostics, reranking, and built-in retrieval evaluation without leaving Postgres.

**Current version:** 0.0.6  
**Recommended API:** `db.query(...)` for new application code  
**Status:** Active beta. The fluent API, docs, examples, and notebooks are being prepared for the next OSS release.

## New in 0.0.6

- LanceDB-style fluent query builder: `.semantic()`, `.keyword()`, `.hybrid()`, `.trigram()`, `.where(...)`, `.limit(...)`, `.select(...)`, and lazy execution with `.to_list()`.
- First-class BM25 and PostgreSQL full-text search controls through `.keyword().bm25()`, `.keyword().fts()`, and hybrid variants.
- Query diagnostics through `explain_plan()`, `analyze_plan()`, and raw PostgreSQL `explain_query()`.
- Multimodal ranking with `.across_spaces(...)` for text, numeric, category, and recency signals.
- Output helpers for Python lists, pandas DataFrames, and PyArrow tables.
- Reranking support for callable scorers and bundled reranker objects.

## Why pgVectorDB

| Capability | What it gives you |
| --- | --- |
| Fluent search API | Compose semantic, keyword, hybrid, trigram, filters, tuning, reranking, and output formats with `db.query(...)`. |
| PostgreSQL-native storage | Keep vectors, content, metadata, scalar indexes, full-text search, and query plans in one database. |
| Recency and multimodal spaces | Combine text with numeric, category, and timestamp signals such as freshness or popularity. |
| Built-in evaluators | Measure precision, recall, MRR, NDCG, hit rate, and K-value tradeoffs from retrieved document IDs. |
| SQL analyzers | Use `explain_plan()`, `analyze_plan()`, and `explain_query()` to inspect real PostgreSQL execution behavior. |
| Optimization tools | Build HNSW, IVFFlat, DiskANN, scalar indexes, BM25 indexes, binary quantized indexes, and subvector indexes. |
| Reranking integrations | Use CrossEncoder, HuggingFace, Cohere, or AWS Bedrock rerankers. |
| LangChain compatibility | Insert LangChain `Document` objects and expose pgVectorDB as a LangChain retriever. |

## Install

```bash
pip install pgvectordb

# Local embeddings
pip install 'pgvectordb[huggingface]'

# AWS Bedrock embeddings
pip install 'pgvectordb[aws]'

# Reranking integrations
pip install 'pgvectordb[rerankers,cohere]'

# Pandas and PyArrow output helpers
pip install 'pgvectordb[dataframe]'
```

## Start PostgreSQL

The repository includes a Docker setup with PostgreSQL 17, `pgvector`, `vectorscale`, and `pg_textsearch`.

```bash
cd docker
docker compose up -d
```

For a manual database, enable the core extensions you need:

```sql
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pg_trgm;

-- Optional, for DiskANN and BM25 features:
CREATE EXTENSION IF NOT EXISTS vectorscale CASCADE;
CREATE EXTENSION IF NOT EXISTS pg_textsearch;
```

## Quickstart

```python
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

from pgvectordb import DistanceMetric, IndexType, pgVectorDB


embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

db = pgVectorDB(
    collection_name="docs",
    embedding_model=embeddings,
    connection_string="postgresql+asyncpg://user:root@localhost:9002/postgres",
    index_type=IndexType.HNSW,
)

await db.initialize()

await db.add_documents([
    Document(
        page_content="pgVectorDB supports fluent PostgreSQL vector search.",
        metadata={"topic": "search", "year": 2024, "tier": "guide"},
    ),
    Document(
        page_content="Built-in evaluators help compare retrieval quality at different K values.",
        metadata={"topic": "evaluation", "year": 2024, "tier": "guide"},
    ),
])

await db.build_index(metric=DistanceMetric.COSINE)

results = await (
    db.query("how do I evaluate vector search quality?")
    .hybrid()
    .where({"year": {"$gte": 2024}})
    .rrf(k=60)
    .ef(100)
    .limit(5)
    .to_list()
)
```

## Fluent Query API

`db.query(...)` is the recommended entry point for new application code.

```python
# Semantic search
rows = await db.query("retrieval augmented generation").semantic().limit(10).to_list()

# Keyword or BM25 search
rows = await db.query("SOC 2 retention policy").keyword().bm25().limit(10).to_list()

# Hybrid search with reciprocal rank fusion
rows = await db.query("database backup policy").hybrid().rrf(k=60).limit(10).to_list()

# Typo-tolerant trigram search
rows = await db.query("vectro databse").trigram().threshold(0.2).limit(10).to_list()

# Output formats
frame = await db.query("search quality").semantic().limit(20).to_pandas()
arrow_table = await db.query("search quality").semantic().limit(20).to_arrow()
```

## Multimodal and Recency Search

Use vector spaces when ranking should combine more than text similarity.

```python
from pgvectordb import CategorySpace, NumberMode, NumberSpace, RecencySpace, TextSpace, TimeUnit

spaces = [
    TextSpace(name="description", weight=0.65),
    NumberSpace(name="price", mode=NumberMode.SIMILAR, weight=0.15),
    CategorySpace(name="property_type", weight=0.10),
    RecencySpace(name="listed_at", decay_period=7, time_unit=TimeUnit.DAY, weight=0.10),
]

results = await (
    db.query("fresh waterfront home near transit")
    .across_spaces(spaces)
    .where({"city": "Seattle"})
    .limit(10)
    .to_list()
)
```

## Evaluation

The evaluator works with retrieved IDs and ground-truth relevant IDs, so you can compare semantic, keyword, hybrid, reranked, or filtered retrieval strategies.

```python
from pgvectordb import RAGEvaluator

evaluator = RAGEvaluator(k=10)

result = evaluator.evaluate_single_query(
    retrieved_ids=["doc-7", "doc-3", "doc-9", "doc-1"],
    relevant_ids=["doc-3", "doc-9"],
)

print(result.precision_at_k, result.recall_at_k, result.ndcg_at_k)
```

## Diagnostics and Optimization

```python
plan = db.query("filtered vector search").semantic().where({"topic": "postgres"}).explain_plan()

metrics = await (
    db.query("filtered vector search")
    .semantic()
    .where({"topic": "postgres"})
    .limit(10)
    .analyze_plan()
)

await db.create_scalar_index("topic", index_type="btree")
await db.build_bm25_index()
await db.build_index_concurrent(index_type=IndexType.HNSW)
```

## Documentation

- [Quickstart](docs/getting_started/quickstart.md)
- [Core Concepts](docs/getting_started/core_concepts.md)
- [Search and Retrieval](docs/user_guide/search_and_retrieval.md)
- [Metadata Filtering](docs/user_guide/filtering.md)
- [Multimodal Search and RecencySpace](docs/user_guide/multimodal_search.md)
- [Metrics and Evaluation](docs/user_guide/metrics_and_evaluation.md)
- [Analytics and Diagnostics](docs/user_guide/analytics_and_diagnostics.md)
- [Indexing and Performance](docs/advanced/indexing.md)
- [LangChain Integration](docs/user_guide/langchain_integration.md)
- [Examples](docs/examples.md)

## Examples

- [Fluent API demo](examples/fluent_api_demo.py)
- [Product search](examples/product_search.py)
- [Real estate natural-language search](examples/real_estate_nlq.py)
- [Evaluation scripts](eval/scripts)
- [Jupyter notebooks](examples)

## Extension Support

| Extension | Required | Used for |
| --- | --- | --- |
| `pgvector` | Yes | Vector storage and similarity search. |
| `pg_trgm` | Yes | Typo-tolerant trigram search. |
| `vectorscale` | Required for DiskANN | DiskANN indexes and label-aware vector search. |
| `pg_textsearch` | Required for BM25 | BM25 ranking. Use PostgreSQL full-text search when unavailable. |

## Development

```bash
uv sync --group dev --group docs
uv run pytest
uv run ruff check .
uv run ruff format --check .
uv run pyright
NO_MKDOCS_2_WARNING=true uv run mkdocs build --strict
```

Useful local checks while working on docs and examples:

```bash
# Execute a notebook in-place against the local Docker database
DB_HOST=localhost DB_PORT=9002 DB_USER=user DB_PASSWORD=root DB_NAME=postgres \
DB_CONNECTION_STRING=postgresql+asyncpg://user:root@localhost:9002/postgres \
uv run jupyter nbconvert --to notebook --execute --inplace examples/01_quickstart.ipynb

# Build the docs without writing into site/
NO_MKDOCS_2_WARNING=true uv run mkdocs build --strict --site-dir /tmp/pgvectordb-mkdocs-check
```

## Contributing

Contributions are welcome. Open an issue or pull request with a focused change, a clear explanation, and tests or docs updates when behavior changes.

## License

pgVectorDB is released under the MIT License. See [LICENSE](LICENSE).