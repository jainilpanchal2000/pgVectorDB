# Reranking

Vector search (HNSW/IVFFlat) is optimized for **speed and recall** — finding the top 100 candidates out of millions in milliseconds. But ranking those 100 candidates precisely is harder; the bi-encoder model that generated embeddings sees query and document **independently**.

**Reranking** solves this with a second, more accurate model that sees the query and document **together** (cross-encoding), trading latency for precision. The canonical pattern is: retrieve many candidates cheaply → rerank them expensively → return the best few.

---

## The `rerank_search()` Method

`rerank_search()` is the recommended one-stop method for retrieve-then-rerank pipelines. It handles both stages internally:

```python
from pgvectordb.rerankers import CrossEncoderReranker

reranker = CrossEncoderReranker(model="cross-encoder/ms-marco-MiniLM-L-6-v2")

results = await db.rerank_search(
    query="best noise-cancelling headphones under $200",
    reranker=reranker,
    k=100,             # Stage 1: retrieve 100 candidates via first-stage retrieval
    rerank_top_k=5,    # Stage 2: return the 5 best by rerank score
    search_method="hybrid",  # Which retrieval backend to use
)

for r in results:
    print(f"{r['score']:.4f} | {r['content'][:80]}")
```

### `search_method` Options

| Value | Retrieval Backend |
|-------|------------------|
| `"semantic"` | `semantic_search()` |
| `"hybrid"` | `hybrid_search()` |
| `"keyword"` | `keyword_search()` (FTS) |
| `"bm25"` | `keyword_search()` (BM25) |
| `"multimodal"` | `multimodal_search()` — pass `query_params` in kwargs |

### `k` vs `rerank_top_k`

The `k` / `rerank_top_k` split is the core trade-off:

| `k` (first stage) | `rerank_top_k` (output) | Effect |
|-------------------|------------------------|--------|
| 20 | 5 | Fast, lower recall |
| 100 | 5 | Balanced (recommended) |
| 500 | 10 | High recall, higher reranker cost |

!!! tip
    A good starting point is `k=100, rerank_top_k=5`. The reranker processes 100 documents but you only pay for the latency of the reranker model, not the vector search stage.

---

## Reranker Options

### 1. `CrossEncoderReranker` — Local HuggingFace

Highly accurate cross-encoder model running locally via PyTorch/Transformers. No API key needed.

*Requires: `pip install "pgvectordb[rerankers]"`*

```python
from pgvectordb.rerankers import CrossEncoderReranker

reranker = CrossEncoderReranker(
    model="cross-encoder/ms-marco-MiniLM-L-6-v2"   # 80MB model, fast on CPU
    # model="cross-encoder/ms-marco-electra-base"  # Larger, more accurate
)

results = await db.rerank_search(
    query="What causes a PostgreSQL deadlock?",
    reranker=reranker,
    k=50,
    rerank_top_k=5
)
```

### 2. `HuggingFaceReranker` — HuggingFace API

Uses the HuggingFace Inference API for reranking — no local GPU needed.

*Requires: `pip install "pgvectordb[huggingface]"` and `HUGGINGFACE_API_KEY`*

```python
import os
from pgvectordb.rerankers import HuggingFaceReranker

os.environ["HUGGINGFACE_API_KEY"] = "hf_..."

reranker = HuggingFaceReranker(
    model="BAAI/bge-reranker-large"
)

results = await db.rerank_search(
    query="database indexing strategies",
    reranker=reranker,
    k=100,
    rerank_top_k=10,
    search_method="semantic"
)
```

### 3. `CohereReranker` — Cohere API

State-of-the-art managed reranking API. Fast, no local GPU, requires network.

*Requires: `pip install "pgvectordb[cohere]"` and `COHERE_API_KEY`*

```python
import os
from pgvectordb.rerankers import CohereReranker

os.environ["COHERE_API_KEY"] = "your-api-key"

reranker = CohereReranker(model_name="rerank-english-v3.0")

results = await db.rerank_search(
    query="How do I tune PostgreSQL memory?",
    reranker=reranker,
    k=100,
    rerank_top_k=5,
    search_method="hybrid"
)
```

### 4. `AWSBedrockReranker` — AWS Bedrock

Use Cohere or other rerankers hosted on AWS Bedrock — keeps data within your VPC.

*Requires: `pip install "pgvectordb[aws]"` and configured AWS credentials*

```python
from pgvectordb.rerankers import AWSBedrockReranker

reranker = AWSBedrockReranker(
    model_id="cohere.rerank-v3-1:0",
    region_name="us-east-1"
)

results = await db.rerank_search(
    query="Vector database indexing strategies",
    reranker=reranker,
    k=100,
    rerank_top_k=5,
    search_method="semantic"
)
```

---

## Reranker Comparison

| Reranker | Latency | Cost | Privacy | GPU Required |
|----------|---------|------|---------|--------------|
| `CrossEncoderReranker` | Medium (100ms–2s) | Free | ✅ Local | No (CPU OK) |
| `HuggingFaceReranker` | Low–Medium | Metered | ❌ External | No |
| `CohereReranker` | Low (~50ms) | Metered | ❌ External | No |
| `AWSBedrockReranker` | Low (~100ms) | Metered | ✅ VPC | No |

---

## `create_reranker()` Factory

Use the factory for dynamic configuration (e.g., from environment variables):

```python
from pgvectordb.rerankers import create_reranker

# Type can be: "cross_encoder", "huggingface", "cohere", "bedrock"
reranker = create_reranker(
    "cohere",
    api_key="co_..."
)

results = await db.rerank_search(query="...", reranker=reranker, k=100, rerank_top_k=5)
```

---

## Multimodal + Reranking

Combine multimodal search with reranking for the highest-precision pipeline:

```python
from pgvectordb.rerankers import CohereReranker

reranker = CohereReranker(model_name="rerank-english-v3.0")

results = await db.rerank_search(
    query="modern 2BR apartment downtown with park views",
    reranker=reranker,
    k=100,
    rerank_top_k=10,
    search_method="multimodal",
    query_params={
        "description": "modern 2BR apartment downtown with park views",
        "price": 800_000,
        "city": "NYC",
    },
    weights={"description": 0.5, "price": 0.3, "city": 0.2},
)
```

---

## How It Works Internally

When you call any search method with a reranker:

1. **Stage 1 — Retrieval:** pgVectorDB runs the first-stage retrieval (`k` candidates) using the specified `search_method`
2. **Stage 2 — Text Extraction:** Raw text is extracted from all `k` `QueryResult` objects
3. **Stage 3 — Cross-Encoding:** The reranker scores each `(query, document)` pair together
4. **Stage 4 — Re-sorting:** Results are sorted by new rerank scores, descending
5. **Stage 5 — Truncation:** Top `rerank_top_k` results are returned

```
Query ──► [HNSW/BM25 Retrieval] ──► 100 candidates
                                          │
                                          ▼
                               [Cross-Encoder Reranker]
                               scores each (query, doc) pair
                                          │
                                          ▼
                               Top 5 by rerank score ──► Results
```
