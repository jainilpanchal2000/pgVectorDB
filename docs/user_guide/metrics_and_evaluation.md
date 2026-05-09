# Metrics and Evaluation: Statistical Deep Dive

Building a RAG system is easy; proving it works is hard. pgVectorDB includes a dedicated `metrics.py` module containing the `RAGEvaluator` class. This provides scientifically rigorous statistical evaluation of your specific retrieval pipelines, allowing you to quantify exactly how much an index tune (like increasing HNSW `ef_search`) or a new reranker improves your accuracy.

---

## Supported Metrics

The `RAGEvaluator` calculates three distinct, industry-standard metrics for Information Retrieval (IR) systems.

### 1. Hit Rate
**What it is**: A binary boolean value (0.0 or 1.0). Did the relevant document appear *anywhere* within the top-K returned results?
**Why use it**: This is the most basic metric. If your hit rate is low, your embeddings are likely poor, or your K value is too small. It doesn't care if the right answer was result #1 or result #10, as long as it was retrieved.

### 2. MRR (Mean Reciprocal Rank)
**What it is**: Measures *where* the first relevant document appeared in the results.
**Formula**: `MRR = 1 / rank_of_first_relevant_item`
**Why use it**: If the correct document is returned at rank 1, the score is 1.0. If it is returned at rank 2, the score is 0.5. Rank 3 is 0.33, etc. MRR heavily penalizes systems that bury the right answer deep in the results, making it the perfect metric for evaluating LLM context window stuffing (where the top documents get the most attention).

### 3. NDCG (Normalized Discounted Cumulative Gain)
**What it is**: The most rigorous metric. It evaluates the ranking quality of *all* retrieved relevant documents, heavily discounting the value of documents the further down the list they appear using a logarithmic scale.
**Formula**: `DCG = sum(rel_i / log2(i + 1))`, then normalized against the Ideal DCG (IDCG).
**Why use it**: Essential when a query has *multiple* relevant documents, and you want to ensure the system clustered all the best answers right at the top.

---

## Evaluating a Search Pipeline

To run an evaluation, you must provide a "Ground Truth" dataset. This is a list of tuples: `(query_string, expected_document_id)`.

```python
from pgvectordb import pgVectorDB, RAGEvaluator

# Define your test suite
ground_truth = [
    ("How do I reset my password?", "doc_user_auth_12"),
    ("What is the refund policy?", "doc_billing_45"),
    ("Database tuning guide", "doc_tech_99")
]

# Instantiate the evaluator
evaluator = RAGEvaluator(db)

# Run the evaluation!
# By default, this uses semantic_search with k=5.
metrics = await evaluator.evaluate(
    ground_truth=ground_truth,
    k=5
)

print(f"Hit Rate: {metrics['hit_rate']}") # e.g., 0.95
print(f"MRR: {metrics['mrr']}")           # e.g., 0.82
print(f"NDCG: {metrics['ndcg']}")         # e.g., 0.85

```

---

## Customizing the Evaluation Function

The power of `RAGEvaluator` is that it allows you to inject *any* custom asynchronous retrieval function. This is critical for A/B testing different search methods.

For example, to test if `hybrid_search` outperforms `semantic_search`:

```python
# Define a custom function using hybrid search
async def hybrid_retriever(query: str, k: int):
    # Notice we can pass custom weights!
    return await db.hybrid_search(query, k=k, weights=(0.7, 0.3))

metrics_hybrid = await evaluator.evaluate(
    ground_truth=ground_truth,
    k=5,
    retrieval_fn=hybrid_retriever
)

print(f"Hybrid MRR: {metrics_hybrid['mrr']}")

```

---

## The `evaluate_k_range` Method

LLMs have limited context windows and charge per token. You want the smallest `K` (number of documents returned) that still maintains an acceptable Hit Rate. 

The `evaluate_k_range` method runs the evaluation multiple times across a spectrum of K values, helping you plot the exact point of diminishing returns.

```python
# Evaluates the pipeline for K=1, K=3, K=5, and K=10
results = await evaluator.evaluate_k_range(
    ground_truth=ground_truth,
    k_values=[1, 3, 5, 10]
)

for result in results:
    print(f"K={result['k']} | Hit Rate: {result['metrics']['hit_rate']} | MRR: {result['metrics']['mrr']}")

```

**Analyzing the output:**
If K=3 yields a Hit Rate of 0.92, and K=10 yields a Hit Rate of 0.94, you now have the empirical data to prove that sending 7 extra documents to the LLM (and paying for those tokens) is absolutely not worth a 2% increase in recall.
