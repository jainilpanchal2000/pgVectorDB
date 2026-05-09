"""
RAG Retrieval Evaluation System
================================

Implements comprehensive retrieval metrics for RAG evaluation:
- Precision@K: (Relevant docs in top K) / K
- Recall@K: (Relevant docs in top K) / (Total relevant docs)
- F1@K: Harmonic mean of Precision@K and Recall@K
- Mean Average Precision (mAP@K): Rank-aware precision
- Mean Reciprocal Rank (MRR): 1 / (Rank of first relevant document)
- Normalized Discounted Cumulative Gain (NDCG@K): Position-discounted ranking quality
- Hit Rate@K: Percentage of queries with at least one relevant result in top K

Key Distinctions:
- Precision@K always divides by K (not by number of results)
- Recall@K divides by total relevant docs (not by K)
- MRR focuses only on position of FIRST relevant document
- NDCG@K penalizes relevant documents appearing lower in rankings

Based on: https://medium.com/@autorag/tips-to-understand-rag-retrieval-metrics-71e9a2bd4b96
"""

import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
import json


@dataclass
class EvaluationResult:
    """Container for evaluation results."""

    precision: float
    recall: float
    f1_score: float
    map_score: float
    mrr_score: float
    ndcg_score: float
    hit_rate: float

    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary."""
        return {
            "precision": self.precision,
            "recall": self.recall,
            "f1_score": self.f1_score,
            "map": self.map_score,
            "mrr": self.mrr_score,
            "ndcg": self.ndcg_score,
            "hit_rate": self.hit_rate,
        }

    def __str__(self) -> str:
        """Pretty print results."""
        return f"""
Retrieval Evaluation Results:
=============================
Precision@K:  {self.precision:.4f}  (relevant in top K / K)
Recall@K:     {self.recall:.4f}  (relevant in top K / total relevant)
F1@K:         {self.f1_score:.4f}  (harmonic mean of P@K and R@K)
MAP@K:        {self.map_score:.4f}  (mean average precision)
MRR:          {self.mrr_score:.4f}  (1 / rank of first relevant)
NDCG@K:       {self.ndcg_score:.4f}  (ranking quality with discount)
Hit Rate@K:   {self.hit_rate:.4f}  (queries with ≥1 relevant)
"""


@dataclass
class QueryGroundTruth:
    """Ground truth for a single query."""

    query: str
    relevant_doc_ids: List[str]  # List of relevant document IDs
    metadata: Optional[Dict[str, Any]] = None


class RAGEvaluator:
    """
    Comprehensive RAG retrieval evaluator.

    Computes multiple metrics to assess retrieval quality:
    - Precision@K: (Relevant docs in top K) / K
    - Recall@K: (Relevant docs in top K) / (Total relevant docs)
    - F1@K: Harmonic mean of Precision@K and Recall@K
    - MAP@K: Mean average precision across queries (rank-aware)
    - MRR: Mean reciprocal rank of first relevant document (1 / rank)
    - NDCG@K: Normalized discounted cumulative gain (penalizes low-ranked relevant docs)
    - Hit Rate@K: Percentage of queries with at least one relevant doc in top K

    All metrics are computed at K (default: 5), meaning only the top K retrieved
    documents are considered for evaluation.
    """

    def __init__(self, k: int = 5):
        """
        Initialize evaluator.

        Args:
            k: Number of top results to consider (default: 5)
        """
        self.k = k

    def evaluate(
        self,
        queries: List[str],
        retrieved_results: List[List[str]],
        ground_truth: List[List[str]],
    ) -> EvaluationResult:
        """
        Evaluate retrieval results against ground truth.

        Args:
            queries: List of query strings
            retrieved_results: List of retrieved document IDs for each query
            ground_truth: List of relevant document IDs for each query

        Returns:
            EvaluationResult with all computed metrics
        """
        if not (len(queries) == len(retrieved_results) == len(ground_truth)):
            raise ValueError("Queries, results, and ground truth must have same length")

        # Compute metrics
        precision = self._compute_precision(retrieved_results, ground_truth)
        recall = self._compute_recall(retrieved_results, ground_truth)
        f1 = self._compute_f1(precision, recall)
        map_score = self._compute_map(retrieved_results, ground_truth)
        mrr = self._compute_mrr(retrieved_results, ground_truth)
        ndcg = self._compute_ndcg(retrieved_results, ground_truth)
        hit_rate = self._compute_hit_rate(retrieved_results, ground_truth)

        return EvaluationResult(
            precision=precision,
            recall=recall,
            f1_score=f1,
            map_score=map_score,
            mrr_score=mrr,
            ndcg_score=ndcg,
            hit_rate=hit_rate,
        )

    def evaluate_single_query(
        self, retrieved_docs: List[str], relevant_docs: List[str]
    ) -> Dict[str, float]:
        """
        Evaluate a single query.

        Args:
            retrieved_docs: Retrieved document IDs
            relevant_docs: Relevant document IDs (ground truth)

        Returns:
            Dictionary with metrics for this query
        """
        retrieved_k = retrieved_docs[: self.k]
        retrieved_set = set(retrieved_k)
        relevant_set = set(relevant_docs)

        # Precision & Recall
        true_positives = len(retrieved_set & relevant_set)
        if self.k == 0:
            precision = 0.0
        else:
            precision = true_positives / self.k

        recall = true_positives / len(relevant_set) if relevant_set else 0.0

        # F1
        f1 = (
            (2 * precision * recall / (precision + recall))
            if (precision + recall) > 0
            else 0.0
        )

        # Average Precision
        ap = self._average_precision(retrieved_docs, relevant_docs)

        # Reciprocal Rank
        rr = self._reciprocal_rank(retrieved_docs, relevant_docs)

        # NDCG
        ndcg = self._ndcg_at_k(retrieved_docs, relevant_docs)

        # Hit Rate
        hit = 1.0 if true_positives > 0 else 0.0

        return {
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "average_precision": ap,
            "reciprocal_rank": rr,
            "ndcg": ndcg,
            "hit": hit,
        }

    def _compute_precision(
        self, retrieved_results: List[List[str]], ground_truth: List[List[str]]
    ) -> float:
        """
        Compute average Precision@K across all queries.

        Precision@K = (# relevant docs in top K) / K

        Example: If 7 out of 10 retrieved docs are relevant, Precision@10 = 0.7
        This measures the proportion of retrieved documents that are relevant.
        """
        precisions = []

        for retrieved, relevant in zip(retrieved_results, ground_truth):
            # Take only top K results
            retrieved_k = retrieved[: self.k]
            relevant_set = set(relevant)

            if not retrieved_k:
                precisions.append(0.0)
                continue

            # Count relevant docs in top K
            true_positives = len([doc for doc in retrieved_k if doc in relevant_set])
            # Precision@K = relevant_in_k / K (always divide by K, not by len(retrieved))
            precision_at_k = true_positives / self.k
            precisions.append(precision_at_k)

        return np.mean(precisions) if precisions else 0.0

    def _compute_recall(
        self, retrieved_results: List[List[str]], ground_truth: List[List[str]]
    ) -> float:
        """
        Compute average Recall@K across all queries.

        Recall@K = (# relevant docs in top K) / (# total relevant docs)

        Example: If 7 relevant docs retrieved out of 15 total relevant, Recall@K = 0.47
        This measures the proportion of all relevant documents that were retrieved.
        """
        recalls = []

        for retrieved, relevant in zip(retrieved_results, ground_truth):
            # Take only top K results
            retrieved_k = retrieved[: self.k]
            relevant_set = set(relevant)

            if not relevant_set:
                recalls.append(0.0)
                continue

            # Count how many relevant docs are in top K
            true_positives = len([doc for doc in retrieved_k if doc in relevant_set])
            # Recall@K = relevant_in_k / total_relevant
            recall_at_k = true_positives / len(relevant_set)
            recalls.append(recall_at_k)

        return np.mean(recalls) if recalls else 0.0

    def _compute_f1(self, precision: float, recall: float) -> float:
        """
        Compute F1@K score (harmonic mean of Precision@K and Recall@K).

        F1@K = 2 * (Precision@K * Recall@K) / (Precision@K + Recall@K)
        """
        if precision + recall == 0:
            return 0.0
        return 2 * precision * recall / (precision + recall)

    def _compute_map(
        self, retrieved_results: List[List[str]], ground_truth: List[List[str]]
    ) -> float:
        """
        Compute Mean Average Precision (mAP).

        mAP is the mean of average precision scores across all queries.
        It's rank-aware and considers precision at each relevant position.
        """
        average_precisions = []

        for retrieved, relevant in zip(retrieved_results, ground_truth):
            ap = self._average_precision(retrieved, relevant)
            average_precisions.append(ap)

        return np.mean(average_precisions) if average_precisions else 0.0

    def _average_precision(self, retrieved: List[str], relevant: List[str]) -> float:
        """
        Compute Average Precision for a single query.

        AP = (sum of precision@k for each relevant doc) / (# relevant docs)
        """
        relevant_set = set(relevant)

        if not relevant_set:
            return 0.0

        retrieved_k = retrieved[: self.k]
        precision_sum = 0.0
        num_relevant_seen = 0

        for i, doc_id in enumerate(retrieved_k, 1):
            if doc_id in relevant_set:
                num_relevant_seen += 1
                # Precision at position i
                precision_at_i = num_relevant_seen / i
                precision_sum += precision_at_i

        return precision_sum / len(relevant_set)

    def _compute_mrr(
        self, retrieved_results: List[List[str]], ground_truth: List[List[str]]
    ) -> float:
        """
        Compute Mean Reciprocal Rank (MRR).

        MRR = mean(1 / rank of first relevant document)
        Focuses on position of first relevant result.
        """
        reciprocal_ranks = []

        for retrieved, relevant in zip(retrieved_results, ground_truth):
            rr = self._reciprocal_rank(retrieved, relevant)
            reciprocal_ranks.append(rr)

        return np.mean(reciprocal_ranks) if reciprocal_ranks else 0.0

    def _reciprocal_rank(self, retrieved: List[str], relevant: List[str]) -> float:
        """
        Compute Reciprocal Rank for a single query.

        MRR = 1 / (Rank of first relevant document)

        Measures how high the first relevant document appears in results.
        Higher is better (max = 1.0 when first doc is relevant)

        Examples:
        - First relevant at position 1: RR = 1/1 = 1.0
        - First relevant at position 2: RR = 1/2 = 0.5
        - First relevant at position 5: RR = 1/5 = 0.2
        - No relevant found: RR = 0.0
        """
        relevant_set = set(relevant)
        retrieved_k = retrieved[: self.k]

        for rank, doc_id in enumerate(retrieved_k, start=1):
            if doc_id in relevant_set:
                return 1.0 / rank

        return 0.0

    def _compute_ndcg(
        self, retrieved_results: List[List[str]], ground_truth: List[List[str]]
    ) -> float:
        """
        Compute Normalized Discounted Cumulative Gain (NDCG).

        NDCG measures ranking quality with position discount.
        Higher scores for relevant docs appearing earlier.
        """
        ndcg_scores = []

        for retrieved, relevant in zip(retrieved_results, ground_truth):
            ndcg = self._ndcg_at_k(retrieved, relevant)
            ndcg_scores.append(ndcg)

        return np.mean(ndcg_scores) if ndcg_scores else 0.0

    def _ndcg_at_k(self, retrieved: List[str], relevant: List[str]) -> float:
        """
        Compute NDCG@K for a single query.

        Normalized Discounted Cumulative Gain considers both:
        1. Relevance of documents
        2. Position/ranking of documents (penalizes relevant docs appearing lower)

        Formula:
        DCG@K = Σ(rel_i / log2(i + 1)) for i in 1..K
        IDCG@K = DCG for perfect ranking (all relevant docs at top)
        NDCG@K = DCG@K / IDCG@K

        Range: 0 to 1 (1 = perfect ranking)

        Examples:
        - Retrieved: [rel, rel, irrel, rel, irrel] (3 relevant out of 5)
        - DCG = 1/log2(2) + 1/log2(3) + 0/log2(4) + 1/log2(5) + 0/log2(6)
        - IDCG = 1/log2(2) + 1/log2(3) + 1/log2(4) (perfect: all 3 relevant at top)
        - NDCG = DCG / IDCG
        """
        relevant_set = set(relevant)
        retrieved_k = retrieved[: self.k]

        # Compute DCG@K
        dcg = 0.0
        for position, doc_id in enumerate(retrieved_k, start=1):
            relevance = 1.0 if doc_id in relevant_set else 0.0
            # Discount factor: 1/log2(position + 1)
            dcg += relevance / np.log2(position + 1)

        # Compute IDCG@K (ideal DCG - best possible ranking)
        # Place all relevant docs at top positions
        idcg = 0.0
        num_relevant = min(len(relevant), self.k)
        for position in range(1, num_relevant + 1):
            idcg += 1.0 / np.log2(position + 1)

        if idcg == 0:
            return 0.0

        return dcg / idcg

    def _compute_hit_rate(
        self, retrieved_results: List[List[str]], ground_truth: List[List[str]]
    ) -> float:
        """
        Compute Hit Rate (% of queries with at least one relevant result).

        Hit Rate = (# queries with ≥1 relevant doc) / (# total queries)
        """
        hits = 0

        for retrieved, relevant in zip(retrieved_results, ground_truth):
            retrieved_set = set(retrieved[: self.k])
            relevant_set = set(relevant)

            if retrieved_set & relevant_set:  # Any intersection
                hits += 1

        return hits / len(retrieved_results) if retrieved_results else 0.0


class EvaluationDataset:
    """
    Container for RAG evaluation dataset.

    Stores queries, ground truth relevance, and metadata.
    """

    def __init__(self):
        self.queries: List[str] = []
        self.ground_truth: List[List[str]] = []
        self.metadata: List[Dict[str, Any]] = []

    def add_query(
        self,
        query: str,
        relevant_doc_ids: List[str],
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """Add a query with ground truth to dataset."""
        self.queries.append(query)
        self.ground_truth.append(relevant_doc_ids)
        self.metadata.append(metadata or {})

    def save(self, filepath: str):
        """Save dataset to JSON file."""
        data = {
            "queries": self.queries,
            "ground_truth": self.ground_truth,
            "metadata": self.metadata,
        }

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    @classmethod
    def load(cls, filepath: str) -> "EvaluationDataset":
        """Load dataset from JSON file."""
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)

        dataset = cls()
        dataset.queries = data["queries"]
        dataset.ground_truth = data["ground_truth"]
        dataset.metadata = data.get("metadata", [{}] * len(dataset.queries))

        return dataset

    def __len__(self) -> int:
        return len(self.queries)

    def __getitem__(self, idx: int) -> Tuple[str, List[str], Dict[str, Any]]:
        """Get query, ground truth, and metadata by index."""
        return self.queries[idx], self.ground_truth[idx], self.metadata[idx]


def create_sample_evaluation_dataset() -> EvaluationDataset:
    """
    Create a sample evaluation dataset for testing.

    Returns:
        EvaluationDataset with diverse queries and ground truth
    """
    dataset = EvaluationDataset()

    # Programming queries
    dataset.add_query(
        query="How to implement a binary search tree in Python?",
        relevant_doc_ids=["doc_10", "doc_25", "doc_45"],
        metadata={"category": "programming", "difficulty": "medium"},
    )

    dataset.add_query(
        query="What are Python decorators and how do they work?",
        relevant_doc_ids=["doc_15", "doc_33"],
        metadata={"category": "programming", "difficulty": "medium"},
    )

    dataset.add_query(
        query="Explain async/await in Python",
        relevant_doc_ids=["doc_22", "doc_41", "doc_58", "doc_67"],
        metadata={"category": "programming", "difficulty": "advanced"},
    )

    # AI/ML queries
    dataset.add_query(
        query="What is transformer architecture in deep learning?",
        relevant_doc_ids=["doc_12", "doc_28", "doc_55"],
        metadata={"category": "ai", "difficulty": "advanced"},
    )

    dataset.add_query(
        query="How does gradient descent optimization work?",
        relevant_doc_ids=["doc_8", "doc_31", "doc_49", "doc_62"],
        metadata={"category": "ai", "difficulty": "medium"},
    )

    dataset.add_query(
        query="Best practices for training neural networks",
        relevant_doc_ids=["doc_18", "doc_39", "doc_54"],
        metadata={"category": "ai", "difficulty": "medium"},
    )

    # Database queries
    dataset.add_query(
        query="How to optimize PostgreSQL query performance?",
        relevant_doc_ids=["doc_5", "doc_27", "doc_44", "doc_63"],
        metadata={"category": "database", "difficulty": "medium"},
    )

    dataset.add_query(
        query="What are database indexes and when to use them?",
        relevant_doc_ids=["doc_19", "doc_36", "doc_51"],
        metadata={"category": "database", "difficulty": "easy"},
    )

    dataset.add_query(
        query="ACID properties in database transactions",
        relevant_doc_ids=["doc_14", "doc_29"],
        metadata={"category": "database", "difficulty": "medium"},
    )

    # Web development queries
    dataset.add_query(
        query="How to build a REST API with FastAPI?",
        relevant_doc_ids=["doc_7", "doc_23", "doc_38", "doc_56"],
        metadata={"category": "web", "difficulty": "medium"},
    )

    dataset.add_query(
        query="What is JWT authentication and how to implement it?",
        relevant_doc_ids=["doc_16", "doc_34", "doc_48"],
        metadata={"category": "web", "difficulty": "medium"},
    )

    dataset.add_query(
        query="React hooks vs class components",
        relevant_doc_ids=["doc_21", "doc_42", "doc_59"],
        metadata={"category": "web", "difficulty": "easy"},
    )

    # DevOps queries
    dataset.add_query(
        query="Docker container best practices",
        relevant_doc_ids=["doc_11", "doc_30", "doc_47", "doc_64"],
        metadata={"category": "devops", "difficulty": "medium"},
    )

    dataset.add_query(
        query="How to set up CI/CD pipeline with GitHub Actions?",
        relevant_doc_ids=["doc_24", "doc_40", "doc_57"],
        metadata={"category": "devops", "difficulty": "medium"},
    )

    dataset.add_query(
        query="Kubernetes deployment strategies",
        relevant_doc_ids=["doc_13", "doc_32", "doc_50", "doc_68"],
        metadata={"category": "devops", "difficulty": "advanced"},
    )

    # Security queries
    dataset.add_query(
        query="Common web application security vulnerabilities",
        relevant_doc_ids=["doc_9", "doc_26", "doc_43"],
        metadata={"category": "security", "difficulty": "medium"},
    )

    dataset.add_query(
        query="How to prevent SQL injection attacks?",
        relevant_doc_ids=["doc_17", "doc_35", "doc_52", "doc_65"],
        metadata={"category": "security", "difficulty": "easy"},
    )

    dataset.add_query(
        query="OAuth 2.0 authentication flow explained",
        relevant_doc_ids=["doc_20", "doc_37", "doc_53"],
        metadata={"category": "security", "difficulty": "advanced"},
    )

    # Cloud queries
    dataset.add_query(
        query="AWS Lambda vs EC2: When to use each?",
        relevant_doc_ids=["doc_6", "doc_28", "doc_46", "doc_61"],
        metadata={"category": "cloud", "difficulty": "medium"},
    )

    dataset.add_query(
        query="How to design a scalable cloud architecture?",
        relevant_doc_ids=["doc_4", "doc_25", "doc_45", "doc_66"],
        metadata={"category": "cloud", "difficulty": "advanced"},
    )

    dataset.add_query(
        query="Cloud storage options comparison: S3 vs Blob vs GCS",
        relevant_doc_ids=["doc_3", "doc_19", "doc_39"],
        metadata={"category": "cloud", "difficulty": "medium"},
    )

    return dataset


def print_detailed_results(
    evaluator: RAGEvaluator,
    queries: List[str],
    retrieved_results: List[List[str]],
    ground_truth: List[List[str]],
):
    """
    Print detailed per-query results and overall metrics.

    Args:
        evaluator: RAGEvaluator instance
        queries: List of query strings
        retrieved_results: Retrieved document IDs for each query
        ground_truth: Relevant document IDs for each query
    """
    print("\n" + "=" * 80)
    print("DETAILED QUERY-BY-QUERY RESULTS")
    print("=" * 80)

    for i, (query, retrieved, relevant) in enumerate(
        zip(queries, retrieved_results, ground_truth), 1
    ):
        metrics = evaluator.evaluate_single_query(retrieved, relevant)

        print(f"\nQuery {i}: {query[:60]}...")
        print(f"  Retrieved: {len(retrieved)} docs | Relevant: {len(relevant)} docs")
        print(
            f"  Precision: {metrics['precision']:.3f} | Recall: {metrics['recall']:.3f} | F1: {metrics['f1_score']:.3f}"
        )
        print(
            f"  AP: {metrics['average_precision']:.3f} | RR: {metrics['reciprocal_rank']:.3f} | NDCG: {metrics['ndcg']:.3f}"
        )

    print("\n" + "=" * 80)
    print("OVERALL METRICS")
    print("=" * 80)

    result = evaluator.evaluate(queries, retrieved_results, ground_truth)
    print(result)


if __name__ == "__main__":
    # Example usage
    print("RAG Evaluation System - Quick Test\n")

    # Create sample dataset
    dataset = create_sample_evaluation_dataset()
    print(f"Created evaluation dataset with {len(dataset)} queries\n")

    # Simulate retrieval results (in practice, these come from your RAG system)
    simulated_results = []
    for i in range(len(dataset)):
        query, relevant_docs, meta = dataset[i]

        # Simulate: include some relevant docs and some random docs
        num_relevant_to_include = min(2, len(relevant_docs))
        retrieved = relevant_docs[:num_relevant_to_include].copy()

        # Add some irrelevant docs
        for j in range(5 - len(retrieved)):
            retrieved.append(f"doc_{(i * 7 + j) % 100}")

        simulated_results.append(retrieved)

    # Evaluate
    evaluator = RAGEvaluator(k=5)
    result = evaluator.evaluate(
        queries=dataset.queries,
        retrieved_results=simulated_results,
        ground_truth=dataset.ground_truth,
    )

    print(result)

    # Print tips based on results
    print("\n" + "=" * 80)
    print("INTERPRETATION TIPS")
    print("=" * 80)

    if result.precision < 0.5:
        print("⚠️  Low Precision: Many irrelevant documents retrieved.")
        print(
            "   → Consider: Stricter filtering, better embeddings, or tuning similarity threshold"
        )

    if result.recall < 0.5:
        print("⚠️  Low Recall: Missing many relevant documents.")
        print(
            "   → Consider: Retrieving more documents (increase k), or improving query expansion"
        )

    if result.mrr_score < 0.5:
        print("⚠️  Low MRR: First relevant document appears late in results.")
        print("   → Consider: Improving ranking algorithm or re-ranking strategies")

    if result.ndcg_score < 0.6:
        print("⚠️  Low NDCG: Poor ranking quality overall.")
        print(
            "   → Consider: Hybrid search, query rewriting, or advanced ranking models"
        )

    if result.hit_rate < 0.8:
        print("⚠️  Low Hit Rate: Many queries return no relevant documents.")
        print(
            "   → Consider: Expanding document corpus or improving query understanding"
        )

    print("\n✅ Evaluation complete!")


class KValueAnalysis:
    """
    Analyzes optimal K value for a RAG system by testing multiple K values.

    Helps answer: "What K value is best for my use case?"

    Examples:
        >>> analyzer = KValueAnalysis()
        >>> results = analyzer.analyze(
        ...     queries=queries,
        ...     retrieved_results_by_k={1: results_k1, 5: results_k5, 10: results_k10},
        ...     ground_truth=ground_truth
        ... )
        >>> analyzer.print_analysis()
        >>> best_k = analyzer.get_recommendation()
    """

    def __init__(self):
        """Initialize K-value analyzer."""
        self.results_by_k = {}
        self.k_values = []

    def analyze(
        self,
        queries: List[str],
        retrieved_results_by_k: Dict[int, List[List[str]]],
        ground_truth: List[List[str]],
    ) -> Dict[int, EvaluationResult]:
        """
        Analyze multiple K values.

        Args:
            queries: List of query strings
            retrieved_results_by_k: Dict mapping K -> retrieved results for that K
                Example: {
                    1: [["doc_1"], ["doc_5"], ...],      # K=1 results
                    5: [["doc_1", "doc_2", ...], ...],   # K=5 results
                    10: [["doc_1", "doc_2", ...], ...]   # K=10 results
                }
            ground_truth: List of relevant doc IDs per query

        Returns:
            Dict mapping K value to EvaluationResult
        """
        self.k_values = sorted(retrieved_results_by_k.keys())
        self.results_by_k = {}

        for k, retrieved_results in retrieved_results_by_k.items():
            evaluator = RAGEvaluator(k=k)
            result = evaluator.evaluate(queries, retrieved_results, ground_truth)
            self.results_by_k[k] = result

        return self.results_by_k

    def print_analysis(self) -> None:
        """Print comprehensive K-value analysis."""
        if not self.results_by_k:
            print("No analysis results available. Run analyze() first.")
            return

        print("\n" + "=" * 100)
        print("K-VALUE ANALYSIS")
        print("=" * 100)

        # Header
        print(
            f"\n{'K':<6} {'Precision':<12} {'Recall':<12} {'F1':<12} {'MAP':<12} {'MRR':<12} {'NDCG':<12} {'Hit Rate':<12}"
        )
        print("-" * 100)

        # Results for each K
        for k in self.k_values:
            result = self.results_by_k[k]
            print(
                f"{k:<6} "
                f"{result.precision:<12.4f} "
                f"{result.recall:<12.4f} "
                f"{result.f1_score:<12.4f} "
                f"{result.map_score:<12.4f} "
                f"{result.mrr_score:<12.4f} "
                f"{result.ndcg_score:<12.4f} "
                f"{result.hit_rate:<12.4f}"
            )

        print("\n" + "=" * 100)
        print("METRIC TRENDS")
        print("=" * 100)

        # Precision trend
        print("\nPrecision@K (Higher = Better Quality):")
        for k in self.k_values:
            bars = "█" * int(self.results_by_k[k].precision * 50)
            print(f"  K={k:>3}: {bars} {self.results_by_k[k].precision:.3f}")

        # Recall trend
        print("\nRecall@K (Higher = Better Coverage):")
        for k in self.k_values:
            bars = "█" * int(self.results_by_k[k].recall * 50)
            print(f"  K={k:>3}: {bars} {self.results_by_k[k].recall:.3f}")

        # F1 trend
        print("\nF1@K (Higher = Better Balance):")
        for k in self.k_values:
            bars = "█" * int(self.results_by_k[k].f1_score * 50)
            print(f"  K={k:>3}: {bars} {self.results_by_k[k].f1_score:.3f}")

        # NDCG trend
        print("\nNDCG@K (Higher = Better Ranking):")
        for k in self.k_values:
            bars = "█" * int(self.results_by_k[k].ndcg_score * 50)
            print(f"  K={k:>3}: {bars} {self.results_by_k[k].ndcg_score:.3f}")

    def get_recommendation(self) -> Dict[str, Any]:
        """
        Get K value recommendations based on different criteria.

        Returns:
            Dict with recommendations:
            {
                'max_precision': (k, score),
                'max_recall': (k, score),
                'best_f1': (k, score),
                'best_ndcg': (k, score),
                'optimal_balanced': k,
                'summary': str
            }
        """
        if not self.results_by_k:
            return {}

        # Find best K for each metric
        max_precision = max(self.results_by_k.items(), key=lambda x: x[1].precision)
        max_recall = max(self.results_by_k.items(), key=lambda x: x[1].recall)
        best_f1 = max(self.results_by_k.items(), key=lambda x: x[1].f1_score)
        best_ndcg = max(self.results_by_k.items(), key=lambda x: x[1].ndcg_score)
        best_mrr = max(self.results_by_k.items(), key=lambda x: x[1].mrr_score)

        # Determine optimal balanced K (best F1)
        optimal_k = best_f1[0]

        # Generate summary
        summary = f"""
K-Value Recommendations:
========================

📊 Metric-Specific Best K:
  • Max Precision (Quality):  K={max_precision[0]} (P={max_precision[1].precision:.3f})
  • Max Recall (Coverage):    K={max_recall[0]} (R={max_recall[1].recall:.3f})
  • Best F1 (Balance):        K={best_f1[0]} (F1={best_f1[1].f1_score:.3f})
  • Best NDCG (Ranking):      K={best_ndcg[0]} (NDCG={best_ndcg[1].ndcg_score:.3f})
  • Best MRR (First Result):  K={best_mrr[0]} (MRR={best_mrr[1].mrr_score:.3f})

🎯 Recommended K by Use Case:
  • FAQ/Q&A (First answer matters):         K={best_mrr[0]}
  • General RAG (Balanced):                 K={optimal_k}
  • Research/Analysis (Comprehensive):      K={max_recall[0]}
  • LLM with limited context (Quality):     K={max_precision[0]}

⚡ Optimal Balanced Choice: K={optimal_k}
   - Precision: {best_f1[1].precision:.3f}
   - Recall: {best_f1[1].recall:.3f}
   - F1: {best_f1[1].f1_score:.3f}
   - NDCG: {best_f1[1].ndcg_score:.3f}
"""

        return {
            "max_precision": (max_precision[0], max_precision[1].precision),
            "max_recall": (max_recall[0], max_recall[1].recall),
            "best_f1": (best_f1[0], best_f1[1].f1_score),
            "best_ndcg": (best_ndcg[0], best_ndcg[1].ndcg_score),
            "best_mrr": (best_mrr[0], best_mrr[1].mrr_score),
            "optimal_balanced": optimal_k,
            "summary": summary,
        }

    def print_recommendation(self) -> None:
        """Print K value recommendations."""
        recommendation = self.get_recommendation()
        if recommendation:
            print(recommendation["summary"])
        else:
            print("No recommendations available. Run analyze() first.")

    def export_results(self, filepath: str) -> None:
        """
        Export analysis results to JSON file.

        Args:
            filepath: Path to save JSON file
        """
        data = {
            "k_values": self.k_values,
            "results": {k: result.to_dict() for k, result in self.results_by_k.items()},
            "recommendations": self.get_recommendation(),
        }

        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)

        print(f"✅ Results exported to {filepath}")
