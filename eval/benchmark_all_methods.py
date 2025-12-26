"""
Comprehensive Benchmark of All Search Methods
==============================================

Evaluates all 10 search methods across retrieval quality metrics:
1. keyword_search (FTS)
2. keyword_search (BM25)
3. semantic_search
4. hybrid_search (FTS + Semantic)
5. hybrid_search (BM25 + Semantic)
6. hybrid_search (BM25 + Semantic + RRF)
7. metadata_semantic_search
8. ensemble_search (BM25 + Semantic + RRF + Metadata)
9. trigram_search
10. metadata_trigram_search

Metrics computed:
- Precision@K, Recall@K, F1@K
- Mean Average Precision (MAP)
- Mean Reciprocal Rank (MRR)
- Normalized Discounted Cumulative Gain (NDCG)
- Hit Rate@K
"""

import asyncio
import time
import sys
from pathlib import Path
from typing import List, Dict, Any

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from src.core import pgVectorDB, IndexType, KeywordSearchType
from src.evaluation import RAGEvaluator, EvaluationResult
import pandas as pd


class BenchmarkDataset:
    """Sample dataset with ground truth for benchmarking."""
    
    def __init__(self):
        self.documents = self._create_documents()
        self.queries = self._create_queries()
        self.ground_truth = self._create_ground_truth()
    
    def _create_documents(self) -> List[Document]:
        """Create 50 realistic documents across 5 categories."""
        docs = []
        
        # AI/ML Category (10 docs)
        ai_ml_topics = [
            "Neural networks and deep learning architectures",
            "Machine learning model optimization techniques",
            "Natural language processing with transformers",
            "Computer vision and image recognition systems",
            "Reinforcement learning for game AI",
            "Transfer learning in deep neural networks",
            "Generative AI and large language models",
            "AutoML and neural architecture search",
            "Explainable AI and model interpretability",
            "Edge AI and model compression techniques"
        ]
        
        # Database Category (10 docs)
        database_topics = [
            "PostgreSQL indexing strategies and query optimization",
            "Vector databases for similarity search",
            "NoSQL databases and document stores",
            "Database sharding and horizontal scaling",
            "ACID transactions and consistency models",
            "Database replication and high availability",
            "Time-series databases for IoT data",
            "Graph databases and relationship queries",
            "Database performance tuning and monitoring",
            "Distributed databases and CAP theorem"
        ]
        
        # Web Development Category (10 docs)
        web_topics = [
            "React and modern frontend frameworks",
            "RESTful API design and best practices",
            "GraphQL for efficient data fetching",
            "Web application security and authentication",
            "Server-side rendering with Next.js",
            "WebSocket for real-time communication",
            "Progressive Web Apps and offline functionality",
            "Web performance optimization techniques",
            "Microservices architecture for web apps",
            "Web accessibility standards and WCAG"
        ]
        
        # DevOps Category (10 docs)
        devops_topics = [
            "Docker containerization best practices",
            "Kubernetes orchestration and deployment",
            "CI/CD pipelines with GitHub Actions",
            "Infrastructure as Code with Terraform",
            "Monitoring and observability with Prometheus",
            "Log aggregation and analysis systems",
            "Cloud migration strategies and patterns",
            "Automated testing in deployment pipelines",
            "Service mesh and microservices networking",
            "GitOps and declarative infrastructure"
        ]
        
        # Security Category (10 docs)
        security_topics = [
            "OAuth2 and OpenID Connect authentication",
            "SQL injection prevention techniques",
            "Cross-site scripting (XSS) attack mitigation",
            "Encryption at rest and in transit",
            "Zero-trust security architecture",
            "Penetration testing and vulnerability scanning",
            "Security auditing and compliance (SOC2, GDPR)",
            "API security and rate limiting",
            "Secret management and key rotation",
            "DDoS protection and mitigation strategies"
        ]
        
        categories = [
            ("ai_ml", ai_ml_topics, "high"),
            ("database", database_topics, "medium"),
            ("web_dev", web_topics, "medium"),
            ("devops", devops_topics, "high"),
            ("security", security_topics, "critical")
        ]
        
        doc_id = 0
        for category, topics, priority in categories:
            for i, topic in enumerate(topics):
                docs.append(Document(
                    page_content=f"{topic}. This document provides comprehensive information about {topic.lower()}.",
                    metadata={
                        "doc_id": doc_id,
                        "category": category,
                        "priority": priority,
                        "year": 2020 + (i % 5)
                    }
                ))
                doc_id += 1
        
        return docs
    
    def _create_queries(self) -> List[str]:
        """Create test queries."""
        return [
            "neural networks deep learning",           # AI/ML
            "vector database similarity search",       # Database
            "React frontend development",              # Web Dev
            "Docker Kubernetes deployment",            # DevOps
            "OAuth authentication security",           # Security
            "machine learning optimization",           # AI/ML
            "PostgreSQL query optimization",           # Database
            "API design best practices",               # Web Dev
            "CI/CD pipeline automation",               # DevOps
            "encryption data security"                 # Security
        ]
    
    def _create_ground_truth(self) -> List[List[str]]:
        """Define ground truth (relevant doc IDs) for each query."""
        return [
            # Query 0: "neural networks deep learning"
            ["doc_0", "doc_1", "doc_5"],
            
            # Query 1: "vector database similarity search"
            ["doc_11", "doc_10"],
            
            # Query 2: "React frontend development"
            ["doc_20", "doc_24"],
            
            # Query 3: "Docker Kubernetes deployment"
            ["doc_30", "doc_31", "doc_38"],
            
            # Query 4: "OAuth authentication security"
            ["doc_40", "doc_43"],
            
            # Query 5: "machine learning optimization"
            ["doc_1", "doc_7"],
            
            # Query 6: "PostgreSQL query optimization"
            ["doc_10", "doc_18"],
            
            # Query 7: "API design best practices"
            ["doc_21", "doc_27"],
            
            # Query 8: "CI/CD pipeline automation"
            ["doc_32", "doc_37"],
            
            # Query 9: "encryption data security"
            ["doc_43", "doc_48"]
        ]


async def benchmark_method(
    rag: pgVectorDB,
    method_name: str,
    queries: List[str],
    k: int,
    search_fn
) -> tuple[List[List[str]], float]:
    """
    Benchmark a single search method.
    
    Returns:
        (retrieved_results, average_latency_ms)
    """
    retrieved_results = []
    latencies = []
    
    for query in queries:
        start = time.time()
        results = await search_fn(query, k)
        latency = (time.time() - start) * 1000  # Convert to ms
        latencies.append(latency)
        
        # Extract doc IDs
        doc_ids = [f"doc_{r['metadata']['doc_id']}" for r in results]
        retrieved_results.append(doc_ids)
    
    avg_latency = sum(latencies) / len(latencies)
    return retrieved_results, avg_latency


async def main():
    """Main benchmark workflow."""
    
    print("=" * 80)
    print("COMPREHENSIVE SEARCH METHOD BENCHMARK")
    print("=" * 80)
    
    # 1. Setup
    print("\n📦 Step 1: Setting up RAG system...")
    
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    
    rag = pgVectorDB(
        collection_name="benchmark_test",
        embedding_model=embeddings,
        connection_string="postgresql+asyncpg://user:root@localhost:9002/postgres",
        index_type=IndexType.HNSW
    )
    
    await rag.initialize(overwrite_existing=True)
    
    # 2. Load benchmark dataset
    print("\n📝 Step 2: Loading benchmark dataset...")
    dataset = BenchmarkDataset()
    
    print(f"   - Documents: {len(dataset.documents)}")
    print(f"   - Queries: {len(dataset.queries)}")
    print(f"   - Categories: ai_ml, database, web_dev, devops, security")
    
    # 3. Add documents and build indexes
    print("\n🔨 Step 3: Building indexes...")
    
    await rag.add_documents(dataset.documents)
    await rag.create_metadata_index(["category", "priority"])
    
    # Build vector index (HNSW)
    await rag.build_index()
    
    # Build BM25 index
    await rag.build_bm25_index()
    
    print("   ✓ HNSW vector index built")
    print("   ✓ BM25 keyword index built")
    print("   ✓ Metadata indexes built")
    
    # 4. Define search methods to benchmark
    print("\n🔍 Step 4: Benchmarking all search methods...")
    print("   Testing K=5 across 10 queries\n")
    
    k = 5
    methods = []
    
    # Method 1: FTS keyword search
    methods.append({
        "name": "1. Keyword (FTS)",
        "search_fn": lambda q, k: rag.keyword_search(q, k, search_type=KeywordSearchType.FTS)
    })
    
    # Method 2: BM25 keyword search
    methods.append({
        "name": "2. Keyword (BM25)",
        "search_fn": lambda q, k: rag.keyword_search(q, k, search_type=KeywordSearchType.BM25)
    })
    
    # Method 3: Semantic search
    methods.append({
        "name": "3. Semantic",
        "search_fn": lambda q, k: rag.semantic_search(q, k)
    })
    
    # Method 4: Hybrid (FTS + Semantic, weighted)
    methods.append({
        "name": "4. Hybrid (FTS + Semantic)",
        "search_fn": lambda q, k: rag.hybrid_search(
            q, k, weights=(0.5, 0.5), 
            keyword_type=KeywordSearchType.FTS
        )
    })
    
    # Method 5: Hybrid (BM25 + Semantic, weighted)
    methods.append({
        "name": "5. Hybrid (BM25 + Semantic)",
        "search_fn": lambda q, k: rag.hybrid_search(
            q, k, weights=(0.5, 0.5),
            keyword_type=KeywordSearchType.BM25
        )
    })
    
    # Method 6: Hybrid (BM25 + Semantic, RRF)
    methods.append({
        "name": "6. Hybrid (BM25 + Semantic + RRF)",
        "search_fn": lambda q, k: rag.hybrid_search(
            q, k, use_rrf=True,
            keyword_type=KeywordSearchType.BM25
        )
    })
    
    # Method 7: Metadata + Semantic (high priority only)
    methods.append({
        "name": "7. Metadata + Semantic",
        "search_fn": lambda q, k: rag.metadata_semantic_search(
            q, {"priority": {"$in": ["high", "critical"]}}, k
        )
    })
    
    # Method 8: Ensemble (BM25 + Semantic + RRF + Metadata)
    methods.append({
        "name": "8. Ensemble (Full)",
        "search_fn": lambda q, k: rag.ensemble_search(
            q, {"priority": {"$in": ["high", "critical", "medium"]}}, k,
            use_rrf=True, keyword_type=KeywordSearchType.BM25
        )
    })
    
    # Method 9: Trigram (fuzzy matching)
    methods.append({
        "name": "9. Trigram (Fuzzy)",
        "search_fn": lambda q, k: rag.trigram_search(q, k, threshold=0.1)
    })
    
    # Method 10: Metadata + Trigram
    methods.append({
        "name": "10. Metadata + Trigram",
        "search_fn": lambda q, k: rag.metadata_trigram_search(
            q, {"category": {"$in": ["ai_ml", "database", "web_dev", "devops", "security"]}}, 
            k, threshold=0.1
        )
    })
    
    # 5. Run benchmarks
    evaluator = RAGEvaluator(k=k)
    results_data = []
    
    for method in methods:
        print(f"   Testing: {method['name']}...", end=" ")
        
        try:
            retrieved, latency = await benchmark_method(
                rag, method['name'], dataset.queries, k, method['search_fn']
            )
            
            # Evaluate
            eval_result = evaluator.evaluate(
                dataset.queries,
                retrieved,
                dataset.ground_truth
            )
            
            results_data.append({
                "Method": method['name'],
                "Precision@5": f"{eval_result.precision:.3f}",
                "Recall@5": f"{eval_result.recall:.3f}",
                "F1@5": f"{eval_result.f1_score:.3f}",
                "MAP@5": f"{eval_result.map_score:.3f}",
                "MRR": f"{eval_result.mrr_score:.3f}",
                "NDCG@5": f"{eval_result.ndcg_score:.3f}",
                "Hit Rate": f"{eval_result.hit_rate:.3f}",
                "Latency (ms)": f"{latency:.1f}"
            })
            
            print(f"✓ (P={eval_result.precision:.3f}, {latency:.1f}ms)")
            
        except Exception as e:
            print(f"✗ Error: {e}")
            results_data.append({
                "Method": method['name'],
                "Precision@5": "ERROR",
                "Recall@5": "ERROR",
                "F1@5": "ERROR",
                "MAP@5": "ERROR",
                "MRR": "ERROR",
                "NDCG@5": "ERROR",
                "Hit Rate": "ERROR",
                "Latency (ms)": "ERROR"
            })
    
    # 6. Display results
    print("\n" + "=" * 80)
    print("BENCHMARK RESULTS")
    print("=" * 80)
    
    df = pd.DataFrame(results_data)
    print("\n" + df.to_string(index=False))
    
    # 7. Analysis and Recommendations
    print("\n" + "=" * 80)
    print("ANALYSIS & RECOMMENDATIONS")
    print("=" * 80)
    
    print("""
📊 Metric Explanations:
  • Precision@5: Of 5 retrieved docs, how many are relevant? (Higher = less noise)
  • Recall@5: Of all relevant docs, how many did we retrieve? (Higher = more coverage)
  • F1@5: Balance between precision and recall (Higher = better overall)
  • MAP@5: Rank-aware precision (rewards relevant docs at top positions)
  • MRR: 1/rank of first relevant doc (Higher = faster to find answer)
  • NDCG@5: Ranking quality with position discount (Higher = better ranking)
  • Hit Rate: % of queries with ≥1 relevant result (Higher = more reliable)
  • Latency: Average query time in milliseconds

🎯 Use Case Recommendations:
  • FAQ/Chatbot (speed + first result):    Check MRR + Latency
  • General RAG (balanced):                 Check F1 + NDCG
  • Research/Comprehensive (coverage):      Check Recall + Hit Rate
  • Production (quality + speed):           Check Precision + Latency
  
🔍 Expected Trends:
  • BM25 > FTS:              Better keyword ranking
  • Semantic:                Best for conceptual matches
  • Hybrid (BM25+Semantic):  Best overall accuracy
  • RRF:                     No weight tuning needed
  • Ensemble:                Highest precision (filtered results)
  • Trigram:                 Best for typo tolerance
""")
    
    # 8. Export results
    print("\n💾 Exporting results...")
    df.to_csv("benchmark_results.csv", index=False)
    df.to_json("benchmark_results.json", orient="records", indent=2)
    print("   ✓ Results saved to benchmark_results.csv and benchmark_results.json")
    
    print("\n" + "=" * 80)
    print("✅ Benchmark Complete!")
    print("=" * 80)


if __name__ == "__main__":
    asyncio.run(main())
