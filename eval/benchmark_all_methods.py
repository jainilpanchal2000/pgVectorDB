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
from src.core import pgVectorDB, IndexType, KeywordSearchType
from src.evaluation import RAGEvaluator, EvaluationResult
from src.config import Config
import pandas as pd


class BenchmarkDataset:
    """Comprehensive dataset with 200 documents and complex queries for realistic benchmarking."""
    
    def __init__(self):
        self.documents = self._create_documents()
        self.queries = self._create_queries()
        self.ground_truth = self._create_ground_truth()
    
    def _create_documents(self) -> List[Document]:
        """Create 200 realistic documents across 8 categories with detailed content."""
        docs = []
        
        # AI/ML Category (30 docs)
        ai_ml_topics = [
            "Neural networks and deep learning architectures for image classification",
            "Machine learning model optimization using gradient descent and backpropagation",
            "Natural language processing with transformer models like BERT and GPT",
            "Computer vision and image recognition using convolutional neural networks",
            "Reinforcement learning algorithms for game AI and robotics",
            "Transfer learning in deep neural networks for domain adaptation",
            "Generative AI and large language models for text generation",
            "AutoML and neural architecture search for automated model design",
            "Explainable AI and model interpretability using SHAP and LIME",
            "Edge AI and model compression techniques for mobile deployment",
            "Recurrent neural networks and LSTM for sequence prediction",
            "Attention mechanisms and self-attention in transformer architectures",
            "Few-shot learning and meta-learning for data-efficient AI",
            "Adversarial machine learning and robustness testing",
            "Federated learning for privacy-preserving distributed training",
            "Graph neural networks for social network analysis",
            "Time series forecasting with deep learning models",
            "Multi-modal learning combining vision and language",
            "Active learning strategies for efficient data labeling",
            "Hyperparameter tuning and optimization techniques",
            "Ensemble methods combining multiple models",
            "Anomaly detection using autoencoders and isolation forests",
            "Dimensionality reduction with PCA and t-SNE",
            "Clustering algorithms k-means DBSCAN hierarchical",
            "Decision trees random forests gradient boosting XGBoost",
            "Support vector machines kernel methods",
            "Bayesian optimization for model selection",
            "Neural style transfer and image generation",
            "Object detection YOLO Faster R-CNN",
            "Semantic segmentation and instance segmentation models"
        ]
        
        # Database Category (30 docs)
        database_topics = [
            "PostgreSQL indexing strategies B-tree GiST GIN and query optimization techniques",
            "Vector databases for similarity search and nearest neighbor retrieval",
            "NoSQL databases MongoDB Cassandra document stores key-value stores",
            "Database sharding and horizontal scaling for distributed systems",
            "ACID transactions and consistency models in relational databases",
            "Database replication and high availability with master-slave architecture",
            "Time-series databases InfluxDB TimescaleDB for IoT sensor data",
            "Graph databases Neo4j relationship queries and graph traversal",
            "Database performance tuning query planning and monitoring",
            "Distributed databases and CAP theorem eventual consistency",
            "Column-oriented databases for analytical workloads",
            "In-memory databases Redis Memcached for caching",
            "Database normalization denormalization trade-offs",
            "Full-text search engines Elasticsearch Solr",
            "Database backup recovery disaster recovery strategies",
            "SQL injection attacks prevention and parameterized queries",
            "Database connection pooling and resource management",
            "Multi-version concurrency control MVCC in PostgreSQL",
            "Database partitioning strategies range hash list",
            "Query optimization techniques join algorithms index selection",
            "Database migrations and schema evolution",
            "Data warehousing ETL processes star schema",
            "OLTP vs OLAP database design patterns",
            "Database security encryption authentication authorization",
            "Stored procedures triggers and database programming",
            "Database monitoring tools pgAdmin Datadog",
            "Vector similarity search with pgvector extension",
            "Database clustering Galera Cluster Patroni",
            "Change data capture CDC for real-time sync",
            "Database indexing for JSON JSONB fields"
        ]
        
        # Web Development Category (30 docs)
        web_topics = [
            "React hooks useState useEffect modern frontend development patterns",
            "RESTful API design principles HTTP methods status codes",
            "GraphQL schema design queries mutations subscriptions",
            "Web application security JWT authentication OAuth2 sessions",
            "Server-side rendering SSR with Next.js for SEO optimization",
            "WebSocket protocol for real-time bidirectional communication",
            "Progressive Web Apps PWA offline functionality service workers",
            "Web performance optimization lazy loading code splitting",
            "Microservices architecture API gateway service mesh",
            "Web accessibility WCAG ARIA screen readers keyboard navigation",
            "Vue.js composition API reactive state management",
            "Angular dependency injection RxJS observables",
            "TypeScript type safety interfaces generics decorators",
            "CSS frameworks Tailwind Bootstrap responsive design",
            "State management Redux Zustand Pinia patterns",
            "Testing frameworks Jest Cypress integration tests",
            "Build tools Webpack Vite Rollup bundling",
            "Cross-origin resource sharing CORS policies",
            "Content delivery networks CDN edge caching",
            "Single page applications SPA routing navigation",
            "Web components custom elements shadow DOM",
            "Browser developer tools debugging performance profiling",
            "HTTP/2 HTTP/3 QUIC protocol improvements",
            "Web animations CSS transitions JavaScript libraries",
            "Form validation client-side server-side",
            "Internationalization i18n localization l10n",
            "Error boundaries error handling logging",
            "Web vitals LCP FID CLS metrics",
            "API versioning backward compatibility",
            "Rate limiting throttling debouncing techniques"
        ]
        
        # DevOps Category (30 docs)
        devops_topics = [
            "Docker containerization Dockerfile multi-stage builds best practices",
            "Kubernetes orchestration deployments services ingress controllers",
            "CI/CD pipelines GitHub Actions Jenkins automated testing deployment",
            "Infrastructure as Code Terraform AWS CloudFormation configuration management",
            "Monitoring and observability Prometheus Grafana metrics alerting",
            "Log aggregation ELK stack Splunk centralized logging analysis",
            "Cloud migration strategies lift-and-shift refactoring patterns",
            "Automated testing unit tests integration tests end-to-end testing pipelines",
            "Service mesh Istio Linkerd microservices networking traffic management",
            "GitOps ArgoCD Flux declarative infrastructure continuous deployment",
            "Container orchestration Docker Swarm vs Kubernetes comparison",
            "Blue-green deployments canary releases rolling updates",
            "Configuration management Ansible Puppet Chef",
            "Secrets management HashiCorp Vault AWS Secrets Manager",
            "Load balancing HAProxy NGINX reverse proxy",
            "Auto-scaling horizontal pod autoscaler cluster autoscaler",
            "Disaster recovery backup strategies RPO RTO",
            "Observability tracing Jaeger OpenTelemetry distributed tracing",
            "Container security vulnerability scanning image signing",
            "Continuous integration build automation artifact management",
            "Environment management dev staging production",
            "Feature flags dark launches progressive rollouts",
            "Incident management on-call runbooks postmortems",
            "Performance testing load testing stress testing",
            "Database migrations schema changes zero-downtime",
            "API gateway Kong Traefik routing authentication",
            "Serverless functions AWS Lambda event-driven architecture",
            "Cost optimization cloud resource management tagging",
            "Compliance automation security scanning policy enforcement",
            "Service level objectives SLOs error budgets"
        ]
        
        # Security Category (30 docs)
        security_topics = [
            "OAuth2 OpenID Connect PKCE authorization code flow authentication",
            "SQL injection prevention parameterized queries prepared statements ORM",
            "Cross-site scripting XSS attack mitigation Content Security Policy",
            "Encryption at rest AES-256 encryption in transit TLS SSL",
            "Zero-trust security architecture least privilege network segmentation",
            "Penetration testing vulnerability scanning ethical hacking OWASP",
            "Security auditing compliance SOC2 HIPAA GDPR ISO27001",
            "API security rate limiting API keys OAuth tokens",
            "Secret management key rotation credential scanning",
            "DDoS protection rate limiting WAF mitigation strategies",
            "Multi-factor authentication MFA TOTP biometrics",
            "Session management CSRF tokens secure cookies",
            "Password hashing bcrypt Argon2 salting",
            "Certificate management PKI SSL/TLS certificates",
            "Network security firewalls VPN IPSec",
            "Container security image scanning runtime protection",
            "Intrusion detection IDS IPS SIEM systems",
            "Security headers HSTS X-Frame-Options CSP",
            "Access control RBAC ABAC authorization models",
            "Threat modeling STRIDE DREAD risk assessment",
            "Data masking tokenization PII protection",
            "Security monitoring log analysis anomaly detection",
            "Incident response forensics breach handling",
            "Supply chain security dependency scanning SCA",
            "API authentication bearer tokens JWT validation",
            "Cryptography asymmetric symmetric hashing signing",
            "Web application firewall WAF ModSecurity",
            "Security testing SAST DAST IAST tools",
            "Compliance automation policy as code",
            "Identity and access management IAM SSO"
        ]
        
        # Cloud Computing Category (25 docs)
        cloud_topics = [
            "AWS EC2 instance types auto-scaling elastic load balancing",
            "Azure virtual machines resource groups ARM templates",
            "Google Cloud Platform Compute Engine App Engine services",
            "Serverless computing AWS Lambda Azure Functions event-driven",
            "Cloud storage S3 Blob Storage object storage lifecycle",
            "Cloud networking VPC subnets security groups routing",
            "Managed databases RDS Aurora DynamoDB Cosmos DB",
            "Container services ECS EKS AKS GKE",
            "Cloud IAM roles policies service accounts",
            "Cloud monitoring CloudWatch Azure Monitor stackdriver",
            "CDN CloudFront Azure CDN edge locations",
            "Message queues SQS SNS Service Bus Pub/Sub",
            "Cloud migration assessment planning execution",
            "Multi-cloud hybrid cloud strategies vendor lock-in",
            "Cloud cost optimization reserved instances spot instances",
            "Infrastructure as code CloudFormation Terraform ARM",
            "Cloud backup disaster recovery region failover",
            "API management API Gateway APIM",
            "Cloud security best practices encryption compliance",
            "DevOps on cloud CI/CD pipelines automation",
            "Cloud-native applications twelve-factor app methodology",
            "Container registry ECR ACR GCR image management",
            "Cloud logging centralized logs audit trails",
            "Cloud automation Lambda functions automation scripts",
            "Cloud governance policies tagging cost allocation"
        ]
        
        # Data Science Category (25 docs)
        data_science_topics = [
            "Exploratory data analysis pandas numpy visualization techniques",
            "Statistical hypothesis testing t-tests ANOVA chi-square",
            "Feature engineering selection extraction transformation",
            "Data cleaning missing values outliers normalization",
            "A/B testing experimental design statistical significance",
            "Predictive modeling regression classification algorithms",
            "Model evaluation metrics accuracy precision recall F1",
            "Cross-validation k-fold stratified time-series splits",
            "Imbalanced data SMOTE undersampling oversampling",
            "Data visualization matplotlib seaborn plotly dashboards",
            "Big data processing Spark Hadoop MapReduce",
            "ETL pipelines data integration data quality",
            "Data warehousing Snowflake Redshift BigQuery",
            "Stream processing Kafka Flink real-time analytics",
            "Natural language processing text mining sentiment analysis",
            "Recommendation systems collaborative filtering content-based",
            "Time series analysis ARIMA forecasting seasonality",
            "Causal inference propensity scores matching",
            "Bayesian statistics probabilistic modeling MCMC",
            "Survival analysis Cox proportional hazards Kaplan-Meier",
            "Principal component analysis dimensionality reduction",
            "Network analysis social graphs community detection",
            "Geospatial analysis GIS mapping spatial statistics",
            "Experiment tracking MLflow Weights Biases",
            "Data governance lineage quality metadata management"
        ]
        
        categories = [
            ("ai_ml", ai_ml_topics, "high", 2018),
            ("database", database_topics, "medium", 2019),
            ("web_dev", web_topics, "medium", 2020),
            ("devops", devops_topics, "high", 2021),
            ("security", security_topics, "critical", 2022),
            ("cloud", cloud_topics, "high", 2020),
            ("data_science", data_science_topics, "medium", 2019),
            ("mobile_dev", [
                "iOS Swift SwiftUI UIKit mobile app development",
                "Android Kotlin Jetpack Compose Material Design",
                "React Native cross-platform mobile development",
                "Flutter Dart widgets state management",
                "Mobile CI/CD Fastlane automated testing deployment",
                "Push notifications FCM APNs messaging",
                "Mobile analytics Firebase Analytics crash reporting",
                "App store optimization ASO keywords ratings",
                "Mobile security encryption data protection",
                "Offline-first architecture local storage sync",
                "Mobile performance optimization battery life",
                "Deep linking universal links app navigation",
                "Mobile testing XCTest Espresso UI automation",
                "App architecture MVVM MVP Clean Architecture",
                "Mobile databases Realm SQLite CoreData"
            ], "medium", 2021)
        ]
        
        doc_id = 0
        for category, topics, priority, base_year in categories:
            for i, topic in enumerate(topics):
                docs.append(Document(
                    page_content=f"{topic}. This comprehensive technical document covers {topic.lower()} with detailed explanations, best practices, implementation patterns, common pitfalls, performance considerations, and real-world use cases for production environments.",
                    metadata={
                        "doc_id": doc_id,
                        "category": category,
                        "priority": priority,
                        "year": base_year + (i % 6),
                        "author": f"Expert_{(i % 5) + 1}",
                        "difficulty": ["beginner", "intermediate", "advanced", "expert"][i % 4]
                    }
                ))
                doc_id += 1
        
        return docs
    
    def _create_queries(self) -> List[str]:
        """Create complex and realistic test queries."""
        return [
            # Complex multi-term queries
            "how to implement transformer models with attention mechanisms for natural language processing",
            "PostgreSQL vector similarity search indexing strategies and performance optimization techniques",
            "React hooks useState useEffect server-side rendering Next.js SEO optimization",
            "Kubernetes deployment strategies blue-green canary rolling updates with Docker containers",
            "OAuth2 authorization code flow PKCE security best practices JWT token validation",
            
            # Technical deep-dive queries
            "deep learning neural network architectures CNN RNN LSTM for image classification",
            "database sharding horizontal scaling replication high availability distributed systems",
            "microservices architecture API gateway service mesh monitoring observability",
            "CI/CD pipeline automation GitHub Actions Jenkins testing deployment strategies",
            "web application security XSS CSRF SQL injection prevention authentication",
            
            # Specific technology queries
            "transfer learning fine-tuning pre-trained models BERT GPT for NLP tasks",
            "NoSQL MongoDB Cassandra vs relational databases PostgreSQL MySQL comparison",
            "GraphQL schema design queries mutations subscriptions versus REST API",
            "Infrastructure as Code Terraform AWS CloudFormation configuration management",
            "encryption AES-256 TLS SSL certificate management PKI security",
            
            # Problem-solving queries
            "machine learning model overfitting regularization cross-validation techniques",
            "database performance tuning slow queries index optimization monitoring",
            "React state management Redux Zustand context API best practices",
            "container orchestration Kubernetes vs Docker Swarm production deployment",
            "penetration testing vulnerability scanning OWASP security audit compliance",
            
            # Advanced technical queries
            "federated learning privacy-preserving distributed machine learning training",
            "time-series database InfluxDB TimescaleDB sensor data IoT real-time analytics",
            "Progressive Web Apps PWA service workers offline functionality caching",
            "serverless computing AWS Lambda event-driven architecture cost optimization",
            "zero-trust security architecture network segmentation least privilege access"
        ]
    
    def _create_ground_truth(self) -> List[List[str]]:
        """Define ground truth (relevant doc IDs) for each complex query."""
        return [
            # Query 0: transformer models attention NLP
            ["doc_2", "doc_11", "doc_17"],
            
            # Query 1: PostgreSQL vector similarity indexing
            ["doc_30", "doc_31", "doc_39", "doc_56"],
            
            # Query 2: React hooks SSR Next.js
            ["doc_60", "doc_64", "doc_72"],
            
            # Query 3: Kubernetes deployment Docker
            ["doc_90", "doc_91", "doc_100"],
            
            # Query 4: OAuth2 PKCE JWT security
            ["doc_120", "doc_127", "doc_144"],
            
            # Query 5: deep learning CNN RNN LSTM
            ["doc_0", "doc_3", "doc_10"],
            
            # Query 6: database sharding replication distributed
            ["doc_33", "doc_35", "doc_39"],
            
            # Query 7: microservices API gateway service mesh
            ["doc_68", "doc_98"],
            
            # Query 8: CI/CD GitHub Actions Jenkins
            ["doc_92", "doc_97"],
            
            # Query 9: web security XSS CSRF SQL injection
            ["doc_63", "doc_121", "doc_122"],
            
            # Query 10: transfer learning BERT GPT NLP
            ["doc_2", "doc_5", "doc_6"],
            
            # Query 11: NoSQL MongoDB PostgreSQL comparison
            ["doc_32", "doc_30"],
            
            # Query 12: GraphQL schema vs REST
            ["doc_62", "doc_61"],
            
            # Query 13: Terraform CloudFormation IaC
            ["doc_93", "doc_145"],
            
            # Query 14: encryption TLS SSL PKI
            ["doc_123", "doc_143"],
            
            # Query 15: ML overfitting regularization
            ["doc_1", "doc_19", "doc_176"],
            
            # Query 16: database performance tuning
            ["doc_30", "doc_38", "doc_39"],
            
            # Query 17: React state management Redux
            ["doc_74", "doc_60"],
            
            # Query 18: Kubernetes Docker Swarm comparison
            ["doc_90", "doc_91", "doc_100"],
            
            # Query 19: penetration testing OWASP security
            ["doc_125", "doc_126"],
            
            # Query 20: federated learning privacy
            ["doc_14", "doc_8"],
            
            # Query 21: time-series database InfluxDB IoT
            ["doc_36", "doc_183"],
            
            # Query 22: PWA service workers offline
            ["doc_66", "doc_189"],
            
            # Query 23: serverless Lambda event-driven
            ["doc_153", "doc_96"],
            
            # Query 24: zero-trust architecture security
            ["doc_124", "doc_138"]
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
    
    # Display current configuration
    Config.print_config()
    
    # Get embeddings based on configuration
    embeddings = Config.get_embeddings()
    
    # Get connection string (respects ENVIRONMENT setting)
    connection_string = Config.get_connection_string()
    
    rag = pgVectorDB(
        collection_name="benchmark_test",
        embedding_model=embeddings,
        connection_string=connection_string,
        index_type=IndexType.HNSW
    )
    
    await rag.initialize(overwrite_existing=True)
    
    # 2. Load benchmark dataset
    print("\n📝 Step 2: Loading benchmark dataset...")
    dataset = BenchmarkDataset()
    
    print(f"   - Documents: {len(dataset.documents)}")
    print(f"   - Queries: {len(dataset.queries)}")
    print(f"   - Categories: ai_ml, database, web_dev, devops, security, cloud, data_science, mobile_dev")
    
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
    print(f"   Testing K=5 across {len(dataset.queries)} complex queries\n")
    
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
    df.to_csv("eval/benchmark_results.csv", index=False)
    df.to_json("eval/benchmark_results.json", orient="records", indent=2)
    print("   ✓ Results saved to eval/benchmark_results.csv and eval/benchmark_results.json")
    
    print("\n" + "=" * 80)
    print("✅ Benchmark Complete!")
    print("=" * 80)


if __name__ == "__main__":
    asyncio.run(main())
