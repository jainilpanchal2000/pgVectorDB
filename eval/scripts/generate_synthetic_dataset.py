import json
import random
from pathlib import Path
from typing import Any

# Configuration
NUM_DOCUMENTS = 1000
OUTPUT_FILE = "eval/data/benchmark_dataset_1k.json"

CATEGORIES = {
    "ai_ml": [
        "Neural Networks",
        "Deep Learning",
        "Reinforcement Learning",
        "Computer Vision",
        "NLP",
        "Transformers",
        "GANs",
        "Diffusion Models",
        "LLMs",
        "Gradient Descent",
        "Backpropagation",
        "Overfitting",
        "Regularization",
        "Dropout",
        "Batch Normalization",
        "Activation Functions",
        "Loss Functions",
        "Optimizers",
        "Transfer Learning",
        "Fine-tuning",
    ],
    "database": [
        "PostgreSQL",
        "MySQL",
        "MongoDB",
        "Redis",
        "Cassandra",
        "Elasticsearch",
        "Neo4j",
        "SQL",
        "NoSQL",
        "Indexing",
        "B-Tree",
        "LSM Tree",
        "ACID",
        "CAP Theorem",
        "Sharding",
        "Replication",
        "Partitioning",
        "Normalization",
        "Denormalization",
        "Transactions",
    ],
    "web_dev": [
        "React",
        "Vue",
        "Angular",
        "Svelte",
        "Node.js",
        "Django",
        "Flask",
        "FastAPI",
        "HTML5",
        "CSS3",
        "JavaScript",
        "TypeScript",
        "WebAssembly",
        "REST",
        "GraphQL",
        "gRPC",
        "WebSockets",
        "PWA",
        "SPA",
        "SSR",
    ],
    "devops": [
        "Docker",
        "Kubernetes",
        "Jenkins",
        "GitLab CI",
        "GitHub Actions",
        "Terraform",
        "Ansible",
        "Prometheus",
        "Grafana",
        "ELK Stack",
        "AWS",
        "Azure",
        "GCP",
        "Microservices",
        "Serverless",
        "IaC",
        "GitOps",
        "Blue-Green Deployment",
        "Canary",
        "Service Mesh",
    ],
    "security": [
        "OAuth2",
        "OIDC",
        "JWT",
        "SAML",
        "MFA",
        "RBAC",
        "ABAC",
        "SQL Injection",
        "XSS",
        "CSRF",
        "DDoS",
        "Phishing",
        "Ransomware",
        "Encryption",
        "Hashing",
        "PKI",
        "TLS/SSL",
        "Firewalls",
        "VPN",
        "Zero Trust",
    ],
    "cloud": [
        "EC2",
        "S3",
        "Lambda",
        "DynamoDB",
        "RDS",
        "VPC",
        "CloudFront",
        "Route53",
        "Azure VM",
        "Azure Blob",
        "Azure Functions",
        "GCE",
        "GCS",
        "Cloud Run",
        "BigQuery",
        "Redshift",
        "Snowflake",
        "CloudFormation",
        "ARM Templates",
        "CloudWatch",
    ],
    "data_science": [
        "Pandas",
        "NumPy",
        "Scikit-learn",
        "Matplotlib",
        "Seaborn",
        "Plotly",
        "Jupyter",
        "Data Cleaning",
        "Feature Engineering",
        "EDA",
        "Hypothesis Testing",
        "A/B Testing",
        "Regression",
        "Classification",
        "Clustering",
        "Dimensionality Reduction",
        "PCA",
        "Time Series",
        "Bayesian Statistics",
        "MCMC",
    ],
    "blockchain": [
        "Bitcoin",
        "Ethereum",
        "Smart Contracts",
        "Solidity",
        "DeFi",
        "NFTs",
        "DAO",
        "Consensus Algorithms",
        "PoW",
        "PoS",
        "Layer 2",
        "Rollups",
        "Zero Knowledge Proofs",
        "Web3",
        "DApps",
        "Wallets",
        "Cryptography",
        "Hashing",
        "Merkle Trees",
        "Gas",
    ],
    "iot": [
        "Sensors",
        "Actuators",
        "Microcontrollers",
        "Arduino",
        "Raspberry Pi",
        "MQTT",
        "CoAP",
        "Zigbee",
        "LoRaWAN",
        "Edge Computing",
        "IoT Security",
        "Smart Home",
        "Industrial IoT",
        "Wearables",
        "Embedded Systems",
        "RTOS",
        "Firmware",
        "OTA Updates",
        "Digital Twins",
        "Fleet Management",
    ],
    "mobile": [
        "iOS",
        "Android",
        "Swift",
        "Kotlin",
        "React Native",
        "Flutter",
        "Objective-C",
        "Java",
        "Dart",
        "Xcode",
        "Android Studio",
        "Mobile UI/UX",
        "App Store",
        "Play Store",
        "Push Notifications",
        "Deep Linking",
        "Offline Storage",
        "Performance Optimization",
        "Mobile Security",
        "Cross-Platform",
    ],
}

TEMPLATES = [
    "The core concept of {topic} involves {action} to achieve {goal}.",
    "In the realm of {category}, {topic} plays a crucial role by {action}.",
    "{topic} is a fundamental technology in {category} that enables {goal}.",
    "When implementing {topic}, one must consider {action} for optimal {goal}.",
    "Advanced {topic} techniques include {action}, which significantly improves {goal}.",
    "{topic} has revolutionized {category} by allowing developers to {action}.",
    "One of the key challenges in {topic} is {action}, but this leads to better {goal}.",
    "Comparing {topic} with other {category} solutions reveals its strength in {goal}.",
    "Future trends in {topic} point towards more efficient {action} for {goal}.",
    "Ideally, {topic} should be used when the objective is {goal} through {action}.",
]

ACTIONS = [
    "optimizing performance",
    "enhancing security",
    "scaling infrastructure",
    "improving user experience",
    "automating workflows",
    "analyzing data patterns",
    "securing transactions",
    "managing state",
    "handling concurrency",
    "reducing latency",
    "increasing throughput",
    "ensuring consistency",
    "facilitating communication",
    "abstracting complexity",
    "monitoring health",
    "detecting anomalies",
    "generating insights",
    "predicting outcomes",
    "classifying inputs",
    "clustering entities",
]

GOALS = [
    "robustness",
    "scalability",
    "maintainability",
    "reliability",
    "efficiency",
    "accuracy",
    "precision",
    "observability",
    "interoperability",
    "flexibility",
    "security",
    "compliance",
    "usability",
    "accessibility",
    "availability",
    "consistency",
    "integrity",
    "confidentiality",
    "transparency",
    "auditability",
]


def generate_document(doc_id: int) -> dict[str, Any]:
    category = random.choice(list(CATEGORIES.keys()))
    topic = random.choice(CATEGORIES[category])
    action = random.choice(ACTIONS)
    goal = random.choice(GOALS)
    template = random.choice(TEMPLATES)

    content = template.format(
        topic=topic, category=category.replace("_", " "), action=action, goal=goal
    )

    # Add some random noise/extra sentences to make it more like a paragraph
    for _ in range(random.randint(1, 4)):
        t2 = random.choice(CATEGORIES[category])
        a2 = random.choice(ACTIONS)
        g2 = random.choice(GOALS)
        content += f" Furthermore, {t2} contributes to {g2} by {a2}."

    return {
        "page_content": content,
        "metadata": {
            "doc_id": doc_id,
            "category": category,
            "topic": topic,
            "year": random.randint(2015, 2025),
            "priority": random.choice(["low", "medium", "high", "critical"]),
        },
    }


def generate_query(docs: list[dict[str, Any]], query_id: int) -> dict[str, Any]:
    # Select a target document to base the query on
    target_doc = random.choice(docs)
    topic = target_doc["metadata"]["topic"]
    category = target_doc["metadata"]["category"]

    # Create a query that SHOULD match this topic
    query_templates = [
        "How does {topic} work?",
        "Explain {topic} in {category}",
        "What are the benefits of {topic}?",
        "Best practices for {topic}",
        "{topic} implementation details",
        "Issues with {topic} and how to solve them",
        "Comparison of {topic} and alternatives",
        "Future of {topic} in {category}",
    ]
    query_text = random.choice(query_templates).format(
        topic=topic, category=category.replace("_", " ")
    )

    # Find relevant documents (ground truth)
    # We define relevance as documents containing the SAME topic
    relevant_ids = []
    for doc in docs:
        if doc["metadata"]["topic"] == topic:
            relevant_ids.append(f"doc_{doc['metadata']['doc_id']}")

    return {
        "query": query_text,
        "ground_truth": relevant_ids,
        "metadata": {"category": category, "target_topic": topic},
    }


def main():
    print(f"Generating {NUM_DOCUMENTS} documents...")

    documents = []
    for i in range(NUM_DOCUMENTS):
        documents.append(generate_document(i))

    print("Generating queries...")
    queries = []
    # Generate 50 queries covering various topics
    for i in range(50):
        queries.append(generate_query(documents, i))

    dataset = {
        "documents": documents,
        "queries": [q["query"] for q in queries],
        "ground_truth": [q["ground_truth"] for q in queries],
        "query_metadata": [q["metadata"] for q in queries],
    }

    output_path = Path(OUTPUT_FILE)
    output_path.parent.mkdir(exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=2)

    print(f"Dataset saved to {OUTPUT_FILE}")
    print(f"- Documents: {len(documents)}")
    print(f"- Queries: {len(queries)}")


if __name__ == "__main__":
    main()
