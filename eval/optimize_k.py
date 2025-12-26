"""
K-Value Optimization Example
=============================

This script demonstrates how to find the optimal K value for your RAG system.

Run this after you have:
1. A RAG system set up
2. An evaluation dataset with ground truth
3. Time to test multiple K values

The script will:
- Test K values: 1, 3, 5, 10, 20, 50
- Show precision/recall trade-offs
- Recommend optimal K for different use cases
"""

import asyncio
from typing import List
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from src.core import pgVectorDB, IndexType
from src.evaluation import (
    create_sample_evaluation_dataset,
    KValueAnalysis
)


async def test_k_value(
    rag: pgVectorDB,
    queries: List[str],
    k: int
) -> List[List[str]]:
    """
    Retrieve results for a specific K value.
    
    Args:
        rag: Initialized pgVectorDB system
        queries: List of query strings
        k: Number of results to retrieve
    
    Returns:
        List of retrieved doc IDs for each query
    """
    results = []
    
    for query in queries:
        # Retrieve documents
        docs = await rag.semantic_search(query, k=k)
        
        # Extract doc IDs (handle both dict and Document objects)
        if docs and isinstance(docs[0], dict):
            doc_ids = [f"doc_{doc.get('metadata', {}).get('doc_id', 'unknown')}" for doc in docs]
        else:
            doc_ids = [f"doc_{doc.metadata.get('doc_id', 'unknown')}" for doc in docs]
        results.append(doc_ids)
    
    return results


async def main():
    """Main K-value optimization workflow."""
    
    print("=" * 80)
    print("K-VALUE OPTIMIZATION FOR RAG SYSTEMS")
    print("=" * 80)
    
    # 1. Setup
    print("\n📦 Step 1: Setting up RAG system...")
    
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    
    rag = pgVectorDB(
        collection_name="k_value_optimization",
        embedding_model=embeddings,
        connection_string="postgresql+asyncpg://user:root@localhost:9002/postgres",
        index_type=IndexType.HNSW
    )
    
    await rag.initialize(overwrite_existing=True)
    
    # 2. Generate sample data
    print("\n📝 Step 2: Generating sample documents...")
    
    categories = ["programming", "ai", "database", "web", "devops", "security", "cloud", "mobile"]
    documents = []
    
    for i in range(100):
        category = categories[i % len(categories)]
        documents.append(
            Document(
                page_content=f"Document {i} about {category}: This is sample content for testing.",
                metadata={
                    "doc_id": i,
                    "category": category,
                    "year": 2020 + (i % 5)
                }
            )
        )
    
    await rag.add_documents(documents)
    await rag.build_index()
    
    # 3. Create evaluation dataset
    print("\n🎯 Step 3: Creating evaluation dataset...")
    
    dataset = create_sample_evaluation_dataset()
    
    # 4. Test multiple K values
    print("\n🔍 Step 4: Testing multiple K values...")
    print("   This may take a minute...\n")
    
    k_values_to_test = [1, 3, 5, 10, 20, 50]
    retrieved_results_by_k = {}
    
    for k in k_values_to_test:
        print(f"   Testing K={k}...")
        results = await test_k_value(rag, dataset.queries, k)
        retrieved_results_by_k[k] = results
    
    # 5. Analyze results
    print("\n📊 Step 5: Analyzing results...\n")
    
    analyzer = KValueAnalysis()
    analyzer.analyze(
        queries=dataset.queries,
        retrieved_results_by_k=retrieved_results_by_k,
        ground_truth=dataset.ground_truth
    )
    
    # 6. Display analysis
    analyzer.print_analysis()
    analyzer.print_recommendation()
    
    # 7. Export results
    print("\n💾 Step 6: Exporting results...")
    analyzer.export_results("k_value_analysis_results.json")
    
    print("\n" + "=" * 80)
    print("INTERPRETATION GUIDE")
    print("=" * 80)
    
    print("""
📈 Understanding the Results:

1. PRECISION@K (Quality)
   - Higher K → Lower precision (more noise)
   - Lower K → Higher precision (fewer results, more focused)
   - Use Case: LLM with limited context needs high precision

2. RECALL@K (Coverage)
   - Higher K → Higher recall (find more relevant docs)
   - Lower K → Lower recall (miss some relevant docs)
   - Use Case: Research/analysis needs high recall

3. F1@K (Balance)
   - Optimal K usually maximizes F1
   - Best general-purpose choice
   - Use Case: Most production RAG applications

4. NDCG@K (Ranking Quality)
   - Shows if relevant docs appear at top
   - Important for user experience
   - Use Case: When order matters

5. MRR (First Result)
   - High MRR = First result usually relevant
   - Critical for FAQ/Q&A systems
   - Use Case: User reads only first result

💡 Recommendations by Use Case:

• FAQ/Customer Support (K=1-3)
  - High precision needed
  - First result critical
  - Example: "How do I reset password?"

• General RAG Application (K=5-10)
  - Balanced precision/recall
  - Most common use case
  - Example: Technical documentation

• Research/Analysis (K=20-50)
  - High recall needed
  - Comprehensive coverage
  - Example: Literature review

• Limited LLM Context (K=1-5)
  - Token limits constrain K
  - Need high precision
  - Example: GPT-3.5 (4K context)

• Large LLM Context (K=10-50)
  - Can handle more docs
  - Optimize for recall
  - Example: GPT-4 (32K context)

📝 Next Steps:

1. Review the analysis above
2. Choose K based on your use case
3. Test with your actual data
4. Monitor metrics in production
5. Adjust K based on user feedback
""")
    
    print("✅ K-Value optimization complete!\n")


if __name__ == "__main__":
    # Run async main
    asyncio.run(main())
