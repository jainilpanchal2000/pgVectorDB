"""
pgVectorDB v0.0.6 Demo - Fluent API
===================================

This script demonstrates the new LanceDB-style fluent API:
- Query builder pattern with method chaining
- Explain/Analyze query plans with real PostgreSQL EXPLAIN ANALYZE
- Advanced query parameters (ef, nprobes, refine_factor)
- Scalar indexes for metadata filtering
- NEW: Metadata-only search (metadata_only)
- NEW: Ensemble search (ensemble)
- NEW: Label filtering (labels) for DiskANN

Run with: python examples/fluent_api_demo.py
"""

import asyncio
import logging

from langchain_core.documents import Document

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Configuration
DB_HOST = "localhost"
DB_PORT = "5433"  # Test container port
DB_NAME = "testdb"
DB_USER = "testuser"
DB_PASSWORD = "testpass"
CONNECTION_STRING = f"postgresql+asyncpg://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"


async def main():
    """Main demo function."""
    print("=" * 80)
    print("pgVectorDB v0.0.6 - Fluent API Demo")
    print("Search methods now use: search().where().limit().to_list()")
    print("=" * 80)

    # Import pgVectorDB
    print("\n📦 Importing pgVectorDB...")
    from pgvectordb import DistanceMetric, IndexType, pgVectorDB

    # Initialize embeddings
    try:
        from langchain_huggingface import HuggingFaceEmbeddings

        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        print("✓ HuggingFace embeddings loaded")
    except ImportError:
        print("❌ Install langchain-huggingface: pip install langchain-huggingface")
        return

    # Create database instance
    db = pgVectorDB(
        collection_name="fluent_demo",
        embedding_model=embeddings,
        connection_string=CONNECTION_STRING,
        schema_name="public",
        index_type=IndexType.HNSW,
    )
    print("✓ pgVectorDB created")

    # Initialize database
    print("\n💾 Initializing database...")
    await db.initialize(overwrite_existing=True)
    print("✓ Database initialized")

    # Add sample documents
    print("\n📄 Adding sample documents...")
    sample_docs = [
        Document(
            page_content="Machine learning enables computers to learn from data.",
            metadata={"category": "ai", "topic": "ml", "year": 2024, "priority": 9},
        ),
        Document(
            page_content="Deep learning uses neural networks with multiple layers.",
            metadata={"category": "ai", "topic": "deep_learning", "year": 2024, "priority": 8},
        ),
        Document(
            page_content="Natural language processing allows computers to understand text.",
            metadata={"category": "ai", "topic": "nlp", "year": 2023, "priority": 7},
        ),
        Document(
            page_content="PostgreSQL is a powerful open-source database system.",
            metadata={"category": "database", "topic": "postgres", "year": 2024, "priority": 8},
        ),
        Document(
            page_content="Vector databases enable efficient similarity search.",
            metadata={"category": "database", "topic": "vector_db", "year": 2024, "priority": 9},
        ),
        Document(
            page_content="Python is popular for data science and machine learning.",
            metadata={"category": "programming", "topic": "python", "year": 2024, "priority": 10},
        ),
        Document(
            page_content="Transformers revolutionized NLP with attention mechanisms.",
            metadata={"category": "ai", "topic": "transformers", "year": 2024, "priority": 10},
        ),
        Document(
            page_content="RAG combines search with language models for accurate answers.",
            metadata={"category": "ai", "topic": "pgvdb", "year": 2024, "priority": 9},
        ),
    ]

    doc_ids = await db.add_documents(sample_docs)
    print(f"✓ Added {len(doc_ids)} documents")

    # Build indexes
    print("\n🔨 Building indexes...")
    await db.build_index(metric=DistanceMetric.COSINE, m=16, ef_construction=64)
    print("✓ HNSW index built")

    # Create scalar indexes for faster filtering
    await db.create_scalar_index("category", index_type="bitmap")
    await db.create_scalar_index("year", index_type="btree")
    print("✓ Scalar indexes created (category: bitmap, year: btree)")

    # ============================================================
    # DEMO 1: Basic Semantic Search
    # ============================================================
    print("\n" + "=" * 60)
    print("DEMO 1: Basic Semantic Search")
    print("Code: db.query('machine learning').semantic().limit(3).to_list()")
    print("=" * 60)

    test_query = "machine learning AI"

    results = await db.query(test_query).semantic().limit(3).to_list()
    for i, r in enumerate(results, 1):
        print(f"  {i}. [{r['score']:.4f}] {r['content'][:60]}...")

    # ============================================================
    # DEMO 2: Filtered Search
    # ============================================================
    print("\n" + "=" * 60)
    print("DEMO 2: Filtered Search (category='ai')")
    print(
        "Code: db.query('machine learning').semantic().where({'category': 'ai'}).limit(3).to_list()"
    )
    print("=" * 60)

    results = await db.query(test_query).semantic().where({"category": "ai"}).limit(3).to_list()
    for i, r in enumerate(results, 1):
        print(f"  {i}. [{r['score']:.4f}] {r['content'][:60]}...")
        print(f"      Metadata: {r['metadata']}")

    # ============================================================
    # DEMO 3: Complex Filter
    # ============================================================
    print("\n" + "=" * 60)
    print("DEMO 3: Complex Filter (AI + year=2024 + priority >= 8)")
    print("=" * 60)

    results = await (
        db.query("AI technology")
        .semantic()
        .where({"$and": [{"category": "ai"}, {"year": 2024}, {"priority": {"$gte": 8}}]})
        .limit(5)
        .to_list()
    )
    for i, r in enumerate(results, 1):
        print(f"  {i}. [{r['score']:.4f}] {r['content'][:60]}...")
        print(f"      Metadata: {r['metadata']}")

    # ============================================================
    # DEMO 4: Explain Query Plan
    # ============================================================
    print("\n" + "=" * 60)
    print("DEMO 4: Explain Query Plan")
    print("Code: db.query('query').semantic().where({'category': 'ai'}).explain_plan()")
    print("=" * 60)

    plan = db.query("machine learning").semantic().where({"category": "ai"}).explain_plan()
    print(f"  Search Method: {plan.get('search_method', 'N/A')}")
    print(f"  Filter: {plan.get('filter', 'N/A')}")
    print(f"  Limit: {plan.get('limit', 'N/A')}")
    print(f"  Index Type: {plan.get('index_type', 'N/A')}")

    # ============================================================
    # DEMO 5: Analyze Query Plan (with timings)
    # ============================================================
    print("\n" + "=" * 60)
    print("DEMO 5: Analyze Query Plan (with execution metrics)")
    print("Code: await db.query('query').semantic().analyze_plan()")
    print("=" * 60)

    metrics = await db.query("database systems").semantic().analyze_plan()
    print(f"  Execution Time: {metrics.get('execution_time_ms', 'N/A')} ms")
    print(f"  Rows Returned: {metrics.get('rows_returned', 'N/A')}")
    print(f"  Search Method: {metrics.get('search_method', 'N/A')}")

    # ============================================================
    # DEMO 6: Advanced Query Parameters
    # ============================================================
    print("\n" + "=" * 60)
    print("DEMO 6: Advanced Query Parameters (ef=100 for better recall)")
    print("Code: db.query('query').semantic().ef(100).limit(3).to_list()")
    print("=" * 60)

    results = await db.query("neural networks").semantic().ef(100).limit(3).to_list()
    for i, r in enumerate(results, 1):
        print(f"  {i}. [{r['score']:.4f}] {r['content'][:60]}...")

    # ============================================================
    # DEMO 7: Bypass Vector Index (Exact Search)
    # ============================================================
    print("\n" + "=" * 60)
    print("DEMO 7: Exact Search (bypass_vector_index)")
    print("Code: db.query('query').semantic().bypass_vector_index().limit(3).to_list()")
    print("=" * 60)

    exact_results = (
        await db.query("machine learning").semantic().bypass_vector_index().limit(3).to_list()
    )
    ann_results = await db.query("machine learning").semantic().limit(3).to_list()

    print("  Exact search results:")
    for i, r in enumerate(exact_results, 1):
        print(f"    {i}. [{r['score']:.4f}]")

    print("  ANN search results:")
    for i, r in enumerate(ann_results, 1):
        print(f"    {i}. [{r['score']:.4f}]")

    # Calculate simple recall
    exact_ids = {r["id"] for r in exact_results}
    ann_ids = {r["id"] for r in ann_results}
    recall = len(exact_ids & ann_ids) / len(exact_ids) if exact_ids else 0
    print(f"  Recall@3: {recall:.0%}")

    # ============================================================
    # DEMO 8: Output Formats
    # ============================================================
    print("\n" + "=" * 60)
    print("DEMO 8: Different Output Formats")
    print("=" * 60)

    # to_list() - default
    list_results = await db.query("python").semantic().limit(2).to_list()
    print(f"  to_list() type: {type(list_results).__name__}")
    print(f"  to_list() sample: {list_results[0]['content'][:40]}...")

    # to_pandas()
    df = await db.query("python").semantic().limit(2).to_pandas()
    print(f"  to_pandas() type: {type(df).__name__}")
    print(f"  to_pandas() columns: {list(df.columns)}")

    # to_arrow()
    table = await db.query("python").semantic().limit(2).to_arrow()
    print(f"  to_arrow() type: {type(table).__name__}")
    print(f"  to_arrow() rows: {len(table)}")

    # ============================================================
    # DEMO 9: Hybrid Search
    # ============================================================
    print("\n" + "=" * 60)
    print("DEMO 9: Hybrid Search (Vector + Text)")
    print(
        "Code: db.query('query').hybrid().fts().weights(semantic=0.6, keyword=0.4).limit(3).to_list()"
    )
    print("=" * 60)

    hybrid_results = await (
        db.query("neural networks")
        .hybrid()
        .fts()
        .weights(semantic=0.6, keyword=0.4)
        .limit(3)
        .to_list()
    )
    print("  Hybrid search combines semantic and keyword ranking")
    for i, r in enumerate(hybrid_results, 1):
        print(f"  {i}. [{r['score']:.4f}] {r['content'][:60]}...")

    # ============================================================
    # DEMO 10: NEW - Metadata-Only Search
    # ============================================================
    print("\n" + "=" * 60)
    print("DEMO 10: Metadata-Only Search (NEW in v0.0.6)")
    print("Code: db.query('').metadata_only().where({'category': 'ai'}).limit(5).to_list()")
    print("=" * 60)
    print("  Search documents by metadata without text query")

    try:
        meta_results = await (
            db.query("").metadata_only().where({"category": "ai"}).limit(5).to_list()
        )
        for i, r in enumerate(meta_results, 1):
            print(f"  {i}. {r['content'][:60]}...")
            print(f"      Metadata: {r['metadata']}")
    except Exception as e:
        print(f"  Note: metadata_only() requires pgVectorDB v0.0.6+ - {e}")

    # ============================================================
    # DEMO 11: NEW - Ensemble Search
    # ============================================================
    print("\n" + "=" * 60)
    print("DEMO 11: Ensemble Search (NEW in v0.0.6)")
    print("Code: db.query('query').ensemble().where({'category': 'ai'}).limit(5).to_list()")
    print("=" * 60)
    print("  Hybrid search on filtered subset - convenient for RAG applications")

    try:
        ensemble_results = await (
            db.query("machine learning")
            .ensemble()
            .where({"category": "ai"})
            .weights(semantic=0.7, keyword=0.3)
            .limit(5)
            .to_list()
        )
        for i, r in enumerate(ensemble_results, 1):
            print(f"  {i}. [{r['score']:.4f}] {r['content'][:60]}...")
    except Exception as e:
        print(f"  Note: ensemble() requires pgVectorDB v0.0.6+ - {e}")

    # ============================================================
    # DEMO 12: Statistics
    # ============================================================
    print("\n" + "=" * 60)
    print("DEMO 12: Collection Statistics")
    print("=" * 60)

    stats = await db.get_stats()
    print(f"  Documents: {stats.get('document_count', 'N/A')}")
    print(f"  Index Type: {stats.get('index_type', 'N/A')}")
    print(f"  Index Built: {stats.get('index_built', 'N/A')}")
    print(f"  Table Size: {stats.get('table_size', 'N/A')}")

    # ============================================================
    # Summary
    # ============================================================
    print("\n" + "=" * 60)
    print("Demo Complete!")
    print("=" * 60)
    print("\nKey takeaways:")
    print("  1. Use db.query() for semantic search with fluent API")
    print("  2. Use .where() for metadata filtering")
    print("  3. Use .explain_plan() to check query plans without executing")
    print("  4. Use .analyze_plan() for real PostgreSQL EXPLAIN ANALYZE")
    print("  5. Use .ef(), .nprobes() for tuning recall")
    print("  6. Use .bypass_vector_index() for exact search")
    print("  7. Use .metadata_only() for pure metadata filtering (v0.0.6+)")
    print("  8. Use .ensemble() for filtered hybrid search (v0.0.6+)")
    print("  9. Use .to_list(), .to_pandas(), .to_arrow() for outputs")

    # Cleanup
    await db.close()
    print("\n✓ Connection closed")


if __name__ == "__main__":
    asyncio.run(main())
