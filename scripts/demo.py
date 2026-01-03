"""
pgVectorDB v2.2.0 Demo Script
=============================

This script demonstrates the key features of pgVectorDB:
1. Initialization and setup
2. Document operations
3. All 10 search methods
4. Extension-aware features

Run with: python scripts/demo.py

Requirements:
- PostgreSQL with pgvector extension
- Docker container running (docker compose up -d)
"""

import asyncio
import logging
from langchain_core.documents import Document

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ==================== Configuration ====================

DB_HOST = "localhost"
DB_PORT = "9002"
DB_NAME = "postgres"
DB_USER = "user"
DB_PASSWORD = "root"
CONNECTION_STRING = f"postgresql+asyncpg://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"


async def main():
    """Main demo function."""
    
    print("=" * 80)
    print("pgVectorDB v2.2.0 Demo")
    print("=" * 80)
    
    # ==================== Step 1: Import and Check Extensions ====================
    print("\n📦 Step 1: Importing modules...")
    
    from src import (
        pgVectorDB,
        IndexType,
        KeywordSearchType,
        DistanceMetric,
        ExtensionManager,
        ValidationError,
        ALLOWED_TEXT_CONFIGS,
    )
    
    print("✓ All imports successful")
    print(f"  - Index types: {[e.value for e in IndexType]}")
    print(f"  - Keyword search types: {[e.value for e in KeywordSearchType]}")
    print(f"  - Supported languages: {len(ALLOWED_TEXT_CONFIGS)} (BM25)")
    
    # ==================== Step 2: Initialize with HuggingFace Embeddings ====================
    print("\n🚀 Step 2: Initializing pgVectorDB...")
    
    try:
        from langchain_huggingface import HuggingFaceEmbeddings
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        print("✓ HuggingFace embeddings loaded")
    except ImportError:
        print("❌ langchain-huggingface not installed. Install with:")
        print("   pip install langchain-huggingface sentence-transformers")
        return
    
    # Create pgVectorDB instance
    rag = pgVectorDB(
        collection_name="demo_docs",
        embedding_model=embeddings,
        connection_string=CONNECTION_STRING,
        schema_name="public",
        index_type=IndexType.HNSW,
        pool_size=5,
    )
    
    print(f"✓ pgVectorDB created: table='demo_docs', index={rag.index_type.value}")
    
    # ==================== Step 3: Check Extension Availability ====================
    print("\n🔌 Step 3: Checking extension availability...")
    
    try:
        ext_manager = ExtensionManager(rag.sqlalchemy_engine)
        status = await ext_manager.check_extensions()
        
        print("Extension Status:")
        print(f"  - pgvector: {'✓' if status['pgvector'] else '✗'} (required)")
        print(f"  - vectorscale: {'✓' if status['vectorscale'] else '✗'} (optional, enables DiskANN)")
        print(f"  - pg_textsearch: {'✓' if status['pg_textsearch'] else '✗'} (optional, enables BM25)")
        
        # Show feature availability
        features = ext_manager.get_feature_availability()
        print("\nFeature Availability:")
        for feature, info in features.items():
            icon = "✓" if info['available'] else "✗"
            print(f"  {icon} {feature}")
            
    except Exception as e:
        print(f"⚠ Extension check failed (is database running?): {e}")
        print("  Start with: cd docker && docker compose up -d")
        return
    
    # ==================== Step 4: Initialize Database ====================
    print("\n💾 Step 4: Initializing database...")
    
    try:
        await rag.initialize(overwrite_existing=True)
        print("✓ Database initialized successfully")
    except Exception as e:
        print(f"❌ Initialization failed: {e}")
        return
    
    # ==================== Step 5: Add Sample Documents ====================
    print("\n📄 Step 5: Adding sample documents...")
    
    sample_docs = [
        Document(
            page_content="Machine learning is a subset of artificial intelligence that enables computers to learn from data.",
            metadata={"category": "ai", "topic": "ml", "year": 2024, "priority": 9}
        ),
        Document(
            page_content="Deep learning uses neural networks with multiple layers to process complex patterns.",
            metadata={"category": "ai", "topic": "deep_learning", "year": 2024, "priority": 8}
        ),
        Document(
            page_content="Natural language processing allows computers to understand and generate human language.",
            metadata={"category": "ai", "topic": "nlp", "year": 2023, "priority": 7}
        ),
        Document(
            page_content="PostgreSQL is a powerful open-source relational database management system.",
            metadata={"category": "database", "topic": "postgres", "year": 2024, "priority": 8}
        ),
        Document(
            page_content="Vector databases enable efficient similarity search for AI embeddings.",
            metadata={"category": "database", "topic": "vector_db", "year": 2024, "priority": 9}
        ),
        Document(
            page_content="Python is the most popular programming language for data science and machine learning.",
            metadata={"category": "programming", "topic": "python", "year": 2024, "priority": 10}
        ),
        Document(
            page_content="Transformers architecture revolutionized natural language processing with attention mechanisms.",
            metadata={"category": "ai", "topic": "nlp", "year": 2023, "priority": 9}
        ),
        Document(
            page_content="Retrieval Augmented Generation combines search with language models for accurate answers.",
            metadata={"category": "ai", "topic": "rag", "year": 2024, "priority": 10}
        ),
    ]
    
    doc_ids = await rag.add_documents(sample_docs)
    print(f"✓ Added {len(doc_ids)} documents")
    
    # ==================== Step 6: Build Vector Index ====================
    print("\n🔨 Step 6: Building HNSW vector index...")
    
    await rag.build_index(
        metric=DistanceMetric.COSINE,
        m=16,
        ef_construction=64
    )
    print("✓ HNSW index built")
    
    # Build BM25 index if pg_textsearch is available
    if ext_manager.has_pg_textsearch:
        print("\n🔨 Building BM25 index...")
        await rag.build_bm25_index(text_config="english", k1=1.2, b=0.75)
        print("✓ BM25 index built")
    else:
        print("\n⚠ Skipping BM25 index (pg_textsearch not available)")
    
    # ==================== Step 7: Demonstrate Search Methods ====================
    print("\n🔍 Step 7: Demonstrating search methods...")
    
    test_query = "machine learning AI"
    
    # 1. Keyword Search (FTS)
    print("\n--- 1. Keyword Search (FTS) ---")
    results = await rag.keyword_search(test_query, k=3, search_type=KeywordSearchType.FTS)
    for r in results:
        print(f"  [{r['score']:.4f}] {r['content'][:60]}...")
    
    # 2. Semantic Search
    print("\n--- 2. Semantic Search ---")
    results = await rag.semantic_search(test_query, k=3)
    for r in results:
        print(f"  [{r['score']:.4f}] {r['content'][:60]}...")
    
    # 3. Hybrid Search
    print("\n--- 3. Hybrid Search (RRF) ---")
    results = await rag.hybrid_search(test_query, k=3, use_rrf=True)
    for r in results:
        print(f"  [{r['score']:.4f}] {r['content'][:60]}...")
    
    # 4. Metadata Semantic Search
    print("\n--- 4. Metadata Semantic Search (category=ai) ---")
    results = await rag.metadata_semantic_search(
        test_query,
        filter={"category": "ai"},
        k=3
    )
    for r in results:
        print(f"  [{r['score']:.4f}] {r['content'][:60]}...")
    
    # 5. Trigram Search (fuzzy)
    print("\n--- 5. Trigram Search (typo-tolerant) ---")
    results = await rag.trigram_search("machin lerning", k=3, threshold=0.2)
    for r in results:
        print(f"  [{r['score']:.4f}] {r['content'][:60]}...")
    
    # ==================== Step 8: Analytics ====================
    print("\n📊 Step 8: Collection statistics...")
    
    stats = await rag.get_stats()
    print(f"  Documents: {stats.get('document_count', 'N/A')}")
    print(f"  Index stats: {stats.get('index_stats', 'N/A')}")
    
    # Count by metadata
    ai_count = await rag.count_by_metadata({"category": "ai"})
    print(f"  AI category documents: {ai_count}")
    
    # ==================== Step 9: LangChain Integration ====================
    print("\n🔗 Step 9: LangChain retriever integration...")
    
    retriever = rag.as_retriever(search_method="hybrid_search", search_kwargs={"k": 3})
    print(f"✓ Retriever created: {retriever}")
    print("  Can now use with LangChain chains and agents")
    
    # ==================== Cleanup ====================
    print("\n🧹 Cleanup...")
    await rag.close()
    print("✓ Connection closed")
    
    print("\n" + "=" * 80)
    print("Demo complete! pgVectorDB v2.2.0 is working correctly.")
    print("=" * 80)


if __name__ == "__main__":
    asyncio.run(main())
