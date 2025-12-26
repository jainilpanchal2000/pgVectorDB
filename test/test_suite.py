"""
Complete Test Suite for Production RAG System
==============================================

Comprehensive testing of all pgVectorDB functionality:
- Initialization (3 index types: HNSW, IVFFlat, DiskANN)
- Document operations (CRUD, batch, metadata)
- All 11 search methods
- All 13 filter operators
- Index operations and optimization
- Analytics, validation, monitoring
- Data export/import
- LangChain integration
- Error handling
- Performance benchmarks

Usage:
    python test/test_suite.py
"""

import sys
import os
from pathlib import Path

# Add parent directory to path
test_dir = Path(__file__).parent
parent_dir = test_dir.parent
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

import asyncio
import time
import logging
from typing import List
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from src.core import (
    pgVectorDB,
    IndexType,
    DistanceMetric,
    StorageLayout,
    ValidationError,
    InitializationError
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Database configuration
DB_HOST = "localhost"
DB_PORT = "9002"
DB_NAME = "postgres"
DB_USER = "user"
DB_PASSWORD = "root"
SCHEMA_NAME = "test"

CONNECTION_STRING = f"postgresql+asyncpg://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"


# ==================== Test Tracking ====================
class TestResults:
    """Track test results."""
    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.errors = []
    
    def add_pass(self, test_name: str):
        self.passed += 1
        print(f"✅ [PASS] {test_name}")
    
    def add_fail(self, test_name: str, error: str):
        self.failed += 1
        error_msg = f"❌ [FAIL] {test_name} | {error}"
        self.errors.append(error_msg)
        print(error_msg)
    
    def print_summary(self):
        total = self.passed + self.failed
        success_rate = (self.passed / total * 100) if total > 0 else 0
        
        print("\n" + "=" * 80)
        print("TEST SUMMARY")
        print("=" * 80)
        print(f"✅ PASSED: {self.passed}")
        print(f"❌ FAILED: {self.failed}")
        print(f"📊 TOTAL:  {total}")
        print(f"📈 SUCCESS RATE: {success_rate:.1f}%")
        print("=" * 80)
        
        if self.failed > 0:
            print("\n⚠️  FAILED TESTS:")
            for error in self.errors:
                print(f"  {error}")
            print()

results = TestResults()


# ==================== Test Data Generation ====================
def generate_test_documents(num_docs: int = 100) -> tuple[List[Document], List[List[int]]]:
    """Generate diverse test documents with metadata and labels."""
    
    categories = ["programming", "ai", "database", "web", "devops", "security", "cloud", "mobile"]
    languages = ["Python", "JavaScript", "Java", "Go", "Rust", "SQL", "TypeScript", "C++"]
    authors = ["Tech Expert", "AI Researcher", "DB Admin", "DevOps Engineer", "Security Analyst"]
    
    content_templates = [
        "{language} is a {adjective} programming language used for {purpose}.",
        "Machine learning with {language} enables {capability} in production systems.",
        "Database optimization using {tool} improves {metric} by significant margins.",
        "Cloud-native applications built with {framework} provide {benefit} for enterprises.",
        "Security best practices include {practice} to prevent {threat} attacks.",
        "Web development frameworks like {framework} simplify {purpose} for developers.",
        "Mobile applications using {platform} deliver {feature} to end users.",
        "DevOps automation with {tool} reduces {problem} and increases reliability.",
        "API design patterns such as {pattern} enhance {quality} in microservices.",
        "Data engineering pipelines processing {data_type} enable {outcome} for analytics."
    ]
    
    adjectives = ["powerful", "versatile", "modern", "efficient", "scalable", "robust"]
    purposes = ["web development", "data science", "system programming", "automation"]
    capabilities = ["pattern recognition", "predictive analytics", "natural language processing"]
    tools = ["indexes", "partitioning", "caching", "replication"]
    metrics = ["query performance", "throughput", "response time"]
    frameworks = ["React", "FastAPI", "Django", "Express", "Spring Boot"]
    benefits = ["scalability", "reliability", "flexibility", "performance"]
    practices = ["input validation", "encryption", "authentication", "authorization"]
    threats = ["SQL injection", "XSS", "CSRF", "DDoS"]
    platforms = ["React Native", "Flutter", "Swift", "Kotlin"]
    features = ["offline support", "push notifications", "real-time updates"]
    problems = ["deployment time", "manual errors", "configuration drift"]
    patterns = ["REST", "GraphQL", "gRPC", "WebSocket"]
    qualities = ["maintainability", "testability", "observability"]
    data_types = ["streaming data", "batch data", "real-time events"]
    outcomes = ["business insights", "data-driven decisions", "predictive models"]
    
    documents = []
    labels = []
    
    for i in range(num_docs):
        category = categories[i % len(categories)]
        language = languages[i % len(languages)]
        author = authors[i % len(authors)]
        year = 2020 + (i % 5)
        
        # Generate content
        template = content_templates[i % len(content_templates)]
        content = template.format(
            language=language,
            adjective=adjectives[i % len(adjectives)],
            purpose=purposes[i % len(purposes)],
            capability=capabilities[i % len(capabilities)],
            tool=tools[i % len(tools)],
            metric=metrics[i % len(metrics)],
            framework=frameworks[i % len(frameworks)],
            benefit=benefits[i % len(benefits)],
            practice=practices[i % len(practices)],
            threat=threats[i % len(threats)],
            platform=platforms[i % len(platforms)],
            feature=features[i % len(features)],
            problem=problems[i % len(problems)],
            pattern=patterns[i % len(patterns)],
            quality=qualities[i % len(qualities)],
            data_type=data_types[i % len(data_types)],
            outcome=outcomes[i % len(outcomes)]
        )
        
        # Create document
        doc = Document(
            page_content=content.strip(),
            metadata={
                "doc_id": i,
                "category": category,
                "language": language,
                "author": author,
                "year": year,
                "priority": (i % 10) + 1,
                "status": "active" if i % 4 != 0 else "archived",
                "tags": f"tag{i % 5}"
            }
        )
        documents.append(doc)
        
        # Assign labels for DiskANN
        category_to_label = {
            "programming": 1, "ai": 2, "database": 3, "web": 4,
            "devops": 5, "security": 6, "cloud": 7, "mobile": 8
        }
        label = category_to_label.get(category, 1)
        labels.append([label])
    
    return documents, labels


# ==================== Schema Management ====================
async def create_test_schema():
    """Create the test schema before running tests."""
    from sqlalchemy.ext.asyncio import create_async_engine
    from sqlalchemy import text
    
    engine = create_async_engine(CONNECTION_STRING, pool_pre_ping=True)
    
    try:
        async with engine.connect() as conn:
            await conn.execute(text(f"CREATE SCHEMA IF NOT EXISTS {SCHEMA_NAME}"))
            await conn.commit()
            logger.info(f"Created test schema: '{SCHEMA_NAME}'")
    except Exception as e:
        logger.error(f"Schema creation error: {e}")
        raise
    finally:
        await engine.dispose()


async def cleanup_test_schema():
    """Drop the test schema after testing."""
    from sqlalchemy.ext.asyncio import create_async_engine
    from sqlalchemy import text
    
    engine = create_async_engine(CONNECTION_STRING, pool_pre_ping=True)
    
    try:
        async with engine.connect() as conn:
            await conn.execute(text(f"DROP SCHEMA IF EXISTS {SCHEMA_NAME} CASCADE"))
            await conn.commit()
            logger.info(f"Dropped test schema: '{SCHEMA_NAME}'")
    except Exception as e:
        logger.error(f"Schema cleanup error: {e}")
    finally:
        await engine.dispose()


# ==================== Test Suites ====================

async def test_initialization(embeddings):
    """Test initialization with all 3 index types."""
    print("\n" + "=" * 80)
    print("📦 INITIALIZATION TESTS")
    print("=" * 80)
    
    # Test HNSW
    try:
        rag = pgVectorDB(
            collection_name="test_hnsw",
            embedding_model=embeddings,
            connection_string=CONNECTION_STRING,
            schema_name=SCHEMA_NAME,
            index_type=IndexType.HNSW
        )
        await rag.initialize(overwrite_existing=True)
        stats = await rag.get_stats()
        assert stats['index_type'] == 'hnsw'
        await rag.close()
        results.add_pass("HNSW Initialization")
    except Exception as e:
        results.add_fail("HNSW Initialization", str(e))
    
    # Test IVFFlat
    try:
        rag = pgVectorDB(
            collection_name="test_ivfflat",
            embedding_model=embeddings,
            connection_string=CONNECTION_STRING,
            schema_name=SCHEMA_NAME,
            index_type=IndexType.IVFFLAT
        )
        await rag.initialize(overwrite_existing=True)
        stats = await rag.get_stats()
        assert stats['index_type'] == 'ivfflat'
        await rag.close()
        results.add_pass("IVFFlat Initialization")
    except Exception as e:
        results.add_fail("IVFFlat Initialization", str(e))
    
    # Test DiskANN
    try:
        rag = pgVectorDB(
            collection_name="test_diskann",
            embedding_model=embeddings,
            connection_string=CONNECTION_STRING,
            schema_name=SCHEMA_NAME,
            index_type=IndexType.DISKANN
        )
        await rag.initialize(overwrite_existing=True)
        stats = await rag.get_stats()
        assert stats['index_type'] == 'diskann'
        await rag.close()
        results.add_pass("DiskANN Initialization")
    except Exception as e:
        results.add_fail("DiskANN Initialization", str(e))


async def test_document_operations(embeddings):
    """Test document CRUD operations."""
    print("\n" + "=" * 80)
    print("📄 DOCUMENT OPERATIONS TESTS")
    print("=" * 80)
    
    docs, _ = generate_test_documents(100)
    
    try:
        rag = pgVectorDB(
            collection_name="test_docs",
            embedding_model=embeddings,
            connection_string=CONNECTION_STRING,
            schema_name=SCHEMA_NAME,
            index_type=IndexType.HNSW
        )
        await rag.initialize(overwrite_existing=True)
        
        # Add documents
        doc_ids = await rag.add_documents(docs[:50])
        assert len(doc_ids) == 50, f"Expected 50 doc IDs, got {len(doc_ids)}"
        results.add_pass("Add Documents")
        
        # Batch add
        batch_ids = await rag.add_documents_batch(docs[50:], batch_size=10, show_progress=False)
        assert len(batch_ids) == 50, f"Expected 50 batch IDs, got {len(batch_ids)}"
        results.add_pass("Batch Add Documents")
        
        # Count by metadata
        total = await rag.count_by_metadata(None)
        assert total == 100, f"Expected 100 docs, got {total}"
        results.add_pass("Count Documents")
        
        active_count = await rag.count_by_metadata({"status": "active"})
        assert active_count > 0, "Should have active documents"
        results.add_pass("Count by Metadata Filter")
        
        # Get by IDs
        retrieved_docs = await rag.aget_by_ids(doc_ids[:5])
        assert len(retrieved_docs) > 0, "Should retrieve documents by IDs"
        results.add_pass("Get Documents by IDs")
        
        # Update documents
        update_doc = Document(
            page_content=docs[0].page_content,
            metadata={"langchain_id": doc_ids[0], "updated": True, "status": "reviewed"}
        )
        updated_ids = await rag.aupdate_documents([update_doc], update_embeddings=False)
        assert len(updated_ids) == 1, "Should update 1 document"
        results.add_pass("Update Documents")
        
        # Bulk metadata update
        update_count = await rag.update_metadata(
            ids=doc_ids[:10],
            metadata_updates={"tagged": True, "batch": 1}
        )
        assert update_count == 10, f"Expected 10 updates, got {update_count}"
        results.add_pass("Bulk Metadata Update")
        
        # Delete documents
        delete_count = await rag.adelete(doc_ids[-5:])
        assert delete_count == 5, f"Expected 5 deletes, got {delete_count}"
        results.add_pass("Delete Documents")
        
        await rag.close()
        
    except Exception as e:
        results.add_fail("Document Operations", str(e))


async def test_all_search_methods(embeddings):
    """Test all 11 search methods."""
    print("\n" + "=" * 80)
    print("🔍 SEARCH METHODS TESTS (11 methods)")
    print("=" * 80)
    
    docs, _ = generate_test_documents(100)
    
    try:
        rag = pgVectorDB(
            collection_name="test_search",
            embedding_model=embeddings,
            connection_string=CONNECTION_STRING,
            schema_name=SCHEMA_NAME,
            index_type=IndexType.HNSW
        )
        await rag.initialize(overwrite_existing=True)
        await rag.add_documents(docs)
        await rag.create_metadata_index(["category", "language", "year"])
        await rag.build_index(metric=DistanceMetric.COSINE)
        
        # Test each search method
        search_tests = [
            ("1. Semantic Search", lambda: rag.semantic_search("programming Python", k=5)),
            ("2. Keyword Search", lambda: rag.keyword_search("database", k=5)),
            ("3. Universal Keyword", lambda: rag.universal_keyword_search("Python", k=5, metadata_fields=["category", "language"])),
            ("4. Metadata Filter", lambda: rag.metadata_filter({"category": "ai"}, k=10)),
            ("5. Metadata + Keyword", lambda: rag.metadata_keyword_search("dev", {"category": "programming"}, k=5)),
            ("6. Metadata + Semantic", lambda: rag.metadata_semantic_search("Python", {"year": 2023}, k=5)),
            ("7. Hybrid Search", lambda: rag.hybrid_search("programming", k=5, weights=(0.6, 0.4))),
            ("8. Hybrid (RRF)", lambda: rag.hybrid_search("database", k=5, use_rrf=True)),
            ("9. Ensemble Search", lambda: rag.ensemble_search("Python", {"category": "programming"}, k=5)),
            ("10. Trigram Search", lambda: rag.trigram_search("programing", k=5, threshold=0.3)),
            ("11. Metadata + Trigram", lambda: rag.metadata_trigram_search("dev", {"category": "programming"}, k=5)),
        ]
        
        for test_name, search_func in search_tests:
            try:
                res = await search_func()
                results.add_pass(test_name)
            except Exception as e:
                results.add_fail(test_name, str(e))
        
        # Additional similarity search variants
        try:
            embedding = embeddings.embed_query("Python")
            res = await rag.asimilarity_search_by_vector(embedding, k=5)
            results.add_pass("Similarity Search by Vector")
        except Exception as e:
            results.add_fail("Similarity Search by Vector", str(e))
        
        try:
            res = await rag.asimilarity_search_with_score("database", k=5)
            assert all(isinstance(r, tuple) and len(r) == 2 for r in res)
            results.add_pass("Similarity Search with Score")
        except Exception as e:
            results.add_fail("Similarity Search with Score", str(e))
        
        await rag.close()
        
    except Exception as e:
        results.add_fail("Search Methods", str(e))


async def test_filter_operators(embeddings):
    """Test all 13 filter operators."""
    print("\n" + "=" * 80)
    print("🔧 FILTER OPERATORS TESTS (13 operators)")
    print("=" * 80)
    
    docs, _ = generate_test_documents(100)
    
    try:
        rag = pgVectorDB(
            collection_name="test_filters",
            embedding_model=embeddings,
            connection_string=CONNECTION_STRING,
            schema_name=SCHEMA_NAME,
            index_type=IndexType.HNSW
        )
        await rag.initialize(overwrite_existing=True)
        await rag.add_documents(docs)
        await rag.create_metadata_index(["category", "language", "year", "priority", "status", "author"])
        await rag.build_index()
        
        operators = [
            ("$eq", {"category": {"$eq": "programming"}}, 
             lambda r: r['metadata'].get('category') == 'programming'),
            ("$ne", {"category": {"$ne": "web"}}, 
             lambda r: r['metadata'].get('category') != 'web'),
            ("$gt", {"priority": {"$gt": 5}}, 
             lambda r: r['metadata'].get('priority', 0) > 5),
            ("$gte", {"priority": {"$gte": 5}}, 
             lambda r: r['metadata'].get('priority', 0) >= 5),
            ("$lt", {"priority": {"$lt": 5}}, 
             lambda r: r['metadata'].get('priority', 0) < 5),
            ("$lte", {"priority": {"$lte": 5}}, 
             lambda r: r['metadata'].get('priority', 0) <= 5),
            ("$in", {"year": {"$in": [2023, 2024]}}, 
             lambda r: r['metadata'].get('year') in [2023, 2024]),
            ("$nin", {"status": {"$nin": ["archived"]}}, 
             lambda r: r['metadata'].get('status') != 'archived'),
            ("$between", {"priority": {"$between": [3, 7]}}, 
             lambda r: 3 <= r['metadata'].get('priority', 0) <= 7),
            ("$exists", {"category": {"$exists": True}}, 
             lambda r: 'category' in r['metadata']),
            ("$like", {"author": {"$like": "%Expert%"}}, 
             lambda r: 'Expert' in r['metadata'].get('author', '')),
            ("$and", {"$and": [{"category": "programming"}, {"year": {"$gte": 2022}}]}, 
             lambda r: r['metadata'].get('category') == 'programming' and r['metadata'].get('year', 0) >= 2022),
            ("$or", {"$or": [{"category": "ai"}, {"category": "database"}]}, 
             lambda r: r['metadata'].get('category') in ['ai', 'database'])
        ]
        
        for op_name, filter_dict, validator in operators:
            try:
                res = await rag.metadata_filter(filter=filter_dict, k=20)
                # Validate all results match the filter criteria
                for result in res:
                    assert validator(result), f"Result doesn't match filter {op_name}: {result['metadata']}"
                results.add_pass(f"Filter: {op_name}")
            except Exception as e:
                results.add_fail(f"Filter: {op_name}", str(e))
        
        await rag.close()
        
    except Exception as e:
        results.add_fail("Filter Operators", str(e))


async def test_index_operations(embeddings):
    """Test index building and operations."""
    print("\n" + "=" * 80)
    print("🔨 INDEX OPERATIONS TESTS")
    print("=" * 80)
    
    docs, labels = generate_test_documents(50)
    
    # Test HNSW index
    try:
        rag = pgVectorDB(
            collection_name="test_hnsw_idx",
            embedding_model=embeddings,
            connection_string=CONNECTION_STRING,
            schema_name=SCHEMA_NAME,
            index_type=IndexType.HNSW
        )
        await rag.initialize(overwrite_existing=True)
        await rag.add_documents(docs)
        await rag.build_index(m=16, ef_construction=64, metric=DistanceMetric.COSINE)
        
        stats = await rag.get_stats()
        assert stats['index_built']
        results.add_pass("Build HNSW Index")
        
        # Test reindex
        await rag.areindex()
        results.add_pass("Reindex")
        
        # Test vacuum analyze
        await rag.vacuum_analyze(full=False)
        results.add_pass("Vacuum Analyze")
        
        await rag.close()
    except Exception as e:
        results.add_fail("HNSW Index Operations", str(e))
    
    # Test IVFFlat index
    try:
        rag = pgVectorDB(
            collection_name="test_ivf_idx",
            embedding_model=embeddings,
            connection_string=CONNECTION_STRING,
            schema_name=SCHEMA_NAME,
            index_type=IndexType.IVFFLAT
        )
        await rag.initialize(overwrite_existing=True)
        await rag.add_documents(docs)
        await rag.build_index(lists=10)
        
        stats = await rag.get_stats()
        assert stats['index_built']
        results.add_pass("Build IVFFlat Index")
        
        await rag.close()
    except Exception as e:
        results.add_fail("IVFFlat Index Operations", str(e))
    
    # Test DiskANN index with labels
    try:
        rag = pgVectorDB(
            collection_name="test_diskann_idx",
            embedding_model=embeddings,
            connection_string=CONNECTION_STRING,
            schema_name=SCHEMA_NAME,
            index_type=IndexType.DISKANN
        )
        await rag.initialize(overwrite_existing=True)
        await rag.add_documents(docs, labels=labels)
        await rag.build_index(
            num_neighbors=50,
            search_list_size=100,
            storage_layout=StorageLayout.MEMORY_OPTIMIZED,
            include_labels=True
        )
        
        stats = await rag.get_stats()
        assert stats['index_built']
        results.add_pass("Build DiskANN Index with Labels")
        
        # Test label filtering
        res = await rag.semantic_search("programming", k=10, label_filter=[1, 2])
        results.add_pass("Label-Based Filtering")
        
        await rag.close()
    except Exception as e:
        results.add_fail("DiskANN Index Operations", str(e))


async def test_analytics_and_monitoring(embeddings):
    """Test analytics, stats, and monitoring."""
    print("\n" + "=" * 80)
    print("📊 ANALYTICS & MONITORING TESTS")
    print("=" * 80)
    
    docs, _ = generate_test_documents(50)
    
    try:
        rag = pgVectorDB(
            collection_name="test_analytics",
            embedding_model=embeddings,
            connection_string=CONNECTION_STRING,
            schema_name=SCHEMA_NAME,
            index_type=IndexType.HNSW
        )
        await rag.initialize(overwrite_existing=True)
        await rag.add_documents(docs)
        await rag.build_index()
        
        # Get stats
        try:
            stats = await rag.get_stats()
            assert "document_count" in stats
            assert "index_type" in stats
            results.add_pass("Get Stats")
        except Exception as e:
            results.add_fail("Get Stats", str(e))
        
        # Get index stats
        try:
            idx_stats = await rag.get_index_stats()
            assert "index_type" in idx_stats
            assert "indexes" in idx_stats
            results.add_pass("Get Index Stats")
        except Exception as e:
            results.add_fail("Get Index Stats", str(e))
        
        # Validate collection
        try:
            validation = await rag.validate_collection()
            assert "healthy" in validation
            assert "issues" in validation
            results.add_pass("Validate Collection")
        except Exception as e:
            results.add_fail("Validate Collection", str(e))
        
        # Explain query
        try:
            plan = await rag.explain_query("test", "semantic_search", k=5)
            assert len(plan) > 0
            results.add_pass("Explain Query")
        except Exception as e:
            results.add_fail("Explain Query", str(e))
        
        # Benchmark
        try:
            bench = await rag.benchmark_search_methods(["test1", "test2"], k=5)
            assert len(bench) > 0
            results.add_pass("Benchmark Search Methods")
        except Exception as e:
            results.add_fail("Benchmark Search Methods", str(e))
        
        await rag.close()
        
    except Exception as e:
        results.add_fail("Analytics", str(e))


async def test_data_export_import(embeddings):
    """Test data export and import."""
    print("\n" + "=" * 80)
    print("💾 DATA EXPORT/IMPORT TESTS")
    print("=" * 80)
    
    docs, _ = generate_test_documents(30)
    export_file = "test_export.json"
    
    try:
        rag = pgVectorDB(
            collection_name="test_export",
            embedding_model=embeddings,
            connection_string=CONNECTION_STRING,
            schema_name=SCHEMA_NAME,
            index_type=IndexType.HNSW
        )
        await rag.initialize(overwrite_existing=True)
        await rag.add_documents(docs)
        
        # Test export
        try:
            count = await rag.export_to_json(
                export_file,
                filter={"status": "active"},
                include_embeddings=False
            )
            assert count > 0
            assert Path(export_file).exists()
            results.add_pass("Export to JSON")
        except Exception as e:
            results.add_fail("Export to JSON", str(e))
        
        # Test import
        try:
            count = await rag.import_from_json(
                export_file,
                batch_size=10,
                skip_existing=True
            )
            results.add_pass("Import from JSON")
        except Exception as e:
            results.add_fail("Import from JSON", str(e))
        
        await rag.close()
        
        # Cleanup
        if Path(export_file).exists():
            Path(export_file).unlink()
        
    except Exception as e:
        results.add_fail("Data Export/Import", str(e))


async def test_langchain_integration(embeddings):
    """Test LangChain retriever integration."""
    print("\n" + "=" * 80)
    print("🔗 LANGCHAIN INTEGRATION TESTS")
    print("=" * 80)
    
    docs, _ = generate_test_documents(30)
    
    try:
        rag = pgVectorDB(
            collection_name="test_langchain",
            embedding_model=embeddings,
            connection_string=CONNECTION_STRING,
            schema_name=SCHEMA_NAME,
            index_type=IndexType.HNSW
        )
        await rag.initialize(overwrite_existing=True)
        await rag.add_documents(docs)
        await rag.build_index()
        
        # Test as_retriever
        retriever = rag.as_retriever(search_method="semantic_search", search_kwargs={"k": 5})
        assert retriever is not None
        
        # Test async retrieval
        retrieved_docs = await retriever.ainvoke("Python programming")
        assert isinstance(retrieved_docs, list)
        results.add_pass("LangChain Retriever")
        
        await rag.close()
        
    except Exception as e:
        results.add_fail("LangChain Integration", str(e))


async def test_error_handling(embeddings):
    """Test error handling."""
    print("\n" + "=" * 80)
    print("⚠️  ERROR HANDLING TESTS")
    print("=" * 80)
    
    # ValidationError: empty query
    try:
        rag = pgVectorDB(
            collection_name="test_errors",
            embedding_model=embeddings,
            connection_string=CONNECTION_STRING,
            schema_name=SCHEMA_NAME,
            index_type=IndexType.HNSW
        )
        await rag.initialize(overwrite_existing=True)
        
        try:
            await rag.semantic_search("", k=5)
            results.add_fail("ValidationError: Empty Query", "Should raise ValidationError")
        except ValidationError:
            results.add_pass("ValidationError: Empty Query")
        
        # ValidationError: invalid k
        try:
            await rag.semantic_search("test", k=0)
            results.add_fail("ValidationError: Invalid K", "Should raise ValidationError")
        except ValidationError:
            results.add_pass("ValidationError: Invalid K")
        
        # ValidationError: invalid weights
        docs, _ = generate_test_documents(10)
        await rag.add_documents(docs)
        await rag.build_index()
        
        try:
            await rag.hybrid_search("test", k=5, weights=(0.3, 0.5))
            results.add_fail("ValidationError: Invalid Weights", "Should raise ValidationError")
        except ValidationError:
            results.add_pass("ValidationError: Invalid Weights")
        
        await rag.close()
    except Exception as e:
        results.add_fail("Validation Errors", str(e))
    
    # InitializationError
    try:
        rag = pgVectorDB(
            collection_name="test_init_error",
            embedding_model=embeddings,
            connection_string=CONNECTION_STRING,
            schema_name=SCHEMA_NAME,
            index_type=IndexType.HNSW
        )
        
        try:
            await rag.semantic_search("test", k=5)
            results.add_fail("InitializationError", "Should raise InitializationError")
        except InitializationError:
            results.add_pass("InitializationError")
    except Exception as e:
        results.add_fail("InitializationError Test", str(e))


async def test_performance(embeddings):
    """Test performance benchmarks."""
    print("\n" + "=" * 80)
    print("⚡ PERFORMANCE TESTS")
    print("=" * 80)
    
    docs, _ = generate_test_documents(100)
    
    try:
        rag = pgVectorDB(
            collection_name="test_performance",
            embedding_model=embeddings,
            connection_string=CONNECTION_STRING,
            schema_name=SCHEMA_NAME,
            index_type=IndexType.HNSW
        )
        await rag.initialize(overwrite_existing=True)
        await rag.add_documents(docs)
        await rag.build_index()
        
        # Benchmark semantic search
        times = []
        for i in range(10):
            start = time.time()
            await rag.semantic_search(f"query {i}", k=5)
            times.append((time.time() - start) * 1000)
        
        avg_time = sum(times) / len(times)
        print(f"    Average query time: {avg_time:.2f}ms")
        assert avg_time < 1000, f"Query too slow: {avg_time:.2f}ms"
        results.add_pass("Performance Benchmark")
        
        # Test query parameter tuning
        await rag.set_query_params(ef_search=40)
        res = await rag.semantic_search("test query", k=5)
        assert len(res) <= 5
        results.add_pass("Query Parameter Tuning")
        
        await rag.close()
        
    except Exception as e:
        results.add_fail("Performance Tests", str(e))


# ==================== Main Test Runner ====================
async def run_all_tests():
    """Run complete test suite."""
    print("\n" + "=" * 80)
    print("PRODUCTION RAG SYSTEM - COMPLETE TEST SUITE")
    print("=" * 80)
    print(f"Database: {DB_HOST}:{DB_PORT}/{DB_NAME}")
    print(f"Schema: {SCHEMA_NAME}")
    print(f"Testing with 100 example documents")
    print("=" * 80)
    
    # Load embeddings once
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"}
    )
    
    try:
        # Run all test suites
        await test_initialization(embeddings)
        await test_document_operations(embeddings)
        await test_all_search_methods(embeddings)
        await test_filter_operators(embeddings)
        await test_index_operations(embeddings)
        await test_analytics_and_monitoring(embeddings)
        await test_data_export_import(embeddings)
        await test_langchain_integration(embeddings)
        await test_error_handling(embeddings)
        await test_performance(embeddings)
        
    except Exception as e:
        logger.error(f"Critical error: {e}")
        import traceback
        traceback.print_exc()
    
    return results.failed == 0


async def main():
    """Entry point."""
    try:
        # Create test schema
        await create_test_schema()
        
        # Run tests
        success = await run_all_tests()
        
        # Print summary
        results.print_summary()
        
        # Cleanup
        await cleanup_test_schema()
        
        # Exit
        if success:
            print("\n✅ ALL TESTS PASSED! System is production-ready.")
            sys.exit(0)
        else:
            print("\n❌ SOME TESTS FAILED. Please review errors above.")
            sys.exit(1)
    
    except KeyboardInterrupt:
        print("\n\n⚠️  Tests interrupted by user")
        await cleanup_test_schema()
        sys.exit(130)
    
    except Exception as e:
        print(f"\n\n❌ FATAL ERROR: {e}")
        import traceback
        traceback.print_exc()
        await cleanup_test_schema()
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
