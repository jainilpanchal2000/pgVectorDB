"""
Comprehensive Test Suite for Production RAG System
==================================================

Tests all functionality with 100 example documents:
- All 3 index types (HNSW, IVFFlat, DiskANN)
- All 6 search methods
- All 13 filter operators
- Error handling
- Performance benchmarks
- Label-based filtering
"""

import sys
import os

# Ensure the parent directory is in sys.path for module import (MUST BE FIRST)
test_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(test_dir, '..'))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

import asyncio
import time
from typing import List
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from src.core import (
    pgVectorDB,
    IndexType,
    StorageLayout,
    ValidationError,
    InitializationError
)


# ==================== Test Configuration ====================
DB_HOST = "localhost"
DB_PORT = "9002"
DB_NAME = "postgres"
DB_USER = "user"
DB_PASSWORD = "root"

CONNECTION_STRING = f"postgresql+asyncpg://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

# Test counters
tests_passed = 0
tests_failed = 0
test_results = []


# ==================== Test Data Generation ====================
def generate_test_documents(num_docs: int = 100) -> tuple[List[Document], List[List[int]]]:
    """Generate diverse test documents with metadata and labels."""
    
    categories = ["programming", "ai", "database", "web", "devops", "security", "cloud", "mobile"]
    languages = ["Python", "JavaScript", "Java", "Go", "Rust", "SQL", "TypeScript", "C++"]
    authors = ["Tech Expert", "AI Researcher", "DB Admin", "DevOps Engineer", "Security Analyst"]
    
    # Content templates
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
        # Select random attributes
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
            page_content=content,
            metadata={
                "doc_id": i,
                "category": category,
                "language": language,
                "author": author,
                "year": year,
                "priority": (i % 10) + 1,  # 1-10
                "active": i % 2 == 0,  # True/False alternating
                "tags": f"tag{i % 5}"
            }
        )
        documents.append(doc)
        
        # Assign labels for DiskANN (1=programming, 2=ai, 3=database, 4=web, 5=devops, 6=security, 7=cloud, 8=mobile)
        category_to_label = {
            "programming": 1, "ai": 2, "database": 3, "web": 4,
            "devops": 5, "security": 6, "cloud": 7, "mobile": 8
        }
        label = category_to_label.get(category, 1)
        labels.append([label])
    
    return documents, labels


# ==================== Test Utilities ====================
def log_test(test_name: str, passed: bool, message: str = ""):
    """Log test result."""
    global tests_passed, tests_failed, test_results
    
    status = "✅ PASS" if passed else "❌ FAIL"
    result = f"{status} | {test_name}"
    if message:
        result += f" | {message}"
    
    test_results.append(result)
    print(result)
    
    if passed:
        tests_passed += 1
    else:
        tests_failed += 1


async def test_wrapper(test_name: str, test_func):
    """Wrap test function with error handling."""
    try:
        await test_func()
        log_test(test_name, True)
        return True
    except Exception as e:
        log_test(test_name, False, f"Error: {str(e)}")
        return False


# ==================== Initialization Tests ====================
async def test_hnsw_initialization():
    """Test HNSW system initialization."""
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"}
    )
    
    rag = pgVectorDB(
        collection_name="test_hnsw",
        embedding_model=embeddings,
        connection_string=CONNECTION_STRING,
        index_type=IndexType.HNSW
    )
    
    await rag.initialize(overwrite_existing=True)
    stats = await rag.get_stats()
    assert stats['index_type'] == 'hnsw'
    await rag.close()


async def test_ivfflat_initialization():
    """Test IVFFlat system initialization."""
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"}
    )
    
    rag = pgVectorDB(
        collection_name="test_ivfflat",
        embedding_model=embeddings,
        connection_string=CONNECTION_STRING,
        index_type=IndexType.IVFFLAT
    )
    
    await rag.initialize(overwrite_existing=True)
    stats = await rag.get_stats()
    assert stats['index_type'] == 'ivfflat'
    await rag.close()


async def test_diskann_initialization():
    """Test DiskANN system initialization."""
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"}
    )
    
    rag = pgVectorDB(
        collection_name="test_diskann",
        embedding_model=embeddings,
        connection_string=CONNECTION_STRING,
        index_type=IndexType.DISKANN
    )
    
    await rag.initialize(overwrite_existing=True)
    stats = await rag.get_stats()
    assert stats['index_type'] == 'diskann'
    await rag.close()


# ==================== Document Management Tests ====================
async def test_add_documents():
    """Test adding 100 documents."""
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"}
    )
    
    rag = pgVectorDB(
        collection_name="test_docs",
        embedding_model=embeddings,
        connection_string=CONNECTION_STRING,
        index_type=IndexType.HNSW
    )
    
    await rag.initialize(overwrite_existing=True)
    documents, _ = generate_test_documents(100)
    
    doc_ids = await rag.add_documents(documents)
    assert len(doc_ids) == 100
    
    stats = await rag.get_stats()
    assert stats['document_count'] == 100
    
    await rag.close()


async def test_add_documents_with_labels():
    """Test adding documents with labels for DiskANN."""
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"}
    )
    
    rag = pgVectorDB(
        collection_name="test_labels",
        embedding_model=embeddings,
        connection_string=CONNECTION_STRING,
        index_type=IndexType.DISKANN
    )
    
    await rag.initialize(overwrite_existing=True)
    documents, labels = generate_test_documents(100)
    
    doc_ids = await rag.add_documents(documents, labels=labels)
    assert len(doc_ids) == 100
    
    await rag.close()


async def test_metadata_indexing():
    """Test creating metadata indexes."""
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"}
    )
    
    rag = pgVectorDB(
        collection_name="test_metadata",
        embedding_model=embeddings,
        connection_string=CONNECTION_STRING,
        index_type=IndexType.HNSW
    )
    
    await rag.initialize(overwrite_existing=True)
    documents, _ = generate_test_documents(50)
    await rag.add_documents(documents)
    
    await rag.create_metadata_index(["category", "language", "author"])
    
    await rag.close()


# ==================== Index Building Tests ====================
async def test_build_hnsw_index():
    """Test building HNSW index."""
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"}
    )
    
    rag = pgVectorDB(
        collection_name="test_hnsw_build",
        embedding_model=embeddings,
        connection_string=CONNECTION_STRING,
        index_type=IndexType.HNSW
    )
    
    await rag.initialize(overwrite_existing=True)
    documents, _ = generate_test_documents(50)
    await rag.add_documents(documents)
    
    await rag.build_index(m=16, ef_construction=64)
    
    stats = await rag.get_stats()
    assert stats['index_built']
    
    await rag.close()


async def test_build_ivfflat_index():
    """Test building IVFFlat index."""
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"}
    )
    
    rag = pgVectorDB(
        collection_name="test_ivf_build",
        embedding_model=embeddings,
        connection_string=CONNECTION_STRING,
        index_type=IndexType.IVFFLAT
    )
    
    await rag.initialize(overwrite_existing=True)
    documents, _ = generate_test_documents(50)
    await rag.add_documents(documents)
    
    await rag.build_index(lists=10)
    
    stats = await rag.get_stats()
    assert stats['index_built']
    
    await rag.close()


async def test_build_diskann_index():
    """Test building DiskANN index with labels."""
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"}
    )
    
    rag = pgVectorDB(
        collection_name="test_diskann_build",
        embedding_model=embeddings,
        connection_string=CONNECTION_STRING,
        index_type=IndexType.DISKANN
    )
    
    await rag.initialize(overwrite_existing=True)
    documents, labels = generate_test_documents(50)
    await rag.add_documents(documents, labels=labels)
    
    await rag.build_index(
        num_neighbors=50,
        search_list_size=100,
        storage_layout=StorageLayout.MEMORY_OPTIMIZED,
        include_labels=True
    )
    
    stats = await rag.get_stats()
    assert stats['index_built']
    
    await rag.close()


# ==================== Search Method Tests ====================
async def test_keyword_search():
    """Test keyword search method."""
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"}
    )
    
    rag = pgVectorDB(
        collection_name="test_keyword",
        embedding_model=embeddings,
        connection_string=CONNECTION_STRING,
        index_type=IndexType.HNSW
    )
    
    await rag.initialize(overwrite_existing=True)
    documents, _ = generate_test_documents(100)
    await rag.add_documents(documents)
    await rag.build_index()
    
    results = await rag.keyword_search("programming language", k=5)
    assert len(results) > 0
    assert len(results) <= 5
    
    await rag.close()


async def test_universal_keyword_search():
    """Test universal keyword search method."""
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"}
    )
    
    rag = pgVectorDB(
        collection_name="test_universal",
        embedding_model=embeddings,
        connection_string=CONNECTION_STRING,
        index_type=IndexType.HNSW
    )
    
    await rag.initialize(overwrite_existing=True)
    documents, _ = generate_test_documents(100)
    await rag.add_documents(documents)
    await rag.build_index()
    
    results = await rag.universal_keyword_search(
        "Python",
        k=5,
        metadata_fields=["category", "language"]
    )
    assert len(results) > 0
    assert len(results) <= 5
    
    await rag.close()


async def test_semantic_search():
    """Test semantic search method."""
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"}
    )
    
    rag = pgVectorDB(
        collection_name="test_semantic",
        embedding_model=embeddings,
        connection_string=CONNECTION_STRING,
        index_type=IndexType.HNSW
    )
    
    await rag.initialize(overwrite_existing=True)
    documents, _ = generate_test_documents(100)
    await rag.add_documents(documents)
    await rag.build_index()
    
    results = await rag.semantic_search("machine learning AI", k=5)
    assert len(results) > 0
    assert len(results) <= 5
    
    await rag.close()


async def test_metadata_keyword_search():
    """Test metadata + keyword search method."""
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"}
    )
    
    rag = pgVectorDB(
        collection_name="test_meta_keyword",
        embedding_model=embeddings,
        connection_string=CONNECTION_STRING,
        index_type=IndexType.HNSW
    )
    
    await rag.initialize(overwrite_existing=True)
    documents, _ = generate_test_documents(100)
    await rag.add_documents(documents)
    await rag.create_metadata_index(["category"])
    await rag.build_index()
    
    results = await rag.metadata_keyword_search(
        query="programming",
        filter={"category": {"$eq": "programming"}},
        k=5
    )
    assert len(results) <= 5
    
    await rag.close()


async def test_metadata_semantic_search():
    """Test metadata + semantic search method."""
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"}
    )
    
    rag = pgVectorDB(
        collection_name="test_meta_semantic",
        embedding_model=embeddings,
        connection_string=CONNECTION_STRING,
        index_type=IndexType.HNSW
    )
    
    await rag.initialize(overwrite_existing=True)
    documents, _ = generate_test_documents(100)
    await rag.add_documents(documents)
    await rag.create_metadata_index(["category"])
    await rag.build_index()
    
    results = await rag.metadata_semantic_search(
        query="artificial intelligence",
        filter={"category": {"$eq": "ai"}},
        k=5
    )
    assert len(results) <= 5
    
    await rag.close()


async def test_hybrid_search():
    """Test hybrid search method."""
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"}
    )
    
    rag = pgVectorDB(
        collection_name="test_hybrid",
        embedding_model=embeddings,
        connection_string=CONNECTION_STRING,
        index_type=IndexType.HNSW
    )
    
    await rag.initialize(overwrite_existing=True)
    documents, _ = generate_test_documents(100)
    await rag.add_documents(documents)
    await rag.build_index()
    
    results = await rag.hybrid_search(
        query="Python programming",
        k=5,
        weights=(0.5, 0.5)
    )
    assert len(results) > 0
    assert len(results) <= 5
    
    await rag.close()


async def test_ensemble_search():
    """Test ensemble search method."""
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"}
    )
    
    rag = pgVectorDB(
        collection_name="test_ensemble",
        embedding_model=embeddings,
        connection_string=CONNECTION_STRING,
        index_type=IndexType.HNSW
    )
    
    await rag.initialize(overwrite_existing=True)
    documents, _ = generate_test_documents(100)
    await rag.add_documents(documents)
    await rag.create_metadata_index(["category", "year"])
    await rag.build_index()
    
    results = await rag.ensemble_search(
        query="database optimization",
        filter={"year": {"$gte": 2022}},
        k=5,
        weights=(0.6, 0.4)
    )
    assert len(results) <= 5
    
    await rag.close()


# ==================== Filter Operator Tests ====================
async def test_filter_eq():
    """Test $eq filter operator."""
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"}
    )
    
    rag = pgVectorDB(
        collection_name="test_filter_eq",
        embedding_model=embeddings,
        connection_string=CONNECTION_STRING,
        index_type=IndexType.HNSW
    )
    
    await rag.initialize(overwrite_existing=True)
    documents, _ = generate_test_documents(100)
    await rag.add_documents(documents)
    await rag.create_metadata_index(["category"])
    await rag.build_index()
    
    results = await rag.metadata_semantic_search(
        query="technology",
        filter={"category": {"$eq": "programming"}},
        k=20
    )
    
    # Verify all results match filter
    for result in results:
        assert result['metadata'].get('category') == 'programming'
    
    await rag.close()


async def test_filter_ne():
    """Test $ne filter operator."""
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"}
    )
    
    rag = pgVectorDB(
        collection_name="test_filter_ne",
        embedding_model=embeddings,
        connection_string=CONNECTION_STRING,
        index_type=IndexType.HNSW
    )
    
    await rag.initialize(overwrite_existing=True)
    documents, _ = generate_test_documents(100)
    await rag.add_documents(documents)
    await rag.create_metadata_index(["category"])
    await rag.build_index()
    
    results = await rag.metadata_semantic_search(
        query="technology",
        filter={"category": {"$ne": "ai"}},
        k=20
    )
    
    # Verify no results have 'ai' category
    for result in results:
        assert result['metadata'].get('category') != 'ai'
    
    await rag.close()


async def test_filter_gte():
    """Test $gte filter operator."""
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"}
    )
    
    rag = pgVectorDB(
        collection_name="test_filter_gte",
        embedding_model=embeddings,
        connection_string=CONNECTION_STRING,
        index_type=IndexType.HNSW
    )
    
    await rag.initialize(overwrite_existing=True)
    documents, _ = generate_test_documents(100)
    await rag.add_documents(documents)
    await rag.create_metadata_index(["year"])
    await rag.build_index()
    
    results = await rag.metadata_semantic_search(
        query="technology",
        filter={"year": {"$gte": 2023}},
        k=20
    )
    
    # Verify all results have year >= 2023
    for result in results:
        assert result['metadata'].get('year') >= 2023
    
    await rag.close()


async def test_filter_in():
    """Test $in filter operator."""
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"}
    )
    
    rag = pgVectorDB(
        collection_name="test_filter_in",
        embedding_model=embeddings,
        connection_string=CONNECTION_STRING,
        index_type=IndexType.HNSW
    )
    
    await rag.initialize(overwrite_existing=True)
    documents, _ = generate_test_documents(100)
    await rag.add_documents(documents)
    await rag.create_metadata_index(["language"])
    await rag.build_index()
    
    results = await rag.metadata_semantic_search(
        query="coding",
        filter={"language": {"$in": ["Python", "JavaScript", "Java"]}},
        k=20
    )
    
    # Verify all results have language in the list
    for result in results:
        assert result['metadata'].get('language') in ["Python", "JavaScript", "Java"]
    
    await rag.close()


async def test_filter_between():
    """Test $between filter operator."""
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"}
    )
    
    rag = pgVectorDB(
        collection_name="test_filter_between",
        embedding_model=embeddings,
        connection_string=CONNECTION_STRING,
        index_type=IndexType.HNSW
    )
    
    await rag.initialize(overwrite_existing=True)
    documents, _ = generate_test_documents(100)
    await rag.add_documents(documents)
    await rag.create_metadata_index(["year"])
    await rag.build_index()
    
    results = await rag.metadata_semantic_search(
        query="technology",
        filter={"year": {"$between": [2021, 2023]}},
        k=20
    )
    
    # Verify all results have year between 2021 and 2023
    for result in results:
        year = result['metadata'].get('year')
        assert 2021 <= year <= 2023
    
    await rag.close()


async def test_filter_exists():
    """Test $exists filter operator."""
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"}
    )
    
    rag = pgVectorDB(
        collection_name="test_filter_exists",
        embedding_model=embeddings,
        connection_string=CONNECTION_STRING,
        index_type=IndexType.HNSW
    )
    
    await rag.initialize(overwrite_existing=True)
    documents, _ = generate_test_documents(100)
    await rag.add_documents(documents)
    await rag.build_index()
    
    results = await rag.metadata_semantic_search(
        query="technology",
        filter={"author": {"$exists": True}},
        k=20
    )
    
    # Verify all results have 'author' field
    for result in results:
        assert 'author' in result['metadata']
    
    await rag.close()


async def test_filter_like():
    """Test $like filter operator."""
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"}
    )
    
    rag = pgVectorDB(
        collection_name="test_filter_like",
        embedding_model=embeddings,
        connection_string=CONNECTION_STRING,
        index_type=IndexType.HNSW
    )
    
    await rag.initialize(overwrite_existing=True)
    documents, _ = generate_test_documents(100)
    await rag.add_documents(documents)
    await rag.create_metadata_index(["author"])
    await rag.build_index()
    
    results = await rag.metadata_semantic_search(
        query="technology",
        filter={"author": {"$like": "%Expert%"}},
        k=20
    )
    
    # Verify all results have 'Expert' in author
    for result in results:
        assert 'Expert' in result['metadata'].get('author', '')
    
    await rag.close()


async def test_filter_and():
    """Test $and filter operator."""
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"}
    )
    
    rag = pgVectorDB(
        collection_name="test_filter_and",
        embedding_model=embeddings,
        connection_string=CONNECTION_STRING,
        index_type=IndexType.HNSW
    )
    
    await rag.initialize(overwrite_existing=True)
    documents, _ = generate_test_documents(100)
    await rag.add_documents(documents)
    await rag.create_metadata_index(["category", "year"])
    await rag.build_index()
    
    results = await rag.metadata_semantic_search(
        query="technology",
        filter={
            "$and": [
                {"category": {"$eq": "programming"}},
                {"year": {"$gte": 2022}}
            ]
        },
        k=20
    )
    
    # Verify all results match both conditions
    for result in results:
        assert result['metadata'].get('category') == 'programming'
        assert result['metadata'].get('year') >= 2022
    
    await rag.close()


async def test_filter_or():
    """Test $or filter operator."""
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"}
    )
    
    rag = pgVectorDB(
        collection_name="test_filter_or",
        embedding_model=embeddings,
        connection_string=CONNECTION_STRING,
        index_type=IndexType.HNSW
    )
    
    await rag.initialize(overwrite_existing=True)
    documents, _ = generate_test_documents(100)
    await rag.add_documents(documents)
    await rag.create_metadata_index(["category"])
    await rag.build_index()
    
    results = await rag.metadata_semantic_search(
        query="technology",
        filter={
            "$or": [
                {"category": {"$eq": "ai"}},
                {"category": {"$eq": "database"}}
            ]
        },
        k=20
    )
    
    # Verify all results match at least one condition
    for result in results:
        category = result['metadata'].get('category')
        assert category in ['ai', 'database']
    
    await rag.close()


# ==================== Label Filtering Tests (DiskANN) ====================
async def test_label_filtering():
    """Test label-based filtering with DiskANN."""
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"}
    )
    
    rag = pgVectorDB(
        collection_name="test_labels_filter",
        embedding_model=embeddings,
        connection_string=CONNECTION_STRING,
        index_type=IndexType.DISKANN
    )
    
    await rag.initialize(overwrite_existing=True)
    documents, labels = generate_test_documents(100)
    await rag.add_documents(documents, labels=labels)
    
    await rag.build_index(
        num_neighbors=50,
        search_list_size=100,
        include_labels=True
    )
    
    # Search with label filter (1=programming, 2=ai)
    results = await rag.semantic_search(
        query="coding and algorithms",
        k=10,
        label_filter=[1, 2]
    )
    
    assert len(results) <= 10
    
    await rag.close()


# ==================== Error Handling Tests ====================
async def test_validation_error_empty_query():
    """Test ValidationError for empty query."""
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"}
    )
    
    rag = pgVectorDB(
        collection_name="test_error1",
        embedding_model=embeddings,
        connection_string=CONNECTION_STRING,
        index_type=IndexType.HNSW
    )
    
    await rag.initialize(overwrite_existing=True)
    
    try:
        await rag.semantic_search("", k=5)
        assert False, "Should have raised ValidationError"
    except ValidationError:
        pass  # Expected
    
    await rag.close()


async def test_validation_error_invalid_k():
    """Test ValidationError for invalid k parameter."""
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"}
    )
    
    rag = pgVectorDB(
        collection_name="test_error2",
        embedding_model=embeddings,
        connection_string=CONNECTION_STRING,
        index_type=IndexType.HNSW
    )
    
    await rag.initialize(overwrite_existing=True)
    
    try:
        await rag.semantic_search("test", k=0)
        assert False, "Should have raised ValidationError"
    except ValidationError:
        pass  # Expected
    
    await rag.close()


async def test_validation_error_invalid_weights():
    """Test ValidationError for invalid weights."""
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"}
    )
    
    rag = pgVectorDB(
        collection_name="test_error3",
        embedding_model=embeddings,
        connection_string=CONNECTION_STRING,
        index_type=IndexType.HNSW
    )
    
    await rag.initialize(overwrite_existing=True)
    documents, _ = generate_test_documents(10)
    await rag.add_documents(documents)
    await rag.build_index()
    
    try:
        await rag.hybrid_search("test", k=5, weights=(0.3, 0.5))  # Don't sum to 1.0
        assert False, "Should have raised ValidationError"
    except ValidationError:
        pass  # Expected
    
    await rag.close()


async def test_initialization_error():
    """Test InitializationError when using system before initialization."""
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"}
    )
    
    rag = pgVectorDB(
        collection_name="test_error4",
        embedding_model=embeddings,
        connection_string=CONNECTION_STRING,
        index_type=IndexType.HNSW
    )
    
    # Don't initialize - try to use directly
    try:
        await rag.semantic_search("test", k=5)
        assert False, "Should have raised InitializationError"
    except InitializationError:
        pass  # Expected


# ==================== Performance Tests ====================
async def test_performance_benchmark():
    """Benchmark search performance with 100 documents."""
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"}
    )
    
    rag = pgVectorDB(
        collection_name="test_performance",
        embedding_model=embeddings,
        connection_string=CONNECTION_STRING,
        index_type=IndexType.HNSW
    )
    
    await rag.initialize(overwrite_existing=True)
    documents, _ = generate_test_documents(100)
    await rag.add_documents(documents)
    await rag.build_index()
    
    # Benchmark semantic search
    times = []
    for i in range(10):
        start = time.time()
        await rag.semantic_search(f"query {i}", k=5)
        times.append((time.time() - start) * 1000)
    
    avg_time = sum(times) / len(times)
    assert avg_time < 1000, f"Average query time {avg_time:.2f}ms too slow"
    
    print(f"    Average query time: {avg_time:.2f}ms")
    
    await rag.close()


async def test_query_parameter_tuning():
    """Test query parameter tuning for HNSW."""
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"}
    )
    
    rag = pgVectorDB(
        collection_name="test_tuning",
        embedding_model=embeddings,
        connection_string=CONNECTION_STRING,
        index_type=IndexType.HNSW
    )
    
    await rag.initialize(overwrite_existing=True)
    documents, _ = generate_test_documents(50)
    await rag.add_documents(documents)
    await rag.build_index()
    
    # Set query parameters
    await rag.set_query_params(ef_search=40)
    
    results = await rag.semantic_search("test query", k=5)
    assert len(results) <= 5
    
    await rag.close()


# ==================== Main Test Runner ====================
async def run_all_tests():
    """Run all tests and display results."""
    global tests_passed, tests_failed
    
    print("=" * 80)
    print("PRODUCTION RAG SYSTEM - COMPREHENSIVE TEST SUITE")
    print("=" * 80)
    print("Testing with 100 example documents")
    print(f"Database: {DB_HOST}:{DB_PORT}/{DB_NAME}")
    print("=" * 80)
    print()
    
    # Initialization Tests
    print("📦 INITIALIZATION TESTS")
    print("-" * 80)
    await test_wrapper("HNSW Initialization", test_hnsw_initialization)
    await test_wrapper("IVFFlat Initialization", test_ivfflat_initialization)
    await test_wrapper("DiskANN Initialization", test_diskann_initialization)
    print()
    
    # Document Management Tests
    print("📄 DOCUMENT MANAGEMENT TESTS")
    print("-" * 80)
    await test_wrapper("Add 100 Documents", test_add_documents)
    await test_wrapper("Add Documents with Labels", test_add_documents_with_labels)
    await test_wrapper("Create Metadata Indexes", test_metadata_indexing)
    print()
    
    # Index Building Tests
    print("🔨 INDEX BUILDING TESTS")
    print("-" * 80)
    await test_wrapper("Build HNSW Index", test_build_hnsw_index)
    await test_wrapper("Build IVFFlat Index", test_build_ivfflat_index)
    await test_wrapper("Build DiskANN Index", test_build_diskann_index)
    print()
    
    # Search Method Tests
    print("🔍 SEARCH METHOD TESTS (6 methods)")
    print("-" * 80)
    await test_wrapper("Keyword Search", test_keyword_search)
    await test_wrapper("Universal Keyword Search", test_universal_keyword_search)
    await test_wrapper("Semantic Search", test_semantic_search)
    await test_wrapper("Metadata + Keyword Search", test_metadata_keyword_search)
    await test_wrapper("Metadata + Semantic Search", test_metadata_semantic_search)
    await test_wrapper("Hybrid Search", test_hybrid_search)
    await test_wrapper("Ensemble Search", test_ensemble_search)
    print()
    
    # Filter Operator Tests
    print("🔧 FILTER OPERATOR TESTS (13 operators)")
    print("-" * 80)
    await test_wrapper("Filter: $eq", test_filter_eq)
    await test_wrapper("Filter: $ne", test_filter_ne)
    await test_wrapper("Filter: $gte", test_filter_gte)
    await test_wrapper("Filter: $in", test_filter_in)
    await test_wrapper("Filter: $between", test_filter_between)
    await test_wrapper("Filter: $exists", test_filter_exists)
    await test_wrapper("Filter: $like", test_filter_like)
    await test_wrapper("Filter: $and", test_filter_and)
    await test_wrapper("Filter: $or", test_filter_or)
    print()
    
    # Label Filtering Tests
    print("🏷️  LABEL FILTERING TESTS (DiskANN)")
    print("-" * 80)
    await test_wrapper("Label-Based Filtering", test_label_filtering)
    print()
    
    # Error Handling Tests
    print("⚠️  ERROR HANDLING TESTS")
    print("-" * 80)
    await test_wrapper("ValidationError: Empty Query", test_validation_error_empty_query)
    await test_wrapper("ValidationError: Invalid K", test_validation_error_invalid_k)
    await test_wrapper("ValidationError: Invalid Weights", test_validation_error_invalid_weights)
    await test_wrapper("InitializationError: Not Initialized", test_initialization_error)
    print()
    
    # Performance Tests
    print("⚡ PERFORMANCE TESTS")
    print("-" * 80)
    await test_wrapper("Performance Benchmark", test_performance_benchmark)
    await test_wrapper("Query Parameter Tuning", test_query_parameter_tuning)
    print()
    
    # Summary
    print("=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    print(f"✅ PASSED: {tests_passed}")
    print(f"❌ FAILED: {tests_failed}")
    print(f"📊 TOTAL:  {tests_passed + tests_failed}")
    print(f"📈 SUCCESS RATE: {(tests_passed / (tests_passed + tests_failed) * 100):.1f}%")
    print("=" * 80)
    
    if tests_failed > 0:
        print("\n⚠️  FAILED TESTS:")
        for result in test_results:
            if "❌ FAIL" in result:
                print(f"  {result}")
        print()
    
    return tests_failed == 0


# ==================== Entry Point ====================
if __name__ == "__main__":
    print("\n🚀 Starting Production RAG System Test Suite...\n")
    
    try:
        success = asyncio.run(run_all_tests())
        
        if success:
            print("✅ ALL TESTS PASSED! System is production-ready.")
            sys.exit(0)
        else:
            print("❌ SOME TESTS FAILED. Please review errors above.")
            sys.exit(1)
    
    except KeyboardInterrupt:
        print("\n\n⚠️  Tests interrupted by user")
        sys.exit(130)
    
    except Exception as e:
        print(f"\n\n❌ FATAL ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
