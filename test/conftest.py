# ruff: noqa: E402
"""
Shared pytest fixtures for pgVectorDB test suite.

Provides:
  - embeddings      : session-scoped HuggingFace embeddings
  - connection_string: session-scoped DB URL (Docker on 9002)
  - db_schema       : session-scoped — creates/drops the "test" schema
  - rag_hnsw        : function-scoped HNSW pgVectorDB instance (init + close)
  - small_docs      : 20-doc list (function-scoped)
  - medium_docs     : 50-doc list (function-scoped)
  - large_docs      : 100-doc list (function-scoped)
  - docs_and_labels : 50-doc list + matching DiskANN labels
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure package is importable when running from project root or test/
ROOT = Path(__file__).parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
# noqa: E402
import pytest
import pytest_asyncio
from langchain_core.documents import Document
from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy import text

from pgvectordb import pgVectorDB, IndexType


# ---------------------------------------------------------------------------
# Database constants — always target Docker on port 9002
# ---------------------------------------------------------------------------
DB_HOST = "localhost"
DB_PORT = "9002"
DB_NAME = "postgres"
DB_USER = "user"
DB_PASSWORD = "root"
SCHEMA_NAME = "test"
CONNECTION_STRING = (
    f"postgresql+asyncpg://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
)


# ---------------------------------------------------------------------------
# Test document generator
# ---------------------------------------------------------------------------
def generate_test_documents(
    num_docs: int = 50,
) -> tuple[list[Document], list[list[int]]]:
    """Generate diverse test documents with metadata and DiskANN labels."""
    categories = [
        "programming",
        "ai",
        "database",
        "web",
        "devops",
        "security",
        "cloud",
        "mobile",
    ]
    languages = [
        "Python",
        "JavaScript",
        "Java",
        "Go",
        "Rust",
        "SQL",
        "TypeScript",
        "C++",
    ]
    authors = [
        "Tech Expert",
        "AI Researcher",
        "DB Admin",
        "DevOps Engineer",
        "Security Analyst",
    ]

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
        "Data engineering pipelines processing {data_type} enable {outcome} for analytics.",
    ]

    fill = {
        "adjective": [
            "powerful",
            "versatile",
            "modern",
            "efficient",
            "scalable",
            "robust",
        ],
        "purpose": [
            "web development",
            "data science",
            "system programming",
            "automation",
        ],
        "capability": [
            "pattern recognition",
            "predictive analytics",
            "natural language processing",
        ],
        "tool": ["indexes", "partitioning", "caching", "replication"],
        "metric": ["query performance", "throughput", "response time"],
        "framework": ["React", "FastAPI", "Django", "Express", "Spring Boot"],
        "benefit": ["scalability", "reliability", "flexibility", "performance"],
        "practice": [
            "input validation",
            "encryption",
            "authentication",
            "authorization",
        ],
        "threat": ["SQL injection", "XSS", "CSRF", "DDoS"],
        "platform": ["React Native", "Flutter", "Swift", "Kotlin"],
        "feature": ["offline support", "push notifications", "real-time updates"],
        "problem": ["deployment time", "manual errors", "configuration drift"],
        "pattern": ["REST", "GraphQL", "gRPC", "WebSocket"],
        "quality": ["maintainability", "testability", "observability"],
        "data_type": ["streaming data", "batch data", "real-time events"],
        "outcome": ["business insights", "data-driven decisions", "predictive models"],
    }

    cat_to_label = {
        "programming": 1,
        "ai": 2,
        "database": 3,
        "web": 4,
        "devops": 5,
        "security": 6,
        "cloud": 7,
        "mobile": 8,
    }

    documents: list[Document] = []
    labels: list[list[int]] = []

    for i in range(num_docs):
        cat = categories[i % len(categories)]
        lang = languages[i % len(languages)]
        author = authors[i % len(authors)]
        year = 2020 + (i % 5)

        template = content_templates[i % len(content_templates)]
        content = template.format(
            language=lang,
            **{k: v[i % len(v)] for k, v in fill.items()},
        )

        documents.append(
            Document(
                page_content=content.strip(),
                metadata={
                    "doc_id": i,
                    "category": cat,
                    "language": lang,
                    "author": author,
                    "year": year,
                    "priority": (i % 10) + 1,
                    "status": "active" if i % 4 != 0 else "archived",
                    "tags": f"tag{i % 5}",
                },
            )
        )
        labels.append([cat_to_label.get(cat, 1)])

    return documents, labels


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def connection_string() -> str:
    return CONNECTION_STRING


@pytest.fixture(scope="session")
def embeddings():
    """Session-scoped HuggingFace embeddings (all-MiniLM-L6-v2, 384-dim).

    Falls back to a MagicMock when langchain_huggingface is not installed,
    so unit tests that only need the fixture for object construction don't error.
    """
    try:
        from langchain_huggingface import HuggingFaceEmbeddings
        return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    except ImportError:
        from unittest.mock import MagicMock
        mock = MagicMock()
        mock.embed_documents.return_value = [[0.0] * 384]
        mock.embed_query.return_value = [0.0] * 384
        return mock



@pytest_asyncio.fixture(scope="session")
async def db_schema(connection_string):
    """Create the test schema once for the session; drop it at teardown."""
    engine = create_async_engine(connection_string, pool_pre_ping=True)
    async with engine.connect() as conn:
        await conn.execute(text(f"CREATE SCHEMA IF NOT EXISTS {SCHEMA_NAME}"))
        await conn.commit()
    yield SCHEMA_NAME
    async with engine.connect() as conn:
        await conn.execute(text(f"DROP SCHEMA IF EXISTS {SCHEMA_NAME} CASCADE"))
        await conn.commit()
    await engine.dispose()


@pytest_asyncio.fixture
async def rag_hnsw(db_schema, embeddings, connection_string):
    """Function-scoped HNSW pgVectorDB instance, fresh table per test."""
    rag = pgVectorDB(
        collection_name="test_hnsw",
        embedding_model=embeddings,
        connection_string=connection_string,
        schema_name=db_schema,
        index_type=IndexType.HNSW,
    )
    await rag.initialize(overwrite_existing=True)
    yield rag
    await rag.close()


@pytest.fixture
def small_docs() -> tuple[list[Document], list[list[int]]]:
    """20 test documents + labels."""
    return generate_test_documents(20)


@pytest.fixture
def medium_docs() -> tuple[list[Document], list[list[int]]]:
    """50 test documents + labels."""
    return generate_test_documents(50)


@pytest.fixture
def large_docs() -> tuple[list[Document], list[list[int]]]:
    """100 test documents + labels."""
    return generate_test_documents(100)


@pytest.fixture
def docs_and_labels() -> tuple[list[Document], list[list[int]]]:
    """50 test documents with DiskANN labels."""
    return generate_test_documents(50)
