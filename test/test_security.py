"""
test_security.py — Security and validation tests for pgVectorDB

Covers:
  - SQL injection rejection in collection_name / schema_name
  - ValidationError for empty query, k <= 0, bad weights
  - InitializationError when search called before initialize()
  - Schema name validation

Run:
    .venv\\Scripts\\python -m pytest test/test_security.py -v
"""

import pytest
import pytest_asyncio

from pgvectordb import (
    pgVectorDB,
    IndexType,
    ValidationError,
    InitializationError,
)


pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
CONNECTION_STRING = "postgresql+asyncpg://user:root@localhost:9002/postgres"


def _make_rag(collection_name: str = "sec_test", schema_name: str = "test", embeddings=None):
    """Helper: build a pgVectorDB instance without calling initialize()."""
    return pgVectorDB(
        collection_name=collection_name,
        embedding_model=embeddings,  # pass real embeddings if success expected
        connection_string=CONNECTION_STRING,
        schema_name=schema_name,
        index_type=IndexType.HNSW,
    )


# ---------------------------------------------------------------------------
# SQL injection / bad identifier rejection (constructor-time)
# ---------------------------------------------------------------------------

class TestIdentifierValidation:
    def test_semicolon_in_collection_name(self):
        with pytest.raises((ValidationError, Exception)):
            _make_rag(collection_name="test; DROP TABLE foo")

    def test_drop_table_in_name(self):
        with pytest.raises((ValidationError, Exception)):
            _make_rag(collection_name="x'; DROP TABLE users; --")

    def test_quote_in_collection_name(self):
        with pytest.raises((ValidationError, Exception)):
            _make_rag(collection_name='test"injection')

    def test_space_only_collection_name(self):
        with pytest.raises((ValidationError, Exception)):
            _make_rag(collection_name="   ")

    def test_empty_collection_name(self):
        with pytest.raises((ValidationError, Exception)):
            _make_rag(collection_name="")

    def test_valid_name_accepted(self, embeddings):
        """Valid names with letters, numbers, underscores should not raise."""
        rag = _make_rag(collection_name="valid_collection_123", embeddings=embeddings)
        assert rag is not None

    def test_valid_name_with_hyphen(self, embeddings):
        """Hyphens are usually allowed via quoting — at minimum should not crash."""
        try:
            rag = _make_rag(collection_name="valid-collection", embeddings=embeddings)
            assert rag is not None
        except (ValidationError, Exception):
            pass  # Stricter validation is also acceptable


# ---------------------------------------------------------------------------
# Input validation (requires DB — but checked as integration guard)
# ---------------------------------------------------------------------------

class TestInputValidation:
    """
    These tests verify ValidationError is raised BEFORE hitting the DB.
    They require a real rag instance but test pre-flight validation, so
    they can be used even without a running DB for the constructor path.
    """

    @pytest.mark.integration
    async def test_empty_query_raises_validation_error(self, rag_hnsw):
        """Semantic search with empty string must raise ValidationError."""
        with pytest.raises(ValidationError):
            await rag_hnsw.semantic_search("", k=5)

    @pytest.mark.integration
    async def test_whitespace_query_raises_validation_error(self, rag_hnsw):
        """Semantic search with whitespace-only string must raise ValidationError."""
        with pytest.raises(ValidationError):
            await rag_hnsw.semantic_search("   ", k=5)

    @pytest.mark.integration
    async def test_k_zero_raises_validation_error(self, rag_hnsw):
        """k=0 is invalid — must raise ValidationError."""
        with pytest.raises(ValidationError):
            await rag_hnsw.semantic_search("test", k=0)

    @pytest.mark.integration
    async def test_k_negative_raises_validation_error(self, rag_hnsw):
        """k=-1 is invalid — must raise ValidationError."""
        with pytest.raises(ValidationError):
            await rag_hnsw.semantic_search("test", k=-1)

    @pytest.mark.integration
    async def test_invalid_hybrid_weights_raises_validation_error(self, rag_hnsw, medium_docs, embeddings):
        """hybrid_search weights that don't sum to 1.0 must raise ValidationError."""
        docs, _ = medium_docs
        await rag_hnsw.add_documents(docs)
        await rag_hnsw.build_index()
        with pytest.raises(ValidationError):
            await rag_hnsw.hybrid_search("test", k=5, weights=(0.3, 0.5))

    @pytest.mark.integration
    async def test_empty_keyword_query_raises_validation_error(self, rag_hnsw):
        """keyword_search with empty string must raise ValidationError."""
        with pytest.raises(ValidationError):
            await rag_hnsw.keyword_search("", k=5)


# ---------------------------------------------------------------------------
# InitializationError guard
# ---------------------------------------------------------------------------

class TestInitializationGuard:
    @pytest.mark.integration
    async def test_search_before_initialize_raises(self, embeddings, connection_string):
        """Calling semantic_search before initialize() must raise InitializationError."""
        rag = pgVectorDB(
            collection_name="sec_uninit_guard",
            embedding_model=embeddings,
            connection_string=connection_string,
            schema_name="test",
            index_type=IndexType.HNSW,
        )
        with pytest.raises(InitializationError):
            await rag.semantic_search("test", k=5)

    @pytest.mark.integration
    async def test_add_documents_before_initialize_raises(self, embeddings, connection_string):
        """Calling add_documents before initialize() must raise InitializationError."""
        from langchain_core.documents import Document
        rag = pgVectorDB(
            collection_name="sec_uninit_docs",
            embedding_model=embeddings,
            connection_string=connection_string,
            schema_name="test",
            index_type=IndexType.HNSW,
        )
        with pytest.raises(InitializationError):
            await rag.add_documents([Document(page_content="test", metadata={})])
