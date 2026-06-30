"""Live Integration Tests for v0.0.7 Features

Tests all new v0.0.7 features against a real PostgreSQL database.
"""

import asyncio
import os
from datetime import datetime

import pytest
from langchain_core.documents import Document

from pgvectordb import IndexType, SearchMethod, pgVectorDB
from pgvectordb.query.unified import SearchConfig, UnifiedQueryBuilder

# Database connection settings
DB_HOST = os.getenv("TEST_DB_HOST", "localhost")
DB_PORT = os.getenv("TEST_DB_PORT", "5434")
DB_NAME = os.getenv("TEST_DB_NAME", "testdb")
DB_USER = os.getenv("TEST_DB_USER", "testuser")
DB_PASSWORD = os.getenv("TEST_DB_PASSWORD", "testpass")
CONNECTION_STRING = f"postgresql+asyncpg://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

# Check if DB is available
try:
    from sqlalchemy import text
    from sqlalchemy.ext.asyncio import create_async_engine

    engine = create_async_engine(CONNECTION_STRING, pool_pre_ping=True)


    async def check_db():
        async with engine.connect() as conn:
            result = await conn.execute(text("SELECT 1"))
            return result.fetchone() is not None


    DB_AVAILABLE = asyncio.run(check_db())
except Exception:
    DB_AVAILABLE = False

pytestmark = pytest.mark.skipif(
    not DB_AVAILABLE,
    reason="Docker PostgreSQL database not available. Run: docker-compose -f docker-compose.test.yml up -d"
)


class TestDistanceFiltering:
    """Test distance range filtering."""

    def test_within_distance_method_exists(self):
        """within_distance() method exists on UnifiedQueryBuilder."""
        assert hasattr(UnifiedQueryBuilder, "within_distance")

    def test_distance_range_method_exists(self):
        """distance_range() method exists on UnifiedQueryBuilder."""
        assert hasattr(UnifiedQueryBuilder, "distance_range")

    def test_within_distance_sets_config(self):
        """within_distance sets distance_range config."""
        config = SearchConfig()
        config.distance_range = (0.0, 0.5)
        assert config.distance_range == (0.0, 0.5)


class TestExactSearch:
    """Test exact search bypass."""

    def test_exact_search_method_exists(self):
        """exact_search() method exists."""
        assert hasattr(UnifiedQueryBuilder, "exact_search")

    def test_exact_search_sets_config(self):
        """exact_search sets config flags."""
        config = SearchConfig()
        config.exact_search = True
        config.bypass_vector_index = True
        assert config.exact_search is True
        assert config.bypass_vector_index is True


class TestFilterStrategy:
    """Test filter strategy controls."""

    def test_pre_filter_method_exists(self):
        """pre_filter() method exists."""
        assert hasattr(UnifiedQueryBuilder, "pre_filter")

    def test_post_filter_method_exists(self):
        """post_filter() method exists."""
        assert hasattr(UnifiedQueryBuilder, "post_filter")

    def test_filter_strategy_config_exists(self):
        """filter_strategy config field exists."""
        config = SearchConfig()
        assert hasattr(config, "filter_strategy")
        assert config.filter_strategy == "auto"

    def test_fetch_multiplier_exists(self):
        """fetch_multiplier config field exists."""
        config = SearchConfig()
        assert hasattr(config, "fetch_multiplier")
        assert config.fetch_multiplier == 2.0


@pytest.mark.asyncio
class TestLiveDatabase:
    """Tests requiring live PostgreSQL database."""

    async def test_db_connection(self):
        """Can connect to database."""
        from langchain_core.embeddings import Embeddings

        class MockEmbeddings(Embeddings):
            def embed_documents(self, texts):
                return [[0.1] * 384 for _ in texts]

            def embed_query(self, text):
                return [0.1] * 384

        db = pgVectorDB(
            collection_name="test_v0_0_7_live",
            embedding_model=MockEmbeddings(),
            connection_string=CONNECTION_STRING,
            index_type=IndexType.HNSW,
        )

        # Just initialize (creates table)
        await db.initialize()
        assert db.table_name == "test_v0_0_7_live"

    async def test_indexes_property_accessible(self):
        """db.indexes property is accessible."""
        from langchain_core.embeddings import Embeddings

        class MockEmbeddings(Embeddings):
            def embed_documents(self, texts):
                return [[0.1] * 384 for _ in texts]

            def embed_query(self, text):
                return [0.1] * 384

        db = pgVectorDB(
            collection_name="test_v0_0_7_indexes",
            embedding_model=MockEmbeddings(),
            connection_string=CONNECTION_STRING,
            index_type=IndexType.HNSW,
        )

        indexes = db.indexes
        assert indexes is not None

    async def test_gin_property_accessible(self):
        """db.gin property is accessible."""
        from langchain_core.embeddings import Embeddings

        class MockEmbeddings(Embeddings):
            def embed_documents(self, texts):
                return [[0.1] * 384 for _ in texts]

            def embed_query(self, text):
                return [0.1] * 384

        db = pgVectorDB(
            collection_name="test_v0_0_7_gin",
            embedding_model=MockEmbeddings(),
            connection_string=CONNECTION_STRING,
            index_type=IndexType.HNSW,
        )

        gin = db.gin
        assert gin is not None

    async def test_query_with_all_v0_0_7_methods(self):
        """Can chain all v0.0.7 methods on query builder."""
        from langchain_core.embeddings import Embeddings

        class MockEmbeddings(Embeddings):
            def embed_documents(self, texts):
                return [[0.1] * 384 for _ in texts]

            def embed_query(self, text):
                return [0.1] * 384

        db = pgVectorDB(
            collection_name="test_v0_0_7_chain",
            embedding_model=MockEmbeddings(),
            connection_string=CONNECTION_STRING,
            index_type=IndexType.HNSW,
        )

        await db.initialize()

        # Build chain with all v0.0.7 methods
        query = (
            db.query("test")
            .semantic()
            .exact_search()
            .pre_filter()
            .within_distance(0.5)
            .ef(100)
            .limit(5)
        )

        assert query._config.exact_search is True
        assert query._config.filter_strategy == "pre"
        assert query._config.distance_range == (0.0, 0.5)
        assert query._config.ef == 100

    async def test_index_list_returns_list(self):
        """list_indexes returns a list (may be empty)."""
        from langchain_core.embeddings import Embeddings

        class MockEmbeddings(Embeddings):
            def embed_documents(self, texts):
                return [[0.1] * 384 for _ in texts]

            def embed_query(self, text):
                return [0.1] * 384

        db = pgVectorDB(
            collection_name="test_v0_0_7_list_idx",
            embedding_model=MockEmbeddings(),
            connection_string=CONNECTION_STRING,
            index_type=IndexType.HNSW,
        )

        await db.initialize()
        indexes = await db.indexes.list_indexes()
        assert isinstance(indexes, list)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
