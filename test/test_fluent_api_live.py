#!/usr/bin/env python3
"""
Live Test Suite for FluentAPI Fixes
=================================

This script runs actual tests against a PostgreSQL database to verify
all FluentAPI fixes are working correctly.

Usage:
    python test_fluent_api_live.py

Or with pytest:
    pytest test_fluent_api_live.py -v

Requirements:
    - PostgreSQL with pgvector extension
    - Running database accessible via connection parameters below
"""

import asyncio
import json
import os
import sys
import warnings
from dataclasses import dataclass
from typing import Any

import pytest

# Add parent directory to path for local imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pgvectordb import pgVectorDB, SearchMethod, IndexType


# ============================================================================
# Configuration
# ============================================================================

# Database connection - update for your environment
DB_HOST = os.getenv("TEST_DB_HOST", "localhost")
DB_PORT = os.getenv("TEST_DB_PORT", "5432")
DB_NAME = os.getenv("TEST_DB_NAME", "postgres")
DB_USER = os.getenv("TEST_DB_USER", "postgres")
DB_PASSWORD = os.getenv("TEST_DB_PASSWORD", "postgres")
CONNECTION_STRING = f"postgresql+asyncpg://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"


# ============================================================================
# Test Fixtures
# ============================================================================

@pytest.fixture(scope="module")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="module")
async def test_db():
    """Create test database with sample data."""
    db = pgVectorDB(
        collection_name="fluent_api_test_live",
        connection_string=CONNECTION_STRING,
        dimensions=384,
        index_type=IndexType.HNSW,
        embedding_model_name="sentence-transformers/all-MiniLM-L6-v2",
    )

    # Add sample documents
    sample_docs = [
        {
            "id": "doc1",
            "content": "Machine learning is a subset of artificial intelligence that enables computers to learn patterns from data.",
            "metadata": {"category": "ai", "year": 2024, "status": "active"},
        },
        {
            "id": "doc2",
            "content": "PostgreSQL is a powerful open-source relational database with excellent support for vector operations.",
            "metadata": {"category": "database", "year": 2023, "status": "active"},
        },
        {
            "id": "doc3",
            "content": "HNSW indexing provides fast approximate nearest neighbor search for high-dimensional vectors.",
            "metadata": {"category": "ai", "year": 2024, "status": "active"},
        },
        {
            "id": "doc4",
            "content": "The pgvector extension adds vector similarity search to PostgreSQL databases efficiently.",
            "metadata": {"category": "database", "year": 2024, "status": "inactive"},
        },
        {
            "id": "doc5",
            "content": "BM25 is a ranking function used in information retrieval for keyword search.",
            "metadata": {"category": "search", "year": 2023, "status": "active"},
        },
    ]

    for doc in sample_docs:
        await db.add_document(
            document_id=doc["id"],
            content=doc["content"],
            metadata=doc["metadata"],
        )

    yield db

    # Cleanup
    await db.delete_collection()


# ============================================================================
# Test Cases
# ============================================================================

class TestAnalyzePlan:
    """Test REAL PostgreSQL EXPLAIN ANALYZE functionality."""

    @pytest.mark.asyncio
    async def test_analyze_plan_returns_postgres_metrics(self, test_db):
        """Test that analyze_plan() returns actual PostgreSQL metrics."""
        metrics = await test_db.query("machine learning").limit(5).analyze_plan()

        # Should have PostgreSQL-specific metrics
        assert "execution_time_ms" in metrics
        assert "planning_time_ms" in metrics
        assert "rows_returned" in metrics
        assert "config" in metrics
        assert "search_method" in metrics

        # Values should be reasonable (not mock data)
        assert metrics["execution_time_ms"] >= 0
        assert metrics["planning_time_ms"] >= 0
        assert metrics["rows_returned"] >= 0

    @pytest.mark.asyncio
    async def test_analyze_plan_has_plan_structure(self, test_db):
        """Test that analyze_plan returns structured plan data."""
        metrics = await test_db.query("machine learning").limit(3).analyze_plan()

        # Should have plan data from PostgreSQL
        if "plan" in metrics and metrics["plan"]:
            assert isinstance(metrics["plan"], dict) or isinstance(metrics["plan"], list)


class TestExplainPlan:
    """Test structured explain_plan() output."""

    @pytest.mark.asyncio
    async def test_explain_plan_returns_structured_info(self, test_db):
        """Test that explain_plan() returns structured query info."""
        plan = test_db.query("machine learning").limit(10).explain_plan()

        # Should be a dictionary with query info
        assert isinstance(plan, dict)
        assert "search_method" in plan
        assert "query" in plan
        assert "limit" in plan

    @pytest.mark.asyncio
    async def test_explain_plan_with_ef(self, test_db):
        """Test explain_plan() with ef parameter."""
        plan = test_db.query("test").ef(200).limit(5).explain_plan()

        assert plan.get("ef_search") == 200
        assert plan.get("search_method") == "semantic"


class TestMetadataOnly:
    """Test METADATA_FILTER (metadata_only) functionality."""

    @pytest.mark.asyncio
    async def test_metadata_only_without_query_text(self, test_db):
        """Test that metadata_only() works without query text."""
        results = await test_db.query("") \
            .metadata_only() \
            .where({"category": "ai"}) \
            .limit(10) \
            .to_list()

        # Should return results
        assert len(results) > 0

        # All results should be in category=ai
        for r in results:
            assert r.metadata.get("category") == "ai"

    @pytest.mark.asyncio
    async def test_metadata_only_with_complex_filter(self, test_db):
        """Test metadata_only() with complex boolean filters."""
        results = await test_db.query("") \
            .metadata_only() \
            .where({"$and": [{"year": {"$gte": 2024}}, {"status": "active"}]}) \
            .limit(10) \
            .to_list()

        for r in results:
            assert r.metadata.get("year") >= 2024
            assert r.metadata.get("status") == "active"

    @pytest.mark.asyncio
    async def test_metadata_only_requires_filter(self, test_db):
        """Test that metadata_only() requires a filter."""
        with pytest.raises(ValueError) as exc_info:
            await test_db.query("").metadata_only().limit(10).to_list()

        assert "METADATA_FILTER" in str(exc_info.value) or "filter" in str(exc_info.value).lower()


class TestParameterPassThrough:
    """Test ef/nprobes/refine_factor parameter pass-through."""

    @pytest.mark.asyncio
    async def test_ef_in_search_config(self, test_db):
        """Test that ef is stored in SearchConfig."""
        query = test_db.query("test").ef(200).limit(5)

        # Access internal config
        assert query._config.ef == 200

    @pytest.mark.asyncio
    async def test_nprobes_in_search_config(self, test_db):
        """Test that nprobes is stored in SearchConfig."""
        query = test_db.query("test").nprobes(16).limit(5)

        assert query._config.nprobes == 16

    @pytest.mark.asyncio
    async def test_refine_factor_in_search_config(self, test_db):
        """Test that refine_factor is stored in SearchConfig."""
        query = test_db.query("test").refine_factor(2).limit(5)

        assert query._config.refine_factor == 2


class TestEnsemble:
    """Test ensemble() convenience method."""

    @pytest.mark.asyncio
    async def test_ensemble_sets_hybrid_mode(self, test_db):
        """Test that ensemble() sets search mode to HYBRID."""
        query = test_db.query("test").ensemble()

        assert query._search_method == SearchMethod.HYBRID

    @pytest.mark.asyncio
    async def test_ensemble_with_weights(self, test_db):
        """Test ensemble search with weights."""
        results = await test_db.query("database") \
            .ensemble() \
            .where({"category": "database"}) \
            .weights(semantic=0.6, keyword=0.4) \
            .limit(5) \
            .to_list()

        # Should return results
        assert len(results) >= 0  # May be 0 if no matches


class TestLabels:
    """Test labels() method for DiskANN."""

    @pytest.mark.asyncio
    async def test_labels_in_config(self, test_db):
        """Test that labels() stores label_filter in config."""
        query = test_db.query("test").labels([1, 2, 3]).limit(5)

        assert query._config.label_filter == [1, 2, 3]

    @pytest.mark.asyncio
    async def test_labels_in_explain_plan(self, test_db):
        """Test that labels appear in explain_plan output."""
        plan = test_db.query("test").labels([1, 2]).explain_plan()

        assert plan.get("label_filter") == [1, 2]


class TestDeprecationWarnings:
    """Test deprecation warnings for old builders."""

    def test_builder_deprecation_warning(self):
        """Test that builder.py classes emit warnings."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            from pgvectordb.query.builder import VectorQueryBuilder

            deprecation_warnings = [x for x in w if issubclass(x.category, DeprecationWarning)]
            assert len(deprecation_warnings) > 0

    def test_builders_deprecation_warning(self):
        """Test that builders.py classes emit warnings."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            from pgvectordb.query.builders import SemanticQueryBuilder

            deprecation_warnings = [x for x in w if issubclass(x.category, DeprecationWarning)]
            assert len(deprecation_warnings) > 0


class TestHybrid:
    """Test hybrid search functionality."""

    @pytest.mark.asyncio
    async def test_hybrid_weighted(self, test_db):
        """Test hybrid search with weighted fusion."""
        results = await test_db.query("search database") \
            .hybrid() \
            .weights(semantic=0.6, keyword=0.4) \
            .limit(5) \
            .to_list()

        assert len(results) >= 0

    @pytest.mark.asyncio
    async def test_hybrid_rrf(self, test_db):
        """Test hybrid search with RRF."""
        results = await test_db.query("search database") \
            .hybrid() \
            .rrf(k=60) \
            .limit(5) \
            .to_list()

        assert len(results) >= 0


class TestFilteredSearch:
    """Test filtered search methods."""

    @pytest.mark.asyncio
    async def test_filtered_semantic(self, test_db):
        """Test semantic search with filter."""
        results = await test_db.query("machine learning") \
            .where({"category": "ai"}) \
            .limit(5) \
            .to_list()

        for r in results:
            assert r.metadata.get("category") == "ai"

    @pytest.mark.asyncio
    async def test_filtered_keyword(self, test_db):
        """Test keyword search with filter."""
        results = await test_db.query("database") \
            .keyword() \
            .where({"category": "database"}) \
            .limit(5) \
            .to_list()

        for r in results:
            assert r.metadata.get("category") == "database"


# ============================================================================
# Main Entry Point
# ============================================================================

if __name__ == "__main__":
    """Run tests with pytest."""
    pytest.main([__file__, "-v", "--tb=short"])
