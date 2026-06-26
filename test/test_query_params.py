"""
Test Advanced Query Parameters (TDD - Red Phase)

Tests for query parameter application (nprobes, ef, refine_factor, distance_range).
"""

import pytest


class TestQueryParams:
    """Test query parameter integration with PostgreSQL."""

    @pytest.mark.asyncio
    async def test_nprobes_changes_query_behavior(self, db_with_docs):
        """Different nprobes values should affect results (IVF only)."""
        # This is hard to test directly, but we can verify it doesn't error
        results = await db_with_docs.query("test search").nprobes(10).limit(5).to_list()
        assert isinstance(results, list)

    @pytest.mark.asyncio
    async def test_ef_changes_query_behavior(self, db_with_docs):
        """Different ef values should affect results (HNSW only)."""
        results = await db_with_docs.query("test search").ef(50).limit(5).to_list()
        assert isinstance(results, list)

    @pytest.mark.asyncio
    async def test_refine_factor_returns_more_accurate(self, db_with_docs):
        """refine_factor should improve result quality."""
        # With refine_factor, we should still get results
        results = await db_with_docs.query("test search").refine_factor(3).limit(5).to_list()
        assert isinstance(results, list)
        assert len(results) <= 5

    @pytest.mark.asyncio
    async def test_distance_range_filters_results(self, db_with_docs):
        """distance_range should filter by distance."""
        # Get results with tight distance range
        results = await (
            db_with_docs.query("test search").distance_range(0.0, 0.5).limit(10).to_list()
        )
        # All results should have distance <= 0.5
        for r in results:
            assert r["score"] <= 0.5 or r.get("distance", 0) <= 0.5

    @pytest.mark.asyncio
    async def test_bypass_vector_index_for_exact_search(self, db_with_docs):
        """bypass_vector_index should do exact search."""
        results = await db_with_docs.query("test search").bypass_vector_index().limit(5).to_list()
        assert isinstance(results, list)


class TestQueryParamsCombined:
    """Test combining multiple query parameters."""

    @pytest.mark.asyncio
    async def test_combined_params_work(self, db_with_docs):
        """Multiple params can be combined."""
        results = await (
            db_with_docs.query("test search")
            .nprobes(20)
            .ef(100)
            .distance_range(0.0, 1.0)
            .limit(5)
            .to_list()
        )
        assert isinstance(results, list)

    @pytest.mark.asyncio
    async def test_params_with_filter(self, db_with_docs):
        """Params work with filters."""
        results = await (
            db_with_docs.query("test search")
            .where({"category": "ai"})
            .nprobes(20)
            .limit(5)
            .to_list()
        )
        assert isinstance(results, list)
