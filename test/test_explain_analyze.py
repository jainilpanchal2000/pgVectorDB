"""
Test Explain/Analyze functionality (TDD - Red Phase)

Tests for PostgreSQL EXPLAIN/EXPLAIN ANALYZE integration.
"""

import pytest


class TestExplainPlan:
    """Test explain_plan functionality with real PostgreSQL."""

    @pytest.mark.asyncio
    async def test_explain_plan_returns_dict(self, db_with_docs):
        """explain_plan() should return a dictionary with plan info."""
        plan = db_with_docs.query("test search").limit(5).explain_plan()
        assert isinstance(plan, dict)

    @pytest.mark.asyncio
    async def test_explain_plan_contains_expected_keys(self, db_with_docs):
        """explain_plan() should contain plan structure keys."""
        plan = db_with_docs.query("test search").limit(5).explain_plan()
        # UnifiedQueryBuilder explain_plan returns a config summary dict
        assert "search_method" in plan or "plan" in plan or "Plan" in plan or "raw_plan" in plan

    @pytest.mark.asyncio
    async def test_explain_plan_shows_index_usage(self, db_with_docs):
        """explain_plan() should show if index is used."""
        plan = db_with_docs.query("test search").limit(5).explain_plan()
        # Check for index-related info
        plan_str = str(plan)
        assert "Index" in plan_str or "index" in plan_str.lower() or "Seq" in plan_str


class TestAnalyzePlan:
    """Test analyze_plan functionality with real PostgreSQL."""

    @pytest.mark.asyncio
    async def test_analyze_plan_returns_dict(self, db_with_docs):
        """analyze_plan() should return a dictionary with metrics."""
        metrics = await db_with_docs.query("test search").limit(5).analyze_plan()
        assert isinstance(metrics, dict)

    @pytest.mark.asyncio
    async def test_analyze_plan_contains_execution_time(self, db_with_docs):
        """analyze_plan() should contain execution time."""
        metrics = await db_with_docs.query("test search").limit(5).analyze_plan()
        # Should have timing info
        assert any(k in metrics for k in ["execution_time_ms", "Execution Time", "actual_time"])

    @pytest.mark.asyncio
    async def test_analyze_plan_contains_rows_info(self, db_with_docs):
        """analyze_plan() should contain row count info."""
        metrics = await db_with_docs.query("test search").limit(5).analyze_plan()
        # Should have row info
        assert any(k in metrics for k in ["rows_returned", "Actual Rows", "Plan Rows", "rows"])


class TestExplainWithFilters:
    """Test explain with query filters."""

    @pytest.mark.asyncio
    async def test_explain_plan_with_where(self, db_with_docs):
        """explain_plan() should work with filters."""
        plan = db_with_docs.query("test search").where({"category": "ai"}).limit(5).explain_plan()
        assert isinstance(plan, dict)

    @pytest.mark.asyncio
    async def test_analyze_plan_with_where(self, db_with_docs):
        """analyze_plan() should work with filters."""
        metrics = await (
            db_with_docs.query("test search").where({"category": "ai"}).limit(5).analyze_plan()
        )
        assert isinstance(metrics, dict)
