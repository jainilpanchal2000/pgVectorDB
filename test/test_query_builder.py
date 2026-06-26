"""
Test VectorQueryBuilder (TDD - Red Phase)

Tests for the LanceDB-style fluent query builder.
"""

import pytest


class TestSearchEntryPoint:
    """Test the search() entry point on pgVectorDB."""

    @pytest.mark.asyncio
    async def test_search_returns_vector_query_builder(self, rag_hnsw):
        """search() should return a query builder for vector queries."""
        from pgvectordb.query.unified import UnifiedQueryBuilder

        builder = rag_hnsw.query("test search")
        # Accept either legacy VectorQueryBuilder or new SemanticQueryBuilder
        assert isinstance(builder, UnifiedQueryBuilder)

    @pytest.mark.asyncio
    async def test_search_accepts_text_query(self, rag_hnsw):
        """search("text") should work and use embedding internally."""
        from pgvectordb.query.unified import UnifiedQueryBuilder

        builder = rag_hnsw.query("machine learning")
        # Accept either legacy VectorQueryBuilder or new SemanticQueryBuilder
        assert isinstance(builder, UnifiedQueryBuilder)


class TestVectorQueryBuilderChaining:
    """Test method chaining behavior."""

    @pytest.mark.asyncio
    async def test_limit_returns_self(self, rag_hnsw):
        """.limit() should return builder for chaining."""
        builder = rag_hnsw.query("test search")
        result = builder.limit(10)
        assert result is builder

    @pytest.mark.asyncio
    async def test_where_returns_self(self, rag_hnsw):
        """.where() should return builder for chaining."""
        builder = rag_hnsw.query("test search")
        result = builder.where({"category": "ai"})
        assert result is builder

    @pytest.mark.asyncio
    async def test_select_returns_self(self, rag_hnsw):
        """.select() should return builder for chaining."""
        builder = rag_hnsw.query("test search")
        result = builder.select(["content", "metadata"])
        assert result is builder

    @pytest.mark.asyncio
    async def test_offset_returns_self(self, rag_hnsw):
        """.offset() should return builder for chaining."""
        builder = rag_hnsw.query("test search")
        result = builder.offset(5)
        assert result is builder

    @pytest.mark.asyncio
    async def test_chaining_works(self, rag_hnsw):
        """Multiple methods should chain together."""
        builder = (
            rag_hnsw.query("test search").where({"category": "ai"}).limit(5).select(["content"])
        )
        assert builder._config.limit == 5
        assert builder._config.filter == {"category": "ai"}
        assert builder._config.columns == ["content"]


class TestVectorQueryBuilderExecution:
    """Test query execution methods."""

    @pytest.mark.asyncio
    async def test_to_list_executes_query(self, db_with_docs):
        """to_list() should execute and return results."""
        results = await db_with_docs.query("test search").limit(3).to_list()
        assert isinstance(results, list)
        assert len(results) <= 3

    @pytest.mark.asyncio
    async def test_to_list_returns_query_results(self, db_with_docs):
        """Results should be QueryResult-like dicts."""
        results = await db_with_docs.query("test search").limit(2).to_list()
        assert len(results) > 0
        # Each result should have expected keys
        for r in results:
            assert "id" in r or "langchain_id" in r
            assert "content" in r
            assert "score" in r or "distance" in r

    @pytest.mark.asyncio
    async def test_to_pandas_returns_dataframe(self, db_with_docs):
        """to_pandas() should return a pandas DataFrame."""
        import pandas as pd

        df = await db_with_docs.query("test search").limit(3).to_pandas()
        assert isinstance(df, pd.DataFrame)
        assert len(df) <= 3

    @pytest.mark.asyncio
    async def test_to_arrow_returns_table(self, db_with_docs):
        """to_arrow() should return a pyarrow Table."""
        import pyarrow as pa

        table = await db_with_docs.query("test search").limit(3).to_arrow()
        assert isinstance(table, pa.Table)
        assert table.num_rows <= 3


class TestVectorQueryBuilderFiltering:
    """Test filter functionality."""

    @pytest.mark.asyncio
    async def test_where_with_dict_filter(self, db_with_docs):
        """where(dict) should apply metadata filter."""
        results = await (
            db_with_docs.query("test search").where({"category": "ai"}).limit(10).to_list()
        )
        # All results should have category="ai"
        for r in results:
            assert r["metadata"].get("category") == "ai"

    @pytest.mark.asyncio
    async def test_where_with_string_filter(self, db_with_docs):
        """where(dict) should apply metadata filter."""
        results = await (
            db_with_docs.query("test search").where({"year": {"$gt": 2020}}).limit(5).to_list()
        )
        assert isinstance(results, list)


class TestVectorQueryBuilderParameters:
    """Test query parameter methods."""

    @pytest.mark.asyncio
    async def test_nprobes_returns_self(self, rag_hnsw):
        """.nprobes() should return builder."""
        builder = rag_hnsw.query("test search")
        result = builder.nprobes(20)
        assert result is builder
        assert builder._config.nprobes == 20

    @pytest.mark.asyncio
    async def test_ef_returns_self(self, rag_hnsw):
        """.ef() should return builder."""
        builder = rag_hnsw.query("test search")
        result = builder.ef(100)
        assert result is builder
        assert builder._config.ef == 100

    @pytest.mark.asyncio
    async def test_refine_factor_returns_self(self, rag_hnsw):
        """.refine_factor() should return builder."""
        builder = rag_hnsw.query("test search")
        result = builder.refine_factor(3)
        assert result is builder
        assert builder._config.refine_factor == 3

    @pytest.mark.asyncio
    async def test_distance_range_returns_self(self, rag_hnsw):
        """.distance_range() should return builder."""
        builder = rag_hnsw.query("test search")
        result = builder.distance_range(0.0, 1.0)
        assert result is builder
        assert builder._config.distance_range == (0.0, 1.0)

    @pytest.mark.asyncio
    async def test_bypass_vector_index_returns_self(self, rag_hnsw):
        """.bypass_vector_index() should return builder."""
        builder = rag_hnsw.query("test search")
        result = builder.bypass_vector_index()
        assert result is builder
        assert builder._config.bypass_vector_index is True


class TestExplainPlan:
    """Test explain functionality."""

    @pytest.mark.asyncio
    async def test_explain_plan_exists(self, db_with_docs):
        """explain_plan() method should exist."""
        result = db_with_docs.query("test search").limit(5).explain_plan()
        assert result is not None

    @pytest.mark.asyncio
    async def test_explain_plan_returns_string_or_dict(self, db_with_docs):
        """explain_plan() should return execution plan info."""
        plan = db_with_docs.query("test search").limit(5).explain_plan()
        assert isinstance(plan, (str, dict))


class TestAnalyzePlan:
    """Test analyze functionality."""

    @pytest.mark.asyncio
    async def test_analyze_plan_exists(self, db_with_docs):
        """analyze_plan() method should exist and be async."""
        result = await db_with_docs.query("test search").limit(5).analyze_plan()
        assert result is not None

    @pytest.mark.asyncio
    async def test_analyze_plan_returns_metrics(self, db_with_docs):
        """analyze_plan() should return execution metrics."""
        metrics = await db_with_docs.query("test search").limit(5).analyze_plan()
        assert isinstance(metrics, dict)
        # Should have execution-related keys
        assert any(k in metrics for k in ["execution_time_ms", "rows", "plan"])
