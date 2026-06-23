"""
Test UnifiedQueryBuilder (v0.0.6)

Tests for the new unified fluent query builder that supports all search methods
through a single entry point: db.query("...")
"""
import pytest


class TestUnifiedQueryEntryPoint:
    """Test the query() and search() entry points."""

    @pytest.mark.asyncio
    async def test_query_returns_unified_builder(self, rag_hnsw):
        """query() should return UnifiedQueryBuilder."""
        from pgvectordb.query.unified import UnifiedQueryBuilder

        builder = rag_hnsw.query("machine learning")
        assert isinstance(builder, UnifiedQueryBuilder)

    @pytest.mark.asyncio
    async def test_search_returns_semantic_builder(self, rag_hnsw):
        """query() should return UnifiedQueryBuilder."""
        from pgvectordb.query.unified import UnifiedQueryBuilder

        builder = rag_hnsw.query("machine learning")
        assert isinstance(builder, UnifiedQueryBuilder)

    @pytest.mark.asyncio
    async def test_vector_query_returns_unified_builder(self, rag_hnsw):
        """query().semantic() returns UnifiedQueryBuilder in semantic mode."""
        from pgvectordb.query.unified import UnifiedQueryBuilder

        builder = rag_hnsw.query("machine learning").semantic()
        assert isinstance(builder, UnifiedQueryBuilder)
        assert builder._search_method.value == "semantic"


class TestUnifiedQueryChaining:
    """Test method chaining behavior."""

    @pytest.mark.asyncio
    async def test_limit_returns_self(self, rag_hnsw):
        """.limit() should return builder for chaining."""
        from pgvectordb.query.unified import UnifiedQueryBuilder

        builder = rag_hnsw.query("test")
        result = builder.limit(10)
        assert result is builder
        assert isinstance(result, UnifiedQueryBuilder)

    @pytest.mark.asyncio
    async def test_where_returns_self(self, rag_hnsw):
        """.where() should return builder for chaining."""
        builder = rag_hnsw.query("test")
        result = builder.where({"category": "ai"})
        assert result is builder

    @pytest.mark.asyncio
    async def test_search_mode_returns_self(self, rag_hnsw):
        """.search_mode() should return builder for chaining."""
        from pgvectordb import SearchMethod

        builder = rag_hnsw.query("test")
        result = builder.search_mode(SearchMethod.KEYWORD)
        assert result is builder
        assert builder._search_method == SearchMethod.KEYWORD

    @pytest.mark.asyncio
    async def test_semantic_method_sets_mode(self, rag_hnsw):
        """.semantic() should set search mode to SEMANTIC."""
        from pgvectordb import SearchMethod

        builder = rag_hnsw.query("test").semantic()
        assert builder._search_method == SearchMethod.SEMANTIC

    @pytest.mark.asyncio
    async def test_keyword_method_sets_mode(self, rag_hnsw):
        """.keyword() should set search mode to KEYWORD."""
        from pgvectordb import SearchMethod

        builder = rag_hnsw.query("test").keyword()
        assert builder._search_method == SearchMethod.KEYWORD

    @pytest.mark.asyncio
    async def test_hybrid_method_sets_mode(self, rag_hnsw):
        """.hybrid() should set search mode to HYBRID."""
        from pgvectordb import SearchMethod

        builder = rag_hnsw.query("test").hybrid()
        assert builder._search_method == SearchMethod.HYBRID

    @pytest.mark.asyncio
    async def test_chaining_works(self, rag_hnsw):
        """Multiple methods should chain together."""
        from pgvectordb import SearchMethod

        builder = (
            rag_hnsw.query("test")
            .search_mode(SearchMethod.SEMANTIC)
            .where({"category": "ai"})
            .limit(5)
            .select(["content"])
        )
        assert builder._config.limit == 5
        assert builder._config.filter == {"category": "ai"}
        assert builder._config.columns == ["content"]
        assert builder._search_method == SearchMethod.SEMANTIC


class TestUnifiedQuerySearchModes:
    """Test different search modes."""

    @pytest.mark.asyncio
    async def test_default_is_semantic(self, rag_hnsw):
        """Default search mode should be SEMANTIC."""
        from pgvectordb import SearchMethod

        builder = rag_hnsw.query("test")
        assert builder._search_method == SearchMethod.SEMANTIC

    @pytest.mark.asyncio
    async def test_keyword_search_config(self, rag_hnsw):
        """Keyword search should accept BM25 params."""
        builder = (
            rag_hnsw.query("test")
            .keyword()
            .bm25_params(k1=1.5, b=0.75)
        )
        assert builder._config.keyword_type.value == "bm25"
        assert builder._config.bm25_k1 == 1.5
        assert builder._config.bm25_b == 0.75

    @pytest.mark.asyncio
    async def test_hybrid_rrf_config(self, rag_hnsw):
        """Hybrid search should accept RRF params."""
        builder = (
            rag_hnsw.query("test")
            .hybrid()
            .rrf(k=100)
        )
        assert builder._config.hybrid_mode == "rrf"
        assert builder._config.rrf_k == 100

    @pytest.mark.asyncio
    async def test_hybrid_weights_config(self, rag_hnsw):
        """Hybrid search should accept weights."""
        builder = (
            rag_hnsw.query("test")
            .hybrid()
            .weights(semantic=0.7, keyword=0.3)
        )
        assert builder._config.hybrid_mode == "weighted"
        assert builder._config.semantic_weight == 0.7
        assert builder._config.keyword_weight == 0.3

    @pytest.mark.asyncio
    async def test_trigram_threshold_config(self, rag_hnsw):
        """Trigram search should accept threshold."""
        builder = (
            rag_hnsw.query("test")
            .trigram()
            .threshold(0.4)
        )
        assert builder._search_method.value == "trigram"
        assert builder._config.trigram_threshold == 0.4


class TestUnifiedQueryExecution:
    """Test query execution."""

    @pytest.mark.asyncio
    async def test_to_list_executes_semantic_query(self, db_with_docs):
        """to_list() should execute semantic search."""
        results = await db_with_docs.query("machine learning").limit(3).to_list()
        assert isinstance(results, list)
        assert len(results) <= 3

    @pytest.mark.asyncio
    async def test_to_list_returns_query_results(self, db_with_docs):
        """Results should have expected structure."""
        results = await db_with_docs.query("machine learning").limit(2).to_list()
        assert len(results) > 0
        for r in results:
            assert "id" in r or "langchain_id" in r
            assert "content" in r
            assert "score" in r or "distance" in r

    @pytest.mark.asyncio
    async def test_filtered_query(self, db_with_docs):
        """Query with where() should filter results."""
        results = await (
            db_with_docs.query("test")
            .where({"category": "programming"})
            .limit(5)
            .to_list()
        )
        # All results should match filter
        for r in results:
            assert r["metadata"].get("category") == "programming"

    @pytest.mark.asyncio
    async def test_to_pandas_returns_dataframe(self, db_with_docs):
        """to_pandas() should return a DataFrame."""
        df = await db_with_docs.query("test").limit(3).to_pandas()
        import pandas as pd
        assert isinstance(df, pd.DataFrame)

    @pytest.mark.asyncio
    async def test_keyword_search_execution(self, db_with_docs):
        """Keyword search (FTS fallback) should execute."""
        from pgvectordb import SearchMethod

        results = await (
            db_with_docs.query("machine learning")
            .search_mode(SearchMethod.KEYWORD)
            .search_config(keyword_type="fts")
            .limit(3)
            .to_list()
        )
        assert isinstance(results, list)


class TestUnifiedQueryAnalysis:
    """Test query analysis methods."""

    @pytest.mark.asyncio
    async def test_explain_plan_returns_dict(self, rag_hnsw):
        """explain_plan() should return a dictionary."""
        plan = rag_hnsw.query("test").explain_plan()
        assert isinstance(plan, dict)
        assert "search_method" in plan

    @pytest.mark.asyncio
    async def test_analyze_plan_returns_metrics(self, db_with_docs):
        """analyze_plan() should return execution metrics."""
        metrics = await db_with_docs.query("test").analyze_plan()
        assert isinstance(metrics, dict)
        assert "execution_time_ms" in metrics
        assert "rows_returned" in metrics


class TestUnifiedQueryConfig:
    """Test search configuration."""

    @pytest.mark.asyncio
    async def test_search_config_accepts_params(self, rag_hnsw):
        """search_config() should accept arbitrary params."""
        builder = (
            rag_hnsw.query("test")
            .search_config(ef=100, refine_factor=2)
        )
        assert builder._config.ef == 100
        assert builder._config.refine_factor == 2

    @pytest.mark.asyncio
    async def test_ef_sets_config(self, rag_hnsw):
        """.ef() should set HNSW parameter."""
        builder = rag_hnsw.query("test").ef(150)
        assert builder._config.ef == 150

    @pytest.mark.asyncio
    async def test_nprobes_sets_config(self, rag_hnsw):
        """.nprobes() should set IVF parameter."""
        builder = rag_hnsw.query("test").nprobes(20)
        assert builder._config.nprobes == 20

    @pytest.mark.asyncio
    async def test_bypass_vector_index_sets_flag(self, rag_hnsw):
        """.bypass_vector_index() should set exact search flag."""
        builder = rag_hnsw.query("test").bypass_vector_index()
        assert builder._config.bypass_vector_index is True
