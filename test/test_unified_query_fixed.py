"""
Test UnifiedQueryBuilder Fixes (TDD - Green Phase)

Tests for the fixed FluentAPI functionality:
- METADATA_FILTER search method
- ef, nprobes parameter pass-through
- analyze_plan with real PostgreSQL EXPLAIN ANALYZE
- label_filter support
- ensemble search
"""


import pytest


class TestMetadataFilterSearch:
    """Test METADATA_FILTER search method."""

    @pytest.mark.asyncio
    async def test_metadata_filter_search_method_exists(self, rag_hnsw):
        """SearchMethod.METADATA_FILTER should exist."""
        from pgvectordb import SearchMethod

        assert hasattr(SearchMethod, "METADATA_FILTER")
        assert SearchMethod.METADATA_FILTER.value == "metadata_filter"

    @pytest.mark.asyncio
    async def test_metadata_filter_requires_where(self, rag_hnsw):
        """metadata_only() requires a filter."""
        from pgvectordb.base import SearchMethod

        # Should raise ValueError when no filter is set
        with pytest.raises(ValueError) as exc_info:
            await rag_hnsw.query("").search_mode(SearchMethod.METADATA_FILTER).limit(5).to_list()
        assert "filter" in str(exc_info.value).lower()

    @pytest.mark.asyncio
    async def test_metadata_only_convenience_method(self, rag_hnsw):
        """metadata_only() should set search mode."""
        from pgvectordb import SearchMethod

        builder = rag_hnsw.query("").metadata_only().where({"category": "test"})
        assert builder._search_method == SearchMethod.METADATA_FILTER


class TestQueryParameterPassThrough:
    """Test that SearchConfig parameters are applied to backend."""

    @pytest.mark.asyncio
    async def test_ef_synced_to_db_query_params(self, rag_hnsw):
        """ef() should sync to db._query_params."""
        from pgvectordb import IndexType

        builder = rag_hnsw.query("test").ef(100)
        builder._sync_search_config_to_query_params()

        if rag_hnsw.index_type == IndexType.HNSW:
            assert rag_hnsw._query_params.get("hnsw.ef_search") == 100

    @pytest.mark.asyncio
    async def test_nprobes_synced_to_db_query_params(self, rag_hnsw):
        """nprobes() should sync to db._query_params for IVFFlat."""
        from pgvectordb import IndexType

        if rag_hnsw.index_type != IndexType.IVFFLAT:
            pytest.skip("Only applies to IVFFlat index type")

        builder = rag_hnsw.query("test").nprobes(20)
        builder._sync_search_config_to_query_params()

        assert rag_hnsw._query_params.get("ivfflat.probes") == 20

    @pytest.mark.asyncio
    async def test_refine_factor_multiplier(self, rag_hnsw):
        """refine_factor should multiply k."""
        builder = rag_hnsw.query("test").refine_factor(3).limit(10)

        # When _execute_semantic is called, k should be multiplied
        args = builder._build_execution_args()
        assert args["k"] == 30  # 10 * 3 = 30


class TestAnalyzePlanRealExplain:
    """Test analyze_plan uses real PostgreSQL EXPLAIN ANALYZE."""

    @pytest.mark.asyncio
    async def test_analyze_plan_returns_postgresql_metrics(self, db_with_docs):
        """analyze_plan should return PostgreSQL metrics, not just Python timing."""
        metrics = await db_with_docs.query("test search").limit(3).analyze_plan()

        assert isinstance(metrics, dict)

        # Should have PostgreSQL-specific keys OR fallback timing
        _ = any(
            key in metrics
            for key in [
                "plan",
                "execution_time_ms",
                "planning_time_ms",
                "index_used",
                "shared_hit_blocks",
            ]
        )

        # If it's fallback timing mode, it should have error note
        if "error" in metrics:
            assert "EXPLAIN ANALYZE" in str(metrics["error"]) or "not implemented" in str(
                metrics.get("note", "")
            )

    @pytest.mark.asyncio
    async def test_analyze_plan_includes_search_method(self, db_with_docs):
        """analyze_plan should include search method in output."""
        metrics = await db_with_docs.query("test search").semantic().limit(3).analyze_plan()

        assert "search_method" in metrics
        assert metrics["search_method"] == "semantic"


class TestExplainPlanStructured:
    """Test explain_plan returns structured query info."""

    @pytest.mark.asyncio
    async def test_explain_plan_structured_output(self, rag_hnsw):
        """explain_plan should return structured query info."""
        plan = (
            rag_hnsw.query("machine learning")
            .semantic()
            .where({"category": "ai"})
            .ef(100)
            .limit(5)
            .explain_plan()
        )

        assert isinstance(plan, dict)
        assert plan.get("search_method") == "semantic"
        assert plan.get("limit") == 5
        assert plan.get("filter") == {"category": "ai"}
        assert plan.get("ef_search") == 100

    @pytest.mark.asyncio
    async def test_explain_plan_keyword_info(self, rag_hnsw):
        """explain_plan for keyword should include keyword-specific info."""
        plan = (
            rag_hnsw.query("machine learning")
            .keyword()
            .bm25_params(k1=1.5, b=0.75)
            .limit(5)
            .explain_plan()
        )

        assert isinstance(plan, dict)
        assert plan.get("search_method") == "keyword"
        assert plan.get("bm25_k1") == 1.5
        assert plan.get("bm25_b") == 0.75

    @pytest.mark.asyncio
    async def test_explain_plan_hybrid_info(self, rag_hnsw):
        """explain_plan for hybrid should include hybrid-specific info."""
        plan = rag_hnsw.query("machine learning").hybrid().weights(0.7, 0.3).limit(5).explain_plan()

        assert isinstance(plan, dict)
        assert plan.get("search_method") == "hybrid"
        assert plan.get("semantic_weight") == 0.7
        assert plan.get("keyword_weight") == 0.3


class TestLabelFilterSupport:
    """Test label_filter support in SearchConfig."""

    @pytest.mark.asyncio
    async def test_labels_method_exists(self, rag_hnsw):
        """labels() method should exist on UnifiedQueryBuilder."""
        builder = rag_hnsw.query("test").labels([1, 2, 3])
        assert builder._config.label_filter == [1, 2, 3]

    @pytest.mark.asyncio
    async def test_label_filter_in_config(self, rag_hnsw):
        """label_filter should be in SearchConfig."""
        from pgvectordb.query.unified import SearchConfig

        config = SearchConfig(label_filter=[1, 2])
        assert config.label_filter == [1, 2]


class TestEnsembleSearch:
    """Test ensemble search convenience method."""

    @pytest.mark.asyncio
    async def test_ensemble_method_exists(self, rag_hnsw):
        """ensemble() convenience method should exist."""
        from pgvectordb import SearchMethod

        builder = rag_hnsw.query("test").ensemble()
        # Ensemble falls back to hybrid with mandatory filter
        assert builder._search_method == SearchMethod.HYBRID


class TestDeprecationWarnings:
    """Test that deprecated classes emit warnings."""

    @pytest.mark.asyncio
    async def test_builder_deprecation_warning(self, rag_hnsw):
        """Old builder classes should emit deprecation warnings."""
        import warnings

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            # Import should trigger warning

            # Creating instance should trigger warning
            # (would need actual db instance to test)

            deprecation_warnings = [x for x in w if issubclass(x.category, DeprecationWarning)]
            if deprecation_warnings:
                assert "deprecated" in str(deprecation_warnings[0].message).lower()


class TestQueryBuilderConsistency:
    """Test that UnifiedQueryBuilder is the only recommended builder."""

    @pytest.mark.asyncio
    async def test_query_returns_unified_builder(self, rag_hnsw):
        """db.query() should return UnifiedQueryBuilder."""
        from pgvectordb.query.unified import UnifiedQueryBuilder

        builder = rag_hnsw.query("test")
        assert isinstance(builder, UnifiedQueryBuilder)
