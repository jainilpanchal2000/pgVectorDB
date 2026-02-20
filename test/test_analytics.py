"""
test_analytics.py — Integration tests for pgvectordb/mixins/analytics.py

Tests:
  - get_stats()
  - get_index_stats()
  - validate_collection()
  - explain_query()
  - benchmark_search_methods()
  - compute_recall()
  - set_iterative_scan()
  - create_label_definitions() + get_label_ids_by_names()
  - set_maintenance_work_mem()
  - get_bm25_index_stats() (skipped if pg_textsearch absent)

Run:
    .venv\\Scripts\\python -m pytest test/test_analytics.py -v
"""

import pytest
import pytest_asyncio

from pgvectordb import pgVectorDB, IndexType, IterativeScanMode


pytestmark = pytest.mark.integration


# ---------------------------------------------------------------------------
# Fixture: collection with data + index built
# ---------------------------------------------------------------------------


@pytest_asyncio.fixture
async def rag_ready(db_schema, embeddings, connection_string, medium_docs):
    """HNSW collection with 50 docs + index built."""
    inst = pgVectorDB(
        collection_name="test_analytics_col",
        embedding_model=embeddings,
        connection_string=connection_string,
        schema_name=db_schema,
        index_type=IndexType.HNSW,
    )
    await inst.initialize(overwrite_existing=True)
    docs, _ = medium_docs
    await inst.add_documents(docs)
    await inst.build_index()
    yield inst
    await inst.close()


# ---------------------------------------------------------------------------
# get_stats()
# ---------------------------------------------------------------------------


class TestGetStats:
    async def test_returns_dict(self, rag_ready):
        stats = await rag_ready.get_stats()
        assert isinstance(stats, dict)

    async def test_document_count_key(self, rag_ready):
        stats = await rag_ready.get_stats()
        assert "document_count" in stats

    async def test_index_type_key(self, rag_ready):
        stats = await rag_ready.get_stats()
        assert "index_type" in stats

    async def test_index_built_key(self, rag_ready):
        stats = await rag_ready.get_stats()
        assert "index_built" in stats

    async def test_document_count_correct(self, rag_ready):
        stats = await rag_ready.get_stats()
        assert stats["document_count"] == 50

    async def test_index_type_is_hnsw(self, rag_ready):
        stats = await rag_ready.get_stats()
        assert stats["index_type"] == "hnsw"

    async def test_index_built_true(self, rag_ready):
        stats = await rag_ready.get_stats()
        assert stats["index_built"] is True


# ---------------------------------------------------------------------------
# get_index_stats()
# ---------------------------------------------------------------------------


class TestGetIndexStats:
    async def test_returns_dict(self, rag_ready):
        stats = await rag_ready.get_index_stats()
        assert isinstance(stats, dict)

    async def test_index_type_key(self, rag_ready):
        stats = await rag_ready.get_index_stats()
        assert "index_type" in stats

    async def test_indexes_key(self, rag_ready):
        stats = await rag_ready.get_index_stats()
        assert "indexes" in stats


# ---------------------------------------------------------------------------
# validate_collection()
# ---------------------------------------------------------------------------


class TestValidateCollection:
    async def test_returns_dict(self, rag_ready):
        result = await rag_ready.validate_collection()
        assert isinstance(result, dict)

    async def test_healthy_key(self, rag_ready):
        result = await rag_ready.validate_collection()
        assert "healthy" in result

    async def test_issues_key(self, rag_ready):
        result = await rag_ready.validate_collection()
        assert "issues" in result

    async def test_healthy_data_is_healthy(self, rag_ready):
        result = await rag_ready.validate_collection()
        assert result["healthy"] is True
        assert result["issues"] == []


# ---------------------------------------------------------------------------
# explain_query()
# ---------------------------------------------------------------------------


class TestExplainQuery:
    async def test_returns_non_empty(self, rag_ready):
        plan = await rag_ready.explain_query("test", "semantic_search", k=5)
        assert plan is not None
        assert len(str(plan)) > 0

    async def test_keyword_explain(self, rag_ready):
        plan = await rag_ready.explain_query("test", "keyword_search", k=5)
        assert plan is not None


# ---------------------------------------------------------------------------
# benchmark_search_methods()
# ---------------------------------------------------------------------------


class TestBenchmarkSearchMethods:
    async def test_returns_dict(self, rag_ready):
        bench = await rag_ready.benchmark_search_methods(["test1", "test2"], k=5)
        assert isinstance(bench, dict)

    async def test_non_empty(self, rag_ready):
        bench = await rag_ready.benchmark_search_methods(["test1", "test2"], k=5)
        assert len(bench) > 0

    async def test_contains_semantic(self, rag_ready):
        bench = await rag_ready.benchmark_search_methods(["test"], k=5)
        assert any("semantic" in k.lower() for k in bench)


# ---------------------------------------------------------------------------
# compute_recall()
# ---------------------------------------------------------------------------


class TestComputeRecall:
    async def test_returns_dict(self, rag_ready):
        result = await rag_ready.compute_recall(["python", "database"], k=5)
        assert isinstance(result, dict)

    async def test_recall_at_k_in_result(self, rag_ready):
        result = await rag_ready.compute_recall(["python"], k=5)
        assert "recall@k" in result

    async def test_recall_in_range(self, rag_ready):
        result = await rag_ready.compute_recall(["python", "ai"], k=5)
        assert 0.0 <= result["recall@k"] <= 1.0


# ---------------------------------------------------------------------------
# set_iterative_scan()
# ---------------------------------------------------------------------------


class TestSetIterativeScan:
    async def test_relaxed_order(self, rag_ready):
        rag_ready.set_iterative_scan(
            mode=IterativeScanMode.RELAXED_ORDER,
            max_scan_tuples=1000,
        )

    async def test_strict_order(self, rag_ready):
        rag_ready.set_iterative_scan(mode=IterativeScanMode.STRICT_ORDER)

    async def test_off(self, rag_ready):
        rag_ready.set_iterative_scan(mode=IterativeScanMode.OFF)


# ---------------------------------------------------------------------------
# Label definitions
# ---------------------------------------------------------------------------


class TestLabelDefinitions:
    async def test_create_label_definitions(self, rag_ready):
        labels = [
            {"id": 1, "name": "programming", "description": "Programming docs"},
            {"id": 2, "name": "ai", "description": "AI/ML docs"},
        ]
        count = await rag_ready.create_label_definitions(labels)
        assert count == 2

    async def test_get_label_ids_by_names(self, rag_ready):
        labels = [
            {"id": 10, "name": "cloud", "description": "Cloud docs"},
            {"id": 11, "name": "devops", "description": "DevOps docs"},
        ]
        await rag_ready.create_label_definitions(labels)
        ids = await rag_ready.get_label_ids_by_names(["cloud", "devops"])
        assert 10 in ids
        assert 11 in ids

    async def test_get_label_ids_unknown_name(self, rag_ready):
        ids = await rag_ready.get_label_ids_by_names(["nonexistent_label_xyz"])
        assert ids == []


# ---------------------------------------------------------------------------
# set_maintenance_work_mem()
# ---------------------------------------------------------------------------


class TestMaintenanceWorkMem:
    async def test_set_512mb(self, rag_ready):
        await rag_ready.set_maintenance_work_mem("512MB")

    async def test_set_1gb(self, rag_ready):
        await rag_ready.set_maintenance_work_mem("1GB")


# ---------------------------------------------------------------------------
# BM25 index stats (conditional)
# ---------------------------------------------------------------------------


class TestBM25IndexStats:
    async def test_bm25_stats_if_available(
        self, db_schema, embeddings, connection_string, small_docs
    ):
        from pgvectordb import ExtensionManager
        from sqlalchemy.ext.asyncio import create_async_engine

        engine = create_async_engine(connection_string, pool_pre_ping=True)
        mgr = ExtensionManager(engine)
        await mgr.check_extensions()
        await engine.dispose()

        if not mgr.has_pg_textsearch:
            pytest.skip("pg_textsearch not installed")

        rag = pgVectorDB(
            collection_name="test_bm25stats",
            embedding_model=embeddings,
            connection_string=connection_string,
            schema_name=db_schema,
            index_type=IndexType.HNSW,
        )
        await rag.initialize(overwrite_existing=True)
        docs, _ = small_docs
        await rag.add_documents(docs)
        await rag.build_bm25_index()
        stats = await rag.get_bm25_index_stats()
        assert isinstance(stats, dict)
        await rag.close()
