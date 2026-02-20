"""
test_indexing.py — Integration tests for pgvectordb/mixins/indexing.py

Tests:
  - build_index() for HNSW, IVFFlat, DiskANN
  - build_bm25_index() (skipped if pg_textsearch absent)
  - build_index_concurrent()
  - areindex() rebuild
  - adrop_vector_index() and verify gone
  - create_metadata_index() GIN index
  - set_query_params() — ef_search, probes, iterative_scan, max_scan_tuples
  - set_diskann_build_params() API only
  - set_maintenance_work_mem()
  - vacuum_analyze()
  - Label-based filtering with DiskANN

Run:
    .venv\\Scripts\\python -m pytest test/test_indexing.py -v
"""

import pytest
import pytest_asyncio

from pgvectordb import pgVectorDB, IndexType, DistanceMetric, StorageLayout


pytestmark = pytest.mark.integration


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_rag(index_type, collection_name, db_schema, embeddings, connection_string):
    return pgVectorDB(
        collection_name=collection_name,
        embedding_model=embeddings,
        connection_string=connection_string,
        schema_name=db_schema,
        index_type=index_type,
    )


# ---------------------------------------------------------------------------
# create_metadata_index (GIN)
# ---------------------------------------------------------------------------

class TestMetadataIndex:
    async def test_create_gin_index(self, rag_hnsw, medium_docs):
        docs, _ = medium_docs
        await rag_hnsw.add_documents(docs)
        # Should not raise
        await rag_hnsw.create_metadata_index(["category", "language", "year"])

    async def test_create_gin_index_single_column(self, rag_hnsw, small_docs):
        docs, _ = small_docs
        await rag_hnsw.add_documents(docs)
        await rag_hnsw.create_metadata_index(["status"])


# ---------------------------------------------------------------------------
# HNSW index
# ---------------------------------------------------------------------------

class TestHNSWIndex:
    async def test_build_hnsw(self, rag_hnsw, medium_docs):
        docs, _ = medium_docs
        await rag_hnsw.add_documents(docs)
        await rag_hnsw.build_index(m=16, ef_construction=64, metric=DistanceMetric.COSINE)
        stats = await rag_hnsw.get_stats()
        assert stats["index_built"], "HNSW index should be marked as built"

    async def test_build_hnsw_l2(self, db_schema, embeddings, connection_string, small_docs):
        rag = _make_rag(IndexType.HNSW, "test_hnsw_l2", db_schema, embeddings, connection_string)
        await rag.initialize(overwrite_existing=True)
        docs, _ = small_docs
        await rag.add_documents(docs)
        await rag.build_index(metric=DistanceMetric.L2)
        stats = await rag.get_stats()
        assert stats["index_built"]
        await rag.close()

    async def test_reindex_hnsw(self, rag_hnsw, small_docs):
        docs, _ = small_docs
        await rag_hnsw.add_documents(docs)
        await rag_hnsw.build_index()
        await rag_hnsw.areindex()  # should not raise

    async def test_drop_vector_index(self, rag_hnsw, small_docs):
        docs, _ = small_docs
        await rag_hnsw.add_documents(docs)
        await rag_hnsw.build_index()
        await rag_hnsw.adrop_vector_index()
        # After dropping, index_built should be False
        stats = await rag_hnsw.get_stats()
        assert not stats["index_built"]

    async def test_vacuum_analyze(self, rag_hnsw, small_docs):
        docs, _ = small_docs
        await rag_hnsw.add_documents(docs)
        await rag_hnsw.vacuum_analyze(full=False)  # should not raise


# ---------------------------------------------------------------------------
# IVFFlat index
# ---------------------------------------------------------------------------

class TestIVFFlatIndex:
    async def test_build_ivfflat(self, db_schema, embeddings, connection_string, medium_docs):
        rag = _make_rag(IndexType.IVFFLAT, "test_ivf_col", db_schema, embeddings, connection_string)
        await rag.initialize(overwrite_existing=True)
        docs, _ = medium_docs
        await rag.add_documents(docs)
        await rag.build_index(lists=10)
        stats = await rag.get_stats()
        assert stats["index_built"]
        await rag.close()


# ---------------------------------------------------------------------------
# DiskANN index
# ---------------------------------------------------------------------------

class TestDiskANNIndex:
    async def test_build_diskann(self, db_schema, embeddings, connection_string, docs_and_labels):
        """DiskANN requires vectorscale — skipped if absent."""
        from pgvectordb import ExtensionManager
        from sqlalchemy.ext.asyncio import create_async_engine

        engine = create_async_engine(connection_string, pool_pre_ping=True)
        mgr = ExtensionManager(engine)
        await mgr.check_extensions()
        await engine.dispose()

        if not mgr.has_vectorscale:
            pytest.skip("vectorscale not installed — skipping DiskANN test")

        rag = _make_rag(IndexType.DISKANN, "test_diskann_col", db_schema, embeddings, connection_string)
        await rag.initialize(overwrite_existing=True)
        docs, labels = docs_and_labels
        await rag.add_documents(docs, labels=labels)
        await rag.build_index(
            num_neighbors=50,
            search_list_size=100,
            storage_layout=StorageLayout.MEMORY_OPTIMIZED,
            include_labels=True,
        )
        stats = await rag.get_stats()
        assert stats["index_built"]

        # Label filtering
        res = await rag.semantic_search("programming", k=10, label_filter=[1, 2])
        assert isinstance(res, list)
        await rag.close()

    async def test_diskann_build_params_api(self, rag_hnsw):
        """set_diskann_build_params API contract — must not raise."""
        await rag_hnsw.set_diskann_build_params(
            force_parallel_workers=2,
            min_vectors_for_parallel_build=100,
        )


# ---------------------------------------------------------------------------
# BM25 index
# ---------------------------------------------------------------------------

class TestBM25Index:
    async def test_build_bm25(self, db_schema, embeddings, connection_string, small_docs):
        from pgvectordb import ExtensionManager
        from sqlalchemy.ext.asyncio import create_async_engine

        engine = create_async_engine(connection_string, pool_pre_ping=True)
        mgr = ExtensionManager(engine)
        await mgr.check_extensions()
        await engine.dispose()

        if not mgr.has_pg_textsearch:
            pytest.skip("pg_textsearch not installed — skipping BM25 index test")

        rag = _make_rag(IndexType.HNSW, "test_bm25_idx", db_schema, embeddings, connection_string)
        await rag.initialize(overwrite_existing=True)
        docs, _ = small_docs
        await rag.add_documents(docs)
        await rag.build_bm25_index(text_config="english", k1=1.2, b=0.75)
        await rag.close()


# ---------------------------------------------------------------------------
# Query parameter tuning
# ---------------------------------------------------------------------------

class TestQueryParams:
    async def test_set_ef_search(self, rag_hnsw, small_docs):
        docs, _ = small_docs
        await rag_hnsw.add_documents(docs)
        await rag_hnsw.build_index()
        await rag_hnsw.set_query_params(ef_search=100)
        res = await rag_hnsw.semantic_search("test", k=5)
        assert len(res) <= 5

    async def test_set_iterative_scan_relaxed(self, rag_hnsw, small_docs):
        docs, _ = small_docs
        await rag_hnsw.add_documents(docs)
        await rag_hnsw.build_index()
        await rag_hnsw.set_query_params(
            ef_search=100,
            iterative_scan="relaxed_order",
            max_scan_tuples=1000,
            scan_mem_multiplier=2,
        )
        res = await rag_hnsw.semantic_search("test", k=5)
        assert isinstance(res, list)

    async def test_set_iterative_scan_strict(self, rag_hnsw, small_docs):
        docs, _ = small_docs
        await rag_hnsw.add_documents(docs)
        await rag_hnsw.build_index()
        await rag_hnsw.set_query_params(iterative_scan="strict_order")
        res = await rag_hnsw.semantic_search("test", k=5)
        assert isinstance(res, list)

    async def test_maintenance_work_mem(self, rag_hnsw):
        """set_maintenance_work_mem must not raise for valid values."""
        await rag_hnsw.set_maintenance_work_mem("512MB")


# ---------------------------------------------------------------------------
# build_index_concurrent
# ---------------------------------------------------------------------------

class TestConcurrentIndex:
    async def test_build_concurrent(self, db_schema, embeddings, connection_string, small_docs):
        rag = _make_rag(IndexType.HNSW, "test_concurrent_idx", db_schema, embeddings, connection_string)
        await rag.initialize(overwrite_existing=True)
        docs, _ = small_docs
        await rag.add_documents(docs)
        # build_index_concurrent uses `distance=` not `metric=`
        await rag.build_index_concurrent(distance=DistanceMetric.COSINE)
        stats = await rag.get_stats()
        assert stats["index_built"]
        await rag.close()
