"""
Test Scalar Index creation (TDD - Red Phase)

Tests for BTree and Bitmap scalar indexes on metadata.
"""
import pytest


class TestBTreeIndex:
    """Test BTree index creation for metadata columns."""

    @pytest.mark.asyncio
    async def test_create_scalar_index_exists(self, rag_hnsw):
        """create_scalar_index method should exist."""
        assert hasattr(rag_hnsw, "create_scalar_index")

    @pytest.mark.asyncio
    async def test_create_btree_index_creates_index(self, db_with_docs):
        """create_scalar_index with btree should create index."""
        await db_with_docs.create_scalar_index("category", index_type="btree")

        # Verify index was created
        stats = await db_with_docs.get_index_stats()
        index_names = [idx["name"] for idx in stats.get("indexes", [])]
        assert any("category" in name.lower() or "btree" in name.lower() for name in index_names)

    @pytest.mark.asyncio
    async def test_create_btree_index_numeric_column(self, db_with_docs):
        """BTree index works on numeric columns."""
        await db_with_docs.create_scalar_index("year", index_type="btree")
        # Should not raise


class TestBitmapIndex:
    """Test Bitmap index creation for low-cardinality columns."""

    @pytest.mark.asyncio
    async def test_create_bitmap_index_exists(self, rag_hnsw):
        """create_scalar_index should support bitmap."""
        assert hasattr(rag_hnsw, "create_scalar_index")

    @pytest.mark.asyncio
    async def test_create_bitmap_index_creates_gin(self, db_with_docs):
        """Bitmap index falls back to GIN in PostgreSQL."""
        await db_with_docs.create_scalar_index("category", index_type="bitmap")

        # Should create some form of index (GIN or partial BTREE)
        stats = await db_with_docs.get_index_stats()
        # Verify index was created
        assert len(stats.get("indexes", [])) > 0


class TestIndexUsage:
    """Test that created indexes are actually used."""

    @pytest.mark.asyncio
    async def test_index_speeds_up_filtered_query(self, db_with_docs):
        """Index should be used in explain plan for filtered queries."""
        # Create index
        await db_with_docs.create_scalar_index("category", index_type="btree")

        # Get explain plan for filtered query
        plan = (
            db_with_docs.query("test search")
            .where({"category": "ai"})
            .limit(5)
            .explain_plan()
        )

        # Should show index usage
        plan_str = str(plan).lower()
        assert "index" in plan_str

    @pytest.mark.asyncio
    async def test_range_query_with_btree_index(self, db_with_docs):
        """Range queries should benefit from BTree index."""
        # Create index on numeric field
        await db_with_docs.create_scalar_index("year", index_type="btree")

        # Range filter should use index
        results = await (
            db_with_docs.query("test search")
            .where({"year": {"$gte": 2020, "$lte": 2022}})
            .limit(10)
            .to_list()
        )

        # Verify results are in range
        for r in results:
            year = r["metadata"].get("year")
            if year is not None:
                assert 2020 <= year <= 2022
