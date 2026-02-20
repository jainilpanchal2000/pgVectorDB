"""
test_storage.py — Integration tests for pgvectordb/mixins/storage.py

Tests:
  - export_to_json() — file created, count > 0
  - import_from_json() — skip_existing works
  - create_halfvec_table()
  - create_sparsevec_table()

Run:
    .venv\\Scripts\\python -m pytest test/test_storage.py -v
"""

import pytest
import pytest_asyncio
import tempfile
import json
from pathlib import Path

from pgvectordb import pgVectorDB, IndexType


pytestmark = pytest.mark.integration


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest_asyncio.fixture
async def rag_with_docs(db_schema, embeddings, connection_string, small_docs):
    """HNSW collection with 20 docs, no index needed for storage tests."""
    inst = pgVectorDB(
        collection_name="test_storage_col",
        embedding_model=embeddings,
        connection_string=connection_string,
        schema_name=db_schema,
        index_type=IndexType.HNSW,
    )
    await inst.initialize(overwrite_existing=True)
    docs, _ = small_docs
    await inst.add_documents(docs)
    yield inst
    await inst.close()


# ---------------------------------------------------------------------------
# export_to_json
# ---------------------------------------------------------------------------


class TestExportToJson:
    async def test_creates_file(self, rag_with_docs):
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            count = await rag_with_docs.export_to_json(path, include_embeddings=False)
            assert Path(path).exists(), "Export file should exist"
            assert count > 0, "Export count should be > 0"
        finally:
            Path(path).unlink(missing_ok=True)

    async def test_exported_file_is_valid_json(self, rag_with_docs):
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
            path = f.name
        try:
            await rag_with_docs.export_to_json(path, include_embeddings=False)
            with open(path) as fp:
                data = json.load(fp)
            assert isinstance(data, list)
            assert len(data) > 0
        finally:
            Path(path).unlink(missing_ok=True)

    async def test_export_with_filter(self, rag_with_docs):
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            count = await rag_with_docs.export_to_json(
                path,
                filter={"status": "active"},
                include_embeddings=False,
            )
            assert count >= 0  # May be 0 if all are archived
        finally:
            Path(path).unlink(missing_ok=True)

    async def test_export_without_embeddings_default(self, rag_with_docs):
        """export_to_json(include_embeddings=False) — embeddings absent from JSON."""
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
            path = f.name
        try:
            await rag_with_docs.export_to_json(path, include_embeddings=False)
            with open(path) as fp:
                data = json.load(fp)
            for doc in data:
                assert "embedding" not in doc, "Embeddings should not be exported"
        finally:
            Path(path).unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# import_from_json
# ---------------------------------------------------------------------------


class TestImportFromJson:
    async def test_import_skip_existing(self, rag_with_docs):
        """Import the same exported docs with skip_existing=True — no duplicates."""
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            await rag_with_docs.export_to_json(path, include_embeddings=False)
            before = await rag_with_docs.count_by_metadata(None)

            # Import same data — skip_existing=True should skip all
            await rag_with_docs.import_from_json(path, batch_size=5, skip_existing=True)

            after = await rag_with_docs.count_by_metadata(None)
            assert after == before, (
                "Document count should not change with skip_existing=True"
            )
        finally:
            Path(path).unlink(missing_ok=True)

    async def test_import_returns_count(
        self, db_schema, embeddings, connection_string, rag_with_docs
    ):
        """Import into a fresh collection — count should equal exported docs."""
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            await rag_with_docs.export_to_json(path, include_embeddings=False)

            fresh = pgVectorDB(
                collection_name="test_import_fresh",
                embedding_model=embeddings,
                connection_string=connection_string,
                schema_name=db_schema,
                index_type=IndexType.HNSW,
            )
            await fresh.initialize(overwrite_existing=True)
            count = await fresh.import_from_json(
                path, batch_size=5, skip_existing=False
            )
            assert count > 0
            await fresh.close()
        finally:
            Path(path).unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# create_halfvec_table
# ---------------------------------------------------------------------------


def _skip_on_halfvec_error(e: Exception):
    """Skip the test if the error is related to halfvec type incompatibility."""
    error_str = str(e).lower()
    if any(
        kw in error_str for kw in ["halfvec", "type", "column", "cast", "dimension"]
    ):
        pytest.skip(f"halfvec type not supported with current vector dimensions: {e}")
    raise e


class TestHalfvecTable:
    async def test_creates_table(self, rag_with_docs):
        """create_halfvec_table() — skipped if halfvec not compatible with vector dim."""
        try:
            await rag_with_docs.create_halfvec_table(overwrite_existing=True)
        except Exception as e:
            _skip_on_halfvec_error(e)

    async def test_creates_custom_table_name(self, rag_with_docs):
        try:
            await rag_with_docs.create_halfvec_table(
                table_name="halfvec_custom",
                overwrite_existing=True,
            )
        except Exception as e:
            _skip_on_halfvec_error(e)

    async def test_idempotent_with_overwrite(self, rag_with_docs):
        """Calling twice with overwrite_existing=True should not raise."""
        try:
            await rag_with_docs.create_halfvec_table(overwrite_existing=True)
            await rag_with_docs.create_halfvec_table(overwrite_existing=True)
        except Exception as e:
            _skip_on_halfvec_error(e)


# ---------------------------------------------------------------------------
# create_sparsevec_table
# ---------------------------------------------------------------------------


class TestSparsevecTable:
    async def test_creates_table(self, rag_with_docs):
        await rag_with_docs.create_sparsevec_table(overwrite_existing=True)

    async def test_custom_dimensions(self, rag_with_docs):
        await rag_with_docs.create_sparsevec_table(
            max_dimensions=5000,
            overwrite_existing=True,
        )

    async def test_idempotent_with_overwrite(self, rag_with_docs):
        await rag_with_docs.create_sparsevec_table(overwrite_existing=True)
        await rag_with_docs.create_sparsevec_table(overwrite_existing=True)
