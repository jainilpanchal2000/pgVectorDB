"""
test_extensions.py — Integration tests for pgvectordb/extensions.py

Tests the ExtensionManager: checking extension availability, feature matrix,
and graceful-degradation require() methods.

Run:
    .venv\\Scripts\\python -m pytest test/test_extensions.py -v
"""

import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import create_async_engine

from pgvectordb import ExtensionManager

pytestmark = pytest.mark.integration


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest_asyncio.fixture
async def ext_manager(connection_string):
    engine = create_async_engine(connection_string, pool_pre_ping=True)
    mgr = ExtensionManager(engine)
    yield mgr
    await engine.dispose()


@pytest_asyncio.fixture
async def checked_manager(ext_manager):
    """ExtensionManager with check_extensions() already called."""
    await ext_manager.check_extensions()
    return ext_manager


# ---------------------------------------------------------------------------
# check_extensions()
# ---------------------------------------------------------------------------


class TestCheckExtensions:
    async def test_returns_dict(self, ext_manager):
        status = await ext_manager.check_extensions()
        assert isinstance(status, dict)

    async def test_pgvector_always_present(self, ext_manager):
        status = await ext_manager.check_extensions()
        assert "pgvector" in status, "pgvector must always appear in extension status"

    async def test_vectorscale_in_status(self, ext_manager):
        status = await ext_manager.check_extensions()
        assert "vectorscale" in status

    async def test_pg_textsearch_in_status(self, ext_manager):
        status = await ext_manager.check_extensions()
        assert "pg_textsearch" in status

    async def test_pgvector_available(self, ext_manager):
        """pgvector is a required dependency — must be installed."""
        await ext_manager.check_extensions()
        assert ext_manager.has_pgvector, "pgvector extension should be available"


# ---------------------------------------------------------------------------
# has_* properties
# ---------------------------------------------------------------------------


class TestHasProperties:
    async def test_has_pgvector_is_bool(self, checked_manager):
        assert isinstance(checked_manager.has_pgvector, bool)

    async def test_has_vectorscale_is_bool(self, checked_manager):
        assert isinstance(checked_manager.has_vectorscale, bool)

    async def test_has_pg_textsearch_is_bool(self, checked_manager):
        assert isinstance(checked_manager.has_pg_textsearch, bool)

    async def test_pgvector_version_if_available(self, checked_manager):
        if checked_manager.has_pgvector:
            assert checked_manager.pgvector_version is not None

    async def test_vectorscale_version_type(self, checked_manager):
        # version is either a string or None
        assert checked_manager.vectorscale_version is None or isinstance(
            checked_manager.vectorscale_version, str
        )

    async def test_pg_textsearch_version_type(self, checked_manager):
        assert checked_manager.pg_textsearch_version is None or isinstance(
            checked_manager.pg_textsearch_version, str
        )


# ---------------------------------------------------------------------------
# get_feature_availability()
# ---------------------------------------------------------------------------


class TestFeatureAvailability:
    async def test_returns_dict(self, checked_manager):
        features = checked_manager.get_feature_availability()
        assert isinstance(features, dict)

    async def test_hnsw_feature_present(self, checked_manager):
        features = checked_manager.get_feature_availability()
        assert "HNSW index" in features

    async def test_ivfflat_feature_present(self, checked_manager):
        features = checked_manager.get_feature_availability()
        assert "IVFFlat index" in features

    async def test_diskann_feature_present(self, checked_manager):
        features = checked_manager.get_feature_availability()
        assert "DiskANN index" in features

    async def test_bm25_feature_present(self, checked_manager):
        features = checked_manager.get_feature_availability()
        assert "BM25 search" in features

    async def test_feature_values_format(self, checked_manager):
        features = checked_manager.get_feature_availability()
        for key, val in features.items():
            assert isinstance(val, dict), (
                f"Feature '{key}' should be dict, got {type(val)}"
            )
            assert "available" in val
            assert "requires" in val
            assert "version" in val
            assert isinstance(val["available"], bool)


# ---------------------------------------------------------------------------
# require_*() methods
# ---------------------------------------------------------------------------


class TestRequireMethods:
    """Test require_*() methods - now no-ops for graceful degradation."""

    async def test_require_vectorscale_no_longer_raises(self, checked_manager):
        """require_vectorscale() is now a no-op for graceful degradation."""
        # Should NOT raise even when extension is absent
        checked_manager.require_vectorscale("test operation")
        # Method should complete without error

    async def test_require_pg_textsearch_no_longer_raises(self, checked_manager):
        """require_pg_textsearch() is now a no-op for graceful degradation."""
        # Should NOT raise even when extension is absent
        checked_manager.require_pg_textsearch("test operation")
        # Method should complete without error
