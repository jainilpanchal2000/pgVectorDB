"""
test_base.py — Unit tests for pgvectordb/base.py

Tests enums, custom exceptions, and constants.
No database required — all tests are pure Python unit tests.

Run:
    .venv\\Scripts\\python -m pytest test/test_base.py -v
"""

import pytest

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class TestIndexType:
    def test_hnsw_value(self):
        from pgvectordb import IndexType

        assert IndexType.HNSW.value == "hnsw"

    def test_ivfflat_value(self):
        from pgvectordb import IndexType

        assert IndexType.IVFFLAT.value == "ivfflat"

    def test_diskann_value(self):
        from pgvectordb import IndexType

        assert IndexType.DISKANN.value == "diskann"

    def test_all_members(self):
        from pgvectordb import IndexType

        assert set(IndexType) == {IndexType.HNSW, IndexType.IVFFLAT, IndexType.DISKANN}


class TestKeywordSearchType:
    def test_fts_value(self):
        from pgvectordb import KeywordSearchType

        assert KeywordSearchType.FTS.value == "fts"

    def test_bm25_value(self):
        from pgvectordb import KeywordSearchType

        assert KeywordSearchType.BM25.value == "bm25"


class TestStorageLayout:
    def test_memory_optimized(self):
        from pgvectordb import StorageLayout

        assert StorageLayout.MEMORY_OPTIMIZED.value == "memory_optimized"

    def test_plain(self):
        from pgvectordb import StorageLayout

        assert StorageLayout.PLAIN.value == "plain"


class TestDistanceMetric:
    def test_cosine(self):
        from pgvectordb import DistanceMetric

        assert DistanceMetric.COSINE.value == "cosine"

    def test_l2(self):
        from pgvectordb import DistanceMetric

        assert DistanceMetric.L2.value == "l2"

    def test_inner_product(self):
        from pgvectordb import DistanceMetric

        assert DistanceMetric.INNER_PRODUCT.value == "inner_product"

    def test_l1(self):
        from pgvectordb import DistanceMetric

        assert DistanceMetric.L1.value == "l1"

    def test_hamming(self):
        from pgvectordb import DistanceMetric

        assert DistanceMetric.HAMMING.value == "hamming"

    def test_jaccard(self):
        from pgvectordb import DistanceMetric

        assert DistanceMetric.JACCARD.value == "jaccard"


class TestVectorPrecision:
    def test_float32(self):
        from pgvectordb import VectorPrecision

        assert VectorPrecision.FLOAT32.value == "float32"

    def test_float16(self):
        from pgvectordb import VectorPrecision

        assert VectorPrecision.FLOAT16.value == "float16"

    def test_binary(self):
        from pgvectordb import VectorPrecision

        assert VectorPrecision.BINARY.value == "binary"


class TestIterativeScanMode:
    def test_off(self):
        from pgvectordb import IterativeScanMode

        assert IterativeScanMode.OFF.value == "off"

    def test_strict_order(self):
        from pgvectordb import IterativeScanMode

        assert IterativeScanMode.STRICT_ORDER.value == "strict_order"

    def test_relaxed_order(self):
        from pgvectordb import IterativeScanMode

        assert IterativeScanMode.RELAXED_ORDER.value == "relaxed_order"


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class TestExceptionHierarchy:
    def test_retrieval_system_error_is_base(self):
        from pgvectordb import (
            DatabaseError,
            InitializationError,
            RateLimitError,
            RetrievalSystemError,
            ValidationError,
        )

        assert issubclass(InitializationError, RetrievalSystemError)
        assert issubclass(ValidationError, RetrievalSystemError)
        assert issubclass(DatabaseError, RetrievalSystemError)
        assert issubclass(RateLimitError, RetrievalSystemError)

    def test_all_exceptions_are_exceptions(self):
        from pgvectordb import (
            DatabaseError,
            InitializationError,
            RateLimitError,
            RetrievalSystemError,
            ValidationError,
        )

        for exc_cls in [
            RetrievalSystemError,
            InitializationError,
            ValidationError,
            DatabaseError,
            RateLimitError,
        ]:
            assert issubclass(exc_cls, Exception)

    def test_raise_validation_error(self):
        from pgvectordb import ValidationError

        with pytest.raises(ValidationError, match="bad input"):
            raise ValidationError("bad input")

    def test_raise_initialization_error(self):
        from pgvectordb import InitializationError

        with pytest.raises(InitializationError):
            raise InitializationError("not ready")

    def test_raise_database_error(self):
        from pgvectordb import DatabaseError

        with pytest.raises(DatabaseError):
            raise DatabaseError("db fail")

    def test_raise_rate_limit_error(self):
        from pgvectordb import RateLimitError

        with pytest.raises(RateLimitError):
            raise RateLimitError("rate limited")

    def test_catch_as_base(self):
        from pgvectordb import RetrievalSystemError, ValidationError

        with pytest.raises(RetrievalSystemError):
            raise ValidationError("caught as base")


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------


class TestConstants:
    def test_allowed_text_configs_contains_english(self):
        from pgvectordb import ALLOWED_TEXT_CONFIGS

        assert "english" in ALLOWED_TEXT_CONFIGS

    def test_allowed_text_configs_is_frozenset(self):
        from pgvectordb import ALLOWED_TEXT_CONFIGS

        assert isinstance(ALLOWED_TEXT_CONFIGS, frozenset)

    def test_valid_query_params_contains_ef_search(self):
        from pgvectordb import VALID_QUERY_PARAMS

        assert "hnsw.ef_search" in VALID_QUERY_PARAMS

    def test_valid_query_params_contains_ivfflat_probes(self):
        from pgvectordb import VALID_QUERY_PARAMS

        assert "ivfflat.probes" in VALID_QUERY_PARAMS

    def test_valid_query_params_is_frozenset(self):
        from pgvectordb import VALID_QUERY_PARAMS

        assert isinstance(VALID_QUERY_PARAMS, frozenset)

    def test_extension_requirements_keys(self):
        from pgvectordb import EXTENSION_REQUIREMENTS

        # EXTENSION_REQUIREMENTS maps feature names -> extension names
        # Check that known feature names are present
        assert any("DiskANN" in k for k in EXTENSION_REQUIREMENTS), (
            "Should contain a DiskANN feature key"
        )
        assert any("BM25" in k for k in EXTENSION_REQUIREMENTS), (
            "Should contain a BM25 feature key"
        )

    def test_extension_requirements_values_are_extension_names(self):
        from pgvectordb import EXTENSION_REQUIREMENTS

        valid_extensions = {"vectorscale", "pg_textsearch", "pgvector"}
        for feature, ext in EXTENSION_REQUIREMENTS.items():
            assert ext in valid_extensions, (
                f"Feature '{feature}' maps to unknown extension '{ext}'"
            )

    def test_extension_requirements_diskann_needs_vectorscale(self):
        from pgvectordb import EXTENSION_REQUIREMENTS

        diskann_features = {
            k: v for k, v in EXTENSION_REQUIREMENTS.items() if "DiskANN" in k
        }
        assert diskann_features, "Should have at least one DiskANN feature"
        for k, v in diskann_features.items():
            assert v == "vectorscale", (
                f"DiskANN feature '{k}' should require vectorscale"
            )

    def test_extension_requirements_bm25_needs_pg_textsearch(self):
        from pgvectordb import EXTENSION_REQUIREMENTS

        bm25_features = {k: v for k, v in EXTENSION_REQUIREMENTS.items() if "BM25" in k}
        assert bm25_features, "Should have at least one BM25 feature"
        for k, v in bm25_features.items():
            assert v == "pg_textsearch", (
                f"BM25 feature '{k}' should require pg_textsearch"
            )


# ---------------------------------------------------------------------------
# Public API surface (__all__)
# ---------------------------------------------------------------------------


class TestPublicAPI:
    def test_pgvectordb_importable(self):
        from pgvectordb import pgVectorDB

        assert pgVectorDB is not None

    def test_config_importable(self):
        from pgvectordb import Config

        assert Config is not None

    def test_metrics_importable(self):
        from pgvectordb import RAGEvaluator

        assert RAGEvaluator is not None

    def test_spaces_importable(self):
        from pgvectordb import CategorySpace, NumberSpace, RecencySpace, TextSpace

        assert TextSpace is not None
        assert NumberSpace is not None
        assert CategorySpace is not None
        assert RecencySpace is not None

    def test_rerankers_importable(self):
        from pgvectordb import create_reranker

        assert create_reranker is not None

    def test_version_string(self):
        import pgvectordb

        assert hasattr(pgvectordb, "__version__")
        assert isinstance(pgvectordb.__version__, str)
        assert pgvectordb.__version__.startswith("0.")
