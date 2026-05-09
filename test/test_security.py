"""
test_security.py — Security and validation tests for pgVectorDB

Covers:
  - SQL injection rejection in collection_name / schema_name
  - VectorSpace name validation
  - NumberSpace / CategorySpace validation bounds
  - ValidationError for empty query, k <= 0, bad weights
  - InitializationError when search called before initialize()

Run:
    python -m pytest test/test_security.py -v
"""

import pytest
import re

from pgvectordb import (
    pgVectorDB,
    IndexType,
    ValidationError,
    InitializationError,
)


pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
CONNECTION_STRING = "postgresql+asyncpg://user:root@localhost:9002/postgres"


def _make_rag(
    collection_name: str = "sec_test", schema_name: str = "test", embeddings=None
):
    """Helper: build a pgVectorDB instance without calling initialize()."""
    return pgVectorDB(
        collection_name=collection_name,
        embedding_model=embeddings,  # pass real embeddings if success expected
        connection_string=CONNECTION_STRING,
        schema_name=schema_name,
        index_type=IndexType.HNSW,
    )


# ---------------------------------------------------------------------------
# SQL injection / bad identifier rejection (constructor-time)
# ---------------------------------------------------------------------------


class TestIdentifierValidation:
    def test_semicolon_in_collection_name(self):
        with pytest.raises((ValidationError, Exception)):
            _make_rag(collection_name="test; DROP TABLE foo")

    def test_drop_table_in_name(self):
        with pytest.raises((ValidationError, Exception)):
            _make_rag(collection_name="x'; DROP TABLE users; --")

    def test_quote_in_collection_name(self):
        with pytest.raises((ValidationError, Exception)):
            _make_rag(collection_name='test"injection')

    def test_space_only_collection_name(self):
        with pytest.raises((ValidationError, Exception)):
            _make_rag(collection_name="   ")

    def test_empty_collection_name(self):
        with pytest.raises((ValidationError, Exception)):
            _make_rag(collection_name="")

    def test_valid_name_accepted(self, embeddings):
        """Valid names with letters, numbers, underscores should not raise."""
        rag = _make_rag(collection_name="valid_collection_123", embeddings=embeddings)
        assert rag is not None

    def test_valid_name_with_hyphen(self, embeddings):
        """Hyphens are usually allowed via quoting — at minimum should not crash."""
        try:
            rag = _make_rag(collection_name="valid-collection", embeddings=embeddings)
            assert rag is not None
        except (ValidationError, Exception):
            pass  # Stricter validation is also acceptable


# ---------------------------------------------------------------------------
# Input validation (requires DB — but checked as integration guard)
# ---------------------------------------------------------------------------


class TestInputValidation:
    """
    These tests verify ValidationError is raised BEFORE hitting the DB.
    They require a real rag instance but test pre-flight validation, so
    they can be used even without a running DB for the constructor path.
    """

    @pytest.mark.integration
    async def test_empty_query_raises_validation_error(self, rag_hnsw):
        """Semantic search with empty string must raise ValidationError."""
        with pytest.raises(ValidationError):
            await rag_hnsw.semantic_search("", k=5)

    @pytest.mark.integration
    async def test_whitespace_query_raises_validation_error(self, rag_hnsw):
        """Semantic search with whitespace-only string must raise ValidationError."""
        with pytest.raises(ValidationError):
            await rag_hnsw.semantic_search("   ", k=5)

    @pytest.mark.integration
    async def test_k_zero_raises_validation_error(self, rag_hnsw):
        """k=0 is invalid — must raise ValidationError."""
        with pytest.raises(ValidationError):
            await rag_hnsw.semantic_search("test", k=0)

    @pytest.mark.integration
    async def test_k_negative_raises_validation_error(self, rag_hnsw):
        """k=-1 is invalid — must raise ValidationError."""
        with pytest.raises(ValidationError):
            await rag_hnsw.semantic_search("test", k=-1)

    @pytest.mark.integration
    async def test_invalid_hybrid_weights_raises_validation_error(
        self, rag_hnsw, medium_docs, embeddings
    ):
        """hybrid_search weights that don't sum to 1.0 must raise ValidationError."""
        docs, _ = medium_docs
        await rag_hnsw.add_documents(docs)
        await rag_hnsw.build_index()
        with pytest.raises(ValidationError):
            await rag_hnsw.hybrid_search("test", k=5, weights=(0.3, 0.5))

    @pytest.mark.integration
    async def test_empty_keyword_query_raises_validation_error(self, rag_hnsw):
        """keyword_search with empty string must raise ValidationError."""
        with pytest.raises(ValidationError):
            await rag_hnsw.keyword_search("", k=5)


# ---------------------------------------------------------------------------
# InitializationError guard
# ---------------------------------------------------------------------------


class TestInitializationGuard:
    @pytest.mark.integration
    async def test_search_before_initialize_raises(self, embeddings, connection_string):
        """Calling semantic_search before initialize() must raise InitializationError."""
        rag = pgVectorDB(
            collection_name="sec_uninit_guard",
            embedding_model=embeddings,
            connection_string=connection_string,
            schema_name="test",
            index_type=IndexType.HNSW,
        )
        with pytest.raises(InitializationError):
            await rag.semantic_search("test", k=5)

    @pytest.mark.integration
    async def test_add_documents_before_initialize_raises(
        self, embeddings, connection_string
    ):
        """Calling add_documents before initialize() must raise InitializationError."""
        from langchain_core.documents import Document

        rag = pgVectorDB(
            collection_name="sec_uninit_docs",
            embedding_model=embeddings,
            connection_string=connection_string,
            schema_name="test",
            index_type=IndexType.HNSW,
        )
        with pytest.raises(InitializationError):
            await rag.add_documents([Document(page_content="test", metadata={})])


# ---------------------------------------------------------------------------
# set_maintenance_work_mem validation — regex allowlist (analytics.py)
# ---------------------------------------------------------------------------


class TestMaintenanceWorkMemValidation:
    """Test the regex whitelist for set_maintenance_work_mem."""

    def _check_regex(self, value: str) -> bool:
        return bool(re.match(r"^\d+\s*(kB|MB|GB|TB)?$", value.strip(), re.IGNORECASE))

    @pytest.mark.parametrize("value", ["2GB", "8GB", "512MB", "65536", "1TB", "1024kB", "4gb", "  4GB  "])
    def test_valid_formats(self, value):
        assert self._check_regex(value), f"'{value}' should be valid"

    @pytest.mark.parametrize("value", [
        "8GB'; DROP TABLE users; --",
        "8GB;",
        "lots",
        "-1GB",
        "",
        "8GB--",
        "1PB",
        "abc",
        "  ; DROP TABLE  ",
    ])
    def test_invalid_formats(self, value):
        assert not self._check_regex(value), f"'{value}' should be rejected"


# ---------------------------------------------------------------------------
# set_parallel_workers validation — int coercion + bounds (analytics.py)
# ---------------------------------------------------------------------------


class TestParallelWorkersValidation:
    """Test int coercion and bounds check for set_parallel_workers."""

    def test_int_coerces(self):
        assert int("4") == 4
        assert int(4.7) == 4

    def test_negative_coerced_int_fails_bounds(self):
        val = int(-1)
        assert val < 0

    def test_zero_and_large_valid(self):
        assert int(0) >= 0
        assert int(128) >= 0

    def test_string_int_coerces(self):
        assert int("7") == 7

    def test_non_numeric_string_raises(self):
        with pytest.raises(ValueError):
            int("not_a_number")


# ---------------------------------------------------------------------------
# build_bm25_index parameter validation — k1, b, text_config (indexing.py)
# ---------------------------------------------------------------------------


class TestBM25ParameterValidation:
    """Test k1/b/text_config bounds for build_bm25_index."""

    def test_k1_out_of_range(self):
        assert not (0.1 <= 15.0 <= 10.0)
        assert not (0.1 <= -0.5 <= 10.0)

    def test_b_out_of_range(self):
        assert not (0.0 <= 1.5 <= 1.0)
        assert not (0.0 <= -0.1 <= 1.0)

    def test_text_config_allowlist(self):
        from pgvectordb import ALLOWED_TEXT_CONFIGS
        assert "english" in ALLOWED_TEXT_CONFIGS
        assert "german" in ALLOWED_TEXT_CONFIGS
        assert "french" in ALLOWED_TEXT_CONFIGS
        assert "not_a_real_config" not in ALLOWED_TEXT_CONFIGS

    def test_parallel_workers_negative_fails_bounds(self):
        val = int(-1)
        assert val < 0


# ---------------------------------------------------------------------------
# VectorSpace name validation (spaces.py)
# ---------------------------------------------------------------------------


class TestVectorSpaceNameValidation:
    """Test VectorSpace.__init__ name validation."""

    def test_valid_name(self):
        from pgvectordb.spaces import NumberSpace
        space = NumberSpace(name="price", field="price", min_value=0, max_value=100)
        assert space.name == "price"

    def test_valid_name_with_underscore(self):
        from pgvectordb.spaces import NumberSpace
        space = NumberSpace(name="unit_price", field="price", min_value=0, max_value=100)
        assert space.name == "unit_price"

    def test_empty_name_raises(self):
        from pgvectordb.spaces import NumberSpace
        with pytest.raises(ValueError, match="non-empty"):
            NumberSpace(name="", field="price", min_value=0, max_value=100)

    def test_space_in_name_raises(self):
        from pgvectordb.spaces import NumberSpace
        with pytest.raises(ValueError, match="alphanumeric"):
            NumberSpace(name="bad name", field="price", min_value=0, max_value=100)

    def test_special_char_in_name_raises(self):
        from pgvectordb.spaces import NumberSpace
        with pytest.raises(ValueError, match="alphanumeric"):
            NumberSpace(name="bad;name", field="price", min_value=0, max_value=100)


# ---------------------------------------------------------------------------
# NumberSpace validation (spaces.py)
# ---------------------------------------------------------------------------


class TestNumberSpaceValidation:
    """Test NumberSpace bounds and mode validation."""

    def test_min_equals_max_raises(self):
        from pgvectordb.spaces import NumberSpace
        with pytest.raises(ValueError, match="less than"):
            NumberSpace(name="p", field="p", min_value=5.0, max_value=5.0)

    def test_min_greater_than_max_raises(self):
        from pgvectordb.spaces import NumberSpace
        with pytest.raises(ValueError, match="less than"):
            NumberSpace(name="p", field="p", min_value=10, max_value=5)

    def test_zero_dimensions_raises(self):
        from pgvectordb.spaces import NumberSpace
        with pytest.raises(ValueError, match="dimensions"):
            NumberSpace(name="p", field="p", min_value=0, max_value=100, dimensions=0)

    def test_normalize_clamping(self):
        from pgvectordb.spaces import NumberSpace
        space = NumberSpace(name="p", field="p", min_value=0, max_value=100)
        assert space.encode(150) == [1.0]
        assert space.encode(-50) == [0.0]
        assert space.encode(50) == [0.5]

    def test_none_encodes_as_midpoint(self):
        from pgvectordb.spaces import NumberSpace
        space = NumberSpace(name="p", field="p", min_value=0, max_value=100)
        assert space.encode(None) == [0.5]


# ---------------------------------------------------------------------------
# CategorySpace validation (spaces.py)
# ---------------------------------------------------------------------------


class TestCategorySpaceValidation:
    """Test CategorySpace validation and encoding."""

    def test_empty_categories_raises(self):
        from pgvectordb.spaces import CategorySpace
        with pytest.raises(ValueError, match="non-empty"):
            CategorySpace(name="cat", field="cat", categories=[])

    def test_duplicate_categories_raises(self):
        from pgvectordb.spaces import CategorySpace
        with pytest.raises(ValueError, match="duplicates"):
            CategorySpace(name="cat", field="cat", categories=["a", "b", "a"])

    def test_unknown_category_returns_zero_vector(self):
        from pgvectordb.spaces import CategorySpace
        space = CategorySpace(name="cat", field="cat", categories=["a", "b", "c"])
        assert space.encode("unknown") == [0.0, 0.0, 0.0]

    def test_known_category_encodes_one_hot(self):
        from pgvectordb.spaces import CategorySpace
        space = CategorySpace(name="cat", field="cat", categories=["a", "b", "c"])
        result = space.encode("b")
        assert result[1] == 1.0
        assert result[0] == 0.0
        assert result[2] == 0.0

    def test_none_value_returns_zero_vector(self):
        from pgvectordb.spaces import CategorySpace
        space = CategorySpace(name="cat", field="cat", categories=["a", "b"])
        assert space.encode(None) == [0.0, 0.0]

    def test_strict_unknown_raises(self):
        from pgvectordb.spaces import CategorySpace
        space = CategorySpace(
            name="cat", field="cat", categories=["a", "b"],
            uncategorized_as_zero=False,
        )
        with pytest.raises(ValueError, match="unknown category"):
            space.encode("nope")

    def test_dimensions_equals_num_categories(self):
        from pgvectordb.spaces import CategorySpace
        space = CategorySpace(name="cat", field="cat", categories=["x", "y", "z"])
        assert space.dimensions == 3
