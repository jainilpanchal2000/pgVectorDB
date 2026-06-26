"""
Test SearchMethod enum (TDD - Red Phase)

Tests for the new SearchMethod enum to replace string-based search types.
"""


def test_search_method_enum_exists():
    """SearchMethod enum should exist and be importable."""
    from pgvectordb.base import SearchMethod

    # Should have these members
    assert hasattr(SearchMethod, "SEMANTIC")
    assert hasattr(SearchMethod, "KEYWORD")
    assert hasattr(SearchMethod, "HYBRID")
    assert hasattr(SearchMethod, "TRIGRAM")
    assert hasattr(SearchMethod, "METADATA_FILTER")


def test_search_method_enum_is_str_enum():
    """SearchMethod should be a str Enum for backward compat."""
    from pgvectordb.base import SearchMethod

    assert isinstance(SearchMethod.SEMANTIC, str)
    assert SearchMethod.SEMANTIC == "semantic"
    assert SearchMethod.KEYWORD == "keyword"
    assert SearchMethod.HYBRID == "hybrid"


def test_search_method_from_string():
    """Should be able to create from string."""
    from pgvectordb.base import SearchMethod

    # Direct string comparison works because it's a str enum
    assert SearchMethod.SEMANTIC == "semantic"
    assert SearchMethod("semantic") == SearchMethod.SEMANTIC
    assert SearchMethod("keyword") == SearchMethod.KEYWORD


def test_search_method_available_in_init():
    """SearchMethod should be exported from pgvectordb package."""
    from pgvectordb import SearchMethod

    assert SearchMethod.SEMANTIC.value == "semantic"
