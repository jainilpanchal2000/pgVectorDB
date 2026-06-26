import pytest

from pgvectordb import ValidationError
from pgvectordb.query.filters import MetadataFilterCompiler, compile_metadata_filter


@pytest.fixture
def compiler():
    return MetadataFilterCompiler()


def test_empty_filter_matches_all():
    assert compile_metadata_filter({}) == ("1=1", {})


@pytest.mark.parametrize(
    ("operator", "sql_operator"),
    [
        ("$eq", "="),
        ("$ne", "!="),
        ("$lt", "<"),
        ("$lte", "<="),
        ("$gt", ">"),
        ("$gte", ">="),
    ],
)
def test_comparison_operators(compiler, operator, sql_operator):
    clause, params = compiler.build({"year": {operator: 2024}})

    assert clause == f"(langchain_metadata->>'year')::numeric {sql_operator} :param_0"
    assert params == {"param_0": 2024}


def test_default_equality_for_scalar_value(compiler):
    clause, params = compiler.build({"category": "tech"})

    assert clause == "langchain_metadata->>'category' = :param_0"
    assert params == {"param_0": "tech"}


def test_in_and_nin_operators(compiler):
    in_clause, in_params = compiler.build({"category": {"$in": ["tech", "science"]}})
    nin_clause, nin_params = compiler.build({"year": {"$nin": [2020, 2021]}})

    assert in_clause == "langchain_metadata->>'category' = ANY(:param_0)"
    assert in_params == {"param_0": ("tech", "science")}
    assert nin_clause == "langchain_metadata->>'year' != ALL(:param_0)"
    assert nin_params == {"param_0": ("2020", "2021")}


def test_between_operator(compiler):
    clause, params = compiler.build({"year": {"$between": [2020, 2024]}})

    assert clause == "(langchain_metadata->>'year')::numeric BETWEEN :param_0 AND :param_1"
    assert params == {"param_0": 2020, "param_1": 2024}


def test_exists_like_and_ilike_operators(compiler):
    exists_clause, exists_params = compiler.build({"category": {"$exists": True}})
    like_clause, like_params = compiler.build({"title": {"$like": "%Vector%"}})
    ilike_clause, ilike_params = compiler.build({"title": {"$ilike": "%vector%"}})

    assert exists_clause == "langchain_metadata->>'category' IS NOT NULL"
    assert exists_params == {}
    assert like_clause == "langchain_metadata->>'title' LIKE :param_0"
    assert like_params == {"param_0": "%Vector%"}
    assert ilike_clause == "langchain_metadata->>'title' ILIKE :param_0"
    assert ilike_params == {"param_0": "%vector%"}


def test_logical_and_or(compiler):
    clause, params = compiler.build(
        {
            "$and": [
                {"category": "tech"},
                {"$or": [{"year": {"$gte": 2023}}, {"status": "active"}]},
            ]
        }
    )

    assert clause == (
        "(langchain_metadata->>'category' = :param_0) AND "
        "((((langchain_metadata->>'year')::numeric >= :param_1) OR "
        "(langchain_metadata->>'status' = :param_2)))"
    )
    assert params == {"param_0": "tech", "param_1": 2023, "param_2": "active"}


def test_invalid_metadata_key_rejected(compiler):
    with pytest.raises(ValidationError):
        compiler.build({"bad-key": "value"})


def test_invalid_operator_rejected(compiler):
    with pytest.raises(ValidationError):
        compiler.build({"category": {"$contains": "tech"}})


def test_invalid_between_value_rejected(compiler):
    with pytest.raises(ValidationError):
        compiler.build({"year": {"$between": [2020]}})
