"""
test_search.py — Integration tests for pgvectordb/search.py (SearchMixin)

Tests:
  - All 11 search methods
  - All 13 filter operators
  - asimilarity_search_by_vector, asimilarity_search_with_score
  - BM25 positive-score regression fix
  - count_by_metadata with complex filters

Run:
    .venv\\Scripts\\python -m pytest test/test_search.py -v
"""

import pytest
import pytest_asyncio
from langchain_core.documents import Document

from pgvectordb import DistanceMetric, IndexType, KeywordSearchType, pgVectorDB

pytestmark = pytest.mark.integration


# ---------------------------------------------------------------------------
# Fixture: collection with 100 docs + HNSW index built
# ---------------------------------------------------------------------------


@pytest_asyncio.fixture
async def rag_with_data(db_schema, embeddings, connection_string, large_docs):
    """HNSW collection with 100 docs, metadata index and vector index built."""
    inst = pgVectorDB(
        collection_name="test_search_col",
        embedding_model=embeddings,
        connection_string=connection_string,
        schema_name=db_schema,
        index_type=IndexType.HNSW,
    )
    await inst.initialize(overwrite_existing=True)
    docs, _ = large_docs
    await inst.add_documents(docs)
    await inst.create_metadata_index(
        ["category", "language", "year", "priority", "status", "author"]
    )
    await inst.build_index(metric=DistanceMetric.COSINE)
    yield inst
    await inst.close()


# ---------------------------------------------------------------------------
# 1. Semantic Search
# ---------------------------------------------------------------------------


class TestSemanticSearch:
    async def test_returns_results(self, rag_with_data):
        res = await rag_with_data.semantic_search("programming Python", k=5)
        assert len(res) > 0

    async def test_respects_k(self, rag_with_data):
        res = await rag_with_data.semantic_search("database", k=3)
        assert len(res) <= 3

    async def test_result_structure(self, rag_with_data):
        res = await rag_with_data.semantic_search("cloud computing", k=5)
        for r in res:
            assert "content" in r
            assert "metadata" in r
            assert "score" in r

    async def test_score_is_float(self, rag_with_data):
        res = await rag_with_data.semantic_search("machine learning", k=5)
        for r in res:
            assert isinstance(r["score"], float)

    async def test_with_metadata_filter(self, rag_with_data):
        res = await rag_with_data.semantic_search(
            "Python", k=5, filter={"category": "programming"}
        )
        for r in res:
            assert r["metadata"]["category"] == "programming"

    async def test_exact_search(self, rag_with_data):
        res = await rag_with_data.semantic_search("Python", k=10, use_exact_search=True)
        assert len(res) == 10


# ---------------------------------------------------------------------------
# 2. Keyword Search (FTS)
# ---------------------------------------------------------------------------


class TestKeywordSearch:
    async def test_fts_returns_results(self, rag_with_data):
        res = await rag_with_data.keyword_search("database", k=5)
        assert len(res) > 0

    async def test_fts_result_structure(self, rag_with_data):
        res = await rag_with_data.keyword_search("programming", k=5)
        for r in res:
            assert "content" in r
            assert "score" in r

    async def test_fts_with_filter(self, rag_with_data):
        res = await rag_with_data.keyword_search(
            "Python", k=5, filter={"category": "programming"}
        )
        for r in res:
            assert r["metadata"]["category"] == "programming"


# ---------------------------------------------------------------------------
# 3. Universal Keyword Search
# ---------------------------------------------------------------------------


class TestUniversalKeywordSearch:
    async def test_searches_content_and_metadata(self, rag_with_data):
        res = await rag_with_data.universal_keyword_search(
            "Python", k=5, metadata_fields=["category", "language"]
        )
        assert len(res) > 0


# ---------------------------------------------------------------------------
# 4. Metadata Filter
# ---------------------------------------------------------------------------


class TestMetadataFilter:
    async def test_simple_equality(self, rag_with_data):
        res = await rag_with_data.metadata_filter({"category": "ai"}, k=10)
        for r in res:
            assert r["metadata"]["category"] == "ai"

    async def test_respects_k(self, rag_with_data):
        res = await rag_with_data.metadata_filter({"status": "active"}, k=5)
        assert len(res) <= 5


# ---------------------------------------------------------------------------
# 5. Metadata + Keyword
# ---------------------------------------------------------------------------


class TestMetadataKeywordSearch:
    async def test_basic(self, rag_with_data):
        res = await rag_with_data.metadata_keyword_search(
            "dev", {"category": "programming"}, k=5
        )
        assert isinstance(res, list)


# ---------------------------------------------------------------------------
# 6. Metadata + Semantic
# ---------------------------------------------------------------------------


class TestMetadataSemanticSearch:
    async def test_basic(self, rag_with_data):
        res = await rag_with_data.metadata_semantic_search(
            "Python", {"year": 2023}, k=5
        )
        assert isinstance(res, list)


# ---------------------------------------------------------------------------
# 7 & 8. Hybrid Search (weighted + RRF)
# ---------------------------------------------------------------------------


class TestHybridSearch:
    async def test_weighted(self, rag_with_data):
        res = await rag_with_data.hybrid_search("programming", k=5, weights=(0.6, 0.4))
        assert len(res) > 0

    async def test_rrf(self, rag_with_data):
        res = await rag_with_data.hybrid_search("database", k=5, use_rrf=True)
        assert len(res) > 0

    async def test_result_has_score(self, rag_with_data):
        res = await rag_with_data.hybrid_search("security", k=5, weights=(0.7, 0.3))
        for r in res:
            assert "score" in r


# ---------------------------------------------------------------------------
# 9. Ensemble Search
# ---------------------------------------------------------------------------


class TestEnsembleSearch:
    async def test_basic(self, rag_with_data):
        res = await rag_with_data.ensemble_search(
            "Python", {"category": "programming"}, k=5
        )
        assert isinstance(res, list)


# ---------------------------------------------------------------------------
# 10. Trigram Search
# ---------------------------------------------------------------------------


class TestTrigramSearch:
    async def test_basic_trigram(self, rag_with_data):
        res = await rag_with_data.trigram_search("programing", k=5, threshold=0.3)
        assert isinstance(res, list)

    async def test_metadata_trigram(self, rag_with_data):
        res = await rag_with_data.metadata_trigram_search(
            "dev", {"category": "programming"}, k=5
        )
        assert isinstance(res, list)


# ---------------------------------------------------------------------------
# Similarity search variants
# ---------------------------------------------------------------------------


class TestSimilarityVariants:
    async def test_similarity_by_vector(self, rag_with_data, embeddings):
        embedding = embeddings.embed_query("Python")
        res = await rag_with_data.asimilarity_search_by_vector(embedding, k=5)
        assert len(res) > 0

    async def test_similarity_with_score(self, rag_with_data):
        res = await rag_with_data.asimilarity_search_with_score("database", k=5)
        assert len(res) > 0
        assert all(isinstance(item, tuple) and len(item) == 2 for item in res)


# ---------------------------------------------------------------------------
# All 13 Filter Operators
# ---------------------------------------------------------------------------


class TestFilterOperators:
    @pytest.mark.parametrize(
        "op_name,filter_dict,validator",
        [
            (
                "$eq",
                {"category": {"$eq": "programming"}},
                lambda r: r["metadata"].get("category") == "programming",
            ),
            (
                "$ne",
                {"category": {"$ne": "web"}},
                lambda r: r["metadata"].get("category") != "web",
            ),
            (
                "$gt",
                {"priority": {"$gt": 5}},
                lambda r: r["metadata"].get("priority", 0) > 5,
            ),
            (
                "$gte",
                {"priority": {"$gte": 5}},
                lambda r: r["metadata"].get("priority", 0) >= 5,
            ),
            (
                "$lt",
                {"priority": {"$lt": 5}},
                lambda r: r["metadata"].get("priority", 0) < 5,
            ),
            (
                "$lte",
                {"priority": {"$lte": 5}},
                lambda r: r["metadata"].get("priority", 0) <= 5,
            ),
            (
                "$in",
                {"year": {"$in": [2023, 2024]}},
                lambda r: r["metadata"].get("year") in [2023, 2024],
            ),
            (
                "$nin",
                {"status": {"$nin": ["archived"]}},
                lambda r: r["metadata"].get("status") != "archived",
            ),
            (
                "$between",
                {"priority": {"$between": [3, 7]}},
                lambda r: 3 <= r["metadata"].get("priority", 0) <= 7,
            ),
            (
                "$exists",
                {"category": {"$exists": True}},
                lambda r: "category" in r["metadata"],
            ),
            (
                "$like",
                {"author": {"$like": "%Expert%"}},
                lambda r: "Expert" in r["metadata"].get("author", ""),
            ),
            (
                "$and",
                {"$and": [{"category": "programming"}, {"year": {"$gte": 2022}}]},
                lambda r: (
                    r["metadata"].get("category") == "programming"
                    and r["metadata"].get("year", 0) >= 2022
                ),
            ),
            (
                "$or",
                {"$or": [{"category": "ai"}, {"category": "database"}]},
                lambda r: r["metadata"].get("category") in ["ai", "database"],
            ),
        ],
    )
    async def test_filter_operator(
        self, rag_with_data, op_name, filter_dict, validator
    ):
        res = await rag_with_data.metadata_filter(filter=filter_dict, k=20)
        assert len(res) > 0, f"No results for operator {op_name}"
        for r in res:
            assert validator(r), (
                f"Result doesn't match filter {op_name}: {r['metadata']}"
            )


# ---------------------------------------------------------------------------
# BM25 Positive-Score Regression
# ---------------------------------------------------------------------------


class TestBM25PositiveScore:
    async def test_bm25_scores_are_positive(
        self, db_schema, embeddings, connection_string
    ):
        """
        BM25 <@> operator returns raw negative numbers; pgVectorDB must negate
        them so callers receive positive scores (higher = more relevant).
        """
        from sqlalchemy.ext.asyncio import create_async_engine

        from pgvectordb import ExtensionManager

        engine = create_async_engine(connection_string, pool_pre_ping=True)
        mgr = ExtensionManager(engine)
        await mgr.check_extensions()
        await engine.dispose()

        if not mgr.has_pg_textsearch:
            pytest.skip("pg_textsearch not installed — skipping BM25 test")

        pgvdb = pgVectorDB(
            collection_name="test_bm25_score",
            embedding_model=embeddings,
            connection_string=connection_string,
            schema_name=db_schema,
            index_type=IndexType.HNSW,
        )
        await pgvdb.initialize(overwrite_existing=True)
        docs = [
            Document(page_content="PostgreSQL database is great", metadata={"id": 1}),
            Document(page_content="Something else entirely", metadata={"id": 2}),
        ]
        await pgvdb.add_documents(docs)
        await pgvdb.build_bm25_index()

        res = await pgvdb.keyword_search(
            "database", k=1, search_type=KeywordSearchType.BM25
        )
        await pgvdb.close()

        assert res, "BM25 search returned no results"
        assert res[0]["score"] > 0, (
            f"BM25 score should be positive but got {res[0]['score']}"
        )
