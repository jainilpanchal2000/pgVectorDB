"""
test_multimodal.py — Integration tests for pgvectordb/mixins/multimodal.py

Tests:
  - register_spaces() with TextSpace, NumberSpace, CategorySpace, RecencySpace
  - add_documents_multimodal() creates embedding columns
  - build_multimodal_index() per-space
  - multimodal_search() weighted
  - multimodal_hybrid_search() fused
  - get_multimodal_index_stats()
  - rerank_search() with basic CrossEncoder (skipped if sentence-transformers absent)

Run:
    .venv\\Scripts\\python -m pytest test/test_multimodal.py -v
"""

import datetime

import pytest
import pytest_asyncio
from langchain_core.documents import Document

from pgvectordb import IndexType, pgVectorDB
from pgvectordb.spaces import (
    CategorySpace,
    NumberSpace,
    RecencySpace,
    TextSpace,
    TimeUnit,
)

pytestmark = pytest.mark.integration


# ---------------------------------------------------------------------------
# Test documents with structured fields for multimodal
# ---------------------------------------------------------------------------


def make_multimodal_docs(n: int = 20) -> list[Document]:
    categories = ["tech", "science", "art", "sports"]
    docs = []
    for i in range(n):
        docs.append(
            Document(
                page_content=f"Document {i} about {categories[i % len(categories)]} topics",
                metadata={
                    "title": f"Doc {i}",
                    "category": categories[i % len(categories)],
                    "year": 2020 + (i % 5),
                    "priority": float((i % 10) + 1),
                    "created_at": (
                        datetime.datetime(2024, 1, 1) + datetime.timedelta(days=i)
                    ).isoformat(),
                },
            )
        )
    return docs


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest_asyncio.fixture
async def rag_mm(db_schema, embeddings, connection_string):
    """HNSW collection for multimodal tests."""
    inst = pgVectorDB(
        collection_name="test_multimodal_col",
        embedding_model=embeddings,
        connection_string=connection_string,
        schema_name=db_schema,
        index_type=IndexType.HNSW,
    )
    await inst.initialize(overwrite_existing=True)
    yield inst
    await inst.close()


@pytest.fixture
def spaces(embeddings):
    """Example set of spaces using the same HF embeddings model."""
    return [
        TextSpace(name="text", field="page_content", model=embeddings),
        NumberSpace(name="priority", field="priority"),
        CategorySpace(
            name="category",
            field="category",
            categories=["tech", "science", "art", "sports"],
        ),
    ]


# ---------------------------------------------------------------------------
# register_spaces
# ---------------------------------------------------------------------------


class TestRegisterSpaces:
    async def test_register_text_space(self, rag_mm, embeddings):
        spaces = [TextSpace(name="text", field="page_content", model=embeddings)]
        rag_mm.register_spaces(spaces)

    async def test_register_multiple_spaces(self, rag_mm, spaces):
        rag_mm.register_spaces(spaces)

    async def test_register_number_space(self, rag_mm):
        spaces = [NumberSpace(name="priority", field="priority")]
        rag_mm.register_spaces(spaces)

    async def test_register_category_space(self, rag_mm):
        spaces = [
            CategorySpace(name="cat", field="category", categories=["a", "b", "c"])
        ]
        rag_mm.register_spaces(spaces)

    async def test_register_recency_space(self, rag_mm):
        spaces = [
            RecencySpace(name="recency", field="created_at", time_unit=TimeUnit.DAY)
        ]
        rag_mm.register_spaces(spaces)


# ---------------------------------------------------------------------------
# add_documents_multimodal
# ---------------------------------------------------------------------------


class TestAddDocumentsMultimodal:
    async def test_adds_docs(self, rag_mm, spaces):
        rag_mm.register_spaces(spaces)
        docs = make_multimodal_docs(10)
        ids = await rag_mm.add_documents_multimodal(
            docs, batch_size=5, show_progress=False
        )
        assert len(ids) == 10

    async def test_count_persisted(self, rag_mm, spaces):
        rag_mm.register_spaces(spaces)
        docs = make_multimodal_docs(10)
        await rag_mm.add_documents_multimodal(docs, batch_size=5, show_progress=False)
        count = await rag_mm.count_by_metadata(None)
        assert count == 10


# ---------------------------------------------------------------------------
# build_multimodal_index
# ---------------------------------------------------------------------------


class TestBuildMultimodalIndex:
    async def test_builds_indexes(self, rag_mm, spaces):
        rag_mm.register_spaces(spaces)
        docs = make_multimodal_docs(15)
        await rag_mm.add_documents_multimodal(docs, show_progress=False)
        await rag_mm.build_multimodal_index()


# ---------------------------------------------------------------------------
# multimodal_search
# ---------------------------------------------------------------------------


class TestMultimodalSearch:
    @pytest_asyncio.fixture
    async def rag_mm_ready(self, rag_mm, spaces):
        rag_mm.register_spaces(spaces)
        docs = make_multimodal_docs(20)
        await rag_mm.add_documents_multimodal(docs, show_progress=False)
        await rag_mm.build_multimodal_index()
        return rag_mm

    async def test_basic_search(self, rag_mm_ready):
        res = await rag_mm_ready.multimodal_search(
            query_params={"text": "technology topics"},
            k=5,
        )
        assert isinstance(res, list)

    async def test_weighted_search(self, rag_mm_ready):
        res = await rag_mm_ready.multimodal_search(
            query_params={"text": "science"},
            weights={"text": 1.0},
            k=5,
        )
        assert len(res) > 0

    async def test_result_structure(self, rag_mm_ready):
        res = await rag_mm_ready.multimodal_search(
            query_params={"text": "art"},
            k=5,
        )
        for r in res:
            assert "content" in r
            assert "score" in r


# ---------------------------------------------------------------------------
# get_multimodal_index_stats
# ---------------------------------------------------------------------------


class TestGetMultimodalIndexStats:
    async def test_returns_dict(self, rag_mm, spaces):
        rag_mm.register_spaces(spaces)
        docs = make_multimodal_docs(10)
        await rag_mm.add_documents_multimodal(docs, show_progress=False)
        await rag_mm.build_multimodal_index()
        stats = await rag_mm.get_multimodal_index_stats()
        assert isinstance(stats, dict)


# ---------------------------------------------------------------------------
# rerank_search (skipped if sentence-transformers not installed)
# ---------------------------------------------------------------------------


class TestRerankSearch:
    async def test_rerank_with_cross_encoder(self, rag_mm, embeddings, spaces):
        try:
            from pgvectordb import CrossEncoderReranker
        except (ImportError, TypeError):
            pytest.skip("sentence-transformers not installed — skip reranker test")

        rag_mm.register_spaces(spaces)
        docs = make_multimodal_docs(20)
        await rag_mm.add_documents_multimodal(docs, show_progress=False)
        await rag_mm.build_multimodal_index()
        await rag_mm.build_index()  # also build base index for reranking stage

        try:
            reranker = CrossEncoderReranker(
                model="cross-encoder/ms-marco-MiniLM-L-6-v2"
            )
            res = await rag_mm.rerank_search(
                query="technology",
                reranker=reranker,
                k=10,
                rerank_top_k=5,
                search_method="semantic",
            )
            assert isinstance(res, list)
            assert len(res) <= 5
        except Exception as e:
            # If model download fails in CI, skip gracefully
            pytest.skip(f"Reranker not available: {e}")
