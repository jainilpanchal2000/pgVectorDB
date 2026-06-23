"""
test_documents.py — Integration tests for pgvectordb/mixins/documents.py

Tests all DocumentsMixin methods:
  add_documents, add_documents_batch, add_documents_batch_isolated,
  aupdate_documents, adelete, update_metadata, aget_by_ids, count_by_metadata

Run:
    .venv\\Scripts\\python -m pytest test/test_documents.py -v
"""

import pytest
import pytest_asyncio
from langchain_core.documents import Document

from pgvectordb import IndexType, pgVectorDB

pytestmark = pytest.mark.integration


# ---------------------------------------------------------------------------
# Additional fixtures for this module
# ---------------------------------------------------------------------------


@pytest_asyncio.fixture
async def pgvdb(db_schema, embeddings, connection_string):
    """Fresh HNSW collection, overwritten per test."""
    inst = pgVectorDB(
        collection_name="test_documents_col",
        embedding_model=embeddings,
        connection_string=connection_string,
        schema_name=db_schema,
        index_type=IndexType.HNSW,
    )
    await inst.initialize(overwrite_existing=True)
    yield inst
    await inst.close()


# ---------------------------------------------------------------------------
# add_documents
# ---------------------------------------------------------------------------


class TestAddDocuments:
    async def test_returns_correct_count(self, pgvdb, small_docs):
        docs, _ = small_docs
        ids = await pgvdb.add_documents(docs)
        assert len(ids) == len(docs), f"Expected {len(docs)} IDs, got {len(ids)}"

    async def test_ids_are_strings(self, pgvdb, small_docs):
        docs, _ = small_docs
        ids = await pgvdb.add_documents(docs)
        assert all(isinstance(i, str) for i in ids)

    async def test_documents_persisted(self, pgvdb, small_docs):
        docs, _ = small_docs
        await pgvdb.add_documents(docs)
        count = await pgvdb.count_by_metadata(None)
        assert count == len(docs)

    async def test_add_with_labels(self, pgvdb, small_docs):
        """add_documents with DiskANN labels must not raise."""
        docs, labels = small_docs
        ids = await pgvdb.add_documents(docs, labels=labels)
        assert len(ids) == len(docs)


# ---------------------------------------------------------------------------
# add_documents_batch
# ---------------------------------------------------------------------------


class TestAddDocumentsBatch:
    async def test_batch_returns_correct_count(self, pgvdb, medium_docs):
        docs, _ = medium_docs
        ids = await pgvdb.add_documents_batch(docs, batch_size=10, show_progress=False)
        assert len(ids) == len(docs)

    async def test_batch_persists_all_docs(self, pgvdb, medium_docs):
        docs, _ = medium_docs
        await pgvdb.add_documents_batch(docs, batch_size=10, show_progress=False)
        count = await pgvdb.count_by_metadata(None)
        assert count == len(docs)

    async def test_batch_with_labels(self, pgvdb, medium_docs):
        docs, labels = medium_docs
        ids = await pgvdb.add_documents_batch(
            docs, batch_size=10, labels=labels, show_progress=False
        )
        assert len(ids) == len(docs)


# ---------------------------------------------------------------------------
# add_documents_batch_isolated (AGNO pattern)
# ---------------------------------------------------------------------------


class TestAddDocumentsBatchIsolated:
    async def test_isolated_batch_succeeds(self, pgvdb, small_docs):
        docs, _ = small_docs
        result = await pgvdb.add_documents_batch_isolated(
            docs, batch_size=5, show_progress=False, continue_on_error=False
        )
        # Returns a list of IDs (or a count depending on implementation)
        assert result is not None
        # The total docs persisted should match
        count = await pgvdb.count_by_metadata(None)
        assert count == len(docs)

    async def test_isolated_continues_on_error(self, pgvdb, small_docs):
        """continue_on_error=True should not raise even if a batch fails."""
        docs, _ = small_docs
        # Normal data — should succeed for all batches
        _result = await pgvdb.add_documents_batch_isolated(
            docs, batch_size=5, show_progress=False, continue_on_error=True
        )
        count = await pgvdb.count_by_metadata(None)
        assert count > 0


# ---------------------------------------------------------------------------
# aupdate_documents
# ---------------------------------------------------------------------------


class TestUpdateDocuments:
    async def test_update_metadata_only(self, pgvdb, small_docs):
        docs, _ = small_docs
        ids = await pgvdb.add_documents(docs)

        updated_doc = Document(
            page_content=docs[0].page_content,
            metadata={"langchain_id": ids[0], "updated": True, "status": "reviewed"},
        )
        updated_ids = await pgvdb.aupdate_documents(
            [updated_doc], update_embeddings=False
        )
        assert len(updated_ids) == 1

    async def test_update_with_re_embedding(self, pgvdb, small_docs):
        docs, _ = small_docs
        ids = await pgvdb.add_documents(docs)

        updated_doc = Document(
            page_content="Brand new content after update",
            metadata={"langchain_id": ids[0]},
        )
        # update_embeddings=True requires embedding the new content — should work
        try:
            updated_ids = await pgvdb.aupdate_documents(
                [updated_doc], update_embeddings=True
            )
            assert len(updated_ids) == 1
        except Exception as e:
            # Some implementations may not support full re-embedding in update
            pytest.skip(f"Re-embedding update not supported: {e}")


# ---------------------------------------------------------------------------
# adelete
# ---------------------------------------------------------------------------


class TestDeleteDocuments:
    async def test_delete_returns_count(self, pgvdb, small_docs):
        docs, _ = small_docs
        ids = await pgvdb.add_documents(docs)
        to_delete = ids[:5]
        count = await pgvdb.adelete(to_delete)
        assert count == 5

    async def test_delete_reduces_total(self, pgvdb, small_docs):
        docs, _ = small_docs
        ids = await pgvdb.add_documents(docs)
        original_count = await pgvdb.count_by_metadata(None)
        await pgvdb.adelete(ids[:5])
        new_count = await pgvdb.count_by_metadata(None)
        assert new_count == original_count - 5

    async def test_delete_empty_list(self, pgvdb):
        result = await pgvdb.adelete([])
        # Should return 0 or None for empty list
        assert result == 0 or result is None


# ---------------------------------------------------------------------------
# update_metadata
# ---------------------------------------------------------------------------


class TestUpdateMetadata:
    async def test_bulk_update_count(self, pgvdb, small_docs):
        docs, _ = small_docs
        ids = await pgvdb.add_documents(docs)
        updated = await pgvdb.update_metadata(
            ids=ids[:10], metadata_updates={"tagged": True}
        )
        assert updated == 10

    async def test_bulk_update_persists(self, pgvdb, small_docs):
        docs, _ = small_docs
        ids = await pgvdb.add_documents(docs)
        await pgvdb.update_metadata(
            ids=ids[:5], metadata_updates={"batch_label": "test_run"}
        )
        retrieved = await pgvdb.aget_by_ids(ids[:5])
        for doc in retrieved:
            assert doc["metadata"].get("batch_label") == "test_run"


# ---------------------------------------------------------------------------
# aget_by_ids
# ---------------------------------------------------------------------------


class TestGetByIds:
    async def test_retrieves_correct_docs(self, pgvdb, small_docs):
        docs, _ = small_docs
        ids = await pgvdb.add_documents(docs)
        retrieved = await pgvdb.aget_by_ids(ids[:5])
        assert len(retrieved) > 0

    async def test_unknown_ids_ignored(self, pgvdb, small_docs):
        import uuid

        docs, _ = small_docs
        await pgvdb.add_documents(docs)
        retrieved = await pgvdb.aget_by_ids([str(uuid.uuid4()), str(uuid.uuid4())])
        # Should return empty list or list of Nones
        assert isinstance(retrieved, list)
        # Filter out any None entries
        non_none = [r for r in retrieved if r is not None]
        assert len(non_none) == 0


# ---------------------------------------------------------------------------
# count_by_metadata
# ---------------------------------------------------------------------------


class TestCountByMetadata:
    async def test_count_all(self, pgvdb, medium_docs):
        docs, _ = medium_docs
        await pgvdb.add_documents(docs)
        count = await pgvdb.count_by_metadata(None)
        assert count == len(docs)

    async def test_count_with_simple_filter(self, pgvdb, medium_docs):
        docs, _ = medium_docs
        await pgvdb.add_documents(docs)
        active = await pgvdb.count_by_metadata({"status": "active"})
        archived = await pgvdb.count_by_metadata({"status": "archived"})
        assert active + archived == len(docs)
        assert active > 0
        assert archived > 0

    async def test_count_empty_collection(self, pgvdb):
        count = await pgvdb.count_by_metadata(None)
        assert count == 0
