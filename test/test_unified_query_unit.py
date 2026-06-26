import pytest

from pgvectordb.query.unified import UnifiedQueryBuilder
from pgvectordb.spaces import TextSpace


class FakeMultimodalDB:
    def __init__(self):
        self.semantic_called = False

    def _ensure_initialized(self):
        return None

    async def multimodal_search(self, *args, **kwargs):
        raise ValueError("bad multimodal config")

    async def semantic_search(self, *args, **kwargs):
        self.semantic_called = True
        return []


@pytest.mark.asyncio
async def test_multimodal_errors_do_not_fallback_to_semantic():
    db = FakeMultimodalDB()
    builder = UnifiedQueryBuilder(db=db, query_text="query").in_space(
        TextSpace(name="text", field="content")
    )

    with pytest.raises(ValueError, match="bad multimodal config"):
        await builder.to_list()

    assert db.semantic_called is False
