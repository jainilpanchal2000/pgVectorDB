from pgvectordb.base import QueryResult
from pgvectordb.query._postprocessing import post_process_results, select_columns


def make_result(doc_id: str, score: float) -> QueryResult:
    return QueryResult(
        id=doc_id,
        content=f"content {doc_id}",
        metadata={"doc_id": doc_id},
        score=score,
    )


def test_select_columns_preserves_id():
    results = [make_result("a", 0.9)]

    selected = select_columns(results, ["content"])

    assert selected == [{"id": "a", "content": "content a"}]


def test_post_process_applies_offset_limit_rerank_then_columns():
    results = [make_result("a", 0.1), make_result("b", 0.2), make_result("c", 0.3)]

    def reranker(query: str, texts: list[str]) -> list[float]:
        assert query == "query"
        return [1.0 if text.endswith("c") else 0.0 for text in texts]

    processed = post_process_results(
        results,
        offset=1,
        limit=2,
        columns=["content"],
        reranker=reranker,
        rerank_query="query",
    )

    assert processed == [
        {"id": "c", "content": "content c"},
        {"id": "b", "content": "content b"},
    ]


def test_post_process_accepts_reranker_object():
    results = [make_result("a", 0.1), make_result("b", 0.2), make_result("c", 0.3)]

    class ObjectReranker:
        def rerank(self, query: str, documents: list[dict], top_k: int | None = None) -> list[dict]:
            assert query == "query"
            assert top_k == 2
            reranked = [
                {**document, "score": 1.0 if document["id"] == "c" else 0.0}
                for document in documents
            ]
            return sorted(reranked, key=lambda document: document["score"], reverse=True)

    processed = post_process_results(
        results,
        offset=1,
        limit=2,
        reranker=ObjectReranker(),
        rerank_query="query",
    )

    assert processed == [
        {"id": "c", "content": "content c", "metadata": {"doc_id": "c"}, "score": 1.0},
        {"id": "b", "content": "content b", "metadata": {"doc_id": "b"}, "score": 0.0},
    ]
