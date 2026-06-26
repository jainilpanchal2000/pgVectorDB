"""Shared query result post-processing helpers."""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, cast

from ..base import QueryResult

if TYPE_CHECKING:
    import pandas as pd
    import pyarrow as pa

logger = logging.getLogger(__name__)


def select_columns(results: list[QueryResult], columns: list[str] | None) -> list[QueryResult]:
    """Return results restricted to requested columns while preserving ids."""
    if not columns:
        return results
    return cast(
        list[QueryResult],
        [
            {key: value for key, value in row.items() if key in columns or key == "id"}
            for row in results
        ],
    )


def apply_reranking(
    results: list[QueryResult],
    reranker: Callable[[str, list[str]], list[float]] | Any | None,
    query: str,
) -> list[QueryResult]:
    """Rerank results with a scorer function or BaseReranker-like object."""
    if reranker is None or not results:
        return results

    try:
        if hasattr(reranker, "rerank"):
            documents = [dict(row) for row in results]
            return cast(
                list[QueryResult],
                reranker.rerank(query, documents, top_k=len(documents)),
            )

        texts = [row.get("content", "") for row in results]
        scores = reranker(query, texts)
        scored = list(zip(results, scores))
        scored.sort(key=lambda item: item[1], reverse=True)
        return [row for row, _ in scored]
    except Exception as exc:
        logger.warning(f"Reranking failed: {exc}")
        return results


def post_process_results(
    results: list[QueryResult],
    *,
    offset: int = 0,
    limit: int = 10,
    columns: list[str] | None = None,
    reranker: Callable[[str, list[str]], list[float]] | Any | None = None,
    rerank_query: str = "",
) -> list[QueryResult]:
    """Apply the standard builder result pipeline."""
    if offset > 0:
        results = results[offset:]
    results = results[:limit]
    results = apply_reranking(results, reranker, rerank_query)
    return select_columns(results, columns)


def results_to_pandas(results: list[QueryResult]) -> pd.DataFrame:
    """Convert query results to a pandas DataFrame with a clear optional-dep error."""
    try:
        import pandas as pd
    except ImportError as exc:
        raise ImportError(
            "pandas is required for to_pandas(). Install it with `pip install pgvectordb[dataframe]`."
        ) from exc

    return pd.DataFrame(results)


def results_to_arrow(results: list[QueryResult]) -> pa.Table:
    """Convert query results to a PyArrow table with a clear optional-dep error."""
    try:
        import pyarrow as pa
    except ImportError as exc:
        raise ImportError(
            "pyarrow is required for to_arrow(). Install it with `pip install pgvectordb[dataframe]`."
        ) from exc

    if not results:
        return pa.table({})
    return pa.Table.from_pylist(results)
