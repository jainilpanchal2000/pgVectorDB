"""
MixinBase: declares the host-class contract all pgVectorDB mixins depend on.

Each mixin is designed to be composed into ``pgVectorDB`` (which supplies all
these attributes at runtime).  Without explicit declarations pyright cannot
resolve cross-mixin / mixin→host attribute access, so we centralise them here
and have every mixin inherit from ``MixinBase``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncEngine

    from ..base import IndexType
    from ..extensions import ExtensionManager


class MixinBase:
    """
    Declares the attributes and methods that all mixins expect their host
    (``pgVectorDB``) to provide.  This is never instantiated directly.
    """

    # ── Core identity ──────────────────────────────────────────────────────
    table_name: str
    schema_name: str
    connection_string: str
    vector_size: int

    # ── Engine / store ─────────────────────────────────────────────────────
    sqlalchemy_engine: AsyncEngine
    _vector_store: Any | None
    _extensions: ExtensionManager | None

    # ── Config / state ─────────────────────────────────────────────────────
    index_type: IndexType
    embedding_model: Any
    _index_built: bool
    _query_params: dict[str, Any]
    _diskann_build_params: dict[str, Any]
    _nprobes: int | None

    # ── Lifecycle ──────────────────────────────────────────────────────────
    def _ensure_initialized(self) -> None:
        """Raises if the collection has not been initialised yet."""
        raise NotImplementedError

    # ── Internal helpers (used across mixins) ──────────────────────────────
    def _build_filter_clauses(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError

    def _build_filter_clauses_wrapper(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError

    def _get_distance_ops(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError

    def _apply_query_params(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError

    def _fuse_results(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError

    # ── Cross-mixin method references ──────────────────────────────────────
    async def semantic_search(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError

    async def keyword_search(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError

    async def hybrid_search(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError

    async def add_documents_batch(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError

    async def adrop_vector_index(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError

    async def build_index(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError
