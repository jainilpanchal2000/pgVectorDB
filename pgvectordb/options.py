"""Typed option objects for high-configuration pgVectorDB operations."""

from __future__ import annotations

from dataclasses import dataclass

from .base import DistanceMetric, IndexType, KeywordSearchType, StorageLayout


@dataclass(frozen=True)
class IndexBuildOptions:
    """Options for building the primary vector index."""

    metric: DistanceMetric = DistanceMetric.COSINE
    m: int = 16
    ef_construction: int = 64
    lists: int | None = None
    num_neighbors: int = 50
    search_list_size: int = 100
    max_alpha: float = 1.2
    storage_layout: StorageLayout = StorageLayout.MEMORY_OPTIMIZED
    num_dimensions: int = 0
    num_bits_per_dimension: int | None = None
    include_labels: bool = False


@dataclass(frozen=True)
class ConcurrentIndexBuildOptions:
    """Options for non-blocking concurrent vector index builds."""

    index_type: IndexType | None = None
    m: int = 16
    ef_construction: int = 64
    lists: int | None = None
    num_neighbors: int = 50
    search_list_size: int = 100
    max_alpha: float = 1.2
    storage_layout: StorageLayout = StorageLayout.MEMORY_OPTIMIZED
    include_labels: bool = False
    distance: DistanceMetric = DistanceMetric.COSINE


@dataclass(frozen=True)
class HybridSearchOptions:
    """Options for hybrid and ensemble search fusion."""

    k: int = 4
    weights: tuple[float, float] = (0.5, 0.5)
    label_filter: list[int] | None = None
    use_rrf: bool = False
    rrf_k: int = 60
    keyword_type: KeywordSearchType = KeywordSearchType.FTS
    bm25_k1: float = 1.2
    bm25_b: float = 0.75
    text_config: str = "english"
    filter: dict[str, object] | None = None
