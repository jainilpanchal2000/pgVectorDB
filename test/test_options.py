from pgvectordb import (
    ConcurrentIndexBuildOptions,
    DistanceMetric,
    HybridSearchOptions,
    IndexBuildOptions,
    KeywordSearchType,
    StorageLayout,
)


def test_index_build_options_defaults_match_legacy_defaults():
    options = IndexBuildOptions()

    assert options.metric == DistanceMetric.COSINE
    assert options.m == 16
    assert options.ef_construction == 64
    assert options.lists is None
    assert options.num_neighbors == 50
    assert options.search_list_size == 100
    assert options.max_alpha == 1.2
    assert options.storage_layout == StorageLayout.MEMORY_OPTIMIZED
    assert options.include_labels is False


def test_concurrent_index_options_defaults_match_legacy_defaults():
    options = ConcurrentIndexBuildOptions()

    assert options.index_type is None
    assert options.distance == DistanceMetric.COSINE
    assert options.m == 16
    assert options.ef_construction == 64
    assert options.lists is None
    assert options.include_labels is False


def test_hybrid_search_options_defaults_match_legacy_defaults():
    options = HybridSearchOptions()

    assert options.k == 4
    assert options.weights == (0.5, 0.5)
    assert options.label_filter is None
    assert options.use_rrf is False
    assert options.rrf_k == 60
    assert options.keyword_type == KeywordSearchType.FTS
    assert options.bm25_k1 == 1.2
    assert options.bm25_b == 0.75
    assert options.text_config == "english"
