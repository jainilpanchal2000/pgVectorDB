"""
Vector Spaces Module for pgVectorDB
====================================

This module defines vector space abstractions for multi-embedding support.
Each space encodes a different data modality (text, numbers, categories,
timestamps) into a fixed-size vector, enabling **multimodal search** with
weighted fusion.

Inspired by:
    - Superlinked's "mixture of encoders": https://superlinked.com/vectorhub/articles/why-do-not-need-re-ranking
    - Real Estate NLQ agent: https://superlinked.com/vectorhub/articles/real-estate-nlq-agent

Space Types:
    - **TextSpace**: Embeds text fields using a LangChain embedding model.
    - **NumberSpace**: Encodes numeric fields via min-max normalization.
    - **CategorySpace**: Encodes categorical fields as one-hot vectors.
    - **RecencySpace**: Encodes timestamps via exponential time-decay.

Usage:
    >>> from pgvectordb.spaces import TextSpace, NumberSpace, CategorySpace
    >>> spaces = [
    ...     TextSpace(name="description", field="content"),
    ...     NumberSpace(name="price", field="price", min_value=0, max_value=1000000,
    ...                 mode="minimum"),
    ...     CategorySpace(name="city", field="city",
    ...                   categories=["New York", "San Francisco", "Chicago"]),
    ... ]
    >>> rag.register_spaces(spaces)
    >>> await rag.add_documents_multimodal(docs)
    >>> results = await rag.multimodal_search(
    ...     query_params={"description": "modern downtown apartment", "price": 500000,
    ...                   "city": "New York"},
    ...     weights={"description": 0.5, "price": 0.3, "city": 0.2},
    ...     k=10,
    ... )

Author: Jainil Panchal
Version: 0.0.3
"""

import math
import time as _time
import logging
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Union
from enum import Enum

logger = logging.getLogger(__name__)


# ==================== Enums ====================

class NumberMode(str, Enum):
    """
    Mode for NumberSpace encoding direction.

    Determines how numeric values are scored relative to a query value.

    Attributes:
        MINIMUM: Lower values are better (e.g., price — cheaper is better).
            Distance increases as value moves above the query.
        MAXIMUM: Higher values are better (e.g., rating — higher is better).
            Distance increases as value moves below the query.
        SIMILAR: Values closest to query are best (e.g., temperature, square footage).
            Distance increases in both directions from the query.

    Example:
        >>> price_space = NumberSpace(
        ...     name="price", field="price",
        ...     min_value=0, max_value=1000000,
        ...     mode=NumberMode.MINIMUM  # Prefer lower prices
        ... )
    """
    MINIMUM = "minimum"   # Lower is better (e.g., price)
    MAXIMUM = "maximum"   # Higher is better (e.g., rating)
    SIMILAR = "similar"   # Closest to query value is best


class TimeUnit(str, Enum):
    """
    Time unit for RecencySpace decay period.

    Determines the granularity of the exponential decay. The decay
    time-constant ``τ`` is computed as ``period_value × unit.to_seconds()``.

    Attributes:
        SECOND: 1 second.
        MINUTE: 60 seconds.
        HOUR: 3 600 seconds.
        DAY: 86 400 seconds.
        WEEK: 604 800 seconds.

    Example:
        >>> recency = RecencySpace(
        ...     name="updated", field="updated_at",
        ...     time_unit=TimeUnit.DAY, period_value=7  # weekly decay
        ... )
    """
    SECOND = "second"
    MINUTE = "minute"
    HOUR = "hour"
    DAY = "day"
    WEEK = "week"

    def to_seconds(self) -> float:
        """Return the number of seconds in one unit."""
        mapping = {
            TimeUnit.SECOND: 1.0,
            TimeUnit.MINUTE: 60.0,
            TimeUnit.HOUR: 3600.0,
            TimeUnit.DAY: 86400.0,
            TimeUnit.WEEK: 604800.0,
        }
        return mapping[self]


# ==================== Abstract Base ====================

class VectorSpace(ABC):
    """
    Abstract base class for all vector spaces.

    A vector space defines how a specific data field is encoded into a fixed-size
    embedding vector. Multiple spaces can be registered on a single collection,
    enabling multimodal search with dynamic query-time weights.

    Attributes:
        name: Unique space name. Used as column suffix: ``embedding_{name}``.
        field: Source data field name. Use ``"content"`` for ``page_content``,
            or a metadata field name.
        dimensions: Output vector dimensionality.

    Subclass Contract:
        - Implement ``encode(value)`` to convert a field value to a vector.
        - Implement ``encode_query(value)`` to convert a query parameter to a vector.
        - Set ``dimensions`` in ``__init__``.

    Example:
        >>> class MyCustomSpace(VectorSpace):
        ...     def __init__(self, name, field, dims):
        ...         super().__init__(name=name, field=field)
        ...         self._dimensions = dims
        ...     @property
        ...     def dimensions(self): return self._dimensions
        ...     def encode(self, value): return [0.0] * self._dimensions
        ...     def encode_query(self, value): return self.encode(value)
    """

    def __init__(self, name: str, field: str):
        """
        Initialize a vector space.

        Args:
            name: Unique identifier for this space. Must be a valid SQL identifier
                (alphanumeric + underscores, no spaces). This becomes the column
                suffix: ``embedding_{name}``.
            field: The document field to encode. Use ``"content"`` for the
                document's ``page_content``, or a metadata key (e.g., ``"price"``).

        Raises:
            ValueError: If name is empty or contains invalid characters.
        """
        if not name or not name.replace("_", "").isalnum():
            raise ValueError(
                f"Space name must be non-empty and contain only alphanumeric "
                f"characters and underscores, got: '{name}'"
            )
        self.name = name
        self.field = field

    @property
    @abstractmethod
    def dimensions(self) -> int:
        """Output vector dimensionality."""
        ...

    @abstractmethod
    def encode(self, value: Any) -> List[float]:
        """
        Encode a document field value into a vector.

        Args:
            value: The raw field value from the document.

        Returns:
            Fixed-size float vector of length ``self.dimensions``.
        """
        ...

    @abstractmethod
    def encode_query(self, value: Any) -> List[float]:
        """
        Encode a query parameter value into a vector.

        For many spaces, this is identical to ``encode()``. However, some spaces
        (like NumberSpace with directional modes) may encode queries differently
        than document values.

        Args:
            value: The query parameter value.

        Returns:
            Fixed-size float vector of length ``self.dimensions``.
        """
        ...

    @property
    def column_name(self) -> str:
        """
        The PostgreSQL column name for this space's embedding.

        Returns:
            Column name in the format ``embedding_{name}``.
        """
        return f"embedding_{self.name}"

    @property
    def index_name_suffix(self) -> str:
        """
        Suffix for the index name on this space's column.

        Returns:
            Index suffix in the format ``idx_{name}``.
        """
        return f"idx_{self.name}"

    def extract_value(self, document: Any) -> Any:
        """
        Extract the relevant field value from a LangChain Document.

        Args:
            document: A LangChain Document object.

        Returns:
            The field value, or None if not found.
        """
        if self.field == "content":
            return getattr(document, "page_content", None)
        metadata = getattr(document, "metadata", {}) or {}
        return metadata.get(self.field)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}', field='{self.field}', dims={self.dimensions})"


# ==================== TextSpace ====================

class TextSpace(VectorSpace):
    """
    Embed text fields using a LangChain embedding model.

    This space uses the collection's configured embedding model (or a separate
    model) to convert text into dense semantic vectors. It supports both
    document ``page_content`` and metadata text fields.

    Attributes:
        name: Space identifier (column suffix).
        field: Source field — ``"content"`` for page_content, or a metadata key.
        model: Optional separate embedding model. If None, uses the collection's
            default model.
        _dimensions: Detected embedding dimensions (set on first use).

    Example:
        >>> # Embed the document content
        >>> desc_space = TextSpace(name="description", field="content")
        >>>
        >>> # Embed a metadata field with a separate model
        >>> from langchain_community.embeddings import HuggingFaceEmbeddings
        >>> title_space = TextSpace(
        ...     name="title", field="title",
        ...     model=HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        ... )
    """

    def __init__(
        self,
        name: str,
        field: str = "content",
        model: Optional[Any] = None,
        dimensions: Optional[int] = None,
    ):
        """
        Initialize a text embedding space.

        Args:
            name: Unique space name (becomes ``embedding_{name}`` column).
            field: Document field to embed. ``"content"`` uses ``page_content``.
            model: Optional LangChain Embeddings model. If None, the collection's
                default ``embedding_model`` will be used at encode time.
            dimensions: Embedding dimensions. If None, auto-detected on first encode.
        """
        super().__init__(name=name, field=field)
        self.model = model
        self._dimensions = dimensions
        self._detected = dimensions is not None

    @property
    def dimensions(self) -> int:
        """
        Embedding dimensionality.

        Returns:
            The dimension count, or 0 if not yet detected.
        """
        return self._dimensions or 0

    def detect_dimensions(self, embedding_model: Any) -> int:
        """
        Auto-detect dimensions by embedding a test string.

        Args:
            embedding_model: LangChain Embeddings model to use for detection.

        Returns:
            Detected dimension count.
        """
        if self._detected and self._dimensions:
            return self._dimensions
        model = self.model or embedding_model
        test_embedding = model.embed_query("dimension detection")
        self._dimensions = len(test_embedding)
        self._detected = True
        logger.info(f"TextSpace '{self.name}': detected {self._dimensions} dimensions")
        return self._dimensions

    def encode(self, value: Any, embedding_model: Optional[Any] = None) -> List[float]:
        """
        Encode text into an embedding vector.

        Args:
            value: Text string to embed. If None or empty, returns a zero vector.
            embedding_model: Fallback model if this space has no dedicated model.

        Returns:
            Embedding vector of length ``self.dimensions``.

        Raises:
            ValueError: If no embedding model is available.
        """
        model = self.model or embedding_model
        if model is None:
            raise ValueError(
                f"TextSpace '{self.name}' has no embedding model. "
                f"Provide one via constructor or pass embedding_model argument."
            )

        if not self._detected:
            self.detect_dimensions(model)

        if not value or (isinstance(value, str) and not value.strip()):
            return [0.0] * self._dimensions

        text = str(value)
        return model.embed_query(text)

    def encode_query(self, value: Any, embedding_model: Optional[Any] = None) -> List[float]:
        """
        Encode a search query into an embedding vector.

        For text spaces, query encoding is identical to document encoding.

        Args:
            value: Query text string.
            embedding_model: Fallback model if no dedicated model is set.

        Returns:
            Embedding vector of length ``self.dimensions``.
        """
        return self.encode(value, embedding_model=embedding_model)


# ==================== NumberSpace ====================

class NumberSpace(VectorSpace):
    """
    Encode numeric fields into normalized vectors using min-max scaling.

    Numbers are mapped to the [0, 1] range based on configured min/max bounds.
    The ``mode`` parameter controls how the value is encoded for distance calculation:

    - **MINIMUM**: For fields where lower is better (e.g., price). The encoding
      ensures that values below the query score closer.
    - **MAXIMUM**: For fields where higher is better (e.g., rating). The
      encoding ensures that values above the query score closer.
    - **SIMILAR**: For fields where closest-to-query is best (e.g., square footage).

    Attributes:
        name: Space identifier (column suffix).
        field: Metadata field name containing the numeric value.
        min_value: Minimum expected value (maps to 0.0).
        max_value: Maximum expected value (maps to 1.0).
        mode: Encoding mode — MINIMUM, MAXIMUM, or SIMILAR.

    Example:
        >>> price_space = NumberSpace(
        ...     name="price", field="price",
        ...     min_value=0, max_value=2000000,
        ...     mode=NumberMode.MINIMUM  # Cheaper is better
        ... )
        >>>
        >>> rating_space = NumberSpace(
        ...     name="rating", field="rating",
        ...     min_value=0, max_value=5,
        ...     mode=NumberMode.MAXIMUM  # Higher is better
        ... )
    """

    def __init__(
        self,
        name: str,
        field: str,
        min_value: float = 0.0,
        max_value: float = 1.0,
        mode: Union[NumberMode, str] = NumberMode.SIMILAR,
        dimensions: int = 1,
    ):
        """
        Initialize a numeric encoding space.

        Args:
            name: Unique space name (becomes ``embedding_{name}`` column).
            field: Metadata field name containing the numeric value.
            min_value: Minimum expected value. Values below this are clamped.
            max_value: Maximum expected value. Values above this are clamped.
            mode: Encoding direction — ``"minimum"``, ``"maximum"``, or ``"similar"``.
            dimensions: Output dimensions (default 1). Higher dimensions can improve
                indexing separation but increase storage.

        Raises:
            ValueError: If min_value >= max_value or dimensions < 1.
        """
        super().__init__(name=name, field=field)
        if min_value >= max_value:
            raise ValueError(
                f"min_value ({min_value}) must be less than max_value ({max_value})"
            )
        if dimensions < 1:
            raise ValueError(f"dimensions must be >= 1, got {dimensions}")

        self.min_value = float(min_value)
        self.max_value = float(max_value)
        self.mode = NumberMode(mode) if isinstance(mode, str) else mode
        self._dimensions = dimensions

    @property
    def dimensions(self) -> int:
        """Output vector dimensionality (usually 1)."""
        return self._dimensions

    def _normalize(self, value: float) -> float:
        """Normalize a value to [0, 1] range with clamping."""
        clamped = max(self.min_value, min(self.max_value, float(value)))
        return (clamped - self.min_value) / (self.max_value - self.min_value)

    def encode(self, value: Any) -> List[float]:
        """
        Encode a numeric document value into a vector.

        The value is normalized to [0, 1] based on min/max bounds, then expanded
        to the configured number of dimensions.

        Args:
            value: Numeric value (int or float). None defaults to 0.

        Returns:
            Normalized vector of length ``self.dimensions``.
        """
        if value is None:
            normalized = 0.5  # Default to midpoint for missing values
        else:
            normalized = self._normalize(value)

        # For directional modes, adjust encoding so cosine distance works correctly
        if self.mode == NumberMode.MINIMUM:
            # Lower values → lower normalized → want these close to query
            encoded = normalized
        elif self.mode == NumberMode.MAXIMUM:
            # Higher values → higher normalized → invert so higher = closer
            encoded = 1.0 - normalized
        else:
            # SIMILAR mode: encode as-is, query encoding handles directionality
            encoded = normalized

        # Expand to N dimensions (repeating the value for better index separation)
        return [encoded] * self._dimensions

    def encode_query(self, value: Any) -> List[float]:
        """
        Encode a query parameter value for search.

        For NumberSpace, query encoding normalizes the value to match the
        document encoding scheme.

        Args:
            value: Query numeric value. None defaults to midpoint.

        Returns:
            Normalized query vector of length ``self.dimensions``.
        """
        if value is None:
            normalized = 0.5
        else:
            normalized = self._normalize(value)

        if self.mode == NumberMode.MINIMUM:
            # Query asks for "at most X" — encode query value directly
            encoded = normalized
        elif self.mode == NumberMode.MAXIMUM:
            # Query asks for "at least X" — invert to match document encoding
            encoded = 1.0 - normalized
        else:
            encoded = normalized

        return [encoded] * self._dimensions


# ==================== CategorySpace ====================

class CategorySpace(VectorSpace):
    """
    Encode categorical fields as one-hot vectors.

    Each category maps to a unique dimension. If the document's category matches,
    that dimension is 1.0, otherwise 0.0. Unknown categories get a zero vector.

    This enables **fuzzy categorical matching** via cosine similarity — when a
    query specifies a category, exact matches get score 1.0, and with negative
    filtering, non-matching categories are pushed away.

    Attributes:
        name: Space identifier (column suffix).
        field: Metadata field name containing the category value.
        categories: List of valid categories (defines dimensionality).
        negative_filter: Score assigned to non-matching categories.
            ``-1.0`` strongly penalizes mismatches, ``0.0`` is neutral.

    Example:
        >>> city_space = CategorySpace(
        ...     name="city", field="city",
        ...     categories=["New York", "San Francisco", "Chicago", "Austin"],
        ...     negative_filter=-1.0,  # Strongly penalize wrong cities
        ... )
    """

    def __init__(
        self,
        name: str,
        field: str,
        categories: List[str],
        negative_filter: float = 0.0,
        uncategorized_as_zero: bool = True,
    ):
        """
        Initialize a categorical encoding space.

        Args:
            name: Unique space name (becomes ``embedding_{name}`` column).
            field: Metadata field name containing the category string.
            categories: List of valid category values. Dimension = len(categories).
            negative_filter: Score for non-matching dimensions. Use ``-1.0`` to
                strongly penalize mismatches, ``0.0`` for neutral non-matches.
            uncategorized_as_zero: If True, unknown categories encode as zero vector.
                If False, unknown categories raise ValueError.

        Raises:
            ValueError: If categories list is empty or has duplicates.
        """
        super().__init__(name=name, field=field)
        if not categories:
            raise ValueError("categories must be a non-empty list")
        if len(categories) != len(set(categories)):
            raise ValueError(f"categories must not contain duplicates: {categories}")

        self.categories = list(categories)
        self.negative_filter = negative_filter
        self.uncategorized_as_zero = uncategorized_as_zero
        self._category_index: Dict[str, int] = {
            cat: i for i, cat in enumerate(self.categories)
        }

    @property
    def dimensions(self) -> int:
        """Dimensionality equals the number of categories."""
        return len(self.categories)

    def encode(self, value: Any) -> List[float]:
        """
        Encode a category value as a one-hot vector.

        Args:
            value: Category string. None or unknown → zero vector (or error).

        Returns:
            One-hot vector of length ``len(categories)``.

        Raises:
            ValueError: If ``uncategorized_as_zero=False`` and value is unknown.
        """
        vec = [self.negative_filter] * self.dimensions

        if value is None:
            if self.uncategorized_as_zero:
                return [0.0] * self.dimensions
            raise ValueError(f"CategorySpace '{self.name}': value is None")

        category_str = str(value).strip()
        idx = self._category_index.get(category_str)

        if idx is not None:
            vec[idx] = 1.0
        elif self.uncategorized_as_zero:
            return [0.0] * self.dimensions
        else:
            raise ValueError(
                f"CategorySpace '{self.name}': unknown category '{category_str}'. "
                f"Valid: {self.categories}"
            )

        return vec

    def encode_query(self, value: Any) -> List[float]:
        """
        Encode a query category. Same as document encoding.

        Args:
            value: Category string to search for.

        Returns:
            One-hot query vector.
        """
        return self.encode(value)


# ==================== RecencySpace ====================

class RecencySpace(VectorSpace):
    """
    Encode timestamps into vectors using exponential time-decay.

    Recent documents score close to 1.0; older documents decay towards 0.0.
    The decay follows ``score = exp(-age / τ)`` where ``τ = period_value ×
    time_unit.to_seconds()``.

    This enables **time-aware multimodal search** — boost fresh content
    without post-retrieval re-ranking.

    Attributes:
        name: Space identifier (column suffix).
        field: Metadata field containing the timestamp.
        time_unit: Granularity of the decay period.
        period_value: Number of time units for the decay constant τ.
        tau: Precomputed decay constant in seconds.

    Decay behaviour:
        - age = 0  → score ≈ 1.0  (fresh)
        - age = τ  → score ≈ 0.37
        - age = 3τ → score ≈ 0.05

    .. warning::
        ``encode()`` uses the wall-clock time at invocation as "now". This
        means stored embeddings become stale over time. Re-encode periodically
        or compute recency at query time for accuracy.

    Example:
        >>> recency = RecencySpace(
        ...     name="published", field="published_at",
        ...     time_unit=TimeUnit.DAY, period_value=7,
        ... )
        >>> # Document published 1 day ago scores ~0.87
        >>> # Document published 7 days ago scores ~0.37
    """

    def __init__(
        self,
        name: str,
        field: str,
        time_unit: Union[TimeUnit, str] = TimeUnit.DAY,
        period_value: float = 1.0,
        dimensions: int = 1,
    ):
        """
        Initialize a recency (time-decay) space.

        Args:
            name: Unique space name (becomes ``embedding_{name}`` column).
            field: Metadata field containing the timestamp. Accepts ISO-8601
                strings, ``datetime`` objects, or Unix epoch floats/ints.
            time_unit: Time granularity for the decay period.
            period_value: Number of ``time_unit`` units that make up the decay
                constant τ. Must be positive.
            dimensions: Output vector dimensions (default 1).

        Raises:
            ValueError: If period_value ≤ 0 or dimensions < 1.
        """
        super().__init__(name=name, field=field)
        if period_value <= 0:
            raise ValueError(
                f"period_value must be > 0, got {period_value}"
            )
        if dimensions < 1:
            raise ValueError(f"dimensions must be >= 1, got {dimensions}")

        self.time_unit = TimeUnit(time_unit) if isinstance(time_unit, str) else time_unit
        self.period_value = float(period_value)
        self.tau = self.period_value * self.time_unit.to_seconds()
        self._dimensions = dimensions

    @property
    def dimensions(self) -> int:
        """Output vector dimensionality (usually 1)."""
        return self._dimensions

    # ------------------------------------------------------------------
    # Timestamp parsing
    # ------------------------------------------------------------------

    @staticmethod
    def _to_epoch(value: Any) -> float:
        """
        Convert a timestamp value to Unix epoch seconds.

        Accepts:
            - ``datetime`` objects (timezone-aware or naive; naive assumed UTC)
            - ISO-8601 strings (e.g. ``"2025-06-15T10:30:00Z"``)
            - Numeric Unix timestamps (int / float)

        Args:
            value: The timestamp to convert.

        Returns:
            Unix epoch seconds as a float.

        Raises:
            ValueError: If the value cannot be parsed.
        """
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, datetime):
            return value.timestamp()
        if isinstance(value, str):
            try:
                # Try ISO-8601 parsing
                dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
                return dt.timestamp()
            except (ValueError, TypeError):
                pass
            # Try as numeric string
            try:
                return float(value)
            except (ValueError, TypeError):
                pass
        raise ValueError(
            f"RecencySpace: cannot parse timestamp '{value}' "
            f"(type {type(value).__name__}). Expected datetime, ISO-8601 "
            f"string, or numeric Unix timestamp."
        )

    # ------------------------------------------------------------------
    # Encoding
    # ------------------------------------------------------------------

    def encode(self, value: Any) -> List[float]:
        """
        Encode a document timestamp into a time-decay vector.

        ``score = exp(-age_seconds / τ)`` where age is measured from *now*.
        Future timestamps are clamped to 1.0.

        Args:
            value: Timestamp (datetime, ISO string, or Unix epoch).
                ``None`` encodes as 0.5 (neutral midpoint).

        Returns:
            Vector of length ``self.dimensions``.
        """
        if value is None:
            return [0.5] * self._dimensions

        epoch = self._to_epoch(value)
        now = _time.time()
        age = now - epoch

        if age <= 0:
            # Future or exact-now → maximum freshness
            score = 1.0
        else:
            score = math.exp(-age / self.tau)

        # Clamp to [0, 1] for safety (exp is always positive, but guard)
        score = max(0.0, min(1.0, score))
        return [score] * self._dimensions

    def encode_query(self, value: Any = None) -> List[float]:
        """
        Encode a query value for recency search.

        If ``value`` is ``None`` (the typical case), returns ``[1.0]`` —
        meaning "prefer the freshest documents". If a specific timestamp is
        provided, it is encoded the same way as a document timestamp.

        Args:
            value: ``None`` for "prefer newest", or a specific reference
                timestamp.

        Returns:
            Query vector of length ``self.dimensions``.
        """
        if value is None:
            return [1.0] * self._dimensions
        return self.encode(value)


# ==================== Utility Functions ====================

def validate_spaces(spaces: List[VectorSpace]) -> None:
    """
    Validate a list of vector spaces for consistency.

    Checks:
        - At least one space defined
        - No duplicate space names
        - All dimensions are positive (where detectable)

    Args:
        spaces: List of VectorSpace instances.

    Raises:
        ValueError: If validation fails.
    """
    if not spaces:
        raise ValueError("At least one VectorSpace must be defined")

    names = [s.name for s in spaces]
    if len(names) != len(set(names)):
        duplicates = [n for n in names if names.count(n) > 1]
        raise ValueError(f"Duplicate space names found: {set(duplicates)}")

    for space in spaces:
        if space.dimensions > 0 and space.dimensions < 1:
            raise ValueError(
                f"Space '{space.name}' has invalid dimensions: {space.dimensions}"
            )


def get_total_dimensions(spaces: List[VectorSpace]) -> int:
    """
    Get the total dimensions across all spaces.

    Note: TextSpace dimensions may be 0 until ``detect_dimensions()`` is called.

    Args:
        spaces: List of VectorSpace instances.

    Returns:
        Sum of all space dimensions.
    """
    return sum(s.dimensions for s in spaces)


def encode_document_spaces(
    document: Any,
    spaces: List[VectorSpace],
    embedding_model: Optional[Any] = None,
) -> Dict[str, List[float]]:
    """
    Encode a single document across all registered spaces.

    Extracts the relevant field from the document for each space and encodes it.

    Args:
        document: A LangChain Document object.
        spaces: List of VectorSpace instances.
        embedding_model: Default embedding model for TextSpaces without a dedicated model.

    Returns:
        Dictionary mapping column names to embedding vectors.
        Example: ``{"embedding_description": [0.1, ...], "embedding_price": [0.7]}``
    """
    embeddings = {}
    for space in spaces:
        value = space.extract_value(document)
        if isinstance(space, TextSpace):
            embedding = space.encode(value, embedding_model=embedding_model)
        else:
            embedding = space.encode(value)
        embeddings[space.column_name] = embedding
    return embeddings


def encode_query_spaces(
    query_params: Dict[str, Any],
    spaces: List[VectorSpace],
    embedding_model: Optional[Any] = None,
) -> Dict[str, List[float]]:
    """
    Encode query parameters across all relevant spaces.

    Only encodes spaces whose names are present in ``query_params``.
    Spaces not in ``query_params`` are skipped (not included in search).

    Args:
        query_params: Dictionary mapping space names to query values.
            Example: ``{"description": "modern apartment", "price": 500000}``
        spaces: List of VectorSpace instances.
        embedding_model: Default embedding model for TextSpaces.

    Returns:
        Dictionary mapping column names to query embedding vectors.
        Only includes spaces present in ``query_params``.
    """
    query_embeddings = {}
    for space in spaces:
        if space.name not in query_params:
            continue
        value = query_params[space.name]
        if isinstance(space, TextSpace):
            embedding = space.encode_query(value, embedding_model=embedding_model)
        else:
            embedding = space.encode_query(value)
        query_embeddings[space.column_name] = embedding
    return query_embeddings


# ==================== Module Exports ====================

__all__ = [
    # Enums
    "NumberMode",
    "TimeUnit",
    # Base class
    "VectorSpace",
    # Space types
    "TextSpace",
    "NumberSpace",
    "CategorySpace",
    "RecencySpace",
    # Utilities
    "validate_spaces",
    "get_total_dimensions",
    "encode_document_spaces",
    "encode_query_spaces",
]
