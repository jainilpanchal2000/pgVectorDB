"""
Centralized Schema Definition for pgVectorDB
=============================================

This module provides SQLAlchemy Table definitions for consistent schema management
and safer SQL operations using ORM constructs instead of raw SQL strings.

Based on AGNO's get_table_v1() pattern for schema centralization.

Author: pgVectorDB Team
Version: 1.0
"""

from sqlalchemy import (
    Column,
    DateTime,
    MetaData,
    String,
    Table,
    Text,
    text,
)
from sqlalchemy.dialects.postgresql import ARRAY, JSONB
from sqlalchemy.sql import func

# Try to import pgvector types, fall back to string if not available
try:
    from pgvector.sqlalchemy import Vector

    VECTOR_TYPE_AVAILABLE = True
except ImportError:
    VECTOR_TYPE_AVAILABLE = False
    Vector = None


def get_vector_table(
    table_name: str,
    schema: str,
    dimensions: int,
    include_labels: bool = False,
    include_content_hash: bool = False,
    include_timestamps: bool = False,
) -> Table:
    """
    Create a SQLAlchemy Table object for vector storage.

    This provides a centralized schema definition that enables:
    - Parameterized inserts via `postgresql.insert(table)`
    - Safe upserts via `on_conflict_do_update()`
    - Proper type handling for vector columns

    Args:
        table_name: Name of the table
        schema: Database schema name
        dimensions: Vector embedding dimensions
        include_labels: Include labels column for DiskANN filtering
        include_content_hash: Include content hash for deduplication
        include_timestamps: Include created_at and updated_at columns

    Returns:
        SQLAlchemy Table object

    Examples:
        >>> table = get_vector_table("my_docs", "public", 384)
        >>> insert_stmt = postgresql.insert(table).values(records)
        >>> upsert_stmt = insert_stmt.on_conflict_do_update(
        ...     index_elements=["langchain_id"],
        ...     set_={"content": insert_stmt.excluded.content}
        ... )
    """
    metadata = MetaData(schema=schema)

    # Core columns (always present)
    columns = [
        Column("langchain_id", String, primary_key=True),
        Column("content", Text, nullable=False),
        Column("langchain_metadata", JSONB, server_default=text("'{}'::jsonb")),
    ]

    # Vector column - use pgvector type if available, otherwise fallback
    if VECTOR_TYPE_AVAILABLE and Vector is not None:
        columns.append(Column("embedding", Vector(dimensions), nullable=True))
    else:
        # Fallback: vector will be handled by raw SQL
        # This column definition is for schema introspection only
        columns.append(Column("embedding", Text, nullable=True))

    # Optional: Labels column for DiskANN filtering
    if include_labels:
        columns.append(Column("labels", ARRAY(String), nullable=True))

    # Optional: Content hash for deduplication (AGNO pattern)
    if include_content_hash:
        columns.append(Column("content_hash", String(32), nullable=True))

    # Optional: Timestamps for audit trail
    if include_timestamps:
        columns.extend(
            [
                Column("created_at", DateTime(timezone=True), server_default=func.now()),
                Column("updated_at", DateTime(timezone=True), onupdate=func.now()),
            ]
        )

    # tsvector column for full-text search
    columns.append(
        Column("content_tsvector", Text, nullable=True)  # Actually tsvector type
    )

    table = Table(
        table_name,
        metadata,
        *columns,
        extend_existing=True,
    )

    return table


def get_label_definitions_table(schema: str = "public") -> Table:
    """
    Create a SQLAlchemy Table for label definitions.

    This table maps integer label IDs to human-readable names
    for DiskANN label-based filtering.

    Args:
        schema: Database schema name

    Returns:
        SQLAlchemy Table object for label definitions

    Examples:
        >>> labels_table = get_label_definitions_table("public")
        >>> stmt = insert(labels_table).values([
        ...     {"id": 1, "name": "science", "description": "Scientific content"},
        ...     {"id": 2, "name": "technology", "description": "Tech content"},
        ... ])
    """
    metadata = MetaData(schema=schema)

    return Table(
        "label_definitions",
        metadata,
        Column("id", String, primary_key=True),
        Column("name", String(255), nullable=False, unique=True),
        Column("description", Text, nullable=True),
        Column("attributes", JSONB, server_default=text("'{}'::jsonb")),
        Column("created_at", DateTime(timezone=True), server_default=func.now()),
        extend_existing=True,
    )


# ==================== Multimodal Table (v0.0.3) ====================


def get_multimodal_table(
    table_name: str,
    schema: str,
    spaces: list,
    include_labels: bool = False,
    include_content_hash: bool = False,
    include_timestamps: bool = False,
) -> Table:
    """
    Create a SQLAlchemy Table with multiple vector columns — one per VectorSpace.

    Extends the standard vector table with additional ``embedding_{space.name}``
    columns for multimodal search. The standard ``embedding`` column is ALSO
    included for backward compatibility with single-embedding methods.

    Args:
        table_name: Name of the table
        schema: Database schema name
        spaces: List of VectorSpace instances defining additional embedding columns.
            Each space adds a column named ``embedding_{space.name}`` with
            dimensions matching the space.
        include_labels: Include labels column for DiskANN filtering
        include_content_hash: Include content hash for deduplication
        include_timestamps: Include created_at and updated_at columns

    Returns:
        SQLAlchemy Table object with multiple vector columns

    Examples:
        >>> from pgvectordb.spaces import TextSpace, NumberSpace, CategorySpace
        >>> spaces = [
        ...     TextSpace(name="description", field="content"),
        ...     NumberSpace(name="price", field="price", min_value=0, max_value=1e6),
        ...     CategorySpace(name="city", field="city", categories=["NYC", "LA"]),
        ... ]
        >>> table = get_multimodal_table("products", "public", spaces)
    """
    metadata = MetaData(schema=schema)

    # Core columns (always present)
    columns = [
        Column("langchain_id", String, primary_key=True),
        Column("content", Text, nullable=False),
        Column("langchain_metadata", JSONB, server_default=text("'{}'::jsonb")),
    ]

    # Add a vector column per space
    for space in spaces:
        col_name = f"embedding_{space.name}"
        dims = space.dimensions
        if dims > 0 and VECTOR_TYPE_AVAILABLE and Vector is not None:
            columns.append(Column(col_name, Vector(dims), nullable=True))
        else:
            # Fallback or undetected dimensions (TextSpace before detect)
            columns.append(Column(col_name, Text, nullable=True))

    # Optional: Labels column for DiskANN filtering
    if include_labels:
        columns.append(Column("labels", ARRAY(String), nullable=True))

    # Optional: Content hash for deduplication
    if include_content_hash:
        columns.append(Column("content_hash", String(32), nullable=True))

    # Optional: Timestamps for audit trail
    if include_timestamps:
        columns.extend(
            [
                Column("created_at", DateTime(timezone=True), server_default=func.now()),
                Column("updated_at", DateTime(timezone=True), onupdate=func.now()),
            ]
        )

    # tsvector column for full-text search
    columns.append(Column("content_tsvector", Text, nullable=True))

    table = Table(
        table_name,
        metadata,
        *columns,
        extend_existing=True,
    )

    return table


# ==================== Helper Functions ====================


def quote_identifier(identifier: str) -> str:
    """
    Safely quote a SQL identifier to prevent SQL injection.

    Args:
        identifier: The identifier to quote (table name, column name, etc.)

    Returns:
        Quoted identifier safe for SQL interpolation

    Raises:
        ValueError: If identifier contains invalid characters

    Examples:
        >>> quote_identifier("my_table")
        '"my_table"'
        >>> quote_identifier("schema.table")  # Invalid
        ValueError: Invalid identifier
    """
    import re

    # Only allow alphanumeric and underscore
    if not re.match(r"^[a-zA-Z_][a-zA-Z0-9_]*$", identifier):
        raise ValueError(
            f"Invalid identifier: '{identifier}'. "
            "Identifiers must start with a letter or underscore and contain only "
            "alphanumeric characters and underscores."
        )

    # Double any existing quotes (PostgreSQL escaping)
    escaped = identifier.replace('"', '""')

    return f'"{escaped}"'


def build_qualified_name(schema: str, name: str) -> str:
    """
    Build a fully qualified table/index name with proper quoting.

    Args:
        schema: Schema name
        name: Table or index name

    Returns:
        Quoted qualified name like "schema"."name"

    Examples:
        >>> build_qualified_name("public", "my_table")
        '"public"."my_table"'
    """
    return f"{quote_identifier(schema)}.{quote_identifier(name)}"


# ==================== Column Type Helpers ====================


def get_distance_operator(distance_metric: str) -> str:
    """
    Get the pgvector operator for a distance metric.

    Args:
        distance_metric: One of 'cosine', 'l2', 'inner_product', 'l1'

    Returns:
        PostgreSQL operator string

    Examples:
        >>> get_distance_operator("cosine")
        '<=>'
    """
    operators = {
        "cosine": "<=>",
        "l2": "<->",
        "inner_product": "<#>",
        "l1": "<+>",
        "hamming": "<~>",
        "jaccard": "<%>",
    }

    if distance_metric.lower() not in operators:
        raise ValueError(
            f"Unknown distance metric: '{distance_metric}'. Supported: {list(operators.keys())}"
        )

    return operators[distance_metric.lower()]


def get_index_ops(distance_metric: str, vector_type: str = "vector") -> str:
    """
    Get the pgvector operator class for index creation.

    Args:
        distance_metric: One of 'cosine', 'l2', 'inner_product', 'l1'
        vector_type: One of 'vector', 'halfvec', 'bit', 'sparsevec'

    Returns:
        Operator class name for CREATE INDEX

    Examples:
        >>> get_index_ops("cosine", "vector")
        'vector_cosine_ops'
        >>> get_index_ops("l2", "halfvec")
        'halfvec_l2_ops'
    """
    base_ops = {
        "cosine": "cosine_ops",
        "l2": "l2_ops",
        "inner_product": "ip_ops",
        "l1": "l1_ops",
    }

    bit_ops = {
        "hamming": "hamming_ops",
        "jaccard": "jaccard_ops",
    }

    if vector_type == "bit":
        if distance_metric.lower() not in bit_ops:
            raise ValueError(
                f"Distance metric '{distance_metric}' not supported for bit vectors. "
                f"Supported: {list(bit_ops.keys())}"
            )
        return f"bit_{bit_ops[distance_metric.lower()]}"

    if distance_metric.lower() not in base_ops:
        raise ValueError(
            f"Unknown distance metric: '{distance_metric}'. Supported: {list(base_ops.keys())}"
        )

    type_prefix = {
        "vector": "vector",
        "halfvec": "halfvec",
        "sparsevec": "sparsevec",
    }

    if vector_type not in type_prefix:
        raise ValueError(
            f"Unknown vector type: '{vector_type}'. Supported: {list(type_prefix.keys())}"
        )

    return f"{type_prefix[vector_type]}_{base_ops[distance_metric.lower()]}"
