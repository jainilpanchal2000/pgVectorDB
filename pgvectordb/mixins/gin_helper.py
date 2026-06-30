"""GIN Index Helper for pgVectorDB.

Provides utilities for creating and managing GIN (Generalized Inverted Index)
indexes for metadata filtering, tag lists, and full-text search.

GIN indexes are ideal for:
- JSONB metadata queries with containment operators (@>, ?)
- Array columns with overlap (&&) and containment (@>) operators
- Full-text search with tsvector and tsquery

Example:
    db = pgVectorDB(...)

    # Create GIN index for metadata
    await db.gin.ensure_gin_index("metadata", index_type="jsonb")

    # Now metadata queries are faster
    results = await db.query("").where({"metadata.category": "ai"}).to_list()
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Literal

from sqlalchemy import text

if TYPE_CHECKING:
    from ..core import pgVectorDB


logger = logging.getLogger(__name__)


class GINIndexHelper:
    """Helper for creating and managing GIN indexes for metadata.

    GIN indexes significantly speed up queries on:
    - JSONB columns (metadata storage)
    - Array columns (tag lists)
    - Text columns (full-text search)

    Attributes:
        db: The pgVectorDB instance this helper is bound to.

    Examples:
        db = pgVectorDB(...)

        # Create GIN index for JSONB metadata
        await db.gin.ensure_gin_index("metadata", "jsonb")

        # Create GIN index for tags array
        await db.gin.ensure_gin_index("tags", "array")

        # List existing GIN indexes
        indexes = await db.gin.list_gin_indexes()
    """

    def __init__(self, db: pgVectorDB) -> None:
        """Initialize GINIndexHelper.

        Args:
            db: pgVectorDB instance to create indexes for.
        """
        self._db = db

    async def ensure_gin_index(
        self,
        column: str,
        index_type: Literal["jsonb", "array", "tsvector"] = "jsonb",
        index_name: str | None = None,
        concurrently: bool = True,
    ) -> str:
        """Create GIN index if it doesn't exist.

        Args:
            column: Column to index (e.g., 'metadata', 'tags', 'content')
            index_type: Type of GIN index to create:
                - "jsonb": For JSONB columns with containment ops
                - "array": For array columns with overlap ops
                - "tsvector": For text columns using to_tsvector
            index_name: Custom index name (auto-generated if None)
            concurrently: Create with CONCURRENTLY to avoid locks

        Returns:
            Index name (existing or newly created)

        Raises:
            DatabaseError: If creation fails

        Examples:
            # JSONB metadata index
            await db.gin.ensure_gin_index("metadata", "jsonb")

            # Array tags index
            await db.gin.ensure_gin_index("tags", "array")

            # Full-text on content
            await db.gin.ensure_gin_index("content", "tsvector")

            # Custom name, non-concurrent (faster but locks table)
            await db.gin.ensure_gin_index("metadata", "jsonb",
                                         "idx_custom", concurrently=False)
        """
        table = self._db.collection_name
        if not table:
            raise ValueError("Table name must be set on db")

        # Generate index name if not provided
        if not index_name:
            index_name = f"idx_{table}_{column}_gin"

        # Check if index already exists
        if await self._index_exists(index_name):
            logger.debug(f"GIN index {index_name} already exists")
            return index_name

        # Build index definition based on type
        if index_type == "jsonb":
            # GIN on the jsonb column
            index_def = f"USING GIN ({column})"
        elif index_type == "array":
            # GIN on the array column
            index_def = f"USING GIN ({column})"
        elif index_type == "tsvector":
            # GIN on tsvector expression
            index_def = f"USING GIN (to_tsvector('english', {column}))"
        else:
            raise ValueError(f"Unknown index type: {index_type}")

        # Build CREATE INDEX statement
        concurrent_str = "CONCURRENTLY" if concurrently else ""
        sql = f"CREATE INDEX {concurrent_str} {index_name} ON {table} {index_def}"

        logger.info(f"Creating GIN index: {index_name} on {table}.{column}")

        async with self._db.engine.begin() as conn:
            await conn.execute(text(sql))

        # Run ANALYZE for statistics
        await conn.execute(text(f"ANALYZE {table}"))

        logger.info(f"Created GIN index: {index_name}")
        return index_name

    async def list_gin_indexes(self, table_name: str | None = None) -> list[dict[str, Any]]:
        """List all GIN indexes on a table.

        Args:
            table_name: Table to list indexes for. If None, uses db.collection_name.

        Returns:
            List of GIN index information dictionaries with:
            - index_name: str
            - column_name: str
            - index_type: str ('gin')
            - index_def: str (full index definition)
            - size_bytes: int

        Examples:
            indexes = await db.gin.list_gin_indexes()
            for idx in indexes:
                print(f"{idx['index_name']} on {idx['column_name']}")
        """
        table = table_name or self._db.collection_name
        if not table:
            raise ValueError("Table name must be provided or db.collection_name set")

        async with self._db.engine.begin() as conn:
            query = text("""
                SELECT
                    ci.relname as index_name,
                    pg_get_indexdef(ci.oid) as index_def,
                    pg_relation_size(ci.oid) as size_bytes,
                    a.attname as column_name
                FROM pg_index i
                JOIN pg_class c ON c.oid = i.indrelid
                JOIN pg_class ci ON ci.oid = i.indexrelid
                JOIN pg_attribute a ON a.attrelid = c.oid
                    AND a.attnum = ANY(i.indkey)
                WHERE c.relname = :table
                AND pg_get_indexdef(ci.oid) LIKE '%USING GIN%'
                ORDER BY ci.relname
            """)
            result = await conn.execute(query, {"table": table})

            indexes = []
            for row in result.fetchall():
                indexes.append(
                    {
                        "index_name": row.index_name,
                        "column_name": row.column_name,
                        "index_type": "gin",
                        "index_def": row.index_def,
                        "size_bytes": row.size_bytes,
                    }
                )

            return indexes

    async def create_tsvector_index(
        self,
        column: str = "content",
        config: str = "english",
        index_name: str | None = None,
        weights: dict[str, str] | None = None,
        concurrently: bool = True,
    ) -> str:
        """Create GIN index for full-text search.

        Creates a GIN index on a generated tsvector column using to_tsvector.
        Optimizes full-text queries using @@ operator.

        Args:
            column: Text column to index (default: "content")
            config: Text search configuration (default: "english")
            index_name: Custom index name (auto-generated if None)
            weights: Optional weight mapping for fields (e.g., {"title": "A", "content": "B"})
            concurrently: Create with CONCURRENTLY to avoid locks

        Returns:
            Index name

        Examples:
            # Simple full-text index
            await db.gin.create_tsvector_index("content")

            # With custom config
            await db.gin.create_tsvector_index("title", "simple")

            # With weights for multiple fields (requires generated column)
            await db.gin.create_tsvector_index(
                "content",
                weights={"title": "A", "content": "B"}
            )
        """
        table = self._db.collection_name
        if not table:
            raise ValueError("Table name must be set on db")

        if not index_name:
            index_name = f"idx_{table}_{column}_fts"

        # Check if index exists
        if await self._index_exists(index_name):
            return index_name

        # Build tsvector expression
        if weights:
            # Weighted tsvector with setweight
            weight_expr = " || ".join(
                f"setweight(to_tsvector('{config}', COALESCE({col}, '')), '{weight}')"
                for col, weight in weights.items()
            )
            index_def = f"USING GIN (({weight_expr}))"
        else:
            # Simple tsvector
            index_def = f"USING GIN (to_tsvector('{config}', {column}))"

        # Create index
        concurrent_str = "CONCURRENTLY" if concurrently else ""
        sql = f"CREATE INDEX {concurrent_str} {index_name} ON {table} {index_def}"

        logger.info(f"Creating FTS GIN index: {index_name}")

        async with self._db.engine.begin() as conn:
            await conn.execute(text(sql))
            await conn.execute(text(f"ANALYZE {table}"))

        return index_name

    async def analyze_index(self, index_name: str) -> dict[str, Any]:
        """Run ANALYZE on an index and return statistics.

        Args:
            index_name: Name of index to analyze

        Returns:
            Dictionary with statistics:
            - index_name: str
            - analyzed_at: str (ISO timestamp)
            - row_estimate: int (estimated rows analyzed)

        Examples:
            stats = await db.gin.analyze_index("idx_metadata_gin")
            print(f"Analyzed ~{stats['row_estimate']} rows")
        """
        table = self._db.collection_name

        async with self._db.engine.begin() as conn:
            # Run ANALYZE
            await conn.execute(text(f"ANALYZE {table}"))

            # Get stats
            query = text("""
                SELECT
                    relname as table_name,
                    n_live_tup as row_estimate,
                    last_vacuum,
                    last_autovacuum
                FROM pg_stat_user_tables
                WHERE relname = :table
            """)
            result = await conn.execute(query, {"table": table})
            row = result.fetchone()

            return {
                "index_name": index_name,
                "table_name": table,
                "analyzed_at": str(row.last_vacuum) if row else None,
                "row_estimate": row.row_estimate if row else 0,
            }

    async def drop_gin_index(
        self,
        index_name: str,
        if_exists: bool = True,
        concurrently: bool = True,
    ) -> bool:
        """Drop a GIN index.

        Args:
            index_name: Name of index to drop
            if_exists: Use IF EXISTS to avoid errors on missing index
            concurrently: Use DROP INDEX CONCURRENTLY to avoid locks

        Returns:
            True if index was dropped

        Examples:
            # Drop index if it exists
            await db.gin.drop_gin_index("idx_metadata_gin")

            # Force drop (may lock table briefly)
            await db.gin.drop_gin_index("idx_metadata_gin", concurrently=False)
        """
        exists_str = "IF EXISTS" if if_exists else ""
        concurrent_str = "CONCURRENTLY" if concurrently else ""

        sql = f"DROP INDEX {concurrent_str} {exists_str} {index_name}"

        logger.info(f"Dropping GIN index: {index_name}")

        async with self._db.engine.begin() as conn:
            await conn.execute(text(sql))

        return True

    async def _index_exists(self, index_name: str) -> bool:
        """Check if an index already exists."""
        async with self._db.engine.begin() as conn:
            query = text("""
                SELECT EXISTS (
                    SELECT 1 FROM pg_indexes
                    WHERE indexname = :index_name
                ) as exists
            """)
            result = await conn.execute(query, {"index_name": index_name})
            row = result.fetchone()
            return row.exists if row else False

    async def suggest_indexes(self, table_name: str | None = None) -> list[dict[str, str]]:
        """Analyze query patterns and suggest GIN indexes.

        Examines table structure and suggests GIN indexes that would
        improve query performance.

        Args:
            table_name: Table to analyze. If None, uses db.collection_name.

        Returns:
            List of suggestions with:
            - column: str
            - index_type: str ("jsonb", "array", "tsvector")
            - reason: str (why this index would help)

        Examples:
            suggestions = await db.gin.suggest_indexes()
            for s in suggestions:
                print(f"Consider: GIN on {s['column']} ({s['index_type']})")
                print(f"  Reason: {s['reason']}")
        """
        table = table_name or self._db.collection_name
        if not table:
            raise ValueError("Table name must be set")

        suggestions = []

        async with self._db.engine.begin() as conn:
            # Check for JSONB columns
            jsonb_query = text("""
                SELECT column_name
                FROM information_schema.columns
                WHERE table_name = :table
                AND data_type = 'jsonb'
            """)
            result = await conn.execute(jsonb_query, {"table": table})
            for row in result.fetchall():
                suggestions.append(
                    {
                        "column": row.column_name,
                        "index_type": "jsonb",
                        "reason": f"JSONB column '{row.column_name}' benefits from GIN for @> containment queries",
                    }
                )

            # Check for array columns
            array_query = text("""
                SELECT column_name
                FROM information_schema.columns
                WHERE table_name = :table
                AND data_type = 'ARRAY'
            """)
            result = await conn.execute(array_query, {"table": table})
            for row in result.fetchall():
                suggestions.append(
                    {
                        "column": row.column_name,
                        "index_type": "array",
                        "reason": f"Array column '{row.column_name}' benefits from GIN for && overlap queries",
                    }
                )

            # Check for text columns (suggest FTS)
            text_query = text("""
                SELECT column_name
                FROM information_schema.columns
                WHERE table_name = :table
                AND data_type IN ('text', 'character varying')
                AND column_name IN ('content', 'title', 'description', 'text')
            """)
            result = await conn.execute(text_query, {"table": table})
            for row in result.fetchall():
                suggestions.append(
                    {
                        "column": row.column_name,
                        "index_type": "tsvector",
                        "reason": f"Text column '{row.column_name}' can use GIN for full-text search (@@ operator)",
                    }
                )

        return suggestions
