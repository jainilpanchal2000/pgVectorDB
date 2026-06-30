"""Index Management for pgVectorDB.

Provides utilities for monitoring and managing pgvector indexes including
progress tracking, statistics, and readiness checks.

Example:
    db = pgVectorDB(...)

    # Wait for index creation
    await db.indexes.wait_for_index("my_index", timeout=300)

    # Get index statistics
    stats = await db.indexes.index_stats()
    print(f"Index size: {stats['size_bytes']} bytes")
    print(f"Index ready: {stats['status'] == 'ready'}")
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

from sqlalchemy import text

if TYPE_CHECKING:
    from ..core import pgVectorDB


class IndexManager:
    """Manage and monitor pgvector indexes.

    Provides utilities for index lifecycle management including creation
    progress monitoring, statistics retrieval, and readiness checking.

    Attributes:
        db: The pgVectorDB instance this manager is bound to.

    Examples:
        db = pgVectorDB(...)

        # Access via db.indexes property
        stats = await db.indexes.index_stats()

        # Wait for index creation with progress
        await db.indexes.wait_for_index(timeout=300, poll_interval=1)

        # Check if ready
        if await db.indexes.is_index_ready():
            results = await db.query("test").limit(10).to_list()
    """

    def __init__(self, db: pgVectorDB) -> None:
        """Initialize IndexManager.

        Args:
            db: pgVectorDB instance to manage indexes for.
        """
        self._db = db

    async def wait_for_index(
        self,
        index_name: str | None = None,
        table_name: str | None = None,
        timeout: float = 300.0,
        poll_interval: float = 1.0,
    ) -> bool:
        """Wait for index creation to complete.

        Polls the database until the index is ready or timeout is reached.
        Useful after async index operations (CREATE INDEX CONCURRENTLY).

        Args:
            index_name: Specific index to wait for. If None, waits for any
                index on the collection table.
            table_name: Table name to check. If None, uses db.collection_name.
            timeout: Maximum seconds to wait (default: 300 = 5 minutes).
            poll_interval: Seconds between progress checks (default: 1).

        Returns:
            True if index is ready, False if timeout reached.

        Raises:
            asyncio.TimeoutError: If timeout exceeded and raise_on_timeout=True.

        Examples:
            # Wait for default index
            await db.indexes.wait_for_index()

            # Wait with timeout
            if await db.indexes.wait_for_index(timeout=60):
                print("Index ready!")
            else:
                print("Timeout")

            # Get progress
            print("Waiting for index...")
            await db.indexes.wait_for_index(poll_interval=5)
        """
        table = table_name or self._db.collection_name
        if not table:
            raise ValueError("Table name must be provided or db.collection_name set")

        start_time = asyncio.get_event_loop().time()

        while True:
            # Check if index is ready
            if await self.is_index_ready(index_name, table):
                return True

            # Check timeout
            elapsed = asyncio.get_event_loop().time() - start_time
            if elapsed >= timeout:
                return False

            await asyncio.sleep(min(poll_interval, timeout - elapsed))

    async def is_index_ready(
        self, index_name: str | None = None, table_name: str | None = None
    ) -> bool:
        """Check if index is ready for queries.

        Checks pg_index to see if indisready and indisvalid are true.

        Args:
            index_name: Specific index to check. If None, checks if any
                valid index exists on the table.
            table_name: Table name to check. If None, uses db.collection_name.

        Returns:
            True if index is ready and valid, False otherwise.

        Examples:
            if await db.indexes.is_index_ready():
                results = await db.query("test").limit(10).to_list()
        """
        table = table_name or self._db.collection_name
        if not table:
            return False

        async with self._db.engine.begin() as conn:
            if index_name:
                # Check specific index
                query = text("""
                    SELECT indisready AND indisvalid as is_ready
                    FROM pg_index i
                    JOIN pg_class c ON c.oid = i.indrelid
                    JOIN pg_class ci ON ci.oid = i.indexrelid
                    WHERE c.relname = :table
                    AND ci.relname = :index_name
                """)
                result = await conn.execute(query, {"table": table, "index_name": index_name})
                row = result.fetchone()
                return row.is_ready if row else False
            else:
                # Check if any valid index exists
                query = text("""
                    SELECT EXISTS (
                        SELECT 1 FROM pg_index i
                        JOIN pg_class c ON c.oid = i.indrelid
                        JOIN pg_class ci ON ci.oid = i.indexrelid
                        WHERE c.relname = :table
                        AND i.indisready
                        AND i.indisvalid
                    ) as has_ready_index
                """)
                result = await conn.execute(query, {"table": table})
                row = result.fetchone()
                return row.has_ready_index if row else False

    async def index_stats(
        self, table_name: str | None = None, index_name: str | None = None
    ) -> dict[str, Any]:
        """Get index statistics.

        Retrieves comprehensive statistics about index size, usage, and status
        from PostgreSQL system catalogs.

        Args:
            table_name: Table to get stats for. If None, uses db.collection_name.
            index_name: Specific index. If None, returns stats for first found index.

        Returns:
            Dictionary with index statistics:
            - index_name: str - Index relation name
            - index_type: str - Type (hnsw, ivfflat, diskann, or unknown)
            - status: str - 'ready', 'building', or 'invalid'
            - size_bytes: int - Index size in bytes
            - tuples_count: int - Approximate tuple count
            - pages_count: int | None - Number of pages (HNSW)
            - lists_count: int | None - Number of lists (IVFFlat)
            - scans: int - Number of index scans
            - tuples_read: int - Tuples read via this index

        Examples:
            stats = await db.indexes.index_stats()
            print(f"Index: {stats['index_name']}")
            print(f"Status: {stats['status']}")
            print(f"Size: {stats['size_bytes']:,} bytes")
            print(f"Scans: {stats['scans']:,}")
        """
        table = table_name or self._db.collection_name
        if not table:
            raise ValueError("Table name must be provided or db.collection_name set")

        async with self._db.engine.begin() as conn:
            # Get index information
            if index_name:
                query = text("""
                    SELECT
                        ci.relname as index_name,
                        pg_relation_size(ci.oid) as size_bytes,
                        i.indisready,
                        i.indisvalid,
                        pg_get_indexdef(ci.oid) as index_def
                    FROM pg_index i
                    JOIN pg_class c ON c.oid = i.indrelid
                    JOIN pg_class ci ON ci.oid = i.indexrelid
                    WHERE c.relname = :table
                    AND ci.relname = :index_name
                """)
                result = await conn.execute(query, {"table": table, "index_name": index_name})
            else:
                query = text("""
                    SELECT
                        ci.relname as index_name,
                        pg_relation_size(ci.oid) as size_bytes,
                        i.indisready,
                        i.indisvalid,
                        pg_get_indexdef(ci.oid) as index_def
                    FROM pg_index i
                    JOIN pg_class c ON c.oid = i.indrelid
                    JOIN pg_class ci ON ci.oid = i.indexrelid
                    WHERE c.relname = :table
                    AND i.indisvalid
                    LIMIT 1
                """)
                result = await conn.execute(query, {"table": table})

            row = result.fetchone()
            if not row:
                return {
                    "index_name": None,
                    "index_type": "none",
                    "status": "not_found",
                    "size_bytes": 0,
                }

            # Determine index type
            index_def = row.index_def.lower()
            if "hnsw" in index_def:
                index_type = "hnsw"
            elif "ivfflat" in index_def:
                index_type = "ivfflat"
            elif "diskann" in index_def:
                index_type = "diskann"
            else:
                index_type = "unknown"

            # Determine status
            if row.indisready and row.indisvalid:
                status = "ready"
            elif not row.indisready:
                status = "building"
            else:
                status = "invalid"

            # Get usage stats
            stats_query = text("""
                SELECT
                    idx_scan as scans,
                    idx_tup_read
                FROM pg_stat_user_indexes
                WHERE relname = :table
                AND indexrelname = :index_name
            """)
            stats_result = await conn.execute(
                stats_query, {"table": table, "index_name": row.index_name}
            )
            stats_row = stats_result.fetchone()

            return {
                "index_name": row.index_name,
                "index_type": index_type,
                "status": status,
                "size_bytes": row.size_bytes,
                "tuples_count": row.tuples_count,
                "scans": stats_row.scans if stats_row else 0,
                "tuples_read": stats_row.idx_tup_read if stats_row else 0,
            }

    async def list_indexes(self, table_name: str | None = None) -> list[dict[str, Any]]:
        """List all indexes on a table.

        Args:
            table_name: Table to list indexes for. If None, uses db.collection_name.

        Returns:
            List of index information dictionaries.

        Examples:
            indexes = await db.indexes.list_indexes()
            for idx in indexes:
                print(f"{idx['index_name']}: {idx['index_type']}")
        """
        table = table_name or self._db.collection_name
        if not table:
            raise ValueError("Table name must be provided or db.collection_name set")

        async with self._db.engine.begin() as conn:
            query = text("""
                SELECT
                    ci.relname as index_name,
                    pg_relation_size(ci.oid) as size_bytes,
                    i.indisready,
                    i.indisvalid,
                    pg_get_indexdef(ci.oid) as index_def
                FROM pg_index i
                JOIN pg_class c ON c.oid = i.indrelid
                JOIN pg_class ci ON ci.oid = i.indexrelid
                WHERE c.relname = :table
                ORDER BY ci.relname
            """)
            result = await conn.execute(query, {"table": table})

            indexes = []
            for row in result.fetchall():
                # Determine index type
                index_def = row.index_def.lower()
                if "hnsw" in index_def:
                    index_type = "hnsw"
                elif "ivfflat" in index_def:
                    index_type = "ivfflat"
                elif "diskann" in index_def:
                    index_type = "diskann"
                elif "gin" in index_def:
                    index_type = "gin"
                else:
                    index_type = "btree"

                indexes.append(
                    {
                        "index_name": row.index_name,
                        "index_type": index_type,
                        "size_bytes": row.size_bytes,
                        "is_ready": row.indisready,
                        "is_valid": row.indisvalid,
                        "definition": row.index_def,
                    }
                )

            return indexes

    async def rebuild_index(
        self,
        index_name: str | None = None,
        concurrently: bool = True,
        timeout: float | None = None,
    ) -> bool:
        """Rebuild an index.

        Args:
            index_name: Index to rebuild. If None, rebuilds main vector index.
            concurrently: Use REINDEX CONCURRENTLY (avoids locks).
            timeout: Maximum time to wait for rebuild to complete.

        Returns:
            True if rebuild succeeded.

        Examples:
            # Rebuild default index
            await db.indexes.rebuild_index()

            # Rebuild specific index
            await db.indexes.rebuild_index("my_hnsw_idx")
        """
        # Get current index if not specified
        if not index_name:
            indexes = await self.list_indexes()
            vector_indexes = [
                i for i in indexes if i["index_type"] in ("hnsw", "ivfflat", "diskann")
            ]
            if not vector_indexes:
                raise ValueError("No vector index found to rebuild")
            index_name = vector_indexes[0]["index_name"]

        # Get table name
        table = self._db.collection_name
        if not table:
            raise ValueError("Table name must be set on db")

        prefix = "CONCURRENTLY" if concurrently else ""
        sql = f"REINDEX {prefix} INDEX {index_name}"

        async with self._db.engine.begin() as conn:
            await conn.execute(text(sql))

        # Wait if timeout specified
        if timeout:
            return await self.wait_for_index(index_name, timeout=timeout)

        return True
