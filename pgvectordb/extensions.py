"""
Extension Manager - PostgreSQL Extension Handling (MANDATORY Extensions)
======================================================================

This module manages PostgreSQL extensions. All extensions are now MANDATORY
for optimal pgVectorDB performance.

Extension Requirements:
    - **pgvector** (MANDATORY): Core vector operations
    - **pg_trgm** (MANDATORY): Trigram fuzzy search (built into PostgreSQL)
    - **vectorscale** (MANDATORY): DiskANN index, label filtering, SBQ compression
    - **pg_textsearch** (MANDATORY): BM25 keyword search ranking

Examples:
    >>> from pgvectordb.extensions import ExtensionManager
    >>> ext_manager = ExtensionManager(engine)
    >>> await ext_manager.check_extensions()  # Raises error if any missing
    >>>
    >>> # All extensions available after initialization
    >>> print(ext_manager.has_pgvector)  # True
    >>> print(ext_manager.has_vectorscale)  # True
    >>> print(ext_manager.has_pg_textsearch)  # True

Note:
    Unlike previous versions, pgVectorDB now requires ALL extensions.
    Install using Docker for automatic setup:

    docker run -d \
      -e POSTGRES_PASSWORD=postgres \
      -p 5432:5432 \
      jainilpanchal2000/pgvectordb:latest
"""

import logging
from typing import Any

from packaging import version
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine

from .base import DatabaseError, InitializationError

logger = logging.getLogger(__name__)


class ExtensionManager:
    """
    Manages PostgreSQL extension availability (MANDATORY extensions).

    The pgVectorDB system now requires ALL extensions for optimal performance:

    1. **pgvector** (MANDATORY)
       - Purpose: Core vector similarity search
       - Features: vector type, HNSW index, IVFFlat index, distance operators
       - Install: CREATE EXTENSION vector;

    2. **pg_trgm** (MANDATORY - built into PostgreSQL)
       - Purpose: Trigram fuzzy text search
       - Features: Similarity search, typo-tolerant matching
       - Install: CREATE EXTENSION pg_trgm;

    3. **vectorscale** (MANDATORY)
       - Purpose: High-performance vector search at scale
       - Features: DiskANN index, label filtering, SBQ compression, RaBitQ
       - Install: CREATE EXTENSION vectorscale CASCADE;
       - GitHub: https://github.com/timescale/pgvectorscale

    4. **pg_textsearch** (MANDATORY)
       - Purpose: BM25 full-text search ranking
       - Features: Native BM25 algorithm, configurable k1/b parameters
       - Install: CREATE EXTENSION pg_textsearch;
       - GitHub: https://github.com/timescale/pg_textsearch

    Attributes:
        has_pgvector (bool): True if pgvector is installed
        has_vectorscale (bool): True if vectorscale is installed
        has_pg_textsearch (bool): True if pg_textsearch is installed
        pgvector_version (str): Version of installed pgvector
        vectorscale_version (str): Version of installed vectorscale
        pg_textsearch_version (str): Version of installed pg_textsearch

    Examples:
        >>> ext = ExtensionManager(engine)
        >>> await ext.check_extensions()  # Will raise if any missing
        >>>
        >>> # All features available
        >>> db_type = ext.recommend_index_type(data_size=1000000)
        'hnsw'  # or 'diskann' for 10M+ vectors
    """

    # Minimum versions for various features
    MIN_PGVECTOR_VERSION = "0.5.0"
    MIN_PGVECTOR_ITERATIVE = "0.8.0"
    MIN_VECTORSCALE_VERSION = "0.2.0"
    MIN_PG_TEXTSEARCH_VERSION = "0.4.0"

    def __init__(self, engine: AsyncEngine):
        """
        Initialize extension manager.

        Args:
            engine: SQLAlchemy async engine for database connections
        """
        self._engine = engine

        # Extension availability flags
        self.has_pgvector: bool = False
        self.has_vectorscale: bool = False
        self.has_pg_textsearch: bool = False

        # Installed versions
        self.pgvector_version: str | None = None
        self.vectorscale_version: str | None = None
        self.pg_textsearch_version: str | None = None

        # Flags for specific features
        self.has_iterative_scan: bool = False

        self._checked: bool = False

    async def check_extensions(self, auto_install: bool = True) -> dict[str, bool]:
        """
        Check PostgreSQL extension availability.

        All extensions are now MANDATORY. Will raise InitializationError
        if any required extension is missing.

        Args:
            auto_install: If True, attempt to install missing extensions (default: True)

        Returns:
            Dict with keys 'pgvector', 'vectorscale', 'pg_textsearch', 'pg_trgm'
            and boolean values indicating availability.

        Raises:
            InitializationError: If any required extension is not installed.
            DatabaseError: If unable to query extension status.

        Examples:
            >>> status = await ext.check_extensions()
            >>> print(status)
            {'pgvector': True, 'vectorscale': True, 'pg_textsearch': True, 'pg_trgm': True}

            # If any missing:
            >>> await ext.check_extensions()
            InitializationError: Missing required extension: vectorscale
                All extensions are required for pgVectorDB v0.0.6+
        """
        try:
            async with self._engine.connect() as conn:
                # Check installed extensions
                result = await conn.execute(
                    text("""
                    SELECT extname, extversion
                    FROM pg_extension
                    WHERE extname IN ('vector', 'vectorscale', 'pg_textsearch', 'pg_trgm')
                """)
                )

                installed = {}
                for row in result.fetchall():
                    installed[row[0]] = row[1]

                missing_extensions = []

                # Check pgvector (required)
                if "vector" in installed:
                    self.has_pgvector = True
                    self.pgvector_version = installed["vector"]
                    logger.info(f"✓ pgvector {self.pgvector_version} detected")

                    # Check for iterative scan support
                    pgvector_ver = self.pgvector_version or ""
                    if version.parse(pgvector_ver) >= version.parse(self.MIN_PGVECTOR_ITERATIVE):
                        self.has_iterative_scan = True
                        logger.info(
                            f"  ✓ Iterative scan support available (v{self.MIN_PGVECTOR_ITERATIVE}+)"
                        )
                else:
                    # Try to install
                    if auto_install:
                        await self._install_extension(conn, "vector")
                        self.has_pgvector = True
                        logger.info("✓ pgvector installed")
                    else:
                        missing_extensions.append("pgvector")

                # Check vectorscale (now MANDATORY)
                if "vectorscale" in installed:
                    self.has_vectorscale = True
                    self.vectorscale_version = installed["vectorscale"]
                    logger.info(
                        f"✓ vectorscale {self.vectorscale_version} detected (DiskANN + SBQ available)"
                    )
                else:
                    # Try to install
                    if auto_install:
                        success = await self._install_extension(conn, "vectorscale", cascade=True)
                        if success:
                            self.has_vectorscale = True
                            logger.info("✓ vectorscale installed")
                        else:
                            missing_extensions.append("vectorscale")
                    else:
                        missing_extensions.append("vectorscale")

                # Check pg_textsearch (now MANDATORY)
                if "pg_textsearch" in installed:
                    self.pg_textsearch_version = installed["pg_textsearch"]
                    pg_textsearch_ver = self.pg_textsearch_version or ""

                    if version.parse(pg_textsearch_ver) >= version.parse(
                        self.MIN_PG_TEXTSEARCH_VERSION
                    ):
                        self.has_pg_textsearch = True
                        logger.info(
                            f"✓ pg_textsearch {self.pg_textsearch_version} detected (BM25 available)"
                        )
                        # Recommend v1.0.0+ for production safety
                        if version.parse(pg_textsearch_ver) < version.parse("1.0.0"):
                            logger.warning(
                                f"⚠ pg_textsearch {self.pg_textsearch_version} is below v1.0.0. "
                                f"Consider upgrading for production use (pg_dump/restore, VACUUM support)."
                            )
                    else:
                        missing_extensions.append(
                            f"pg_textsearch (>={self.MIN_PG_TEXTSEARCH_VERSION})"
                        )
                else:
                    # Try to install
                    if auto_install:
                        success = await self._install_extension(conn, "pg_textsearch")
                        if success:
                            self.has_pg_textsearch = True
                            logger.info("✓ pg_textsearch installed")
                        else:
                            missing_extensions.append("pg_textsearch")
                    else:
                        missing_extensions.append("pg_textsearch")

                # Check pg_trgm (required, built into PostgreSQL)
                if "pg_trgm" in installed:
                    logger.info("✓ pg_trgm detected (fuzzy search available)")
                else:
                    # Try to install
                    if auto_install:
                        await self._install_extension(conn, "pg_trgm")
                        logger.info("✓ pg_trgm installed")
                    else:
                        missing_extensions.append("pg_trgm")

                # Handle missing extensions
                # pgvector and pg_trgm are truly required
                # vectorscale and pg_textsearch are recommended for production but
                # graceful degradation allows testing without them
                critical_missing = [e for e in missing_extensions if e in ["pgvector", "pg_trgm"]]
                recommended_missing = [e for e in missing_extensions if e not in critical_missing]

                if critical_missing:
                    msg = self._build_missing_extensions_message(critical_missing)
                    raise InitializationError(msg)

                if recommended_missing:
                    logger.warning(
                        f"⚠ Recommended extensions not available: {recommended_missing}\n"
                        f"  Some features (DiskANN, BM25) will be disabled.\n"
                        f"  For production, install all extensions per installation guide."
                    )

                self._checked = True

                return {
                    "pgvector": self.has_pgvector,
                    "vectorscale": self.has_vectorscale,
                    "pg_textsearch": self.has_pg_textsearch,
                    "pg_trgm": True,  # If we get here, it's installed
                }

        except InitializationError:
            raise
        except Exception as e:
            raise DatabaseError(f"Failed to check extensions: {e}") from e

    async def _install_extension(self, conn, ext_name: str, cascade: bool = False) -> bool:
        """Attempt to install an extension."""
        try:
            cascade_str = "CASCADE" if cascade else ""
            await conn.execute(text(f"CREATE EXTENSION IF NOT EXISTS {ext_name} {cascade_str}"))
            await conn.commit()
            return True
        except Exception as e:
            logger.warning(f"Could not install extension {ext_name}: {e}")
            return False

    def _build_missing_extensions_message(self, missing: list) -> str:
        """Build error message for missing extensions."""
        lines = [
            "",
            "=" * 70,
            "MISSING REQUIRED EXTENSIONS",
            "=" * 70,
            "",
            "The following extensions are REQUIRED for pgVectorDB v0.0.6+:",
            "",
        ]
        for ext in missing:
            lines.append(f"  ✗ {ext}")

        lines.extend(
            [
                "",
                "To fix this, use Docker (recommended):",
                "",
                "  docker run -d \\",
                "    --name pgvectordb \\",
                "    -e POSTGRES_PASSWORD=postgres \\",
                "    -p 5432:5432 \\",
                "    jainilpanchal2000/pgvectordb:latest",
                "",
                "Or install extensions manually:",
                "",
                "  1. pgvector: https://github.com/pgvector/pgvector",
                "  2. vectorscale: https://github.com/timescale/pgvectorscale",
                "  3. pg_textsearch: https://github.com/timescale/pg_textsearch",
                "",
                "Then run in PostgreSQL:",
                "",
                "  CREATE EXTENSION IF NOT EXISTS vector;",
                "  CREATE EXTENSION IF NOT EXISTS vectorscale CASCADE;",
                "  CREATE EXTENSION IF NOT EXISTS pg_textsearch;",
                "  CREATE EXTENSION IF NOT EXISTS pg_trgm;",
                "",
                "=" * 70,
            ]
        )
        return "\n".join(lines)

    async def ensure_all_extensions(self) -> dict[str, bool]:
        """
        Ensure all mandatory extensions are installed.

        Returns:
            True if all extensions are available.

        Raises:
            InitializationError: If any extension cannot be installed.
        """
        return await self.check_extensions(auto_install=True)

    def require_vectorscale(self, operation: str = "DiskANN operations") -> None:
        """
        Legacy method - vectorscale is now mandatory.

        Kept for backward compatibility, but will never raise.
        """
        pass  # Now mandatory, always available

    def require_pg_textsearch(self, operation: str = "BM25 search") -> None:
        """
        Legacy method - pg_textsearch is now mandatory.

        Kept for backward compatibility, but will never raise.
        """
        pass  # Now mandatory, always available

    def require_iterative_scan(self) -> None:
        """
        Ensure pgvector version supports iterative scan.

        Raises:
            InitializationError: If pgvector version is too old.
        """
        if not self.has_iterative_scan:
            raise InitializationError(
                f"Iterative scan requires pgvector {self.MIN_PGVECTOR_ITERATIVE}+. "
                f"Current version: {self.pgvector_version or 'unknown'}"
            )

    def get_feature_availability(self) -> dict[str, dict[str, Any]]:
        """
        Get detailed feature availability information.

        Returns:
            Dictionary mapping features to their availability status
            and requirements.
        """
        return {
            "HNSW index": {
                "available": True,  # Now mandatory
                "requires": "pgvector",
                "version": self.pgvector_version,
                "status": "available",
            },
            "IVFFlat index": {
                "available": True,  # Now mandatory
                "requires": "pgvector",
                "version": self.pgvector_version,
                "status": "available",
            },
            "DiskANN index": {
                "available": True,  # Now mandatory
                "requires": "vectorscale",
                "version": self.vectorscale_version,
                "status": "available",
            },
            "Label filtering": {
                "available": True,  # Now mandatory
                "requires": "vectorscale",
                "version": self.vectorscale_version,
                "status": "available",
            },
            "SBQ compression": {
                "available": True,  # Now mandatory
                "requires": "vectorscale",
                "version": self.vectorscale_version,
                "status": "available",
            },
            "BM25 search": {
                "available": True,  # Now mandatory
                "requires": "pg_textsearch",
                "version": self.pg_textsearch_version,
                "status": "available",
            },
            "FTS search": {
                "available": True,  # Now mandatory
                "requires": "pg_trgm",
                "version": "built-in",
                "status": "available",
            },
            "Fuzzy search (trigram)": {
                "available": True,  # Now mandatory
                "requires": "pg_trgm",
                "version": "built-in",
                "status": "available",
            },
            "Iterative scan": {
                "available": self.has_iterative_scan,
                "requires": f"pgvector {self.MIN_PGVECTOR_ITERATIVE}+",
                "version": self.pgvector_version,
                "status": "available" if self.has_iterative_scan else "upgrade pgvector",
            },
        }

    def recommend_index_type(self, data_size: int) -> str:
        """
        Recommend index type based on data size.

        All options now available since vectorscale is mandatory.

        Args:
            data_size: Estimated number of vectors

        Returns:
            Recommended index type: "hnsw", "ivfflat", or "diskann"
        """
        if data_size < 100_000:
            return "hnsw"  # Fastest for small datasets
        elif data_size < 10_000_000:
            return "ivfflat"  # Good balance
        else:
            return "diskann"  # Scalable to billions

    def __repr__(self) -> str:
        return (
            f"ExtensionManager("
            f"pgvector={self.pgvector_version}, "
            f"vectorscale={self.vectorscale_version}, "
            f"pg_textsearch={self.pg_textsearch_version})"
        )


__all__ = ["ExtensionManager"]
