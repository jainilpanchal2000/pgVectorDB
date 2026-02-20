"""
Extension Manager - PostgreSQL Extension Handling with Graceful Degradation
============================================================================

This module manages PostgreSQL extension availability and provides graceful
degradation when optional extensions are not installed.

Extension Requirements:
    - **pgvector** (REQUIRED): Core vector operations - must be installed
    - **vectorscale** (OPTIONAL): Enables DiskANN index type and label filtering
    - **pg_textsearch** (OPTIONAL): Enables BM25 keyword search ranking

Usage:
    >>> from pgvectordb.extensions import ExtensionManager
    >>> ext_manager = ExtensionManager(engine)
    >>> await ext_manager.check_extensions()
    >>> 
    >>> # Check what's available
    >>> print(f"DiskANN available: {ext_manager.has_vectorscale}")
    >>> print(f"BM25 available: {ext_manager.has_pg_textsearch}")
    >>> 
    >>> # Methods will check requirements before execution
    >>> ext_manager.require_vectorscale("build DiskANN index")
"""

import logging
from typing import Dict, Any, Optional
from packaging import version
from sqlalchemy.ext.asyncio import AsyncEngine
from sqlalchemy import text

from .base import InitializationError, DatabaseError

logger = logging.getLogger(__name__)


class ExtensionManager:
    """
    Manages PostgreSQL extension availability with graceful degradation.
    
    The pgVectorDB system has three extension dependencies:
    
    1. **pgvector** (REQUIRED)
       - Purpose: Core vector similarity search
       - Features: vector type, HNSW index, IVFFlat index, distance operators
       - Install: CREATE EXTENSION vector;
       
    2. **vectorscale** (OPTIONAL)
       - Purpose: High-performance vector search at scale
       - Features: DiskANN index, label filtering, SBQ compression
       - Install: CREATE EXTENSION vectorscale CASCADE;
       - GitHub: https://github.com/timescale/pgvectorscale
       
    3. **pg_textsearch** (OPTIONAL)
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
    
    Example:
        >>> ext = ExtensionManager(engine)
        >>> await ext.check_extensions()
        >>> 
        >>> if ext.has_vectorscale:
        ...     print("DiskANN is available!")
        ... else:
        ...     print("Using HNSW or IVFFlat (vectorscale not installed)")
        >>> 
        >>> if ext.has_pg_textsearch:
        ...     await rag.keyword_search(q, search_type=KeywordSearchType.BM25)
        ... else:
        ...     await rag.keyword_search(q, search_type=KeywordSearchType.FTS)
    """
    
    # Minimum versions for various features
    MIN_PGVECTOR_VERSION = "0.5.0"
    MIN_PGVECTOR_ITERATIVE = "0.8.0"
    MIN_VECTORSCALE_VERSION = "0.2.0"
    MIN_PG_TEXTSEARCH_VERSION = "0.3.0"
    
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
        self.pgvector_version: Optional[str] = None
        self.vectorscale_version: Optional[str] = None
        self.pg_textsearch_version: Optional[str] = None
        
        # Flags for specific features
        self.has_iterative_scan: bool = False
        
        self._checked: bool = False
    
    async def check_extensions(self) -> Dict[str, bool]:
        """
        Check PostgreSQL extension availability.
        
        Queries the database to determine which extensions are installed
        and validates version compatibility.
        
        Returns:
            Dict with keys 'pgvector', 'vectorscale', 'pg_textsearch'
            and boolean values indicating availability.
        
        Raises:
            InitializationError: If required pgvector extension is not installed.
            DatabaseError: If unable to query extension status.
        
        Example:
            >>> status = await ext.check_extensions()
            >>> print(status)
            {'pgvector': True, 'vectorscale': True, 'pg_textsearch': False}
        """
        try:
            async with self._engine.connect() as conn:
                # Check installed extensions
                result = await conn.execute(text("""
                    SELECT extname, extversion 
                    FROM pg_extension 
                    WHERE extname IN ('vector', 'vectorscale', 'pg_textsearch')
                """))
                
                installed = {}
                for row in result.fetchall():
                    installed[row[0]] = row[1]
                
                # Check pgvector (required)
                if 'vector' in installed:
                    self.has_pgvector = True
                    self.pgvector_version = installed['vector']
                    logger.info(f"✓ pgvector {self.pgvector_version} detected")
                    
                    # Check for iterative scan support
                    if version.parse(self.pgvector_version) >= version.parse(self.MIN_PGVECTOR_ITERATIVE):
                        self.has_iterative_scan = True
                        logger.info(f"  ✓ Iterative scan support available (v{self.MIN_PGVECTOR_ITERATIVE}+)")
                else:
                    # Check if it's available but not yet created
                    avail = await conn.execute(text(
                        "SELECT * FROM pg_available_extensions WHERE name = 'vector'"
                    ))
                    if avail.fetchone() is None:
                        raise InitializationError(
                            "pgvector extension is not available. "
                            "Please install pgvector: https://github.com/pgvector/pgvector"
                        )
                    # Available but not created - we'll create it later
                    self.has_pgvector = False
                
                # Check vectorscale (optional)
                if 'vectorscale' in installed:
                    self.has_vectorscale = True
                    self.vectorscale_version = installed['vectorscale']
                    logger.info(f"✓ vectorscale {self.vectorscale_version} detected (DiskANN available)")
                else:
                    # Check availability
                    avail = await conn.execute(text(
                        "SELECT * FROM pg_available_extensions WHERE name = 'vectorscale'"
                    ))
                    if avail.fetchone() is not None:
                        logger.info("○ vectorscale available but not installed (DiskANN disabled)")
                    else:
                        logger.info("○ vectorscale not available (DiskANN disabled)")
                
                # Check pg_textsearch (optional)
                if 'pg_textsearch' in installed:
                    self.has_pg_textsearch = True
                    self.pg_textsearch_version = installed['pg_textsearch']
                    logger.info(f"✓ pg_textsearch {self.pg_textsearch_version} detected (BM25 available)")
                else:
                    # Check availability
                    avail = await conn.execute(text(
                        "SELECT * FROM pg_available_extensions WHERE name = 'pg_textsearch'"
                    ))
                    if avail.fetchone() is not None:
                        logger.info("○ pg_textsearch available but not installed (BM25 disabled, using FTS)")
                    else:
                        logger.info("○ pg_textsearch not available (BM25 disabled, using FTS)")
                
                self._checked = True
                
                return {
                    'pgvector': self.has_pgvector,
                    'vectorscale': self.has_vectorscale,
                    'pg_textsearch': self.has_pg_textsearch
                }
                
        except InitializationError:
            raise
        except Exception as e:
            raise DatabaseError(f"Failed to check extensions: {e}") from e
    
    async def ensure_pgvector(self) -> None:
        """
        Ensure pgvector extension is created.
        
        Creates the pgvector extension if it's available but not yet created.
        
        Raises:
            InitializationError: If pgvector is not available.
            DatabaseError: If extension creation fails.
        """
        try:
            async with self._engine.connect() as conn:
                await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector;"))
                await conn.commit()
                self.has_pgvector = True
                logger.info("✓ Extension 'vector' enabled")
        except Exception as e:
            raise DatabaseError(f"Failed to create pgvector extension: {e}") from e
    
    async def ensure_vectorscale(self) -> bool:
        """
        Attempt to create vectorscale extension if available.
        
        Returns:
            True if vectorscale is now available, False otherwise.
        """
        if self.has_vectorscale:
            return True
            
        try:
            async with self._engine.connect() as conn:
                # Check if available
                result = await conn.execute(text(
                    "SELECT * FROM pg_available_extensions WHERE name = 'vectorscale'"
                ))
                if result.fetchone() is None:
                    return False
                
                # Try to create
                await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vectorscale CASCADE;"))
                await conn.commit()
                self.has_vectorscale = True
                logger.info("✓ Extension 'vectorscale' enabled")
                return True
        except Exception as e:
            logger.warning(f"Could not enable vectorscale: {e}")
            return False
    
    async def ensure_pg_textsearch(self) -> bool:
        """
        Attempt to create pg_textsearch extension if available.
        
        Returns:
            True if pg_textsearch is now available, False otherwise.
        """
        if self.has_pg_textsearch:
            return True
            
        try:
            async with self._engine.connect() as conn:
                # Check if available
                result = await conn.execute(text(
                    "SELECT * FROM pg_available_extensions WHERE name = 'pg_textsearch'"
                ))
                if result.fetchone() is None:
                    return False
                
                # Try to create
                await conn.execute(text("CREATE EXTENSION IF NOT EXISTS pg_textsearch;"))
                await conn.commit()
                self.has_pg_textsearch = True
                logger.info("✓ Extension 'pg_textsearch' enabled")
                return True
        except Exception as e:
            logger.warning(f"Could not enable pg_textsearch: {e}")
            return False
    
    def require_vectorscale(self, operation: str = "DiskANN operations") -> None:
        """
        Ensure vectorscale extension is available for the requested operation.
        
        Call this before any operation that requires DiskANN or label filtering.
        
        Args:
            operation: Description of the operation for error message context.
        
        Raises:
            InitializationError: If vectorscale is not installed.
                Error message includes installation instructions.
        
        Example:
            >>> ext.require_vectorscale("build DiskANN index")
            InitializationError: Cannot build DiskANN index: 
                vectorscale extension is not installed.
                
                To install vectorscale:
                1. Follow installation at https://github.com/timescale/pgvectorscale
                2. Run: CREATE EXTENSION vectorscale CASCADE;
                
                Alternative: Use IndexType.HNSW or IndexType.IVFFLAT instead.
        """
        if not self.has_vectorscale:
            raise InitializationError(
                f"Cannot {operation}: vectorscale extension is not installed.\n\n"
                "To install vectorscale:\n"
                "1. Follow installation at https://github.com/timescale/pgvectorscale\n"
                "2. Run: CREATE EXTENSION vectorscale CASCADE;\n\n"
                "Alternative: Use IndexType.HNSW or IndexType.IVFFLAT instead."
            )
    
    def require_pg_textsearch(self, operation: str = "BM25 search") -> None:
        """
        Ensure pg_textsearch extension is available for the requested operation.
        
        Call this before any operation that requires BM25 search.
        
        Args:
            operation: Description of the operation for error message context.
        
        Raises:
            InitializationError: If pg_textsearch is not installed.
                Error message includes installation instructions.
        
        Example:
            >>> ext.require_pg_textsearch("build BM25 index")
            InitializationError: Cannot build BM25 index: 
                pg_textsearch extension is not installed.
                
                To install pg_textsearch:
                1. Follow installation at https://github.com/timescale/pg_textsearch
                2. Run: CREATE EXTENSION pg_textsearch;
                
                Alternative: Use KeywordSearchType.FTS for PostgreSQL full-text search.
        """
        if not self.has_pg_textsearch:
            raise InitializationError(
                f"Cannot {operation}: pg_textsearch extension is not installed.\n\n"
                "To install pg_textsearch:\n"
                "1. Follow installation at https://github.com/timescale/pg_textsearch\n"
                "2. Run: CREATE EXTENSION pg_textsearch;\n\n"
                "Alternative: Use KeywordSearchType.FTS for PostgreSQL full-text search."
            )
    
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
    
    def get_feature_availability(self) -> Dict[str, Dict[str, Any]]:
        """
        Get detailed feature availability information.
        
        Returns:
            Dictionary mapping features to their availability status
            and requirements.
        
        Example:
            >>> avail = ext.get_feature_availability()
            >>> print(avail['DiskANN index'])
            {'available': False, 'requires': 'vectorscale', 'version': None}
        """
        return {
            "HNSW index": {
                "available": self.has_pgvector,
                "requires": "pgvector",
                "version": self.pgvector_version
            },
            "IVFFlat index": {
                "available": self.has_pgvector,
                "requires": "pgvector",
                "version": self.pgvector_version
            },
            "DiskANN index": {
                "available": self.has_vectorscale,
                "requires": "vectorscale",
                "version": self.vectorscale_version
            },
            "Label filtering": {
                "available": self.has_vectorscale,
                "requires": "vectorscale",
                "version": self.vectorscale_version
            },
            "BM25 search": {
                "available": self.has_pg_textsearch,
                "requires": "pg_textsearch",
                "version": self.pg_textsearch_version
            },
            "FTS search": {
                "available": self.has_pgvector,
                "requires": "pg_trgm (built-in)",
                "version": "built-in"
            },
            "Iterative scan": {
                "available": self.has_iterative_scan,
                "requires": f"pgvector {self.MIN_PGVECTOR_ITERATIVE}+",
                "version": self.pgvector_version
            }
        }
    
    def __repr__(self) -> str:
        return (
            f"ExtensionManager("
            f"pgvector={self.has_pgvector}, "
            f"vectorscale={self.has_vectorscale}, "
            f"pg_textsearch={self.has_pg_textsearch})"
        )


__all__ = ["ExtensionManager"]
