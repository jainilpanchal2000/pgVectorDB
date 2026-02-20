"""
Data export/import, specialized table creation, and table lifecycle mixin for pgVectorDB.

Provides: export_to_json, import_from_json, create_halfvec_table, create_sparsevec_table, delete_table.
"""

import json
import logging
from typing import Dict, Optional, Any

from langchain_core.documents import Document
from sqlalchemy import text

from ..base import (
    ValidationError,
    DatabaseError,
)
from ..schema import build_qualified_name

logger = logging.getLogger(__name__)


class StorageMixin:
    """Mixin providing data export/import and specialized table operations."""

    async def export_to_json(
        self,
        output_file: str,
        filter: Optional[Dict[str, Any]] = None,
        include_embeddings: bool = False,
    ) -> int:
        """
        Export documents to JSON file for backup or migration.

        Args:
            output_file: Path to output JSON file
            filter: Optional metadata filter (None = export all documents)
            include_embeddings: If True, include embedding vectors (makes file much larger)

        Returns:
            Number of documents exported

        Example:
            >>> # Export all documents (without embeddings for smaller file)
            >>> count = await rag.export_to_json("backup.json")
            >>>
            >>> # Export filtered documents with embeddings
            >>> count = await rag.export_to_json(
            ...     "active_docs.json",
            ...     filter={"status": "active"},
            ...     include_embeddings=True
            ... )
        """
        self._ensure_initialized()

        from pathlib import Path

        try:
            if filter:
                filter_clauses, params = self._build_filter_clauses_wrapper(filter)
                where_clause = f"WHERE {filter_clauses}"
            else:
                where_clause = ""
                params = {}

            # Select columns based on include_embeddings
            if include_embeddings:
                select_columns = (
                    '"langchain_id", "content", "langchain_metadata", "embedding"'
                )
            else:
                select_columns = '"langchain_id", "content", "langchain_metadata"'

            full_query = text(f"""
                SELECT {select_columns}
                FROM "{self.schema_name}"."{self.table_name}"
                {where_clause}
            """)

            async with self.sqlalchemy_engine.connect() as conn:
                result = await conn.execute(full_query, params)

                documents = []
                for row in result.fetchall():
                    doc = {
                        "id": str(row[0]),
                        "content": row[1],
                        "metadata": row[2] or {},
                    }
                    if include_embeddings and len(row) > 3:
                        doc["embedding"] = row[3]
                    documents.append(doc)

            # Write to file
            output_path = Path(output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(documents, f, indent=2, ensure_ascii=False)

            logger.info(f"✓ Exported {len(documents)} documents to {output_file}")
            return len(documents)
        except Exception as e:
            raise DatabaseError(f"Failed to export to JSON: {e}") from e

    async def import_from_json(
        self, input_file: str, batch_size: int = 100, skip_existing: bool = True
    ) -> int:
        """
        Import documents from JSON backup file.

        Args:
            input_file: Path to input JSON file
            batch_size: Number of documents per batch
            skip_existing: If True, skip documents with existing IDs

        Returns:
            Number of documents imported

        Example:
            >>> # Restore from backup
            >>> count = await rag.import_from_json("backup.json")
            >>> print(f"Imported {count} documents")
        """
        self._ensure_initialized()

        from pathlib import Path

        try:
            input_path = Path(input_file)
            if not input_path.exists():
                raise ValidationError(f"Input file not found: {input_file}")

            with open(input_path, "r", encoding="utf-8") as f:
                documents_data = json.load(f)

            if not isinstance(documents_data, list):
                raise ValidationError("JSON file must contain an array of documents")

            # Convert to Document objects
            documents = []
            for doc_data in documents_data:
                # Ensure langchain_id is in metadata
                metadata = doc_data.get("metadata", {})
                if "id" in doc_data:
                    metadata["langchain_id"] = doc_data["id"]

                doc = Document(
                    page_content=doc_data.get("content", ""), metadata=metadata
                )
                documents.append(doc)

            # Check for existing IDs if skip_existing is True
            if skip_existing:
                existing_ids = set()
                async with self.sqlalchemy_engine.connect() as conn:
                    result = await conn.execute(
                        text(f"""
                        SELECT langchain_id 
                        FROM "{self.schema_name}"."{self.table_name}"
                    """)
                    )
                    existing_ids = {str(row[0]) for row in result.fetchall()}

                # Filter out existing documents
                documents = [
                    doc
                    for doc in documents
                    if doc.metadata.get("langchain_id") not in existing_ids
                ]
                logger.info(
                    f"Skipping {len(documents_data) - len(documents)} existing documents"
                )

            if not documents:
                logger.info("No new documents to import")
                return 0

            # Import in batches
            imported_ids = await self.add_documents_batch(
                documents, batch_size=batch_size, show_progress=True
            )

            logger.info(f"✓ Imported {len(imported_ids)} documents from {input_file}")
            return len(imported_ids)
        except Exception as e:
            raise DatabaseError(f"Failed to import from JSON: {e}") from e

    async def create_halfvec_table(
        self, table_name: Optional[str] = None, overwrite_existing: bool = False
    ) -> str:
        """
        Create a table with half-precision vectors (halfvec) for 50% storage savings.

        Half-precision vectors use 2 bytes per dimension instead of 4 bytes,
        cutting storage in half with minimal accuracy loss for most use cases.

        Args:
            table_name: Name for the halfvec table (default: {current_table}_halfvec)
            overwrite_existing: Drop existing table if exists (default: False)

        Returns:
            Name of the created table

        Example:
            >>> halfvec_table = await rag.create_halfvec_table()
            >>> print(f"Created {halfvec_table} with half-precision vectors")

        Note:
            - Requires pgvector 0.7.0+
            - Use with halfvec_l2_ops, halfvec_cosine_ops, halfvec_ip_ops
            - Maximum 4,000 dimensions (vs 2,000 for full precision)
        """
        self._ensure_initialized()

        halfvec_table = table_name or f"{self.table_name}_halfvec"
        qualified_table = build_qualified_name(self.schema_name, halfvec_table)

        try:
            async with self.sqlalchemy_engine.connect() as conn:
                if overwrite_existing:
                    await conn.execute(
                        text(f"DROP TABLE IF EXISTS {qualified_table} CASCADE")
                    )

                # Create table with halfvec type
                await conn.execute(
                    text(f"""
                    CREATE TABLE IF NOT EXISTS {qualified_table} (
                        langchain_id VARCHAR(255) PRIMARY KEY,
                        content TEXT NOT NULL,
                        langchain_metadata JSONB DEFAULT '{{}}'::jsonb,
                        embedding halfvec({self.vector_size}),
                        content_tsvector tsvector,
                        labels SMALLINT[],
                        content_hash VARCHAR(32),
                        created_at TIMESTAMPTZ DEFAULT NOW()
                    )
                """)
                )

                # Create tsvector trigger
                await conn.execute(
                    text(f"""
                    CREATE OR REPLACE FUNCTION update_{halfvec_table}_tsvector() RETURNS TRIGGER AS $$
                    BEGIN
                        NEW.content_tsvector := to_tsvector('english', COALESCE(NEW.content, ''));
                        RETURN NEW;
                    END;
                    $$ LANGUAGE plpgsql;
                """)
                )

                await conn.execute(
                    text(f"""
                    DROP TRIGGER IF EXISTS tsvector_update_{halfvec_table} ON {qualified_table}
                """)
                )

                await conn.execute(
                    text(f"""
                    CREATE TRIGGER tsvector_update_{halfvec_table}
                    BEFORE INSERT OR UPDATE ON {qualified_table}
                    FOR EACH ROW EXECUTE FUNCTION update_{halfvec_table}_tsvector()
                """)
                )

                await conn.commit()

            logger.info(f"✓ Created half-precision table: {halfvec_table}")
            return halfvec_table

        except Exception as e:
            raise DatabaseError(f"Failed to create halfvec table: {e}") from e

    # ==================== REMAINING TASK 3: SPARSE VECTOR TABLE (Task 6) ====================

    async def create_sparsevec_table(
        self,
        table_name: Optional[str] = None,
        max_dimensions: int = 10000,
        overwrite_existing: bool = False,
    ) -> str:
        """
        Create a table with sparse vectors for high-dimensional sparse data.

        Sparse vectors are efficient for:
        - TF-IDF vectors
        - One-hot encodings
        - Bag-of-words representations
        - Any data where most values are zero

        Args:
            table_name: Name for the sparsevec table (default: {current_table}_sparse)
            max_dimensions: Maximum sparse vector dimensions (default: 10000)
            overwrite_existing: Drop existing table if exists (default: False)

        Returns:
            Name of the created table

        Example:
            >>> sparse_table = await rag.create_sparsevec_table(max_dimensions=50000)
            >>> print(f"Created {sparse_table} for sparse vectors")

        Note:
            - Format: '{index1:value1,index2:value2}/dimensions'
            - Supports up to 16,000 non-zero elements
            - Uses sparsevec_l2_ops, sparsevec_cosine_ops, sparsevec_ip_ops
        """
        self._ensure_initialized()

        sparse_table = table_name or f"{self.table_name}_sparse"
        qualified_table = build_qualified_name(self.schema_name, sparse_table)

        try:
            async with self.sqlalchemy_engine.connect() as conn:
                if overwrite_existing:
                    await conn.execute(
                        text(f"DROP TABLE IF EXISTS {qualified_table} CASCADE")
                    )

                # Create table with sparsevec type
                await conn.execute(
                    text(f"""
                    CREATE TABLE IF NOT EXISTS {qualified_table} (
                        langchain_id VARCHAR(255) PRIMARY KEY,
                        content TEXT NOT NULL,
                        langchain_metadata JSONB DEFAULT '{{}}'::jsonb,
                        embedding sparsevec({max_dimensions}),
                        content_tsvector tsvector,
                        created_at TIMESTAMPTZ DEFAULT NOW()
                    )
                """)
                )

                await conn.commit()

            logger.info(
                f"✓ Created sparse vector table: {sparse_table} (max {max_dimensions} dims)"
            )
            return sparse_table

        except Exception as e:
            raise DatabaseError(f"Failed to create sparsevec table: {e}") from e

    async def delete_table(self) -> None:
        """
        Drop the collection table and all associated indexes.

        This permanently deletes the table, all documents, embeddings, and
        indexes. Use with caution — this action cannot be undone.

        Useful for:
        - Test cleanup
        - Resetting a collection
        - Removing unused collections

        Raises:
            InitializationError: If system not initialized
            DatabaseError: If table deletion fails

        Example:
            >>> await rag.delete_table()
            >>> await rag.close()
        """
        self._ensure_initialized()

        try:
            qualified_table = build_qualified_name(self.schema_name, self.table_name)
            async with self.sqlalchemy_engine.connect() as conn:
                await conn.execute(
                    text(f"DROP TABLE IF EXISTS {qualified_table} CASCADE")
                )
                await conn.commit()

            logger.info(f"✓ Deleted table: {self.schema_name}.{self.table_name}")
        except Exception as e:
            raise DatabaseError(f"Failed to delete table: {e}") from e
