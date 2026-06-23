"""
Document CRUD operations mixin for pgVectorDB.

Provides: add_documents, aupdate_documents, adelete, add_documents_batch,
add_documents_batch_isolated, update_metadata, aget_by_ids, upsert_documents,
bulk_load_documents, add_documents_orm, and embedding/dedup helpers.
"""

import hashlib
import json
import logging
import uuid
from typing import Any, Dict, List, Optional, Tuple

from langchain_core.documents import Document
from sqlalchemy import inspect, text

from ..base import (
    DatabaseError,
    IndexType,
    QueryResult,
    RateLimitError,
    ValidationError,
)
from ..schema import build_qualified_name, get_vector_table

from ._base import MixinBase

logger = logging.getLogger(__name__)


class DocumentsMixin(MixinBase):
    """Mixin providing document CRUD operations."""

    async def add_documents(
        self, documents: List[Document], labels: Optional[List[List[int]]] = None
    ) -> List[str]:
        """
        Add documents with optional labels for DiskANN filtering.

        Args:
            documents: List of LangChain Document objects
            labels: Optional list of label arrays for DiskANN (one per document)

        Returns:
            List of document IDs

        Raises:
            InitializationError: If system not initialized
            ValidationError: If documents list is empty or labels mismatch
            DatabaseError: If document insertion fails
        """
        self._ensure_initialized()

        if not documents:
            raise ValidationError("documents list cannot be empty")

        if labels is not None:
            if len(labels) != len(documents):
                raise ValidationError(
                    f"labels length ({len(labels)}) must match documents length ({len(documents)})"
                )
            for i, label_list in enumerate(labels):
                for label in label_list:
                    if not isinstance(label, int) or label < -32768 or label > 32767:
                        raise ValidationError(
                            f"Label {label} in document {i} is outside smallint range (-32768 to 32767)"
                        )

        try:
            for i, doc in enumerate(documents):
                if "langchain_id" not in doc.metadata:
                    doc.metadata["langchain_id"] = str(uuid.uuid4())
                if labels is not None and self.index_type == IndexType.DISKANN:
                    doc.metadata["labels"] = labels[i]

            if self._vector_store is None:
                raise RuntimeError("Vector store is not initialized")
            doc_ids = await self._vector_store.aadd_documents(documents)

            if labels is not None and self.index_type == IndexType.DISKANN:
                await self._add_labels_column(doc_ids, labels)

            logger.info(f"Added {len(doc_ids)} documents")
            return doc_ids
        except Exception as e:
            raise DatabaseError(f"Failed to add documents: {e}") from e

    async def _add_labels_column(
        self, doc_ids: List[str], labels: List[List[int]]
    ) -> None:
        """Add labels column for DiskANN filtering."""
        qualified_table = build_qualified_name(self.schema_name, self.table_name)
        try:
            async with self.sqlalchemy_engine.connect() as conn:
                await conn.execute(
                    text(
                        f"ALTER TABLE {qualified_table} "
                        f"ADD COLUMN IF NOT EXISTS labels SMALLINT[]"
                    )
                )

                for doc_id, label_list in zip(doc_ids, labels):
                    await conn.execute(
                        text(
                            f"UPDATE {qualified_table} "
                            f"SET labels = :labels WHERE langchain_id = :doc_id"
                        ),
                        {"labels": label_list, "doc_id": doc_id},
                    )

                await conn.commit()
            logger.info("Labels added for DiskANN filtering")
        except Exception as e:
            raise DatabaseError(f"Failed to add labels column: {e}") from e

    async def aupdate_documents(
        self, documents: List[Document], update_embeddings: bool = True
    ) -> List[str]:
        """
        Update existing documents without having to delete and re-add.

        Efficiently updates document content and/or metadata. Can optionally
        skip re-embedding if only metadata changed.

        Args:
            documents: List of Documents with 'id' in metadata (required for matching)
            update_embeddings: If True, re-compute embeddings for content changes (default: True)
                              Set to False if only updating metadata to save computation

        Returns:
            List of updated document IDs

        Raises:
            ValidationError: If documents missing 'langchain_id' or list is empty
            DatabaseError: If update operation fails

        Examples:
            >>> # Update metadata only (fast - no re-embedding)
            >>> docs[0].metadata['status'] = 'reviewed'
            >>> docs[0].metadata['langchain_id'] = 'existing-id'
            >>> await pgvdb.aupdate_documents(docs, update_embeddings=False)
            >>>
            >>> # Update content (re-embeds automatically)
            >>> docs[1].page_content = "Updated content here"
            >>> docs[1].metadata['langchain_id'] = 'existing-id-2'
            >>> await pgvdb.aupdate_documents(docs, update_embeddings=True)
        """
        self._ensure_initialized()

        if not documents:
            # Nothing to update
            return []

        updated_ids = []

        try:
            async with self.sqlalchemy_engine.connect() as conn:
                for doc in documents:
                    # Validate document has ID
                    doc_id = doc.metadata.get("langchain_id")
                    if not doc_id:
                        raise ValidationError(
                            "Each document must have 'langchain_id' in metadata for updates"
                        )

                    # Build update query based on what needs updating
                    if update_embeddings:
                        # Re-compute embedding for content
                        import json

                        embedding = self.embedding_model.embed_query(doc.page_content)

                        update_query = text(f"""
                            UPDATE "{self.schema_name}"."{self.table_name}"
                            SET content = :content,
                                langchain_metadata = CAST(:metadata AS jsonb),
                                embedding = :embedding
                            WHERE langchain_id = :doc_id
                        """)

                        await conn.execute(
                            update_query,
                            {
                                "content": doc.page_content,
                                "metadata": json.dumps(doc.metadata),
                                "embedding": str(embedding),
                                "doc_id": doc_id,
                            },
                        )
                    else:
                        # Update only content and metadata (no embedding)
                        import json

                        update_query = text(f"""
                            UPDATE "{self.schema_name}"."{self.table_name}"
                            SET content = :content,
                                langchain_metadata = CAST(:metadata AS jsonb)
                            WHERE langchain_id = :doc_id
                        """)

                        await conn.execute(
                            update_query,
                            {
                                "content": doc.page_content,
                                "metadata": json.dumps(doc.metadata),
                                "doc_id": doc_id,
                            },
                        )

                    updated_ids.append(doc_id)

                await conn.commit()

            logger.info(
                f"Updated {len(updated_ids)} documents (embeddings={update_embeddings})"
            )
            return updated_ids
        except Exception as e:
            raise DatabaseError(f"Failed to update documents: {e}") from e

    async def adelete(self, ids: List[str]) -> int:
        """
        Delete documents by their IDs.

        Args:
            ids: List of document IDs (langchain_id) to delete

        Returns:
            Number of documents deleted

        Raises:
            InitializationError: If system not initialized
            ValidationError: If ids list is empty
            DatabaseError: If deletion fails

        Examples:
            >>> doc_ids = await pgvdb.add_documents(documents)
            >>> # Delete first 5 documents
            >>> deleted_count = await pgvdb.adelete(doc_ids[:5])
            >>> print(f"Deleted {deleted_count} documents")
        """
        self._ensure_initialized()

        if not ids:
            return 0

        try:
            async with self.sqlalchemy_engine.connect() as conn:
                # Delete documents matching the provided IDs
                query = text(f"""
                    DELETE FROM "{self.schema_name}"."{self.table_name}"
                    WHERE langchain_id = ANY(:ids)
                """)
                result = await conn.execute(query, {"ids": ids})
                await conn.commit()

                deleted_count = result.rowcount
                logger.info(f"Deleted {deleted_count} documents")
                return deleted_count
        except Exception as e:
            raise DatabaseError(f"Failed to delete documents: {e}") from e

    async def add_documents_batch(
        self,
        documents: List[Document],
        batch_size: int = 100,
        labels: Optional[List[List[int]]] = None,
        show_progress: bool = True,
    ) -> List[str]:
        """
        Add large numbers of documents efficiently with batching and progress tracking.

        Benefits:
        - Prevents memory overflow with large datasets
        - Progress tracking for long operations
        - Resumable if interrupted (returns IDs added so far)
        - Automatic commit batching for performance

        Args:
            documents: List of Documents to add (can be 10K+)
            batch_size: Number of documents per batch (default: 100)
            labels: Optional labels for DiskANN (must match documents length)
            show_progress: Print progress updates (default: True)

        Returns:
            List of all added document IDs

        Examples:
            >>> # Add 50,000 documents efficiently
            >>> all_ids = await pgvdb.add_documents_batch(
            ...     large_doc_list,
            ...     batch_size=500,
            ...     show_progress=True
            ... )
            >>> print(f"Added {len(all_ids)} documents")
        """
        self._ensure_initialized()

        if not documents:
            raise ValidationError("documents list cannot be empty")
        if batch_size <= 0:
            raise ValidationError("batch_size must be positive")
        if labels is not None and len(labels) != len(documents):
            raise ValidationError(
                f"labels length ({len(labels)}) must match documents length ({len(documents)})"
            )

        all_ids = []
        total_docs = len(documents)
        num_batches = (total_docs + batch_size - 1) // batch_size

        try:
            for batch_idx in range(num_batches):
                start_idx = batch_idx * batch_size
                end_idx = min((batch_idx + 1) * batch_size, total_docs)

                batch_docs = documents[start_idx:end_idx]
                batch_labels = labels[start_idx:end_idx] if labels else None

                # Add batch
                batch_ids = await self.add_documents(batch_docs, labels=batch_labels)
                all_ids.extend(batch_ids)

                if show_progress:
                    progress = (batch_idx + 1) / num_batches * 100
                    logger.info(
                        f"Progress: {batch_idx + 1}/{num_batches} batches "
                        f"({end_idx}/{total_docs} docs, {progress:.1f}%)"
                    )

            if show_progress:
                logger.info(
                    f"✓ Batch ingestion complete: {len(all_ids)} documents added"
                )

            return all_ids
        except Exception as e:
            logger.warning(
                f"Batch ingestion interrupted at {len(all_ids)}/{total_docs} documents"
            )
            raise DatabaseError(f"Failed during batch ingestion: {e}") from e

    async def update_metadata(
        self, ids: List[str], metadata_updates: Dict[str, Any]
    ) -> int:
        """
        Bulk metadata updates without re-embedding.

        Useful for:
        - Tagging/categorizing documents
        - Status updates
        - Adding computed fields
        - Fixing metadata errors

        Args:
            ids: List of document IDs to update
            metadata_updates: Dictionary of metadata fields to update/add

        Returns:
            Number of documents updated

        Examples:
            >>> # Tag documents as reviewed
            >>> doc_ids = ["id1", "id2", "id3"]
            >>> count = await pgvdb.update_metadata(
            ...     ids=doc_ids,
            ...     metadata_updates={"status": "reviewed", "reviewer": "alice"}
            ... )
            >>> print(f"Updated {count} documents")
            >>>
            >>> # Add computed field to all documents matching filter
            >>> docs = await pgvdb.metadata_filter({"category": "ai"})
            >>> ids = [d['id'] for d in docs]
            >>> await pgvdb.update_metadata(ids, {"indexed": True})
        """
        self._ensure_initialized()

        if not ids or not isinstance(ids, list):
            raise ValidationError("ids must be a non-empty list")
        if not metadata_updates or not isinstance(metadata_updates, dict):
            raise ValidationError("metadata_updates must be a non-empty dictionary")

        try:
            async with self.sqlalchemy_engine.connect() as conn:
                # Single bulk UPDATE using JSONB `||` merge operator.
                # This merges `metadata_updates` into the existing JSON column
                # for all matching IDs in one round-trip (O(1) vs the previous O(2N)).
                #
                # Note: langchain_metadata is type JSON (not JSONB), so we must
                # cast to jsonb for the || merge, then cast back to json.
                update_query = text(f"""
                    UPDATE "{self.schema_name}"."{self.table_name}"
                    SET langchain_metadata = (
                        COALESCE(langchain_metadata::jsonb, '{{}}'::jsonb)
                        || CAST(:updates AS jsonb)
                    )::json
                    WHERE langchain_id = ANY(:ids)
                """)
                result = await conn.execute(
                    update_query,
                    {
                        "updates": json.dumps(metadata_updates),
                        "ids": ids,
                    },
                )
                await conn.commit()

                update_count = result.rowcount
                logger.info(f"Updated metadata for {update_count} documents")
                return update_count
        except Exception as e:
            raise DatabaseError(f"Failed to update metadata: {e}") from e

    async def aget_by_ids(self, ids: List[str]) -> List[QueryResult]:
        """
        Retrieve specific documents by their IDs.

        Useful for:
        - Quick lookups of known documents
        - Fetching related documents
        - Debugging and validation
        - Building citation/reference features

        Args:
            ids: List of document IDs (langchain_id values)

        Returns:
            List of QueryResult objects (score=1.0 for all results)

        Raises:
            ValidationError: If ids list is empty
            DatabaseError: If retrieval fails

        Examples:
            >>> doc_ids = ["uuid-1", "uuid-2", "uuid-3"]
            >>> docs = await pgvdb.aget_by_ids(doc_ids)
            >>> for doc in docs:
            ...     print(f"ID: {doc['id']}, Content: {doc['content'][:50]}")
        """
        self._ensure_initialized()

        if not ids:
            return []

        if not isinstance(ids, list):
            raise ValidationError("ids must be a list")

        try:
            # Use ANY for efficient batch retrieval
            full_query = text(f"""
                SELECT "langchain_id", "content", "langchain_metadata"
                FROM "{self.schema_name}"."{self.table_name}"
                WHERE "langchain_id" = ANY(:ids)
            """)

            async with self.sqlalchemy_engine.connect() as conn:
                result = await conn.execute(full_query, {"ids": ids})
                return [
                    QueryResult(
                        id=str(row[0]),
                        content=row[1],
                        metadata=row[2] or {},
                        score=1.0,  # No relevance score for direct ID lookup
                    )
                    for row in result.fetchall()
                ]
        except Exception as e:
            raise DatabaseError(f"Failed to get documents by IDs: {e}") from e

    # ==================== NEW FEATURES: BATCH ERROR ISOLATION (Task 24) ====================

    async def add_documents_batch_isolated(
        self,
        documents: List[Document],
        batch_size: int = 100,
        labels: Optional[List[List[int]]] = None,
        show_progress: bool = True,
        continue_on_error: bool = False,
    ) -> Tuple[List[str], List[int]]:
        """
        Add documents with per-batch error isolation (AGNO pattern).

        Each batch is committed independently. If a batch fails, previous batches
        remain committed and subsequent batches can optionally continue.

        Args:
            documents: List of Documents to add
            batch_size: Number of documents per batch (default: 100)
            labels: Optional labels for DiskANN filtering
            show_progress: Print progress updates (default: True)
            continue_on_error: If True, continue processing after batch failure (default: False)

        Returns:
            Tuple of (successfully_added_ids, failed_batch_indices)

        Examples:
            >>> added_ids, failed_batches = await pgvdb.add_documents_batch_isolated(
            ...     documents,
            ...     batch_size=500,
            ...     continue_on_error=True
            ... )
            >>> print(f"Added {len(added_ids)} docs, {len(failed_batches)} batches failed")
        """
        self._ensure_initialized()

        if not documents:
            raise ValidationError("documents list cannot be empty")

        all_ids = []
        failed_batches = []
        total_docs = len(documents)
        num_batches = (total_docs + batch_size - 1) // batch_size

        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, total_docs)

            batch_docs = documents[start_idx:end_idx]
            batch_labels = labels[start_idx:end_idx] if labels else None

            try:
                # Each batch is committed independently
                batch_ids = await self.add_documents(batch_docs, labels=batch_labels)
                all_ids.extend(batch_ids)

                if show_progress:
                    progress = (batch_idx + 1) / num_batches * 100
                    logger.info(
                        f"✓ Batch {batch_idx + 1}/{num_batches} committed "
                        f"({end_idx}/{total_docs} docs, {progress:.1f}%)"
                    )
            except Exception as e:
                logger.error(f"✗ Batch {batch_idx + 1}/{num_batches} failed: {e}")
                failed_batches.append(batch_idx)

                if not continue_on_error:
                    logger.warning(
                        f"Stopping batch ingestion. {len(all_ids)} docs committed before failure."
                    )
                    break

        if show_progress:
            logger.info(
                f"Batch ingestion complete: {len(all_ids)} added, "
                f"{len(failed_batches)} batches failed"
            )

        return all_ids, failed_batches

    # ==================== NEW FEATURES: EMBEDDING FALLBACK (Task 25) ====================

    def _is_rate_limit_error(self, error: Exception) -> bool:
        """Check if an error is a rate limit error that should not be retried."""
        error_str = str(error).lower()
        rate_limit_indicators = [
            "429",
            "rate limit",
            "too many requests",
            "trial key",
            "quota exceeded",
            "throttl",
            "ratelimit",
        ]
        return any(indicator in error_str for indicator in rate_limit_indicators)

    async def _embed_documents_with_fallback(
        self, documents: List[Document]
    ) -> List[Tuple[Document, Optional[List[float]]]]:
        """
        Embed documents with intelligent fallback (AGNO pattern).

        Strategy:
        1. Try batch embedding first
        2. On rate limit: raise immediately (don't retry)
        3. On other errors: fall back to per-document embedding

        Args:
            documents: List of documents to embed

        Returns:
            List of (document, embedding) tuples. Embedding is None if failed.

        Raises:
            RateLimitError: If rate limit is hit (should not be retried)
        """
        results = []

        try:
            # Try batch embedding
            texts = [doc.page_content for doc in documents]
            embeddings = self.embedding_model.embed_documents(texts)

            for doc, emb in zip(documents, embeddings):
                results.append((doc, emb))

            return results

        except Exception as e:
            if self._is_rate_limit_error(e):
                logger.error(f"Rate limit hit during batch embedding: {e}")
                raise RateLimitError(f"Embedding rate limit exceeded: {e}") from e

            logger.warning(f"Batch embedding failed, falling back to individual: {e}")

            # Fall back to per-document embedding
            for doc in documents:
                try:
                    embedding = self.embedding_model.embed_query(doc.page_content)
                    results.append((doc, embedding))
                except Exception as doc_error:
                    if self._is_rate_limit_error(doc_error):
                        raise RateLimitError(
                            f"Embedding rate limit exceeded: {doc_error}"
                        ) from doc_error
                    logger.error(
                        f"Failed to embed document '{doc.metadata.get('langchain_id', 'unknown')}': {doc_error}"
                    )
                    results.append((doc, None))

            return results

    # ==================== NEW FEATURES: SQLALCHEMY INSPECTOR (Task 26) ====================

    async def _index_exists(self, index_name: str) -> bool:
        """
        Check if an index exists using SQLAlchemy inspector (AGNO pattern).

        More robust than querying pg_indexes directly.

        Args:
            index_name: Name of the index to check

        Returns:
            True if index exists, False otherwise
        """
        try:
            async with self.sqlalchemy_engine.connect() as conn:

                def check_sync(sync_conn):
                    inspector = inspect(sync_conn)
                    indexes = inspector.get_indexes(
                        self.table_name, schema=self.schema_name
                    )
                    return any(idx["name"] == index_name for idx in indexes)

                return await conn.run_sync(check_sync)
        except Exception as e:
            logger.warning(f"Could not check index existence via inspector: {e}")
            # Fallback to pg_indexes query
            async with self.sqlalchemy_engine.connect() as conn:
                result = await conn.execute(
                    text("""
                        SELECT 1 FROM pg_indexes
                        WHERE schemaname = :schema
                        AND tablename = :table
                        AND indexname = :index_name
                    """),
                    {
                        "schema": self.schema_name,
                        "table": self.table_name,
                        "index_name": index_name,
                    },
                )
                return result.fetchone() is not None

    # ==================== NEW FEATURES: CONTENT HASH DEDUPLICATION (Task 27) ====================

    def _compute_content_hash(
        self, content: str, filters: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Compute MD5 hash of content + filters for deduplication.

        Args:
            content: Document content
            filters: Optional filter dictionary to include in hash

        Returns:
            32-character MD5 hash string
        """
        hash_input = content
        if filters:
            hash_input += json.dumps(filters, sort_keys=True)
        return hashlib.md5(hash_input.encode("utf-8")).hexdigest()

    async def upsert_documents(
        self,
        documents: List[Document],
        batch_size: int = 100,
        dedup_by_content: bool = True,
    ) -> Tuple[int, int]:
        """
        Upsert documents with content hash deduplication (AGNO pattern).

        Documents with the same content are identified by MD5 hash and updated
        rather than duplicated.

        Args:
            documents: List of documents to upsert
            batch_size: Batch size for processing
            dedup_by_content: If True, deduplicate by content hash

        Returns:
            Tuple of (inserted_count, updated_count)
        """
        self._ensure_initialized()

        if not documents:
            raise ValidationError("documents list cannot be empty")

        inserted = 0
        updated = 0
        qualified_table = build_qualified_name(self.schema_name, self.table_name)

        try:
            # Single connection for all DDL and UPDATE operations.
            # Previously a new connection was opened for every document, which
            # exhausted the pool on large batches.
            async with self.sqlalchemy_engine.connect() as conn:
                # Ensure content_hash column exists (idempotent)
                await conn.execute(
                    text(f"""
                    ALTER TABLE {qualified_table}
                    ADD COLUMN IF NOT EXISTS content_hash VARCHAR(32)
                """)
                )
                await conn.commit()

                for i in range(0, len(documents), batch_size):
                    batch = documents[i : i + batch_size]

                    for doc in batch:
                        content_hash = (
                            self._compute_content_hash(doc.page_content)
                            if dedup_by_content
                            else None
                        )
                        doc_id = doc.metadata.get("langchain_id") or str(uuid.uuid4())
                        doc.metadata["langchain_id"] = doc_id

                        existing_id = None
                        if content_hash:
                            result = await conn.execute(
                                text(f"""
                                    SELECT langchain_id FROM {qualified_table}
                                    WHERE content_hash = :hash
                                """),
                                {"hash": content_hash},
                            )
                            row = result.fetchone()
                            if row:
                                existing_id = row[0]

                        if existing_id:
                            # Update existing document in-place
                            embedding = self.embedding_model.embed_query(doc.page_content)
                            await conn.execute(
                                text(f"""
                                    UPDATE {qualified_table}
                                    SET content = :content,
                                        langchain_metadata = :metadata,
                                        embedding = :embedding
                                    WHERE langchain_id = :doc_id
                                """),
                                {
                                    "content": doc.page_content,
                                    "metadata": doc.metadata,
                                    "embedding": str(embedding),
                                    "doc_id": existing_id,
                                },
                            )
                            await conn.commit()
                            updated += 1
                        else:
                            # Insert new document via add_documents (manages its own connection)
                            await self.add_documents([doc])

                            # Write content hash back on the same shared connection
                            if content_hash:
                                await conn.execute(
                                    text(f"""
                                        UPDATE {qualified_table}
                                        SET content_hash = :hash
                                        WHERE langchain_id = :doc_id
                                    """),
                                    {"hash": content_hash, "doc_id": doc_id},
                                )
                                await conn.commit()

                            inserted += 1

            logger.info(f"Upsert complete: {inserted} inserted, {updated} updated")
            return inserted, updated

        except Exception as e:
            raise DatabaseError(f"Upsert failed: {e}") from e

    async def bulk_load_documents(
        self,
        documents: List[Document],
        labels: Optional[List[List[int]]] = None,
        drop_indexes_first: bool = True,
        show_progress: bool = True,
    ) -> int:
        """
        Bulk load documents using PostgreSQL COPY for maximum performance.

        10-50x faster than INSERT for large batches. Best for initial data loading.

        Strategy:
        1. Drop indexes (optional but recommended for speed)
        2. Pre-compute all embeddings
        3. Use COPY protocol for bulk insert
        4. Rebuild indexes

        Args:
            documents: List of documents to load
            labels: Optional labels for DiskANN filtering
            drop_indexes_first: Drop and rebuild indexes for faster loading (default: True)
            show_progress: Print progress updates (default: True)

        Returns:
            Number of documents loaded

        Examples:
            >>> # Load 100,000 documents quickly
            >>> count = await pgvdb.bulk_load_documents(large_dataset)
            >>> print(f"Loaded {count} documents")

        Note:
            - Best for initial data loading, not incremental updates
            - Embeddings are computed before COPY (may take time)
            - Indexes are rebuilt after COPY (may take time for large datasets)
        """
        self._ensure_initialized()

        if not documents:
            raise ValidationError("documents list cannot be empty")

        total_docs = len(documents)
        qualified_table = build_qualified_name(self.schema_name, self.table_name)

        try:
            # Step 1: Drop indexes if requested
            if drop_indexes_first and show_progress:
                logger.info("Step 1/4: Dropping indexes for faster loading...")
                await self.adrop_vector_index()

            # Step 2: Pre-compute all embeddings
            if show_progress:
                logger.info(
                    f"Step 2/4: Computing embeddings for {total_docs} documents..."
                )

            texts = [doc.page_content for doc in documents]
            embeddings = self.embedding_model.embed_documents(texts)

            if show_progress:
                logger.info(f"✓ Embeddings computed for {total_docs} documents")

            # Step 3: Prepare data and use batch insert (COPY would require raw connection)
            # Using executemany for bulk insert as a practical alternative
            if show_progress:
                logger.info("Step 3/4: Bulk inserting documents...")

            records = []
            for i, (doc, embedding) in enumerate(zip(documents, embeddings)):
                doc_id = doc.metadata.get("langchain_id") or str(uuid.uuid4())
                doc.metadata["langchain_id"] = doc_id

                record = {
                    "id": doc_id,
                    "content": doc.page_content,
                    "metadata": json.dumps(doc.metadata),
                    "embedding": str(embedding),
                }

                if labels is not None and i < len(labels):
                    record["labels"] = labels[i]

                records.append(record)

            # Bulk insert using executemany pattern
            async with self.sqlalchemy_engine.connect() as conn:
                # Use batch insert
                batch_size = 1000
                for i in range(0, len(records), batch_size):
                    batch = records[i : i + batch_size]

                    for record in batch:
                        await conn.execute(
                            text(f"""
                                INSERT INTO {qualified_table}
                                (langchain_id, content, langchain_metadata, embedding)
                                VALUES (:id, :content, CAST(:metadata AS jsonb), :embedding)
                                ON CONFLICT (langchain_id) DO UPDATE SET
                                    content = EXCLUDED.content,
                                    langchain_metadata = EXCLUDED.langchain_metadata,
                                    embedding = EXCLUDED.embedding
                            """),
                            record,
                        )

                    await conn.commit()

                    if show_progress:
                        progress = (
                            min(i + batch_size, len(records)) / len(records) * 100
                        )
                        logger.info(
                            f"  Inserted {min(i + batch_size, len(records))}/{len(records)} ({progress:.1f}%)"
                        )

            if show_progress:
                logger.info(f"✓ Bulk insert complete: {total_docs} documents")

            # Step 4: Rebuild indexes
            if drop_indexes_first:
                if show_progress:
                    logger.info("Step 4/4: Rebuilding indexes...")
                await self.build_index()
                if show_progress:
                    logger.info("✓ Indexes rebuilt")

            logger.info(f"✓ Bulk load complete: {total_docs} documents loaded")
            return total_docs

        except Exception as e:
            raise DatabaseError(f"Bulk load failed: {e}") from e

    async def add_documents_orm(
        self,
        documents: List[Document],
        labels: Optional[List[List[int]]] = None,
        batch_size: int = 100,
    ) -> List[str]:
        """
        Add documents using SQLAlchemy ORM constructs (more secure).

        Uses postgresql.insert() with on_conflict_do_update() instead of
        raw SQL strings for improved security.

        Args:
            documents: List of documents to add
            labels: Optional labels for DiskANN filtering
            batch_size: Batch size for processing (default: 100)

        Returns:
            List of document IDs

        Examples:
            >>> doc_ids = await pgvdb.add_documents_orm(documents)
        """
        self._ensure_initialized()

        if not documents:
            raise ValidationError("documents list cannot be empty")

        all_ids = []

        try:
            # Get table schema if available
            if get_vector_table is not None:
                get_vector_table(
                    self.table_name,
                    self.schema_name,
                    self.vector_size,
                    include_labels=(labels is not None),
                )

            for i in range(0, len(documents), batch_size):
                batch_docs = documents[i : i + batch_size]
                batch_labels = labels[i : i + batch_size] if labels else None

                # Compute embeddings
                texts = [doc.page_content for doc in batch_docs]
                embeddings = self.embedding_model.embed_documents(texts)

                # Prepare records
                records = []
                for j, (doc, embedding) in enumerate(zip(batch_docs, embeddings)):
                    doc_id = doc.metadata.get("langchain_id") or str(uuid.uuid4())
                    doc.metadata["langchain_id"] = doc_id
                    all_ids.append(doc_id)

                    record = {
                        "langchain_id": doc_id,
                        "content": doc.page_content,
                        "langchain_metadata": doc.metadata,
                        "embedding": str(embedding),
                    }

                    if batch_labels and j < len(batch_labels):
                        record["labels"] = batch_labels[j]

                    records.append(record)

                # Insert using parameterized query (not ORM but still parameterized)
                async with self.sqlalchemy_engine.connect() as conn:
                    for record in records:
                        # Use insert with on conflict
                        insert_sql = text(f"""
                            INSERT INTO {build_qualified_name(self.schema_name, self.table_name)}
                            (langchain_id, content, langchain_metadata, embedding)
                            VALUES (:langchain_id, :content, CAST(:langchain_metadata AS jsonb), :embedding)
                            ON CONFLICT (langchain_id) DO UPDATE SET
                                content = EXCLUDED.content,
                                langchain_metadata = EXCLUDED.langchain_metadata,
                                embedding = EXCLUDED.embedding
                        """)

                        await conn.execute(
                            insert_sql,
                            {
                                "langchain_id": record["langchain_id"],
                                "content": record["content"],
                                "langchain_metadata": json.dumps(
                                    record["langchain_metadata"]
                                ),
                                "embedding": record["embedding"],
                            },
                        )

                    await conn.commit()

            logger.info(f"Added {len(all_ids)} documents via ORM-style insert")
            return all_ids

        except Exception as e:
            raise DatabaseError(f"ORM insert failed: {e}") from e
