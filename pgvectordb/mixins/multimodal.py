"""
Multimodal search and reranking mixin for pgVectorDB.

Provides: register_spaces, add_documents_multimodal, build_multimodal_index,
multimodal_search, multimodal_hybrid_search, get_multimodal_index_stats,
rerank_search, and related helpers.
"""

import uuid
import json
import logging
from typing import Dict, List, Optional, Any

from langchain_core.documents import Document
from sqlalchemy import text

from ..base import (
    IndexType,
    KeywordSearchType,
    DistanceMetric,
    ValidationError,
    DatabaseError,
    QueryResult,
)
from ..schema import build_qualified_name, quote_identifier, get_distance_operator

logger = logging.getLogger(__name__)


class MultimodalMixin:
    """Mixin providing multimodal search and reranking operations."""

    # ==================== MULTIMODAL METHODS (v0.0.3) ====================

    def register_spaces(
        self,
        spaces: List[Any],
    ) -> None:
        """
        Register vector spaces for multimodal search on this collection.

        Each space defines how a specific data field (text, number, category)
        is encoded into an embedding vector. After registration, use
        ``add_documents_multimodal()`` and ``multimodal_search()`` to
        leverage multiple embeddings per document.

        Args:
            spaces: List of VectorSpace instances (TextSpace, NumberSpace,
                CategorySpace). At least one space is required.

        Raises:
            ValidationError: If spaces list is empty or has duplicates.

        Examples:
            >>> from pgvectordb.spaces import TextSpace, NumberSpace, CategorySpace
            >>> rag.register_spaces([
            ...     TextSpace(name="description", field="content"),
            ...     NumberSpace(name="price", field="price",
            ...                 min_value=0, max_value=1000000, mode="minimum"),
            ...     CategorySpace(name="city", field="city",
            ...                   categories=["NYC", "LA", "Chicago"]),
            ... ])
        """
        try:
            from ..spaces import validate_spaces, TextSpace
        except ImportError:
            raise ImportError(
                "src.spaces module not found. Ensure spaces.py is present."
            )

        validate_spaces(spaces)

        # Auto-detect dimensions for TextSpaces
        for space in spaces:
            if isinstance(space, TextSpace) and space.dimensions == 0:
                space.detect_dimensions(self.embedding_model)

        self._spaces = list(spaces)
        logger.info(
            f"Registered {len(spaces)} spaces: "
            f"{[f'{s.name}({s.dimensions}d)' for s in spaces]}"
        )

    async def _ensure_multimodal_columns(self) -> None:
        """
        Ensure the table has all required embedding columns for registered spaces.

        For each registered space, adds an ``embedding_{space.name}`` column
        if it doesn't already exist.
        """
        if not hasattr(self, "_spaces") or not self._spaces:
            return

        qualified_table = build_qualified_name(self.schema_name, self.table_name)

        async with self.sqlalchemy_engine.connect() as conn:
            for space in self._spaces:
                col_name = quote_identifier(f"embedding_{space.name}")
                dims = space.dimensions

                # Check if column exists
                check_result = await conn.execute(
                    text("""
                    SELECT 1 FROM information_schema.columns
                    WHERE table_schema = :schema
                    AND table_name = :table
                    AND column_name = :col
                """),
                    {
                        "schema": self.schema_name,
                        "table": self.table_name,
                        "col": f"embedding_{space.name}",
                    },
                )

                if check_result.fetchone() is None:
                    # Add vector column
                    await conn.execute(
                        text(
                            f"ALTER TABLE {qualified_table} "
                            f"ADD COLUMN {col_name} vector({dims})"
                        )
                    )
                    logger.info(
                        f"Added column embedding_{space.name} "
                        f"(vector({dims})) to {self.table_name}"
                    )

            await conn.commit()

    async def add_documents_multimodal(
        self,
        documents: List[Document],
        batch_size: int = 100,
        show_progress: bool = True,
    ) -> List[str]:
        """
        Add documents with embeddings for all registered vector spaces.

        For each document, this method:
        1. Extracts the relevant field for each space
        2. Encodes each field using the space's encoder
        3. Inserts the document with ALL embedding columns populated

        The standard ``embedding`` column is also populated (using the first
        TextSpace or the default model) for backward compatibility.

        Args:
            documents: List of LangChain Document objects.
            batch_size: Documents per batch (default: 100).
            show_progress: Print progress (default: True).

        Returns:
            List of document IDs.

        Raises:
            InitializationError: If system not initialized.
            ValidationError: If no spaces registered or documents empty.
            DatabaseError: If insert fails.

        Examples:
            >>> from langchain_core.documents import Document
            >>> docs = [
            ...     Document(page_content="Modern downtown apartment",
            ...              metadata={"price": 500000, "city": "NYC"}),
            ... ]
            >>> rag.register_spaces(spaces)
            >>> ids = await rag.add_documents_multimodal(docs)
        """
        self._ensure_initialized()

        if not hasattr(self, "_spaces") or not self._spaces:
            raise ValidationError("No spaces registered. Call register_spaces() first.")
        if not documents:
            raise ValidationError("documents list cannot be empty")

        try:
            from ..spaces import encode_document_spaces
        except ImportError:
            raise ImportError("src.spaces module required")

        # Ensure columns exist
        await self._ensure_multimodal_columns()

        qualified_table = build_qualified_name(self.schema_name, self.table_name)
        all_ids = []
        total = len(documents)

        for batch_start in range(0, total, batch_size):
            batch_docs = documents[batch_start : batch_start + batch_size]

            # Also compute standard embedding for backward compatibility
            texts = [doc.page_content for doc in batch_docs]
            standard_embeddings = self.embedding_model.embed_documents(texts)

            for i, (doc, std_emb) in enumerate(zip(batch_docs, standard_embeddings)):
                doc_id = doc.metadata.get("langchain_id") or str(uuid.uuid4())
                doc.metadata["langchain_id"] = doc_id
                all_ids.append(doc_id)

                # Encode all spaces
                space_embeddings = encode_document_spaces(
                    doc, self._spaces, self.embedding_model
                )

                # Build dynamic SQL for multi-column insert
                col_names = [
                    "langchain_id",
                    "content",
                    "langchain_metadata",
                    "embedding",
                ]
                param_names = [
                    ":langchain_id",
                    ":content",
                    "CAST(:langchain_metadata AS jsonb)",
                    ":embedding",
                ]
                params = {
                    "langchain_id": doc_id,
                    "content": doc.page_content,
                    "langchain_metadata": json.dumps(doc.metadata),
                    "embedding": str(std_emb),
                }

                # Add space embedding columns
                for space in self._spaces:
                    col = f"embedding_{space.name}"
                    col_names.append(quote_identifier(col))
                    param_key = f"emb_{space.name}"
                    param_names.append(f":{param_key}")
                    params[param_key] = str(space_embeddings[col])

                # Build update clause
                update_cols = []
                for cn in col_names[1:]:  # Skip langchain_id
                    update_cols.append(f"{cn} = EXCLUDED.{cn}")

                col_str = ", ".join(col_names)
                val_str = ", ".join(param_names)
                upd_str = ", ".join(update_cols)

                async with self.sqlalchemy_engine.connect() as conn:
                    await conn.execute(
                        text(f"""
                            INSERT INTO {qualified_table} ({col_str})
                            VALUES ({val_str})
                            ON CONFLICT (langchain_id) DO UPDATE SET {upd_str}
                        """),
                        params,
                    )
                    await conn.commit()

            if show_progress:
                done = min(batch_start + batch_size, total)
                logger.info(
                    f"Multimodal insert: {done}/{total} ({done / total * 100:.0f}%)"
                )

        logger.info(
            f"✓ Added {len(all_ids)} documents with "
            f"{len(self._spaces)} space embeddings each"
        )
        return all_ids

    async def build_multimodal_index(
        self,
        index_type: Optional[IndexType] = None,
        metric: DistanceMetric = DistanceMetric.COSINE,
        m: int = 16,
        ef_construction: int = 64,
        spaces: Optional[List[str]] = None,
    ) -> Dict[str, str]:
        """
        Build vector indexes on each registered space's embedding column.

        Creates separate HNSW/IVFFlat indexes per space, allowing the database
        to use index scans for each embedding column independently.

        Args:
            index_type: Index type (default: self.index_type). Applied to all spaces.
            metric: Distance metric (default: cosine).
            m: HNSW m parameter (default: 16).
            ef_construction: HNSW ef_construction (default: 64).
            spaces: Optional list of space names to index. If None, indexes all.

        Returns:
            Dictionary mapping space names to created index names.

        Raises:
            InitializationError: If system not initialized.
            ValidationError: If no spaces registered.

        Examples:
            >>> indexes = await rag.build_multimodal_index(
            ...     metric=DistanceMetric.COSINE, m=24
            ... )
            >>> for space_name, idx_name in indexes.items():
            ...     print(f"{space_name}: {idx_name}")
        """
        self._ensure_initialized()

        if not hasattr(self, "_spaces") or not self._spaces:
            raise ValidationError("No spaces registered. Call register_spaces() first.")

        idx_type = index_type or self.index_type
        qualified_table = build_qualified_name(self.schema_name, self.table_name)
        created = {}

        # Determine which spaces to index
        target_spaces = self._spaces
        if spaces:
            target_spaces = [s for s in self._spaces if s.name in spaces]

        ops_class = self._get_distance_ops(metric)

        try:
            async with self.sqlalchemy_engine.connect() as conn:
                for space in target_spaces:
                    col_name = f"embedding_{space.name}"
                    index_name = f"idx_{self.table_name}_{space.index_name_suffix}"

                    # Drop existing
                    await conn.execute(
                        text(
                            f"DROP INDEX IF EXISTS "
                            f"{build_qualified_name(self.schema_name, index_name)}"
                        )
                    )

                    if idx_type == IndexType.HNSW:
                        await conn.execute(
                            text(f"""
                            CREATE INDEX "{index_name}"
                            ON {qualified_table}
                            USING hnsw ({quote_identifier(col_name)} {ops_class})
                            WITH (m = {m}, ef_construction = {ef_construction})
                        """)
                        )
                    elif idx_type == IndexType.IVFFLAT:
                        result = await conn.execute(
                            text(f"SELECT COUNT(*) FROM {qualified_table}")
                        )
                        row_count = result.scalar() or 1000
                        lists = max(int(row_count**0.5), 1)
                        await conn.execute(
                            text(f"""
                            CREATE INDEX "{index_name}"
                            ON {qualified_table}
                            USING ivfflat ({quote_identifier(col_name)} {ops_class})
                            WITH (lists = {lists})
                        """)
                        )

                    created[space.name] = index_name
                    logger.info(
                        f"✓ Created {idx_type.value} index on "
                        f"embedding_{space.name}: {index_name}"
                    )

                await conn.commit()

        except Exception as e:
            raise DatabaseError(f"Multimodal index build failed: {e}") from e

        return created

    async def multimodal_search(
        self,
        query_params: Dict[str, Any],
        weights: Optional[Dict[str, float]] = None,
        k: int = 10,
        filter: Optional[Dict[str, Any]] = None,
        metric: DistanceMetric = DistanceMetric.COSINE,
    ) -> List[QueryResult]:
        """
        Weighted search across all registered vector spaces.

        **This is the key method for multi-embedding RAG search.**

        For each space named in ``query_params``, this method:
        1. Encodes the query value using the space's encoder
        2. Computes the distance on that space's embedding column
        3. Normalizes and weights the distances
        4. Returns top-k results by fused score

        Eliminates the need for post-retrieval re-ranking by embedding multiple
        signals directly into the search.

        Args:
            query_params: Dictionary mapping space names to query values.
                Example: ``{"description": "luxury apartment", "price": 300000,
                "city": "NYC"}``
            weights: Optional dictionary mapping space names to float weights.
                Higher weight = more influence. If None, equal weights.
                Example: ``{"description": 0.5, "price": 0.3, "city": 0.2}``
            k: Number of results (default: 10).
            filter: Optional metadata filter (applied BEFORE vector search).
            metric: Distance metric (default: cosine).

        Returns:
            List of QueryResult sorted by fused weighted score (best first).

        Raises:
            InitializationError: If not initialized.
            ValidationError: If no spaces registered or query_params empty.

        Examples:
            >>> results = await rag.multimodal_search(
            ...     query_params={
            ...         "description": "modern downtown apartment",
            ...         "price": 500000,
            ...         "city": "New York",
            ...     },
            ...     weights={"description": 0.5, "price": 0.3, "city": 0.2},
            ...     k=10,
            ... )
            >>> for r in results:
            ...     print(f"{r['score']:.3f} - {r['content'][:80]}")
        """
        self._ensure_initialized()

        if not hasattr(self, "_spaces") or not self._spaces:
            raise ValidationError("No spaces registered. Call register_spaces() first.")
        if not query_params:
            raise ValidationError("query_params cannot be empty")

        try:
            from ..spaces import encode_query_spaces
        except ImportError:
            raise ImportError("src.spaces module required")

        # Build query embeddings for each relevant space
        query_embeddings = encode_query_spaces(
            query_params, self._spaces, self.embedding_model
        )

        if not query_embeddings:
            raise ValidationError(
                f"None of the query_params keys ({list(query_params.keys())}) "
                f"match registered space names ({[s.name for s in self._spaces]})"
            )

        # Default weights: equal across queried spaces
        if weights is None:
            weights = {s.name: 1.0 for s in self._spaces if s.name in query_params}

        # Normalize weights to sum to 1.0
        total_weight = sum(
            weights.get(s.name, 0.0) for s in self._spaces if s.name in query_params
        )
        if total_weight == 0:
            total_weight = 1.0

        # Get distance operator
        dist_op = get_distance_operator(metric.value)

        qualified_table = build_qualified_name(self.schema_name, self.table_name)

        # Build the weighted-distance SQL expression
        # For each space: weight * (embedding_col <=> query_vec)
        distance_parts = []
        params = {}
        for space in self._spaces:
            if space.name not in query_params:
                continue
            w = weights.get(space.name, 0.0) / total_weight
            col = quote_identifier(f"embedding_{space.name}")
            param_key = f"q_{space.name}"
            distance_parts.append(f"{w} * ({col} {dist_op} :{param_key})")
            params[param_key] = str(query_embeddings[f"embedding_{space.name}"])

        weighted_distance_expr = " + ".join(distance_parts)

        # Build WHERE clause for metadata filter
        where_clause = ""
        if filter:
            filter_parts, filter_params = self._build_filter_clauses(filter)
            if filter_parts:
                where_clause = f"WHERE {filter_parts}"
                params.update(filter_params)

        # Fetch more candidates than k to ensure quality after potential NULLs
        fetch_k = k * 3

        sql = f"""
            SELECT langchain_id, content, langchain_metadata,
                   ({weighted_distance_expr}) AS weighted_distance
            FROM {qualified_table}
            {where_clause}
            ORDER BY ({weighted_distance_expr}) ASC
            LIMIT :limit_k
        """
        params["limit_k"] = fetch_k

        try:
            async with self.sqlalchemy_engine.connect() as conn:
                result = await conn.execute(text(sql), params)
                rows = result.fetchall()

            # Convert distance to similarity score (1 - distance for cosine)
            results = []
            for row in rows:
                dist = float(row[3]) if row[3] is not None else float("inf")
                # For cosine distance, score = 1 - distance
                # For L2, score = 1 / (1 + distance)
                if metric == DistanceMetric.COSINE:
                    score = 1.0 - dist
                elif metric == DistanceMetric.L2:
                    score = 1.0 / (1.0 + dist)
                elif metric == DistanceMetric.INNER_PRODUCT:
                    score = -dist  # inner product is negated in pgvector
                else:
                    score = 1.0 - dist

                results.append(
                    QueryResult(
                        id=str(row[0]),
                        content=row[1],
                        metadata=row[2] or {},
                        score=score,
                    )
                )

            return results[:k]

        except Exception as e:
            raise DatabaseError(f"Multimodal search failed: {e}") from e

    async def multimodal_hybrid_search(
        self,
        query_params: Dict[str, Any],
        weights: Optional[Dict[str, float]] = None,
        keyword_weight: float = 0.3,
        k: int = 10,
        filter: Optional[Dict[str, Any]] = None,
        metric: DistanceMetric = DistanceMetric.COSINE,
        keyword_type: KeywordSearchType = KeywordSearchType.FTS,
    ) -> List[QueryResult]:
        """
        Multimodal search fused with BM25/FTS keyword scores.

        Performs two search passes:
        1. Multimodal weighted vector search (N spaces)
        2. Keyword search (BM25 or FTS)

        Results are fused using weighted scores.

        Args:
            query_params: Space-to-value mapping (``{"description": "...", ...}``).
            weights: Space weights (``{"description": 0.5, ...}``).
            keyword_weight: Weight for keyword scores (0-1). Vector gets ``1 - keyword_weight``.
            k: Number of results (default: 10).
            filter: Optional metadata filter.
            metric: Distance metric for vector search.
            keyword_type: FTS or BM25 (default: FTS).

        Returns:
            Fused results sorted by combined score.

        Examples:
            >>> results = await rag.multimodal_hybrid_search(
            ...     query_params={"description": "cozy apartment near park"},
            ...     weights={"description": 1.0},
            ...     keyword_weight=0.3,
            ...     k=10,
            ... )
        """
        self._ensure_initialized()

        # Get multimodal results
        vector_weight = 1.0 - keyword_weight
        mm_results = await self.multimodal_search(
            query_params=query_params,
            weights=weights,
            k=k * 2,  # Fetch extra for fusion
            filter=filter,
            metric=metric,
        )

        # Get keyword results using the text query (first text param)
        text_query = None
        for space in self._spaces:
            if space.name in query_params:
                from ..spaces import TextSpace

                if isinstance(space, TextSpace):
                    text_query = str(query_params[space.name])
                    break

        if text_query is None:
            # Fallback: use first string value
            for v in query_params.values():
                if isinstance(v, str) and v.strip():
                    text_query = v
                    break

        if text_query:
            kw_results = await self.keyword_search(
                text_query,
                k=k * 2,
                search_type=keyword_type,
            )
        else:
            kw_results = []

        # Fuse results using weighted scores
        fused = self._fuse_results(
            mm_results,
            kw_results,
            weights=(vector_weight, keyword_weight),
            k=k,
        )

        return fused

    async def get_multimodal_index_stats(self) -> Dict[str, Any]:
        """
        Get index statistics for each registered space's embedding column.

        Returns:
            Dictionary mapping space names to their index info.

        Examples:
            >>> stats = await rag.get_multimodal_index_stats()
            >>> for name, info in stats.items():
            ...     print(f"{name}: {info['index_name']} ({info['index_size']})")
        """
        self._ensure_initialized()

        if not hasattr(self, "_spaces") or not self._spaces:
            return {}

        stats = {}
        try:
            async with self.sqlalchemy_engine.connect() as conn:
                for space in self._spaces:
                    index_name = f"idx_{self.table_name}_{space.index_name_suffix}"

                    result = await conn.execute(
                        text("""
                        SELECT indexname, pg_size_pretty(pg_relation_size(indexrelid))
                        FROM pg_stat_user_indexes
                        WHERE schemaname = :schema AND indexrelname = :idx_name
                    """),
                        {
                            "schema": self.schema_name,
                            "idx_name": index_name,
                        },
                    )

                    row = result.fetchone()
                    stats[space.name] = {
                        "space_name": space.name,
                        "column": f"embedding_{space.name}",
                        "dimensions": space.dimensions,
                        "index_name": index_name,
                        "index_exists": row is not None,
                        "index_size": row[1] if row else "N/A",
                    }

        except Exception as e:
            logger.warning(f"Error getting multimodal index stats: {e}")

        return stats

    # ==================== RERANKING METHODS (v0.0.3) ====================

    async def rerank_search(
        self,
        query: str,
        reranker: Any,
        k: int = 100,
        rerank_top_k: int = 5,
        search_method: str = "semantic",
        **search_kwargs,
    ) -> List[QueryResult]:
        """
        Retrieve-then-Rerank: first stage retrieval followed by precision reranking.

        This is the recommended pattern for high-precision RAG:
        1. Retrieve ``k`` candidates quickly (semantic / hybrid / keyword / multimodal)
        2. Rerank all ``k`` candidates using a cross-encoder or API-based reranker
        3. Return top ``rerank_top_k`` by rerank score

        **Why this works:** Bi-encoder retrieval optimizes for speed (independent
        encoding). Cross-encoder reranking optimizes for precision (query+doc seen
        together). Combining them gives you the best of both worlds.

        Supported search_method values:
            - ``"semantic"``   → :meth:`semantic_search`
            - ``"hybrid"``     → :meth:`hybrid_search`
            - ``"keyword"``    → :meth:`keyword_search`
            - ``"bm25"``       → :meth:`keyword_search` with BM25
            - ``"multimodal"`` → :meth:`multimodal_search` (needs query_params in kwargs)

        Args:
            query: Search query string.
            reranker: A ``BaseReranker`` instance (CrossEncoderReranker,
                CohereReranker, AWSBedrockReranker, or HuggingFaceReranker).
            k: Candidates to retrieve (default: 100). More = better recall,
                but more reranker API calls / latency.
            rerank_top_k: Final results to return after reranking (default: 5).
            search_method: Which retrieval method to use (see above).
            **search_kwargs: Extra arguments forwarded to the chosen search method.
                Common: ``filter``, ``metric``, ``query_params``, ``weights``.

        Returns:
            List of QueryResult sorted by rerank score (best first),
            length ≤ ``rerank_top_k``.

        Raises:
            InitializationError: If system not initialized.
            ValueError: If search_method is unknown.

        Examples:
            >>> from pgvectordb.rerankers import CrossEncoderReranker, create_reranker
            >>>
            >>> # Local cross-encoder
            >>> reranker = CrossEncoderReranker(
            ...     model="cross-encoder/ms-marco-MiniLM-L-6-v2"
            ... )
            >>> results = await rag.rerank_search(
            ...     query="best noise cancelling headphones under $200",
            ...     reranker=reranker,
            ...     k=50,
            ...     rerank_top_k=5,
            ...     search_method="hybrid",
            ... )
            >>>
            >>> # Cohere API reranker
            >>> reranker = create_reranker("cohere", api_key="co_...")
            >>> results = await rag.rerank_search(
            ...     query="modern 2BR apartment downtown",
            ...     reranker=reranker,
            ...     k=100,
            ...     rerank_top_k=10,
            ...     search_method="multimodal",
            ...     query_params={"description": "modern 2BR apartment", "price": 400000},
            ...     weights={"description": 0.6, "price": 0.4},
            ... )
        """
        self._ensure_initialized()

        method = search_method.lower().strip()

        # === Stage 1: Retrieve candidates ===
        if method == "semantic":
            candidates = await self.semantic_search(query, k=k, **search_kwargs)

        elif method == "hybrid":
            candidates = await self.hybrid_search(query, k=k, **search_kwargs)

        elif method in ("keyword", "bm25", "fts"):
            from ..base import KeywordSearchType

            search_type = (
                KeywordSearchType.BM25 if method == "bm25" else KeywordSearchType.FTS
            )
            candidates = await self.keyword_search(
                query, k=k, search_type=search_type, **search_kwargs
            )

        elif method == "multimodal":
            query_params = search_kwargs.pop("query_params", {query: query})
            weights = search_kwargs.pop("weights", None)
            candidates = await self.multimodal_search(
                query_params=query_params,
                weights=weights,
                k=k,
                **search_kwargs,
            )

        else:
            raise ValueError(
                f"Unknown search_method: '{search_method}'. "
                f"Supported: semantic, hybrid, keyword, bm25, fts, multimodal"
            )

        if not candidates:
            logger.info(f"rerank_search: no candidates from {method} search")
            return []

        # === Stage 2: Rerank ===
        logger.info(
            f"rerank_search: reranking {len(candidates)} candidates "
            f"with {reranker.__class__.__name__}"
        )

        # Convert QueryResult objects to dicts if needed
        candidate_dicts = []
        for c in candidates:
            if hasattr(c, "__dict__"):
                # Named tuple or dataclass
                d = {
                    "id": getattr(c, "id", ""),
                    "content": getattr(c, "content", ""),
                    "metadata": getattr(c, "metadata", {}),
                    "score": getattr(c, "score", 0.0),
                }
            elif isinstance(c, dict):
                d = c
            else:
                # Try __getitem__
                d = {
                    "id": c[0],
                    "content": c[1],
                    "metadata": c[2],
                    "score": c[3],
                }
            candidate_dicts.append(d)

        reranked = reranker.rerank(query, candidate_dicts, top_k=rerank_top_k)

        # Convert back to QueryResult
        results = []
        for r in reranked:
            results.append(
                QueryResult(
                    id=r.get("id", ""),
                    content=r.get("content", ""),
                    metadata=r.get("metadata", {}),
                    score=r.get("rerank_score", r.get("score", 0.0)),
                )
            )

        logger.info(
            f"rerank_search: ✓ returned {len(results)} results "
            f"(from {len(candidates)} candidates)"
        )
        return results
