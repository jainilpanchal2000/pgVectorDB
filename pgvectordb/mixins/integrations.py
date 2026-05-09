"""
LangChain integration mixin for pgVectorDB.

Provides: as_retriever with custom VectorStoreRetriever for LangChain compatibility.
"""

from __future__ import annotations

import logging
from typing import Dict, Optional, Any

from langchain_core.documents import Document
from langchain_core.callbacks import CallbackManagerForRetrieverRun


logger = logging.getLogger(__name__)


class IntegrationsMixin:
    """Mixin providing LangChain ecosystem integration."""

    def as_retriever(
        self,
        search_method: str = "semantic_search",
        search_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Any:
        """
        Convert to LangChain Retriever for ecosystem compatibility.

        Enables drop-in use with any LangChain RAG pipeline, chains, and agents.

        Args:
            search_method: Name of search method to use:
                - "semantic_search" (default)
                - "keyword_search"
                - "hybrid_search"
                - "ensemble_search"
                - "trigram_search"
                - Any other search method name from this class
            search_kwargs: Arguments to pass to the search method (e.g., {"k": 5, "filter": {...}})

        Returns:
            VectorStoreRetriever object compatible with LangChain

        Examples:
            >>> # Basic semantic retriever
            >>> retriever = rag.as_retriever()
            >>>
            >>> # Hybrid search retriever with custom parameters
            >>> retriever = rag.as_retriever(
            ...     search_method="hybrid_search",
            ...     search_kwargs={"k": 10, "weights": (0.7, 0.3)}
            ... )
            >>>
            >>> # Use in LangChain RAG chain
            >>> from langchain.chains import RetrievalQA
            >>> qa_chain = RetrievalQA.from_chain_type(
            ...     llm=llm,
            ...     retriever=retriever
            ... )
        """
        from langchain_core.retrievers import BaseRetriever
        from typing import List as TypingList

        search_kwargs = search_kwargs or {"k": 4}

        class VectorStoreRetriever(BaseRetriever):
            """Custom retriever wrapping pgVectorDB search methods."""

            vectorstore: Any
            search_method: str
            search_kwargs: Dict[str, Any]

            class Config:
                arbitrary_types_allowed = True

            def _get_relevant_documents(
                self,
                query: str,
                *,
                run_manager: Optional[CallbackManagerForRetrieverRun] = None,
            ) -> TypingList[Document]:
                """Sync version - not implemented (use async version)."""
                raise NotImplementedError(
                    "Sync retrieval not supported. Use async methods with ainvoke() or aget_relevant_documents()"
                )

            async def _aget_relevant_documents(
                self,
                query: str,
                *,
                run_manager: Optional[CallbackManagerForRetrieverRun] = None,
            ) -> TypingList[Document]:
                """Async retrieval using configured search method."""
                # Get the search method from vectorstore
                method = getattr(self.vectorstore, self.search_method, None)
                if not method:
                    raise ValueError(
                        f"Search method '{self.search_method}' not found on vectorstore"
                    )

                # Call search method
                results = await method(query, **self.search_kwargs)

                # Convert QueryResult to Document
                return [
                    Document(
                        page_content=result["content"],
                        metadata={**result["metadata"], "score": result["score"]},
                    )
                    for result in results
                ]

        return VectorStoreRetriever(
            vectorstore=self, search_method=search_method, search_kwargs=search_kwargs
        )
