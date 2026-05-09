"""
Reranker Module for pgVectorDB
================================

This module provides reranking backends that improve retrieval precision
by scoring initial search results against the query using specialized models.

**Why Reranking?**
Bi-encoder models (used for embeddings) are optimized for speed — they encode
queries and documents independently. Cross-encoder/reranker models are slower
but more accurate: they see *both* the query and document together, enabling
richer comparisons.

Typical workflow:
    1. Retrieve top-100 candidates quickly (semantic / hybrid / multimodal search)
    2. Rerank with a cross-encoder or API-based model
    3. Return top-5 or top-10 by rerank score

Supported backends:

| Class | Backend | Requires |
|-------|---------|----------|
| ``CrossEncoderReranker`` | Local sentence-transformers | ``sentence-transformers`` |
| ``CohereReranker`` | Cohere API | ``cohere``, API key |
| ``AWSBedrockReranker`` | AWS Bedrock | ``boto3``, AWS credentials |
| ``HuggingFaceReranker`` | Local transformers pipeline | ``transformers``, ``torch`` |

Examples:
    >>> from pgvectordb.rerankers import CrossEncoderReranker, CohereReranker
    >>>
    >>> # Local cross-encoder (no API key needed)
    >>> reranker = CrossEncoderReranker(
    ...     model="cross-encoder/ms-marco-MiniLM-L-6-v2"
    ... )
    >>>
    >>> # Rerank via core method
    >>> results = await rag.rerank_search(
    ...     query="best noise cancelling headphones under $200",
    ...     reranker=reranker,
    ...     k=100,           # Retrieve 100 candidates
    ...     rerank_top_k=5,  # Return best 5 after reranking
    ... )

Version: 0.0.3
"""

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ==================== Base Class ====================


class BaseReranker(ABC):
    """
    Abstract base class for all reranking backends.

    A reranker takes an initial list of search results and re-orders them
    by computing a relevance score for each (query, document) pair.

    Subclass Contract:
        - Implement ``rerank(query, documents, top_k)``
        - Return results sorted by rerank score, descending

    Examples:
        >>> class MyReranker(BaseReranker):
        ...     def rerank(self, query, documents, top_k=None):
        ...         # Score each document
        ...         scored = [(doc, my_score(query, doc)) for doc in documents]
        ...         scored.sort(key=lambda x: x[1], reverse=True)
        ...         return [(doc, score) for doc, score in scored[:top_k]]
    """

    @abstractmethod
    def rerank(
        self,
        query: str,
        documents: List[Dict[str, Any]],
        top_k: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        Rerank a list of documents by relevance to the query.

        Args:
            query: The search query string.
            documents: List of QueryResult-like dicts with at minimum
                ``{"id": ..., "content": ..., "metadata": ..., "score": ...}``.
            top_k: Maximum results to return. If None, returns all input docs
                (reordered).

        Returns:
            Documents re-ordered by rerank score (best first).
            The ``score`` field is replaced with the rerank score.
        """
        ...

    def _safe_top_k(self, documents: List, top_k: Optional[int]) -> int:
        """Return a valid top_k bounded by the number of documents."""
        return min(top_k, len(documents)) if top_k else len(documents)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


# ==================== CrossEncoderReranker ====================


class CrossEncoderReranker(BaseReranker):
    """
    Local cross-encoder reranker using ``sentence-transformers``.

    Cross-encoders process query and document *together*, providing high
    accuracy at the cost of latency (no pre-computation possible).

    Recommended models:
        - ``cross-encoder/ms-marco-MiniLM-L-6-v2`` — Fast, good accuracy (~100ms/batch)
        - ``cross-encoder/ms-marco-MiniLM-L-12-v2`` — Slower, better accuracy
        - ``cross-encoder/ms-marco-electra-base`` — Best accuracy, slowest
        - ``BAAI/bge-reranker-v2-m3`` — Multilingual
        - ``cross-encoder/nli-deberta-v3-small`` — General NLI (not search-tuned)

    Args:
        model: HuggingFace model name or local path.
        device: Device to run on (``"cpu"``, ``"cuda"``, ``"mps"``). Auto-detected.
        batch_size: Documents to score per batch (default: 32).
        max_length: Max token length per (query, doc) pair (default: 512).

    Examples:
        >>> reranker = CrossEncoderReranker(
        ...     model="cross-encoder/ms-marco-MiniLM-L-6-v2"
        ... )
        >>> results = reranker.rerank(query, candidates, top_k=5)
    """

    def __init__(
        self,
        model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        device: Optional[str] = None,
        batch_size: int = 32,
        max_length: int = 512,
    ):
        try:
            from sentence_transformers import CrossEncoder
        except ImportError:
            raise ImportError(
                "sentence-transformers is required for CrossEncoderReranker. "
                "Install with: pip install sentence-transformers"
            )

        self.model_name = model
        self.batch_size = batch_size
        self.max_length = max_length

        logger.info(f"Loading CrossEncoder: {model}")
        self._model = CrossEncoder(
            model,
            device=device,
            max_length=max_length,
        )
        logger.info(f"✓ CrossEncoder loaded: {model}")

    def rerank(
        self,
        query: str,
        documents: List[Dict[str, Any]],
        top_k: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        Score each document using the cross-encoder and reorder.

        Args:
            query: Search query.
            documents: Initial search results.
            top_k: Number to return (default: all).

        Returns:
            Reordered documents with updated ``score`` field.
        """
        if not documents:
            return []

        # Build (query, passage) pairs
        pairs = [[query, doc["content"]] for doc in documents]

        # Score all pairs
        scores = self._model.predict(
            pairs,
            batch_size=self.batch_size,
            show_progress_bar=False,
        )

        # Zip and sort
        scored = list(zip(documents, scores.tolist()))
        scored.sort(key=lambda x: x[1], reverse=True)

        top = self._safe_top_k(documents, top_k)
        return [
            {**doc, "score": float(score), "rerank_score": float(score)}
            for doc, score in scored[:top]
        ]

    def __repr__(self) -> str:
        return f"CrossEncoderReranker(model='{self.model_name}')"


# ==================== CohereReranker ====================


class CohereReranker(BaseReranker):
    """
    Cloud reranker using the Cohere Rerank API.

    Cohere's rerank endpoint is highly optimized for production use — no GPU
    required locally, and the API handles batching automatically.

    Available models:
        - ``rerank-english-v3.0`` — English-only, frontier model
        - ``rerank-multilingual-v3.0`` — Multiple languages
        - ``rerank-english-v2.0`` — Legacy English (cheaper)

    Pricing: per 1K search units (query + top doc == 1 search unit)

    Args:
        api_key: Cohere API key. If None, reads from ``COHERE_API_KEY`` env var.
        model: Rerank model name (default: ``rerank-english-v3.0``).
        max_chunks_per_doc: Max text chunks per document (default: 10).

    Examples:
        >>> import os
        >>> reranker = CohereReranker(api_key=os.environ["COHERE_API_KEY"])
        >>> results = reranker.rerank(query, candidates, top_k=5)
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "rerank-english-v3.0",
        max_chunks_per_doc: int = 10,
    ):
        try:
            import cohere
        except ImportError:
            raise ImportError(
                "cohere is required for CohereReranker. "
                "Install with: pip install cohere"
            )

        import os

        key = api_key or os.environ.get("COHERE_API_KEY")
        if not key:
            raise ValueError(
                "Cohere API key required. Pass api_key= or set COHERE_API_KEY env var."
            )

        self.model = model
        self.max_chunks_per_doc = max_chunks_per_doc
        self._client = cohere.Client(key)
        logger.info(f"CohereReranker initialized: {model}")

    def rerank(
        self,
        query: str,
        documents: List[Dict[str, Any]],
        top_k: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        Rerank using Cohere Rerank API.

        Args:
            query: Search query.
            documents: Initial search results.
            top_k: Number to return (default: all).

        Returns:
            Reordered documents with updated ``score`` field.
        """
        if not documents:
            return []

        texts = [doc["content"] for doc in documents]
        top = self._safe_top_k(documents, top_k)

        try:
            response = self._client.rerank(
                query=query,
                documents=texts,
                top_n=top,
                model=self.model,
                max_chunks_per_doc=self.max_chunks_per_doc,
            )
        except Exception as e:
            logger.error(f"Cohere rerank API call failed: {e}")
            raise RuntimeError(f"Cohere rerank failed: {e}") from e

        results = []
        for result in response.results:
            doc = documents[result.index]
            score = result.relevance_score
            results.append({**doc, "score": float(score), "rerank_score": float(score)})

        return results

    def __repr__(self) -> str:
        return f"CohereReranker(model='{self.model}')"


# ==================== AWSBedrockReranker ====================


class AWSBedrockReranker(BaseReranker):
    """
    Cloud reranker using AWS Bedrock's rerank endpoint (``amazon.rerank-v1:0``).

    Uses the Amazon Bedrock rerank API, which wraps a hosted reranking model.
    Requires AWS credentials (via environment, IAM role, or profile).

    Available models:
        - ``amazon.rerank-v1:0`` — Amazon's primary reranking model
        - ``cohere.rerank-v3-5:0`` — Cohere v3.5 via Bedrock

    Args:
        region_name: AWS region (default: ``"us-east-1"``).
        model_id: Bedrock model ID (default: ``"amazon.rerank-v1:0"``).
        aws_access_key_id: Optional, falls back to env/IAM.
        aws_secret_access_key: Optional, falls back to env/IAM.
        aws_session_token: Optional, for temporary credentials.

    Examples:
        >>> reranker = AWSBedrockReranker(region_name="us-east-1")
        >>> results = reranker.rerank(query, candidates, top_k=5)
    """

    def __init__(
        self,
        region_name: str = "us-east-1",
        model_id: str = "amazon.rerank-v1:0",
        aws_access_key_id: Optional[str] = None,
        aws_secret_access_key: Optional[str] = None,
        aws_session_token: Optional[str] = None,
    ):
        try:
            import boto3
        except ImportError:
            raise ImportError(
                "boto3 is required for AWSBedrockReranker. "
                "Install with: pip install boto3"
            )

        self.model_id = model_id
        self.region_name = region_name

        session_kwargs = {}
        if aws_access_key_id:
            session_kwargs["aws_access_key_id"] = aws_access_key_id
        if aws_secret_access_key:
            session_kwargs["aws_secret_access_key"] = aws_secret_access_key
        if aws_session_token:
            session_kwargs["aws_session_token"] = aws_session_token

        import boto3

        session = boto3.Session(**session_kwargs) if session_kwargs else boto3.Session()
        self._client = session.client("bedrock-agent-runtime", region_name=region_name)
        logger.info(f"AWSBedrockReranker initialized: {model_id} in {region_name}")

    def rerank(
        self,
        query: str,
        documents: List[Dict[str, Any]],
        top_k: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        Rerank using AWS Bedrock Rerank API.

        Args:
            query: Search query.
            documents: Initial search results.
            top_k: Number to return (default: all).

        Returns:
            Reordered documents with updated ``score`` field.
        """
        if not documents:
            return []

        top = self._safe_top_k(documents, top_k)

        # Build Bedrock sources format
        sources = [
            {
                "inlineDocumentSource": {
                    "textDocument": {"text": doc["content"]},
                    "type": "TEXT",
                }
            }
            for doc in documents
        ]

        try:
            response = self._client.rerank(
                rerankingConfiguration={
                    "bedrockRerankingConfiguration": {
                        "modelConfiguration": {
                            "modelArn": f"arn:aws:bedrock:{self.region_name}::foundation-model/{self.model_id}"
                        },
                        "numberOfResults": top,
                    },
                    "type": "BEDROCK_RERANKING_MODEL",
                },
                sources=sources,
                textSources=[{"text": query}],
            )
        except Exception as e:
            logger.error(f"AWS Bedrock rerank API call failed: {e}")
            raise RuntimeError(f"AWS Bedrock rerank failed: {e}") from e

        results = []
        for item in response.get("rerankingResults", []):
            original_idx = item["index"]
            score = item["relevanceScore"]
            doc = documents[original_idx]
            results.append({**doc, "score": float(score), "rerank_score": float(score)})

        return results

    def __repr__(self) -> str:
        return (
            f"AWSBedrockReranker(model='{self.model_id}', region='{self.region_name}')"
        )


# ==================== HuggingFaceReranker ====================


class HuggingFaceReranker(BaseReranker):
    """
    Local reranker using HuggingFace ``transformers`` text-classification pipeline.

    Any sequence-classification model that scores (query, document) pairs can be
    used here. This is the most flexible option for custom or fine-tuned models.

    Recommended models:
        - ``BAAI/bge-reranker-v2-m3`` — Strong multilingual reranker
        - ``BAAI/bge-reranker-base`` — Lighter version
        - ``cross-encoder/ms-marco-MiniLM-L-6-v2`` — Popular, fast
        - ``jinaai/jina-reranker-v2-base-multilingual`` — Jina's multilingual model

    Args:
        model: HuggingFace model name or local path.
        device: Device (``"cpu"``, ``"cuda"``, ``"mps"``). Auto-detected if None.
        batch_size: Batch size for inference (default: 16).
        max_length: Max token length (default: 512).

    Examples:
        >>> reranker = HuggingFaceReranker(
        ...     model="BAAI/bge-reranker-v2-m3",
        ...     device="cuda",
        ... )
        >>> results = reranker.rerank(query, candidates, top_k=5)
    """

    def __init__(
        self,
        model: str = "BAAI/bge-reranker-v2-m3",
        device: Optional[str] = None,
        batch_size: int = 16,
        max_length: int = 512,
    ):
        try:
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
        except ImportError:
            raise ImportError(
                "transformers and torch are required for HuggingFaceReranker. "
                "Install with: pip install transformers torch"
            )

        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.model_name = model
        self.batch_size = batch_size
        self.max_length = max_length

        # Auto-detect device
        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        self.device = device

        logger.info(f"Loading HuggingFace reranker: {model} on {device}")
        self._tokenizer = AutoTokenizer.from_pretrained(model)
        self._model = AutoModelForSequenceClassification.from_pretrained(model)
        self._model.to(device)
        self._model.eval()
        logger.info(f"✓ HuggingFace reranker loaded: {model}")

    def _score_batch(self, pairs: List[Tuple[str, str]]) -> List[float]:
        """Score a batch of (query, document) pairs."""
        import torch

        inputs = self._tokenizer(
            [q for q, _ in pairs],
            [d for _, d in pairs],
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            logits = self._model(**inputs).logits

        # For binary classifiers, use logit of class 1 (relevant)
        # For single-output models, use the raw logit
        if logits.shape[-1] == 2:
            scores = torch.softmax(logits, dim=-1)[:, 1]
        else:
            scores = logits.squeeze(-1)

        return scores.cpu().tolist()

    def rerank(
        self,
        query: str,
        documents: List[Dict[str, Any]],
        top_k: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        Rerank documents using the local HuggingFace model.

        Args:
            query: Search query.
            documents: Initial search results.
            top_k: Number to return (default: all).

        Returns:
            Reordered documents with updated ``score`` field.
        """
        if not documents:
            return []

        all_pairs = [(query, doc["content"]) for doc in documents]
        all_scores = []

        # Process in batches
        for i in range(0, len(all_pairs), self.batch_size):
            batch = all_pairs[i : i + self.batch_size]
            batch_scores = self._score_batch(batch)
            all_scores.extend(batch_scores)

        # Sort by score
        scored = list(zip(documents, all_scores))
        scored.sort(key=lambda x: x[1], reverse=True)

        top = self._safe_top_k(documents, top_k)
        return [
            {**doc, "score": float(score), "rerank_score": float(score)}
            for doc, score in scored[:top]
        ]

    def __repr__(self) -> str:
        return f"HuggingFaceReranker(model='{self.model_name}', device='{self.device}')"


# ==================== Utility / Factory ====================


def create_reranker(
    backend: str,
    **kwargs: Any,
) -> BaseReranker:
    """
    Factory function to create a reranker by backend name.

    Args:
        backend: One of ``"cross_encoder"``, ``"cohere"``, ``"bedrock"``,
            ``"huggingface"``.
        **kwargs: Arguments forwarded to the reranker constructor.

    Returns:
        Configured reranker instance.

    Raises:
        ValueError: If backend name is unknown.

    Examples:
        >>> reranker = create_reranker("cross_encoder",
        ...     model="cross-encoder/ms-marco-MiniLM-L-6-v2"
        ... )
        >>> reranker = create_reranker("cohere", api_key="...", model="rerank-english-v3.0")
        >>> reranker = create_reranker("bedrock", region_name="us-west-2")
        >>> reranker = create_reranker("huggingface", model="BAAI/bge-reranker-v2-m3")
    """
    backends = {
        "cross_encoder": CrossEncoderReranker,
        "crossencoder": CrossEncoderReranker,
        "cohere": CohereReranker,
        "bedrock": AWSBedrockReranker,
        "aws": AWSBedrockReranker,
        "aws_bedrock": AWSBedrockReranker,
        "huggingface": HuggingFaceReranker,
        "hf": HuggingFaceReranker,
    }

    key = backend.lower().replace("-", "_")
    cls = backends.get(key)

    if cls is None:
        raise ValueError(
            f"Unknown reranker backend: '{backend}'. Supported: {list(backends.keys())}"
        )

    return cls(**kwargs)


# ==================== Module Exports ====================

__all__ = [
    # Base
    "BaseReranker",
    # Implementations
    "CrossEncoderReranker",
    "CohereReranker",
    "AWSBedrockReranker",
    "HuggingFaceReranker",
    # Factory
    "create_reranker",
]
