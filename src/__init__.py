"""Production RAG System - Simple imports"""

from .core import pgVectorDB, IndexType, StorageLayout, DistanceMetric
from .evaluation import (
    RAGEvaluator,
    EvaluationDataset,
    EvaluationResult,
    KValueAnalysis,
    create_sample_evaluation_dataset,
)

__version__ = "2.0.0"

__all__ = [
    "pgVectorDB",
    "IndexType",
    "StorageLayout",
    "DistanceMetric",
    "RAGEvaluator",
    "EvaluationDataset",
    "EvaluationResult",
    "KValueAnalysis",
    "create_sample_evaluation_dataset",
]
