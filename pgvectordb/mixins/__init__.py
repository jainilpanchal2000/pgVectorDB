"""
Mixin modules for pgVectorDB.

Each mixin encapsulates a logical group of functionality that is composed
into the main ``pgVectorDB`` class via multiple inheritance.
"""

from .analytics import AnalyticsMixin
from .documents import DocumentsMixin
from .indexing import IndexingMixin
from .integrations import IntegrationsMixin
from .multimodal import MultimodalMixin
from .storage import StorageMixin

__all__ = [
    "DocumentsMixin",
    "IndexingMixin",
    "AnalyticsMixin",
    "StorageMixin",
    "MultimodalMixin",
    "IntegrationsMixin",
]
