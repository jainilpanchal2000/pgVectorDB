"""
Mixin modules for pgVectorDB.

Each mixin encapsulates a logical group of functionality that is composed
into the main ``pgVectorDB`` class via multiple inheritance.
"""

from .documents import DocumentsMixin
from .indexing import IndexingMixin
from .analytics import AnalyticsMixin
from .storage import StorageMixin
from .multimodal import MultimodalMixin
from .integrations import IntegrationsMixin

__all__ = [
    "DocumentsMixin",
    "IndexingMixin",
    "AnalyticsMixin",
    "StorageMixin",
    "MultimodalMixin",
    "IntegrationsMixin",
]
