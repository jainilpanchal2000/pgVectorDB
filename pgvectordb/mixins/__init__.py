"""Mixin modules for pgVectorDB.

Each mixin encapsulates a logical group of functionality that is composed
into the main ``pgVectorDB`` class via multiple inheritance.
"""

from .analytics import AnalyticsMixin
from .documents import DocumentsMixin
from .gin_helper import GINIndexHelper
from .index_manager import IndexManager
from .indexing import IndexingMixin
from .integrations import IntegrationsMixin
from .multimodal import MultimodalMixin
from .storage import StorageMixin

__all__ = [
    "AnalyticsMixin",
    "DocumentsMixin",
    "GINIndexHelper",
    "IndexManager",
    "IndexingMixin",
    "IntegrationsMixin",
    "MultimodalMixin",
    "StorageMixin",
]
