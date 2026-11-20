"""Recall sources. The pipeline auto-registers every class with ``implemented=True``."""

from ..mf.bpr import BPRRecall
from .itemcf import ItemCFRecall
from .popular import PopularityRecall
from .two_tower import TwoTowerRecall

RECALL_REGISTRY = [PopularityRecall, ItemCFRecall, BPRRecall, TwoTowerRecall]

__all__ = [
    "PopularityRecall",
    "ItemCFRecall",
    "BPRRecall",
    "TwoTowerRecall",
    "RECALL_REGISTRY",
]
