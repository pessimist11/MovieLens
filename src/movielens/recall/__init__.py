"""Recall sources. The pipeline auto-registers every class with ``implemented=True``."""

from .itemcf import ItemCFRecall
from .popular import PopularityRecall
from .two_tower import TwoTowerRecall

RECALL_REGISTRY = [PopularityRecall, ItemCFRecall, TwoTowerRecall]

__all__ = ["PopularityRecall", "ItemCFRecall", "TwoTowerRecall", "RECALL_REGISTRY"]
