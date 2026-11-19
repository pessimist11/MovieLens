"""Ranking models (stage 5). The pipeline registers the implemented one."""

from .deepfm import DeepFMRanker
from .din import DINRanker

RANK_REGISTRY = [DINRanker, DeepFMRanker]

__all__ = ["DINRanker", "DeepFMRanker", "RANK_REGISTRY"]
