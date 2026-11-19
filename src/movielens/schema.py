"""Shared contracts for every layer of the pipeline.

Every model emits ``list[Candidate]`` in continuous-index space. This single
format is what makes the layers pluggable and keeps offline evaluation and
online serving on exactly the same code path.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass(frozen=True)
class Candidate:
    item_idx: int
    score: float
    source: str
    rank_score: float | None = None


@dataclass(frozen=True)
class Recommendation:
    movie_id: int
    title: str
    genres: list[str]
    score: float
    source: str
    rank_score: float | None
    reason: str


@dataclass(frozen=True)
class Metrics:
    recall_at_k: float
    ndcg_at_k: float
    coverage: float
    long_tail_coverage: float
    long_tail_share: float
    novelty: float
    ild: float
    n_users: int
    k: int

    def as_dict(self) -> dict:
        return {
            "recall@k": round(self.recall_at_k, 6),
            "ndcg@k": round(self.ndcg_at_k, 6),
            "coverage": round(self.coverage, 6),
            "long_tail_coverage": round(self.long_tail_coverage, 6),
            "long_tail_share": round(self.long_tail_share, 6),
            "novelty": round(self.novelty, 6),
            "ild": round(self.ild, 6),
            "n_users": self.n_users,
            "k": self.k,
        }


class RecallModel(ABC):
    """Recall source: cheaply produce a few hundred candidates per user."""

    name: str = "base"
    implemented: bool = False

    def fit(self, dataset) -> "RecallModel":
        """Optional offline fitting. Defaults to a stateless model."""
        return self

    @abstractmethod
    def recall(self, user_idx: int, k: int, exclude: set[int]) -> list[Candidate]:
        """Return up to k candidates, excluding already-seen item indices."""


class RankModel(ABC):
    """Ranker: score a few hundred candidates with user-item cross features."""

    name: str = "base"
    implemented: bool = False

    @abstractmethod
    def predict(self, user_idx: int, candidates: list[Candidate]) -> list[Candidate]:
        """Attach ``rank_score`` to each candidate."""


class Reranker(ABC):
    """Reranker: trade a little relevance for diversity / long-tail exposure."""

    name: str = "base"
    implemented: bool = False

    @abstractmethod
    def rerank(self, candidates: list[Candidate], k: int) -> list[Candidate]:
        """Return the final Top-k list."""
