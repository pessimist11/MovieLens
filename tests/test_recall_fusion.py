"""Multi-source recall fusion: RRF weights and the popularity-backfill rule."""

from __future__ import annotations

from dataclasses import replace

import pytest

from movielens.config import load_config
from movielens.pipeline import Recommender
from movielens.schema import Candidate, RecallModel
from tests.test_itemcf import make_dataset


class FakeSource(RecallModel):
    """Deterministic recall source: returns ``items`` in a fixed order."""

    def __init__(self, name: str, items: list[int], fallback_only: bool = False):
        self.name = name
        self.items = items
        self.fallback_only = fallback_only
        self.implemented = True

    def recall(self, user_idx: int, k: int, exclude: set[int]) -> list[Candidate]:
        return [
            Candidate(item_idx=i, score=1.0 / (rank + 1), source=self.name)
            for rank, i in enumerate(self.items)
            if i not in exclude
        ][:k]


@pytest.fixture
def rec():
    rows = [
        (0, 0, 100, "train"), (0, 1, 200, "train"),
        (1, 0, 100, "train"), (1, 1, 200, "train"),
        (2, 0, 100, "train"), (2, 2, 200, "train"),
    ]
    ds = make_dataset(rows, n_users=4, n_items=8)
    cfg = replace(
        load_config(),
        recall=replace(load_config().recall, min_primary_candidates=5),
        rerank=replace(load_config().rerank, enabled=False),
    )
    return Recommender(dataset=ds, cfg=cfg)


def test_fallback_backfills_below_primary(rec):
    rec.recall_models = [
        FakeSource("itemcf", [3, 4]),
        FakeSource("popular", [0, 1, 5], fallback_only=True),
    ]
    merged = rec.recall(0, exclude=set())
    assert [c.item_idx for c in merged] == [3, 4, 0, 1, 5]
    primary_min = min(c.score for c in merged[:2])
    assert all(c.score < primary_min for c in merged[2:])
    assert merged[0].source == "itemcf"
    assert merged[-1].source == "popular"


def test_fallback_serves_cold_start(rec):
    rec.recall_models = [
        FakeSource("itemcf", []),
        FakeSource("popular", [6, 7], fallback_only=True),
    ]
    merged = rec.recall(99, exclude=set())
    assert [c.item_idx for c in merged] == [6, 7]
    assert all(c.source == "popular" for c in merged)


def test_no_backfill_when_primary_is_sufficient(rec):
    rec.recall_models = [
        FakeSource("itemcf", [1, 2, 3, 4, 5]),
        FakeSource("popular", [6, 7], fallback_only=True),
    ]
    merged = rec.recall(0, exclude=set())
    assert [c.item_idx for c in merged] == [1, 2, 3, 4, 5]


def test_backfill_does_not_duplicate_primary_items(rec):
    rec.recall_models = [
        FakeSource("itemcf", [3, 4]),
        FakeSource("popular", [3, 4, 6], fallback_only=True),
    ]
    merged = rec.recall(0, exclude=set())
    assert [c.item_idx for c in merged] == [3, 4, 6]


def test_equal_weight_rrf_lets_popularity_dominate(rec):
    rec.recall_models = [
        FakeSource("popular", [7, 6], fallback_only=False),
        FakeSource("itemcf", [3, 4]),
    ]
    merged = rec.recall(0, exclude=set())
    # Both sources contribute 1/(60+rank); rank 0 of two sources beats rank 1.
    assert merged[0].item_idx in {7, 3}
    assert {c.item_idx for c in merged} == {7, 6, 3, 4}
    scores = [c.score for c in merged]
    assert scores == sorted(scores, reverse=True)


def test_source_weights_scale_rrf(rec):
    rec.cfg = replace(
        rec.cfg, raw={"recall": {"source_weights": {"itemcf": 2.0, "popular": 1.0}}}
    )
    rec.recall_models = [
        FakeSource("popular", [7], fallback_only=False),
        FakeSource("itemcf", [3]),
    ]
    merged = {c.item_idx: c for c in rec.recall(0, exclude=set())}
    assert merged[3].score == pytest.approx(2.0 * merged[7].score, rel=1e-6)
