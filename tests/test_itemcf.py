"""ItemCF correctness: hand-computed similarities, leakage, filtering, cache."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from movielens.data.dataset import Dataset
from movielens.recall.itemcf import ItemCFRecall, _topk_per_row


def make_dataset(rows, n_users: int, n_items: int) -> Dataset:
    """rows: (user_idx, item_idx, timestamp, split)."""
    frame = pd.DataFrame(rows, columns=["user_idx", "item_idx", "timestamp", "split"])
    frame["rating"] = 5.0
    frame["is_positive"] = True
    frame["userId"] = frame["user_idx"]
    frame["movieId"] = frame["item_idx"]

    def group_map(split: str) -> dict[int, set[int]]:
        part = frame[frame["split"] == split]
        return {
            int(u): set(int(x) for x in g["item_idx"])
            for u, g in part.groupby("user_idx")
        }

    train = frame[frame["split"] == "train"]
    popularity = np.bincount(train["item_idx"], minlength=n_items).astype(np.int64)
    return Dataset(
        interactions=frame,
        movies=pd.DataFrame(
            {
                "item_idx": range(n_items),
                "movie_id": range(n_items),
                "title": [f"Movie {i}" for i in range(n_items)],
                "genres": ["Action"] * n_items,
                "year": [2000] * n_items,
            }
        ),
        samples=pd.DataFrame(columns=["user_idx", "pos_item_idx", "neg_item_idx"]),
        user2idx={i: i for i in range(n_users)},
        item2idx={i: i for i in range(n_items)},
        idx2user={i: i for i in range(n_users)},
        idx2movie={i: i for i in range(n_items)},
        seen={
            "train": group_map("train"),
            "valid": group_map("train"),
            "test": group_map("train"),
            "all": group_map("train"),
        },
        ground_truth={"valid": {}, "test": {}},
        genre_matrix=np.eye(n_items, dtype=np.float32),
        genre_list=["Action"],
        item_popularity=popularity,
        tail_items=set(),
        n_users=n_users,
        n_items=n_items,
        split_meta={},
    )


@pytest.fixture
def toy() -> Dataset:
    """user0: {0,1}  user1: {0,1}  user2: {0,2}  (all train)."""
    rows = [
        (0, 0, 100, "train"), (0, 1, 200, "train"),
        (1, 0, 100, "train"), (1, 1, 200, "train"),
        (2, 0, 100, "train"), (2, 2, 200, "train"),
    ]
    return make_dataset(rows, n_users=4, n_items=6)


def model_for(ds: Dataset, **kwargs) -> ItemCFRecall:
    kwargs.setdefault("use_cache", False)
    return ItemCFRecall(ds, **kwargs)


def test_similarity_matches_hand_computed(toy):
    # C_01 = 2 users, N_0 = 3, N_1 = 2 -> cosine = 2 / sqrt(6)
    model = model_for(toy, iuf=False, alpha=0.5, shrink=0.0, top_n_neighbours=6)
    sim = model.sim.toarray()
    assert sim[0, 1] == pytest.approx(2.0 / np.sqrt(6.0), rel=1e-5)
    assert sim[1, 0] == pytest.approx(sim[0, 1], rel=1e-6)
    assert np.allclose(np.diag(sim), 0.0)
    # items 3..5 never co-occur with anything
    assert sim[3:, :].sum() == 0.0


def test_iuf_downweights_active_users(toy):
    plain = model_for(toy, iuf=False).sim.toarray()
    weighted = model_for(toy, iuf=True).sim.toarray()
    # every user has 2 interactions -> w = 1/log(3), C_01 = 2 * w^2
    expected = 2.0 * (1.0 / np.log(3.0)) ** 2 / np.sqrt(6.0)
    assert weighted[0, 1] == pytest.approx(expected, rel=1e-5)
    assert weighted[0, 1] < plain[0, 1]


def test_alpha_controls_popularity_penalty(toy):
    raw = model_for(toy, iuf=False, alpha=0.0).sim.toarray()
    hard = model_for(toy, iuf=False, alpha=1.0).sim.toarray()
    assert raw[0, 1] == pytest.approx(2.0, rel=1e-5)          # pure co-occurrence
    assert hard[0, 1] == pytest.approx(2.0 / 6.0, rel=1e-5)   # C / (N_0 * N_1)
    # sim(0,2) involves the more popular item 0 and N_2 = 1
    assert hard[0, 2] == pytest.approx(1.0 / 3.0, rel=1e-5)


def test_min_cooccurrence_uses_raw_support(toy):
    # With IUF the weighted co-occurrence is < 1; filtering must still keep it.
    model = model_for(toy, iuf=True, min_cooccurrence=2)
    sim = model.sim.toarray()
    assert sim[0, 1] > 0.0     # raw support = 2 users
    assert sim[0, 2] == 0.0    # raw support = 1 user -> dropped


def test_topn_neighbours_truncates_rows(toy):
    model = model_for(toy, top_n_neighbours=1)
    assert np.diff(model.sim.indptr).max() <= 1


def test_topk_helper_handles_empty_and_wide_rows():
    import scipy.sparse as sp

    matrix = sp.csr_matrix(
        np.array([[0.0, 5.0, 1.0, 3.0], [0.0, 0.0, 0.0, 0.0]], dtype=np.float32)
    )
    top = _topk_per_row(matrix, 2).toarray()
    assert top[0, 1] == 5.0 and top[0, 3] == 3.0 and top[0, 2] == 0.0
    assert top[1].sum() == 0.0


def test_recall_excludes_history_and_seen(toy):
    model = model_for(toy)
    # user0 history = {0,1}; item 2 co-occurs with item 0 via user2.
    candidates = model.recall(0, k=6, exclude=set())
    items = [c.item_idx for c in candidates]
    assert items == [2]
    assert 0 not in items and 1 not in items      # own train history
    scores = [c.score for c in candidates]
    assert scores == sorted(scores, reverse=True)
    assert all(c.source == "itemcf" for c in candidates)
    assert all(c.score <= 1.0 for c in candidates)
    # excluding the only reachable item leaves an empty list, not a seen item
    assert model.recall(0, k=6, exclude={2}) == []
    assert model.recall(0, k=1, exclude=set())[0].item_idx == 2


def test_cold_start_user_returns_empty(toy):
    model = model_for(toy)
    assert model.recall(3, k=5, exclude=set()) == []
    assert model.recall(999, k=5, exclude=set()) == []


def test_no_leakage_from_valid_or_test():
    rows = [
        (0, 0, 100, "train"),
        (0, 4, 300, "valid"),
        (0, 5, 400, "test"),
        (1, 0, 100, "train"),
        (1, 1, 200, "train"),
    ]
    ds = make_dataset(rows, n_users=2, n_items=6)
    model = model_for(ds)
    sim = model.sim.toarray()
    assert sim[4:, :].sum() == 0.0          # valid/test items never enter the graph
    assert set(model._history[0][0].tolist()) == {0}   # profile is train-only


def test_history_size_uses_most_recent(toy):
    model = model_for(toy, history_size=1)
    assert model._history[0][0].tolist() == [1]   # ts=200 is newer than ts=100
    candidates = model.recall(0, k=5, exclude=set())
    assert all(c.item_idx != 1 for c in candidates)


def test_history_decay_prefers_recent_items():
    rows = [
        (0, 0, 0, "train"),
        (0, 1, 86400 * 30, "train"),   # 30 days later
        (1, 0, 0, "train"), (1, 2, 1, "train"),
        (2, 1, 0, "train"), (2, 3, 1, "train"),
    ]
    ds = make_dataset(rows, n_users=3, n_items=5)
    model = model_for(ds, iuf=False, history_decay=1.0, history_size=0)
    _, weights = model._history[0]
    assert weights[0] < weights[1]               # older item is down-weighted
    assert weights[1] == pytest.approx(1.0, rel=1e-5)


def test_explain_points_to_strongest_neighbour(toy):
    model = model_for(toy)
    candidates = model.recall(0, k=3, exclude=set())
    target = candidates[0].item_idx
    found = model.explain(0, target)
    assert found is not None
    history_item, contribution = found
    assert history_item in {0, 1}
    assert contribution > 0.0
    assert model.explain(3, target) is None      # cold start has no explanation


def test_similarity_matrix_cache_roundtrip(toy):
    first = ItemCFRecall(toy, use_cache=True)
    assert first.stats["from_cache"] is False
    second = ItemCFRecall(toy, use_cache=True)
    assert second.stats["from_cache"] is True
    assert np.allclose(first.sim.toarray(), second.sim.toarray())
    third = ItemCFRecall(toy, use_cache=True, alpha=0.9)
    assert third.stats["from_cache"] is False    # params changed -> refit
