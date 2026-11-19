"""Time-split correctness, negative sampling and the seen-filter contract."""

from dataclasses import replace

import pandas as pd
import pytest

from movielens.config import load_config
from movielens.data.dataset import load_dataset
from movielens.data.prepare import prepare
from movielens.pipeline import Recommender


@pytest.fixture(scope="module")
def prepared(tmp_path_factory):
    base = tmp_path_factory.mktemp("movielens")
    raw = base / "raw" / "ml-latest-small"
    raw.mkdir(parents=True)

    # 8 users, 20 movies, 6 interactions each, with distinct item sets so the
    # train window always contains unseen items for negative sampling.
    rows = []
    for user_id in range(1, 9):
        for j in range(6):
            rows.append(
                {
                    "userId": user_id,
                    "movieId": ((user_id * 3 + j) % 20) + 1,
                    "rating": float((j % 5) + 1),
                    "timestamp": 1_000_000_000 + user_id * 100 + j,
                }
            )
    pd.DataFrame(rows).to_csv(raw / "ratings.csv", index=False)
    pd.DataFrame(
        {
            "movieId": list(range(1, 21)),
            "title": [f"Movie {i} (200{i})" for i in range(1, 21)],
            "genres": ["Action|Comedy" if i % 2 else "Drama|Romance" for i in range(1, 21)],
        }
    ).to_csv(raw / "movies.csv", index=False)

    cfg = load_config()
    cfg = replace(
        cfg,
        data=replace(
            cfg.data,
            min_user_interactions=1,
            min_item_interactions=1,
            processed_dir=str(base / "processed"),
        ),
        split=replace(cfg.split, strategy="global"),
    )
    summary = prepare(cfg, raw_dir=base / "raw")
    return cfg, load_dataset(base / "processed"), summary


def test_three_non_empty_time_splits(prepared):
    _, _, summary = prepared
    counts = summary["split_counts"]
    assert all(counts[s] > 0 for s in ("train", "valid", "test"))
    assert sum(counts.values()) == summary["n_interactions"]


def test_no_leakage_between_splits(prepared):
    _, ds, _ = prepared
    pairs = {
        split: set(zip(frame["user_idx"], frame["item_idx"]))
        for split in ("train", "valid", "test")
        for frame in [ds.frame(split)]
    }
    assert not (pairs["train"] & pairs["test"])
    assert not (pairs["train"] & pairs["valid"])
    for user_idx, truth in ds.ground_truth["test"].items():
        assert truth.isdisjoint(ds.seen["test"].get(user_idx, set()))
    for user_idx, truth in ds.ground_truth["valid"].items():
        assert truth.isdisjoint(ds.seen["valid"].get(user_idx, set()))


def test_negative_samples_are_unseen_and_same_window(prepared):
    _, ds, _ = prepared
    train_items = set(ds.frame("train")["item_idx"].to_numpy().tolist())
    assert len(ds.samples) > 0
    assert (ds.samples["pos_item_idx"] != ds.samples["neg_item_idx"]).all()
    assert set(ds.samples["neg_item_idx"].to_numpy().tolist()) <= train_items

    seen_train = {
        int(u): set(int(x) for x in g["item_idx"].to_numpy())
        for u, g in ds.frame("train").groupby("user_idx")
    }
    for row in ds.samples.itertuples(index=False):
        assert row.neg_item_idx not in seen_train.get(int(row.user_idx), set())


def test_pipeline_filters_seen_and_handles_cold_start(prepared):
    cfg, ds, _ = prepared
    rec = Recommender(dataset=ds, cfg=cfg)
    raw_user_id = ds.idx2user[0]
    result = rec.recommend(raw_user_id, k=5)
    seen_all = ds.seen_for(0, "all")
    assert len(result["items"]) == 5
    assert all(it["movie_id"] not in {ds.idx2movie[i] for i in seen_all} for it in result["items"])
    assert result["cold_start"] is False

    cold = rec.recommend(999_999, k=3)
    assert cold["cold_start"] is True
    assert len(cold["items"]) == 3

    metrics = rec.evaluate("test", k=5)
    assert 0.0 <= metrics.recall_at_k <= 1.0
