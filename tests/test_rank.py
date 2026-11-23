from dataclasses import replace

import pytest
from fastapi.testclient import TestClient

from movielens.api import main as api
from movielens.config import Config, load_config
from movielens.pipeline import Recommender
from movielens.rank import RANK_REGISTRY
from movielens.rank.base import checkpoint_path
from movielens.rank.features import RankFeatures
from movielens.rank.samples import candidate_groups, make_snapshot
from movielens.rank.train import train_rankers
from tests.test_itemcf import make_dataset


@pytest.fixture
def data_and_config(tmp_path, monkeypatch):
    rows = [
        (0, 0, 100, "train"), (1, 1, 110, "train"),
        (2, 2, 120, "train"), (3, 3, 130, "train"),
        (0, 1, 200, "train"), (1, 2, 210, "train"),
        (2, 0, 220, "train"), (3, 0, 230, "train"),
        (0, 3, 300, "valid"), (0, 2, 400, "test"),
    ]
    ds = make_dataset(rows, n_users=4, n_items=6)
    ds.ground_truth = {"valid": {0: {3}}, "test": {0: {2}}}
    cfg = load_config()
    cfg = replace(cfg, rank=replace(
        cfg.rank, snapshot_fraction=0.5, snapshot_bpr_epochs=1,
        snapshot_two_tower_epochs=1, epochs=1, history_size=3,
    ))
    monkeypatch.setattr(Config, "artifacts_dir", lambda self: tmp_path)
    return ds, cfg


def test_candidate_labels_and_history_respect_the_snapshot(data_and_config):
    ds, cfg = data_and_config
    rec, truth, excluded, cutoff = make_snapshot(ds, cfg)
    assert rec.dataset.frame("train")["timestamp"].max() < cutoff
    assert truth == {0: {1}, 1: {2}, 2: {0}, 3: {0}}
    assert excluded == {4, 5}
    assert RankFeatures(rec.dataset, cfg).history[0].tolist() == [0, 0, 1]

    groups = candidate_groups(rec, truth, rec.dataset.seen["valid"],
                              unavailable=excluded, negative_limit=1)
    group = next(g for g in groups if g.user == 0)
    assert {c.item_idx for c in group.candidates} <= {1, 2, 3}
    assert 1 in {c.item_idx for c in group.candidates}
    assert group.weights.sum() == pytest.approx(3.0)


def test_checkpoint_key_is_independent_of_test_rows(data_and_config):
    ds, cfg = data_and_config
    original = checkpoint_path("din", ds, cfg)
    ds.interactions.loc[ds.interactions["split"] == "test", "timestamp"] += 100
    assert checkpoint_path("din", ds, cfg) == original
    ds.interactions.loc[ds.interactions["split"] == "valid", "timestamp"] += 100
    assert checkpoint_path("din", ds, cfg) != original


def test_trained_rankers_reload_and_score_via_the_api(data_and_config, monkeypatch):
    ds, cfg = data_and_config
    paths = train_rankers(ds, cfg, RANK_REGISTRY)
    assert all(path.is_file() for path in paths.values())

    for name in paths:
        rank_cfg = replace(cfg, rank=replace(cfg.rank, model=name))
        monkeypatch.setattr(api, "Recommender", lambda: Recommender(ds, rank_cfg))
        with TestClient(api.app) as client:
            stage = client.get("/health").json()["stages"]["rank"]
            assert [model["name"] for model in stage if model["active"]] == [name]
            response = client.get("/recommend/0?k=3&use_rerank=false")
            assert response.status_code == 200
            items = response.json()["items"]
            assert {row["movie_id"] for row in items}.isdisjoint({0, 1})
            scores = [row["rank_score"] for row in items]
            assert len(scores) == 3 and all(0.0 <= score <= 1.0 for score in scores)
            assert scores == sorted(scores, reverse=True)
            cold = client.get("/recommend/999?k=3").json()
            assert cold["cold_start"] and len(cold["items"]) == 3
            assert all(row["source"] == "popular" and row["rank_score"] is None
                       for row in cold["items"])
