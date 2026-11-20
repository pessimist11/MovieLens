"""Two-tower learning, leakage guards, InfoNCE masking and FAISS retrieval."""

from __future__ import annotations

import os
import subprocess
import sys

import numpy as np
import pytest
import torch

from movielens.recall import RECALL_REGISTRY
from movielens.recall.two_tower import (
    TwoTowerModel,
    TwoTowerRecall,
    info_nce_loss,
)
from tests.test_itemcf import make_dataset


@pytest.fixture
def toy():
    rows = [
        (0, 0, 100, "train"), (0, 1, 200, "train"), (0, 2, 300, "train"),
        (1, 0, 100, "train"), (1, 1, 200, "train"), (1, 3, 300, "train"),
        (2, 0, 100, "train"), (2, 2, 200, "train"), (2, 4, 300, "train"),
        (3, 1, 100, "train"), (3, 3, 200, "train"), (3, 5, 300, "train"),
    ]
    return make_dataset(rows, n_users=4, n_items=8)


def small_kwargs(**extra):
    kwargs = dict(
        dim=8,
        hidden_dim=16,
        batch_size=16,
        history_size=3,
        extra_negatives=4,
        epochs=4,
        eval_every=2,
        patience=2,
        dropout=0.0,
    )
    kwargs.update(extra)
    return kwargs


def test_info_nce_masks_false_negative_but_keeps_target():
    users = torch.tensor([0, 1])
    positives = torch.tensor([0, 1])
    seen = torch.tensor([[True, True], [False, True]])
    user = torch.eye(2)
    item = torch.eye(2)
    loss = info_nce_loss(user, item, users, positives, seen_bool=seen, temperature=1.0)
    # Row 0: [1, -1e4] -> ~0. Row 1: [0, 1] -> log(1 + exp(-1)).
    expected = np.log1p(np.exp(-1.0)) / 2
    assert float(loss) == pytest.approx(expected, abs=1e-6)


def test_info_nce_masks_extra_seen_negative():
    users = torch.tensor([0])
    positives = torch.tensor([0])
    seen = torch.tensor([[True, True]])
    loss = info_nce_loss(
        torch.tensor([[1.0, 0.0]]),
        torch.tensor([[1.0, 0.0]]),
        users,
        positives,
        seen_bool=seen,
        temperature=1.0,
        negative_vectors=torch.tensor([[[0.0, 1.0]]]),
        negative_items=torch.tensor([[1]]),
    )
    assert float(loss) == pytest.approx(0.0, abs=1e-6)


def test_history_is_strict_prefix_then_full_train_profile(toy):
    model = TwoTowerModel(toy, epochs=0, use_cache=False, history_size=3)
    items, mask = model.history_batch(np.array([0]), np.array([2]))
    assert items[mask.astype(bool)].tolist() == [0, 1]  # target item 2 excluded
    full_items, full_mask = model.history_batch(np.array([0]))
    assert full_items[full_mask.astype(bool)].tolist() == [0, 1, 2]
    # valid/test interactions never enter any train profile
    changed = make_dataset(
        [(0, 0, 100, "train"), (0, 1, 200, "train"), (0, 6, 300, "valid")],
        n_users=1,
        n_items=8,
    )
    other = TwoTowerModel(changed, epochs=0, use_cache=False)
    items, mask = other.history_batch(np.array([0]))
    assert 6 not in items[mask.astype(bool)]


def test_training_exports_unit_norm_embeddings(toy):
    model = TwoTowerModel(toy, use_cache=False, **small_kwargs())
    assert model.user_vectors.shape == (4, 8)
    assert model.item_vectors.shape == (8, 8)
    assert np.allclose(np.linalg.norm(model.user_vectors, axis=1), 1.0, atol=1e-5)
    assert np.allclose(np.linalg.norm(model.item_vectors, axis=1), 1.0, atol=1e-5)
    assert len(model.history) == 4
    assert model.stats["best_epoch"] is not None


def test_flat_recall_filters_seen_and_handles_cold_start(toy):
    rec = TwoTowerRecall(toy, use_cache=False, index_type="flat", **small_kwargs(epochs=2))
    candidates = rec.recall(0, k=3, exclude={6})
    items = [c.item_idx for c in candidates]
    assert len(items) == 3
    assert 6 not in items
    assert not set(items) & toy.seen["valid"][0]
    assert all(c.source == "two_tower" for c in candidates)
    assert rec.stats["index_ntotal"] == toy.n_items
    assert rec.recall(999, 3, set()) == []


def test_embedding_and_index_cache_roundtrip(toy):
    kwargs = small_kwargs(epochs=2, use_cache=True)
    first = TwoTowerRecall(toy, **kwargs)
    second = TwoTowerRecall(toy, **kwargs)
    assert first.stats["from_cache"] is False
    assert second.stats["from_cache"] is True
    assert np.allclose(first.model.user_vectors, second.model.user_vectors)
    assert np.allclose(first.model.item_vectors, second.model.item_vectors)
    changed = TwoTowerRecall(toy, **{**kwargs, "dim": 12})
    assert changed.stats["from_cache"] is False


def test_hnsw_index_and_id_mapping_in_isolated_process(tmp_path):
    """Keep native FAISS tests independent from the torch OpenMP runtime."""
    code = r'''
import sys
import numpy as np
from movielens.faiss_index import FaissItemIndex

rng = np.random.default_rng(0)
item = rng.normal(size=(100, 4)).astype(np.float32)
item /= np.linalg.norm(item, axis=1, keepdims=True)
rec = FaissItemIndex(
    item, cache_dir=sys.argv[1], fingerprint="test", index_type="hnsw",
    hnsw_m=8, hnsw_ef_construction=20, hnsw_ef_search=32,
)
assert rec.ntotal == 100
scores, ids = rec.search(item[:2], 5)
assert np.array_equal(ids[:, 0], [0, 1])  # nearest item is itself, id == row
first = ids.tolist()
# Reload the persisted IndexIDMap2 and ensure ids still map to item_idx.
rec.build("hnsw")
assert rec.search(item[:2], 5)[1].tolist() == first
print("hnsw integration ok")
'''
    env = {**os.environ, "PYTHONPATH": str(__import__("pathlib").Path(__file__).parents[1])}
    completed = subprocess.run(
        [sys.executable, "-c", code, str(tmp_path)],
        check=False,
        capture_output=True,
        text=True,
        env=env,
        timeout=60,
    )
    assert completed.returncode == 0, completed.stderr
    assert "hnsw integration ok" in completed.stdout


def test_two_tower_is_registered_as_primary_recall():
    assert TwoTowerRecall in RECALL_REGISTRY
    assert TwoTowerRecall.implemented is True
    assert TwoTowerRecall.fallback_only is False
