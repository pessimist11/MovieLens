"""BPR-MF: hand-computed loss, legal negatives, learning, filtering, caching."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from movielens.mf.bpr import BPRMF, BPRRecall, _BPRNet
from movielens.recall import RECALL_REGISTRY
from tests.test_itemcf import make_dataset


@pytest.fixture
def toy():
    rows = [
        (0, 0, 100, "train"), (0, 1, 200, "train"),
        (1, 0, 100, "train"), (1, 1, 200, "train"),
        (2, 0, 100, "train"), (2, 2, 200, "train"),
    ]
    return make_dataset(rows, n_users=4, n_items=6)


@pytest.fixture
def bigger():
    rows = [
        (u, ((u * 3 + j) % 20), 1_000_000_000 + u * 100 + j, "train")
        for u in range(8)
        for j in range(6)
    ]
    return make_dataset(rows, n_users=8, n_items=20)


def test_bpr_loss_matches_hand_computed():
    net = _BPRNet(n_users=2, n_items=3, dim=2)
    with torch.no_grad():
        net.user.weight[:] = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        net.item.weight[:] = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    u = torch.tensor([0, 1])
    i = torch.tensor([0, 1])     # user0->item0=[1,0], user1->item1=[0,1]
    j = torch.tensor([1, 0])     # the orthogonal item as the negative
    x_ui, x_uj, *_ = net(u, i, j)
    # both users: x_ui = 1, x_uj = 0 -> margin = 1
    assert x_ui.tolist() == pytest.approx([1.0, 1.0])
    assert x_uj.tolist() == pytest.approx([0.0, 0.0])
    loss = -torch.nn.functional.logsigmoid(x_ui - x_uj).mean()
    expected = -np.log(1.0 / (1.0 + np.exp(-1.0)))
    assert float(loss) == pytest.approx(expected, rel=1e-6)


def test_negative_sampling_is_legal(toy):
    model = BPRMF(toy, epochs=0, use_cache=False)
    rng = np.random.default_rng(0)
    users, pos, neg = model.sample_pairs(rng)
    assert len(users) > 0
    assert np.all(pos != neg)
    assert not model._seen_bool[users, neg].any()      # never a seen item
    assert set(np.unique(neg)) <= set(model._train_items.tolist())
    assert len(users) <= len(toy.frame("train")) * model.negatives_per_positive


def test_sampling_mixes_uniform_and_hard_negatives():
    """Item 0 is popular but only seen by half the users, so it stays a legal
    negative for the other half and must be over-sampled by the hard policy."""
    rows = []
    for u in range(100):
        items = ([0] if u < 50 else []) + [((u * 4 + j) % 200) + 1 for j in range(4)]
        rows += [(u, i, 100 + j, "train") for j, i in enumerate(items)]
    ds = make_dataset(rows, n_users=100, n_items=201)
    hard_model = BPRMF(ds, epochs=0, use_cache=False, hard_negative_ratio=1.0)
    _, _, hard = hard_model.sample_pairs(np.random.default_rng(1))
    uniform_model = BPRMF(ds, epochs=0, use_cache=False, hard_negative_ratio=0.0)
    _, _, uniform = uniform_model.sample_pairs(np.random.default_rng(1))
    hard_freq = float((hard == 0).mean())
    uniform_freq = float((uniform == 0).mean())
    assert hard_freq > 3 * uniform_freq
    # users 0..49 have item 0 in their history, so it can never be their negative
    users, _, neg = hard_model.sample_pairs(np.random.default_rng(2))
    assert not ((neg == 0) & (users < 50)).any()


def test_training_reduces_loss(bigger):
    model = BPRMF(bigger, epochs=12, eval_every=6, dim=8, use_cache=False)
    losses = [h["train_loss"] for h in model.history]
    assert len(losses) == 12
    assert losses[-1] < losses[0]
    assert model.stats["best_valid_recall@10"] >= 0.0
    assert model.user_factors.shape == (8, 8)
    assert model.item_factors.shape == (20, 8)


def test_early_stopping_restores_best_epoch(bigger):
    model = BPRMF(bigger, epochs=40, eval_every=2, patience=1, dim=8, use_cache=False)
    assert model.stats["epochs_ran"] <= 40
    assert model.stats["best_epoch"] is not None
    evaluated = [h for h in model.history if "valid_recall@10" in h]
    best = max(h["valid_recall@10"] for h in evaluated)
    assert model.stats["best_valid_recall@10"] == pytest.approx(best)


def test_recall_filters_seen_and_handles_cold_start(bigger):
    rec = BPRRecall(bigger, epochs=6, eval_every=3, dim=8, use_cache=False)
    candidates = rec.recall(0, k=5, exclude={7})
    items = [c.item_idx for c in candidates]
    assert len(items) == 5
    assert 7 not in items
    history = set(bigger.seen["valid"][0])
    assert not (set(items) & history)
    assert all(c.source == "bpr" for c in candidates)
    scores = [c.score for c in candidates]
    assert scores == sorted(scores, reverse=True)
    assert rec.recall(999, k=5, exclude=set()) == []


def test_score_and_embedding_exports(toy):
    model = BPRMF(toy, epochs=0, use_cache=False, dim=4)
    assert model.user_embedding().shape == (4, 4)
    assert model.item_embedding().shape == (6, 4)
    assert model.score(0, 1) == pytest.approx(
        float(model.user_factors[0] @ model.item_factors[1]), rel=1e-6
    )


def test_factor_cache_roundtrip(bigger):
    kwargs = dict(epochs=4, eval_every=2, dim=8, use_cache=True)
    first = BPRMF(bigger, **kwargs)
    assert first.stats["from_cache"] is False
    second = BPRMF(bigger, **kwargs)
    assert second.stats["from_cache"] is True
    assert np.allclose(first.user_factors, second.user_factors)
    assert np.allclose(first.item_factors, second.item_factors)
    third = BPRMF(bigger, **{**kwargs, "dim": 16})
    assert third.stats["from_cache"] is False


def test_bpr_is_registered_as_recall_source():
    assert BPRRecall in RECALL_REGISTRY
    assert BPRRecall.implemented is True
    assert BPRRecall.fallback_only is False
