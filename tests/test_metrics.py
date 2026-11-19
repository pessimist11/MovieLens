import numpy as np
from types import SimpleNamespace

from movielens.eval.metrics import (
    coverage,
    evaluate_all,
    long_tail_coverage,
    ndcg_at_k,
    recall_at_k,
)
from movielens.rerank.mmr import MMRReranker
from movielens.schema import Candidate


def test_recall_and_ndcg():
    assert recall_at_k([1, 2, 3], {2, 5}, k=3) == 0.5
    assert ndcg_at_k([2, 1, 3], {2}, k=3) == 1.0
    assert ndcg_at_k([1, 2, 3], {2}, k=3) == 1.0 / np.log2(3)
    assert ndcg_at_k([1, 2, 3], set(), k=3) == 0.0


def test_coverage_and_long_tail():
    recs = [[1, 2], [2, 3]]
    assert coverage(recs, 4) == 0.75
    assert long_tail_coverage(recs, {1, 3, 4}) == 2 / 3
    assert long_tail_coverage(recs, set()) == 0.0


def test_evaluate_all_shapes():
    m = evaluate_all(
        {0: [1, 2], 1: [3]},
        {0: {2}, 1: {5}},
        n_items=6,
        tail_items={1, 3},
        item_popularity=np.array([10, 1, 1, 1, 1, 1]),
        genre_matrix=np.eye(6, dtype=np.float32),
        k=2,
    )
    assert m.k == 2 and m.n_users == 2
    assert 0.0 <= m.recall_at_k <= 1.0
    assert 0.0 <= m.ndcg_at_k <= 1.0
    assert set(m.as_dict()) >= {"recall@k", "ndcg@k", "coverage", "long_tail_coverage"}


def test_mmr_trades_relevance_for_diversity():
    ds = SimpleNamespace(
        genre_matrix=np.array([[1, 0], [1, 0], [0, 1]], dtype=np.float32)
    )
    candidates = [
        Candidate(0, 0.9, "popular"),
        Candidate(1, 0.8, "popular"),
        Candidate(2, 0.1, "popular"),
    ]
    diverse = [c.item_idx for c in MMRReranker(ds, lam=0.0).rerank(candidates, 3)]
    relevant = [c.item_idx for c in MMRReranker(ds, lam=1.0).rerank(candidates, 3)]
    assert diverse == [0, 2, 1]  # the off-genre item is promoted
    assert relevant == [0, 1, 2]  # pure relevance order
