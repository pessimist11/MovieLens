"""Offline metrics.

Definitions (kept explicit because these are exactly what interviewers ask):
  * Recall@K   = |TopK ∩ relevant| / |relevant|, binary relevance by default;
  * NDCG@K     = DCG / IDCG, gain = 2^rel - 1 when a graded ``gains`` dict is
                 supplied, otherwise binary gain 1 for a hit;
  * Coverage   = |∪ TopK lists| / |all catalog items|;
  * Long-tail coverage = |∪ TopK ∩ tail| / |tail|, where ``tail`` is the set of
                 items whose TRAIN interaction count is at or below the
                 configured tail threshold (see configs/default.yaml);
  * Long-tail share = share of recommendation slots occupied by tail items;
  * Novelty    = mean -log2(item popularity share), higher = less obvious;
  * ILD        = mean pairwise (1 - genre cosine) inside each TopK list.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np

from ..schema import Metrics


def recall_at_k(recommended: list[int], relevant: set[int], k: int = 10) -> float:
    if not relevant:
        return 0.0
    hits = len(set(recommended[:k]) & relevant)
    return hits / len(relevant)


def ndcg_at_k(
    recommended: list[int],
    relevant: set[int],
    k: int = 10,
    gains: dict[int, float] | None = None,
) -> float:
    if not relevant:
        return 0.0

    def gain(item: int) -> float:
        if gains is None:
            return 1.0 if item in relevant else 0.0
        rel = float(gains.get(item, 0.0))
        return (2.0**rel - 1.0) if rel > 0 else 0.0

    discounts = np.log2(np.arange(2, k + 2))
    gains_in_order = np.array([gain(i) for i in recommended[:k]], dtype=float)
    dcg = float(np.sum(gains_in_order / discounts[: len(gains_in_order)]))
    ideal = sorted((gain(i) for i in relevant), reverse=True)[:k]
    idcg = float(np.sum(np.array(ideal, dtype=float) / discounts[: len(ideal)]))
    return dcg / idcg if idcg > 0 else 0.0


def coverage(all_recommendations: Iterable[list[int]], n_items: int) -> float:
    unique = {i for rec in all_recommendations for i in rec}
    return len(unique) / n_items if n_items else 0.0


def long_tail_coverage(
    all_recommendations: Iterable[list[int]], tail_items: set[int]
) -> float:
    unique = {i for rec in all_recommendations for i in rec}
    return len(unique & tail_items) / len(tail_items) if tail_items else 0.0


def novelty(all_recommendations: Iterable[list[int]], item_popularity: np.ndarray) -> float:
    total = float(item_popularity.sum())
    if total <= 0:
        return 0.0
    probs = (item_popularity + 1.0) / (total + len(item_popularity))
    values = [-np.log2(float(probs[i])) for rec in all_recommendations for i in rec]
    return float(np.mean(values)) if values else 0.0


def intra_list_diversity(
    all_recommendations: Iterable[list[int]], genre_matrix: np.ndarray
) -> float:
    per_user = []
    for rec in all_recommendations:
        if len(rec) < 2:
            continue
        vectors = genre_matrix[np.asarray(rec, dtype=int)]
        sim = vectors @ vectors.T
        n = len(rec)
        upper = sim[np.triu_indices(n, k=1)]
        per_user.append(float(np.mean(1.0 - upper)))
    return float(np.mean(per_user)) if per_user else 0.0


def evaluate_all(
    recommendations: dict[int, list[int]],
    ground_truth: dict[int, set[int]],
    *,
    n_items: int,
    tail_items: set[int],
    item_popularity: np.ndarray,
    genre_matrix: np.ndarray,
    k: int = 10,
) -> Metrics:
    """Evaluate every user in ``ground_truth``; users without a list score 0."""
    lists = {u: recommendations.get(u, []) for u in ground_truth}
    all_lists = list(lists.values())
    return Metrics(
        recall_at_k=float(
            np.mean([recall_at_k(lists[u], truth, k) for u, truth in ground_truth.items()])
        ),
        ndcg_at_k=float(
            np.mean([ndcg_at_k(lists[u], truth, k) for u, truth in ground_truth.items()])
        ),
        coverage=coverage(all_lists, n_items),
        long_tail_coverage=long_tail_coverage(all_lists, tail_items),
        long_tail_share=(
            sum(1 for rec in all_lists for i in rec if i in tail_items)
            / max(1, sum(len(rec) for rec in all_lists))
        ),
        novelty=novelty(all_lists, item_popularity),
        ild=intra_list_diversity(all_lists, genre_matrix),
        n_users=len(ground_truth),
        k=k,
    )
