"""Time-sliced pointwise samples from the serving recall pipeline."""

from dataclasses import dataclass, replace

import numpy as np

from ..data.prepare import _seen_map, _truth_map
from ..recall import BPRRecall, ItemCFRecall, PopularityRecall, TwoTowerRecall
from ..schema import Candidate


@dataclass
class CandidateGroup:
    user: int
    candidates: list[Candidate]
    truth: set[int]
    weights: np.ndarray


def make_snapshot(dataset, cfg):
    """Both the upstream fit and its seen sets stop strictly before the cutoff."""
    from ..pipeline import Recommender

    train = dataset.frame("train")
    fraction = cfg.rank.snapshot_fraction
    if not 0 < fraction < 1:
        raise ValueError("rank.snapshot_fraction must be between 0 and 1")
    cutoff = float(np.quantile(train["timestamp"], fraction))
    past = train[train["timestamp"] < cutoff]
    labels = train[train["timestamp"] >= cutoff]
    if past.empty or labels.empty or not past["is_positive"].any():
        raise ValueError("Ranking needs training interactions on both sides of the snapshot cutoff")

    seen = _seen_map(past)
    snapshot = replace(
        dataset, interactions=past, seen={"valid": seen}, ground_truth={"valid": {}}
    )
    # Fixed training budgets keep the ranking label window out of recall's early stopping.
    bpr_epochs = cfg.rank.snapshot_bpr_epochs
    tower_epochs = cfg.rank.snapshot_two_tower_epochs
    recall = [
        PopularityRecall(snapshot),
        ItemCFRecall(snapshot),
        BPRRecall(snapshot, epochs=bpr_epochs, eval_every=bpr_epochs),
        TwoTowerRecall(snapshot, epochs=tower_epochs, eval_every=tower_epochs),
    ]
    no_rank = replace(cfg, rank=replace(cfg.rank, model="none"))
    rec = Recommender(snapshot, no_rank, recall_models=recall)
    # Use films known from the prefix as the catalogue for this historical request.
    unavailable = set(range(dataset.n_items)) - set(past["item_idx"])
    return rec, _truth_map(labels), unavailable, cutoff


def candidate_groups(rec, truth, seen, *, unavailable=frozenset(), negative_limit=None):
    """Retain all retrieved positives and optionally subsample retrieved negatives.

    Negative weights recover the original candidate frequency in the loss.
    """
    rng = np.random.default_rng(rec.cfg.project.seed)
    groups = []
    for user, relevant in truth.items():
        candidates = rec.recall(user, seen.get(user, set()) | unavailable)
        if not candidates:
            continue
        weights = np.ones(len(candidates), np.float32)
        if negative_limit is not None:
            neg = np.flatnonzero([c.item_idx not in relevant for c in candidates])
            if len(neg) > negative_limit:
                keep = rng.choice(neg, negative_limit, replace=False)
                weights[neg] = 0
                weights[keep] = len(neg) / negative_limit
        groups.append(CandidateGroup(
            user, [c for c, w in zip(candidates, weights) if w], relevant, weights[weights > 0]
        ))
    return groups
