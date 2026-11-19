"""Popularity baseline.

Non-personalised, but it is the lower bound for every other model, the
cold-start fallback, and the source of popularity-biased hard negatives.
"""

from __future__ import annotations

import numpy as np

from ..schema import Candidate, RecallModel


class PopularityRecall(RecallModel):
    name = "popular"
    implemented = True

    def __init__(self, dataset, decay: float = 0.0):
        train = dataset.frame("train")
        counts = np.bincount(
            train["item_idx"].to_numpy(), minlength=dataset.n_items
        ).astype(float)
        if decay > 0:
            ts = train["timestamp"].to_numpy(dtype=float)
            newest = ts.max()
            age_days = (newest - ts) / 86400.0
            counts = np.bincount(
                train["item_idx"].to_numpy(),
                weights=np.exp(-decay * age_days),
                minlength=dataset.n_items,
            ).astype(float)
        self.scores = counts / max(counts.max(), 1.0)
        self.ranking = np.argsort(-self.scores, kind="stable")

    def recall(self, user_idx: int, k: int, exclude: set[int]) -> list[Candidate]:
        out: list[Candidate] = []
        for item_idx in self.ranking:
            if int(item_idx) in exclude:
                continue
            out.append(
                Candidate(
                    item_idx=int(item_idx),
                    score=float(self.scores[item_idx]),
                    source=self.name,
                )
            )
            if len(out) >= k:
                break
        return out
