"""MMR reranking: trade a little relevance for genre diversity.

    MMR = argmax_i [ lambda * rel(i) - (1 - lambda) * max_{j in S} sim(i, j) ]

``sim`` is the cosine similarity between L2-normalised genre multi-hot vectors,
so no full item-item matrix is materialised (9.7k x 9.7k would be ~380 MB).
"""

from __future__ import annotations

import numpy as np

from ..schema import Candidate, Reranker


class MMRReranker(Reranker):
    name = "mmr"
    implemented = True

    def __init__(self, dataset, lam: float = 0.7):
        if not 0.0 <= lam <= 1.0:
            raise ValueError("lambda must be in [0, 1]")
        self.lam = lam
        self.genre_matrix = dataset.genre_matrix

    @staticmethod
    def _normalise(values: np.ndarray) -> np.ndarray:
        lo, hi = float(values.min()), float(values.max())
        if hi - lo < 1e-12:
            return np.full_like(values, 0.5)
        return (values - lo) / (hi - lo)

    def rerank(self, candidates: list[Candidate], k: int) -> list[Candidate]:
        if not candidates:
            return []
        k = min(k, len(candidates))
        raw = np.array(
            [
                c.rank_score if c.rank_score is not None else c.score
                for c in candidates
            ],
            dtype=float,
        )
        rel = self._normalise(raw)
        item_idx = np.array([c.item_idx for c in candidates], dtype=int)
        vectors = self.genre_matrix[item_idx]

        chosen = [int(np.argmax(rel))]
        remaining = [i for i in range(len(candidates)) if i != chosen[0]]
        while len(chosen) < k and remaining:
            sims = vectors[remaining] @ vectors[chosen].T
            max_sim = sims.max(axis=1)
            scores = self.lam * rel[remaining] - (1.0 - self.lam) * max_sim
            pick = remaining[int(np.argmax(scores))]
            chosen.append(pick)
            remaining.remove(pick)
        return [candidates[i] for i in chosen]
