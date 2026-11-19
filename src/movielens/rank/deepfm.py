"""DeepFM ranker - stage 5 placeholder (alternative to DIN).

Implementation plan:
  * shared embeddings feeding both an FM second-order interaction term and a
    DNN, so low-order and high-order feature crosses are learned jointly;
  * fields: user_idx, item_idx, genre multi-hot, user activity / mean rating,
    item popularity, time-decayed popularity, and the three upstream scores
    (ItemCF, BPR-MF, two-tower) as stacked features;
  * train on pointwise samples built from TRAIN-window candidates only, monitor
    AUC / LogLoss on VALID, and keep the full-pipeline Recall@K / NDCG@K as the
    real objective.
"""

from __future__ import annotations

from ..schema import Candidate, RankModel


class DeepFMRanker(RankModel):
    name = "deepfm"
    implemented = False

    def __init__(self, dataset, dim: int = 16):
        self.dataset = dataset
        self.dim = dim

    def predict(self, user_idx: int, candidates: list[Candidate]) -> list[Candidate]:
        raise NotImplementedError("DeepFM is scheduled for stage 5")
