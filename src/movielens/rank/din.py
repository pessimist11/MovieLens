"""DIN ranker - stage 5 placeholder (preferred option).

Implementation plan:
  * build a time-ordered behaviour sequence per user from TRAIN interactions;
  * activation unit: the candidate item attends over the history sequence, so
    the user representation changes with the candidate (unlike a fixed tower
    vector); add Dice / PReLU and mini-batch-aware regularisation as needed;
  * features: id / genre / popularity / user stats + upstream recall scores;
  * guard against sample selection bias: the ranker only ever scores recall
    candidates, so its training samples must be drawn from the same candidate
    distribution, not from uniformly sampled negatives.

Output contract: returns the same candidates with ``rank_score`` attached.
"""

from __future__ import annotations

from ..schema import Candidate, RankModel


class DINRanker(RankModel):
    name = "din"
    implemented = False

    def __init__(self, dataset, dim: int = 16, max_history: int = 50):
        self.dataset = dataset
        self.dim = dim
        self.max_history = max_history

    def predict(self, user_idx: int, candidates: list[Candidate]) -> list[Candidate]:
        raise NotImplementedError("DIN is scheduled for stage 5")
