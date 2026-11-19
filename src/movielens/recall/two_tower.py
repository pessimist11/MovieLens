"""Two-tower recall + FAISS - stage 4 placeholder.

Implementation plan:
  * user tower: id embedding + pooled (mean / attention) history item embeddings
    + demographic features; item tower: id embedding + genre multi-hot + year;
  * L2-normalise both towers, dot product / cosine as the matching score;
  * train with in-batch negatives plus popularity-sampled and ItemCF-derived
    hard negatives, InfoNCE with temperature tau; apply a false-negative mask
    so a positive appearing elsewhere in the batch is never treated as negative;
  * export all item vectors to ``IndexFlatIP`` (exact) and ``IndexHNSWFlat``
    (approximate), persist the index + its item_idx -> faiss id mapping;
  * serve user vectors either precomputed offline (T+1) or computed online.

Output contract: ``list[Candidate]`` with ``source="two_tower"``.
"""

from __future__ import annotations

from ..schema import Candidate, RecallModel


class TwoTowerRecall(RecallModel):
    name = "two_tower"
    implemented = False

    def __init__(self, dataset, dim: int = 64):
        self.dataset = dataset
        self.dim = dim

    def build_index(self, index_type: str = "flat") -> None:
        raise NotImplementedError("Two-tower + FAISS is scheduled for stage 4")

    def save(self, path: str) -> None:
        raise NotImplementedError("Two-tower + FAISS is scheduled for stage 4")

    def recall(self, user_idx: int, k: int, exclude: set[int]) -> list[Candidate]:
        raise NotImplementedError("Two-tower + FAISS is scheduled for stage 4")
