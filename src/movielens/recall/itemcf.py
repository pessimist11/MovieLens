"""ItemCF recall - stage 2 placeholder.

Implementation plan (kept as a contract so the pipeline already wires it in):
  * co-occurrence ``C = R.T @ R`` on the TRAIN split only, as a scipy CSR matrix;
  * similarity ``sim(i,j) = C_ij / sqrt(|N(i)| * |N(j)|)`` with an IUF / popularity
    penalty so blockbusters do not dominate every neighbour list;
  * keep only the top-N neighbours per item, optionally time-weight the
    co-occurrence pairs;
  * score a user's candidates by ``sum_j sim(i,j) * w(u,j)`` over their history,
    then apply ``exclude`` before returning ``list[Candidate]``.

Output contract: ``list[Candidate]`` with ``source="itemcf"``.
"""

from __future__ import annotations

from ..schema import Candidate, RecallModel


class ItemCFRecall(RecallModel):
    name = "itemcf"
    implemented = False

    def __init__(self, dataset, top_n_neighbours: int = 50):
        self.dataset = dataset
        self.top_n_neighbours = top_n_neighbours

    def fit(self, dataset=None) -> "ItemCFRecall":
        raise NotImplementedError("ItemCF is scheduled for stage 2 (baseline models)")

    def recall(self, user_idx: int, k: int, exclude: set[int]) -> list[Candidate]:
        raise NotImplementedError("ItemCF is scheduled for stage 2 (baseline models)")
