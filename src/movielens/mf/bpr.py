"""BPR-MF - stage 3 placeholder.

Implementation plan:
  * pairwise ranking loss ``-log sigmoid(x_ui - x_uj) + lambda * ||theta||^2``
    over ``samples.parquet`` (user_idx, pos_item_idx, neg_item_idx, is_hard);
  * user / item bias terms optional; L2 regularisation and early stopping on
    the VALID split (never on test);
  * expose ``user_embedding`` / ``item_embedding`` so the two-tower model can
    be initialised from MF, and ``score(user_idx, item_idx)`` so the ranker can
    use the MF score as a feature;
  * keep a biased-SVD / implicit-ALS reference implementation for comparison
    (pointwise vs pairwise, explicit ratings vs implicit feedback).
"""

from __future__ import annotations


class BPRMF:
    implemented = False

    def __init__(self, dataset, dim: int = 64, lr: float = 0.01, reg: float = 1e-4):
        self.dataset = dataset
        self.dim = dim

    def fit(self, epochs: int = 20) -> "BPRMF":
        raise NotImplementedError("BPR-MF is scheduled for stage 3")

    def score(self, user_idx: int, item_idx: int) -> float:
        raise NotImplementedError("BPR-MF is scheduled for stage 3")
