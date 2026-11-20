"""Item-based collaborative filtering recall.

    R   = user x item implicit matrix built from the TRAIN split only
    C   = R.T @ R                      (item-item co-occurrence)
    sim = C_ij / (|N(i)|^alpha * |N(j)|^alpha + shrink)
    s(u, i) = sum_{j in history(u)} sim(j, i) * w(u, j)

Design notes, i.e. the things an interviewer will push on:
  * ``alpha = 0.5`` is classic cosine ItemCF; ``alpha = 0`` degenerates to raw
    co-occurrence (blockbusters dominate), ``alpha = 1`` penalises popularity
    hard. ``shrink`` damps pairs supported by very few co-occurrences.
  * IUF (inverse user frequency) down-weights very active users inside ``R`` so
    that a handful of 1000-rating accounts do not create fake co-occurrence.
  * Only the top-N neighbours per item are kept: a full 9742 x 9742 similarity
    matrix is both useless (long tail of noise) and needlessly large.
  * Scores come from the user's TRAIN history only. Using valid/test items as
    the profile would leak the answer into the candidate generator.
  * Cold start: no history -> no candidates, the popularity source covers it.

Output contract: ``list[Candidate]`` with ``source="itemcf"``.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import scipy.sparse as sp

from ..config import load_config
from ..schema import Candidate, RecallModel

_EPS = 1e-12


def _topk_per_row(matrix: sp.csr_matrix, k: int) -> sp.csr_matrix:
    """Keep the k largest entries per row of a CSR matrix, sorted descending."""
    matrix = matrix.tocsr()
    if k <= 0:
        return matrix
    indptr, indices, data = matrix.indptr, matrix.indices, matrix.data
    out_indices: list[np.ndarray] = []
    out_data: list[np.ndarray] = []
    out_indptr = [0]
    for row in range(matrix.shape[0]):
        start, end = indptr[row], indptr[row + 1]
        n = end - start
        if n == 0:
            out_indptr.append(out_indptr[-1])
            continue
        cols, vals = indices[start:end], data[start:end]
        if n > k:
            top = np.argpartition(vals, -k)[-k:]
            top = top[np.argsort(-vals[top], kind="stable")]
            cols, vals = cols[top], vals[top]
        out_indices.append(np.asarray(cols, dtype=np.int32))
        out_data.append(np.asarray(vals, dtype=np.float32))
        out_indptr.append(out_indptr[-1] + len(cols))
    if not out_indices:
        return sp.csr_matrix(matrix.shape, dtype=np.float32)
    return sp.csr_matrix(
        (
            np.concatenate(out_data),
            np.concatenate(out_indices),
            np.asarray(out_indptr, dtype=np.int64),
        ),
        shape=matrix.shape,
    )


class ItemCFRecall(RecallModel):
    name = "itemcf"
    implemented = True

    def __init__(
        self,
        dataset,
        *,
        top_n_neighbours: int | None = None,
        alpha: float | None = None,
        shrink: float | None = None,
        iuf: bool | None = None,
        min_cooccurrence: int | None = None,
        history_size: int | None = None,
        history_decay: float | None = None,
        use_cache: bool | None = None,
    ):
        # Defaults come from configs/default.yaml -> recall.itemcf; explicit
        # keyword arguments win, which is what the ablation script uses.
        cfg = load_config().raw.get("recall", {}).get("itemcf", {})
        self.dataset = dataset
        self.top_n_neighbours = int(top_n_neighbours or cfg.get("top_n_neighbours", 80))
        self.alpha = float(cfg.get("alpha", 0.5) if alpha is None else alpha)
        self.shrink = float(cfg.get("shrink", 0.0) if shrink is None else shrink)
        self.iuf = bool(cfg.get("iuf", True) if iuf is None else iuf)
        self.min_cooccurrence = int(
            cfg.get("min_cooccurrence", 1) if min_cooccurrence is None else min_cooccurrence
        )
        self.history_size = int(cfg.get("history_size", 100) if history_size is None else history_size)
        self.history_decay = float(
            cfg.get("history_decay", 0.0) if history_decay is None else history_decay
        )
        self.use_cache = bool(cfg.get("cache", True) if use_cache is None else use_cache)

        self.sim: sp.csr_matrix = sp.csr_matrix((dataset.n_items, dataset.n_items), dtype=np.float32)
        self.stats: dict = {}
        self._history: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        self._build_history()
        self._load_or_fit()

    # ------------------------------------------------------------------ setup
    def _params(self) -> dict:
        return {
            "top_n_neighbours": self.top_n_neighbours,
            "alpha": self.alpha,
            "shrink": self.shrink,
            "iuf": self.iuf,
            "min_cooccurrence": self.min_cooccurrence,
            "history_size": self.history_size,
            "history_decay": self.history_decay,
        }

    def _train_frame(self):
        train = self.dataset.frame("train")
        return train[train["is_positive"]] if "is_positive" in train else train

    def _build_history(self) -> None:
        """Per-user TRAIN profile: most recent items + optional time decay."""
        train = self._train_frame().sort_values("timestamp")
        decay = self.history_decay
        for user_idx, group in train.groupby("user_idx", sort=False):
            items = group["item_idx"].to_numpy(dtype=np.int32)
            if self.history_size > 0 and len(items) > self.history_size:
                items = items[-self.history_size :]
                ts = group["timestamp"].to_numpy(dtype=float)[-self.history_size :]
            else:
                ts = group["timestamp"].to_numpy(dtype=float)
            weights = np.ones(len(items), dtype=np.float32)
            if decay > 0 and len(items):
                age_days = (ts.max() - ts) / 86400.0
                weights = np.exp(-decay * age_days).astype(np.float32)
            self._history[int(user_idx)] = (items, weights)

    def _fingerprint(self) -> str:
        train = self._train_frame()
        digest = hashlib.blake2b(digest_size=16)
        digest.update(train["user_idx"].to_numpy(dtype=np.int64).tobytes())
        digest.update(train["item_idx"].to_numpy(dtype=np.int64).tobytes())
        digest.update(json.dumps(self._params(), sort_keys=True).encode())
        return digest.hexdigest()

    def _cache_paths(self) -> tuple[Path, Path]:
        base = load_config().artifacts_dir() / "models" / "itemcf"
        fp = self._fingerprint()
        return base / f"sim_{fp}.npz", base / f"sim_{fp}.json"

    def _load_or_fit(self) -> None:
        if self.use_cache:
            sim_path, meta_path = self._cache_paths()
            if sim_path.exists():
                sim = sp.load_npz(sim_path).tocsr().astype(np.float32)
                if sim.shape == (self.dataset.n_items, self.dataset.n_items):
                    self.sim = sim
                    self.stats = (
                        json.loads(meta_path.read_text(encoding="utf-8"))
                        if meta_path.exists()
                        else {}
                    )
                    self.stats["from_cache"] = True
                    return
        self.fit()
        if self.use_cache:
            sim_path, meta_path = self._cache_paths()
            sim_path.parent.mkdir(parents=True, exist_ok=True)
            sp.save_npz(sim_path, self.sim.tocsr())
            meta_path.write_text(
                json.dumps(self.stats, indent=2, ensure_ascii=False), encoding="utf-8"
            )

    # -------------------------------------------------------------------- fit
    def fit(self, dataset=None) -> "ItemCFRecall":
        t0 = perf_counter()
        if dataset is not None:
            self.dataset = dataset
            self._build_history()
        train = self._train_frame()
        n_users, n_items = self.dataset.n_users, self.dataset.n_items
        users = train["user_idx"].to_numpy(dtype=np.int64)
        items = train["item_idx"].to_numpy(dtype=np.int64)

        weights = np.ones(len(train), dtype=np.float32)
        if self.iuf:
            per_user = np.bincount(users, minlength=n_users).astype(np.float32)
            weights = 1.0 / np.log1p(per_user[users])

        r = sp.csr_matrix(
            (weights, (users, items)), shape=(n_users, n_items), dtype=np.float32
        )
        binary = sp.csr_matrix(
            (np.ones(len(train), dtype=np.float32), (users, items)),
            shape=(n_users, n_items),
            dtype=np.float32,
        )
        cooc = (r.T @ r).tocsr()
        # Support filtering must use the RAW co-occurrence count: after IUF the
        # weights are < 1, so thresholding the weighted value would silently
        # delete most real pairs.
        if self.min_cooccurrence > 1:
            support = (binary.T @ binary).tocsr()
            cooc = cooc.multiply(support >= self.min_cooccurrence).tocsr()
        cooc.setdiag(0)
        cooc.eliminate_zeros()
        cooc = cooc.tocoo()
        popularity = np.asarray(self.dataset.item_popularity, dtype=np.float64)

        rows, cols, vals = cooc.row, cooc.col, cooc.data.astype(np.float64)
        denom = popularity[rows] ** self.alpha * popularity[cols] ** self.alpha + self.shrink
        sim = sp.csr_matrix(
            (vals / np.maximum(denom, _EPS), (rows, cols)),
            shape=(n_items, n_items),
            dtype=np.float32,
        )
        sim = _topk_per_row(sim, self.top_n_neighbours)
        self.sim = sim

        degrees = np.diff(sim.indptr)
        self.stats = {
            **self._params(),
            "n_items": int(n_items),
            "n_train_interactions": int(len(train)),
            "cooccurrence_nnz": int(len(vals)),
            "sim_nnz": int(sim.nnz),
            "mean_neighbours": round(float(degrees.mean()), 2) if len(degrees) else 0.0,
            "items_without_neighbours": int((degrees == 0).sum()),
            "fit_seconds": round(perf_counter() - t0, 2),
            "from_cache": False,
        }
        return self

    # ----------------------------------------------------------------- recall
    def _score(self, user_idx: int) -> np.ndarray | None:
        profile = self._history.get(int(user_idx))
        if not profile:
            return None
        items, weights = profile
        if len(items) == 0 or self.sim.nnz == 0:
            return None
        row = sp.csr_matrix(
            (weights, (np.zeros(len(items), dtype=np.int64), items)),
            shape=(1, self.dataset.n_items),
            dtype=np.float32,
        )
        return np.asarray((row @ self.sim).todense()).ravel()

    def recall(self, user_idx: int, k: int, exclude: set[int]) -> list[Candidate]:
        scores = self._score(user_idx)
        if scores is None:
            return []
        blocked = set(exclude)
        profile = self._history.get(int(user_idx))
        if profile:
            blocked.update(int(i) for i in profile[0])
        if blocked:
            scores[np.asarray(sorted(blocked), dtype=np.int64)] = -np.inf
        positive = np.flatnonzero(scores > 0)
        if positive.size == 0:
            return []
        k = min(k, positive.size)
        top = positive[np.argpartition(scores[positive], -k)[-k:]]
        top = top[np.argsort(-scores[top], kind="stable")]
        best = float(scores[top[0]])
        return [
            Candidate(
                item_idx=int(i),
                score=float(scores[i] / best) if best > _EPS else float(scores[i]),
                source=self.name,
            )
            for i in top
        ]

    def explain(self, user_idx: int, item_idx: int) -> tuple[int, float] | None:
        """Return the history item that contributed most to this recommendation."""
        profile = self._history.get(int(user_idx))
        if not profile:
            return None
        items, weights = profile
        column = self.sim[:, int(item_idx)].toarray().ravel()[items]
        if not len(column) or float(column.max()) <= 0:
            return None
        contributions = column * weights
        best = int(np.argmax(contributions))
        return int(items[best]), float(contributions[best])
