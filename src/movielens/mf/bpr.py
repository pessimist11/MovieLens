"""BPR-MF: matrix factorisation trained with the Bayesian pairwise ranking loss.

    x_ui   = p_u . q_i
    L      = -ln sigmoid(x_ui - x_uj) + lambda * (||p_u||^2 + ||q_i||^2 + ||q_j||^2)

Why this layer exists after ItemCF:
  * ItemCF memorises explicit item-item co-occurrence, so items with few
    interactions have no neighbours at all. MF embeds every user and item into
    a shared latent space, which generalises to pairs never seen together.
  * BPR optimises the ORDERING (positive must outrank a sampled negative)
    instead of reconstructing ratings, which matches the Top-K objective.

Implementation notes:
  * Negatives are re-sampled every epoch (uniform + popularity-biased hard
    negatives), not frozen once: a fixed pair list overfits to those specific
    negatives within a few epochs. Sampling is vectorised with a boolean seen
    matrix, and negatives only come from items present in the train window.
  * Early stopping uses Recall@10 on the VALID split via one dense score-matrix
    product per evaluation; the test split is never touched during training.
  * ``user_factors`` / ``item_factors`` are exported so the two-tower model can
    be initialised from MF, and ``score()`` is available as a ranker feature.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from ..config import load_config
from ..eval.metrics import evaluate_score_matrix
from ..schema import Candidate, RecallModel


class _BPRNet(nn.Module):
    def __init__(self, n_users: int, n_items: int, dim: int):
        super().__init__()
        self.user = nn.Embedding(n_users, dim)
        self.item = nn.Embedding(n_items, dim)
        nn.init.normal_(self.user.weight, mean=0.0, std=0.01)
        nn.init.normal_(self.item.weight, mean=0.0, std=0.01)

    def forward(self, users, pos_items, neg_items):
        p_u = self.user(users)
        q_i = self.item(pos_items)
        q_j = self.item(neg_items)
        x_ui = (p_u * q_i).sum(dim=-1)
        x_uj = (p_u * q_j).sum(dim=-1)
        return x_ui, x_uj, p_u, q_i, q_j


class BPRMF:
    implemented = True

    def __init__(
        self,
        dataset,
        *,
        dim: int | None = None,
        lr: float | None = None,
        reg: float | None = None,
        epochs: int | None = None,
        batch_size: int | None = None,
        negatives_per_positive: int | None = None,
        hard_negative_ratio: float | None = None,
        eval_every: int | None = None,
        patience: int | None = None,
        seed: int | None = None,
        use_cache: bool | None = None,
    ):
        base = load_config()
        cfg = base.raw.get("mf", {}).get("bpr", {})
        self.dataset = dataset
        self.dim = int(cfg.get("dim", 64) if dim is None else dim)
        self.lr = float(cfg.get("lr", 0.01) if lr is None else lr)
        self.reg = float(cfg.get("reg", 1e-4) if reg is None else reg)
        self.epochs = int(cfg.get("epochs", 60) if epochs is None else epochs)
        self.batch_size = int(cfg.get("batch_size", 8192) if batch_size is None else batch_size)
        self.negatives_per_positive = int(
            cfg.get("negatives_per_positive", 4)
            if negatives_per_positive is None
            else negatives_per_positive
        )
        self.hard_negative_ratio = float(
            cfg.get("hard_negative_ratio", 0.5)
            if hard_negative_ratio is None
            else hard_negative_ratio
        )
        self.eval_every = int(cfg.get("eval_every", 5) if eval_every is None else eval_every)
        self.patience = int(cfg.get("patience", 4) if patience is None else patience)
        self.seed = int(base.project.seed if seed is None else seed)
        self.use_cache = bool(cfg.get("cache", True) if use_cache is None else use_cache)

        self.user_factors = np.zeros((dataset.n_users, self.dim), dtype=np.float32)
        self.item_factors = np.zeros((dataset.n_items, self.dim), dtype=np.float32)
        self.history: list[dict] = []
        self.stats: dict = {}
        self._prepare()
        self._load_or_fit()

    # ------------------------------------------------------------------- data
    def _params(self) -> dict:
        return {
            "dim": self.dim,
            "lr": self.lr,
            "reg": self.reg,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "negatives_per_positive": self.negatives_per_positive,
            "hard_negative_ratio": self.hard_negative_ratio,
            "eval_every": self.eval_every,
            "patience": self.patience,
            "seed": self.seed,
        }

    def _prepare(self) -> None:
        train = self.dataset.frame("train")
        positives = train[train["is_positive"]] if "is_positive" in train else train
        self._positives = positives[["user_idx", "item_idx"]].to_numpy(dtype=np.int64)
        self._train_items = np.unique(self._positives[:, 1])
        popularity = np.bincount(
            self._positives[:, 1], minlength=self.dataset.n_items
        ).astype(float)
        weights = popularity[self._train_items] ** 0.75
        self._neg_weights = weights / weights.sum()
        self._seen_bool = np.zeros(
            (self.dataset.n_users, self.dataset.n_items), dtype=bool
        )
        self._seen_bool[self._positives[:, 0], self._positives[:, 1]] = True

    def sample_pairs(self, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Vectorised (user, positive, negative) triples with rejection sampling."""
        n_neg = max(1, self.negatives_per_positive)
        users = np.repeat(self._positives[:, 0], n_neg)
        pos = np.repeat(self._positives[:, 1], n_neg)
        is_hard = rng.random(len(users)) < self.hard_negative_ratio
        neg = np.where(
            is_hard,
            rng.choice(self._train_items, size=len(users), p=self._neg_weights),
            rng.choice(self._train_items, size=len(users)),
        )
        for _ in range(12):
            bad = self._seen_bool[users, neg] | (neg == pos)
            if not bad.any():
                break
            neg[bad] = rng.choice(self._train_items, size=int(bad.sum()), p=self._neg_weights)
        good = ~(self._seen_bool[users, neg] | (neg == pos))
        return users[good], pos[good], neg[good]

    # ------------------------------------------------------------------ cache
    def _fingerprint(self) -> str:
        digest = hashlib.blake2b(digest_size=16)
        digest.update(self._positives.tobytes())
        digest.update(json.dumps(self._params(), sort_keys=True).encode())
        return digest.hexdigest()

    def _cache_paths(self) -> tuple[Path, Path]:
        base = load_config().artifacts_dir() / "models" / "bpr"
        fp = self._fingerprint()
        return base / f"factors_{fp}.npz", base / f"factors_{fp}.json"

    def _load_or_fit(self) -> None:
        if self.use_cache:
            npz_path, meta_path = self._cache_paths()
            if npz_path.exists():
                with np.load(npz_path) as data:
                    user, item = data["user"], data["item"]
                if user.shape == (self.dataset.n_users, self.dim) and item.shape == (
                    self.dataset.n_items,
                    self.dim,
                ):
                    self.user_factors, self.item_factors = user, item
                    payload = (
                        json.loads(meta_path.read_text(encoding="utf-8"))
                        if meta_path.exists()
                        else {}
                    )
                    self.history = payload.pop("history", [])
                    self.stats = {**payload, "from_cache": True}
                    return
        self.fit()
        if self.use_cache:
            npz_path, meta_path = self._cache_paths()
            npz_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(npz_path, user=self.user_factors, item=self.item_factors)
            meta_path.write_text(
                json.dumps({**self.stats, "history": self.history}, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )

    # ------------------------------------------------------------------- fit
    def fit(self) -> "BPRMF":
        t0 = perf_counter()
        torch.manual_seed(self.seed)
        rng = np.random.default_rng(self.seed)
        net = _BPRNet(self.dataset.n_users, self.dataset.n_items, self.dim)
        optimiser = torch.optim.Adam(net.parameters(), lr=self.lr)

        best_recall, best_factors, stale = -1.0, None, 0
        for epoch in range(1, self.epochs + 1):
            users, pos, neg = self.sample_pairs(rng)
            order = rng.permutation(len(users))
            users, pos, neg = users[order], pos[order], neg[order]
            net.train()
            total_loss, n_batches = 0.0, 0
            for start in range(0, len(users), self.batch_size):
                u = torch.from_numpy(users[start : start + self.batch_size])
                i = torch.from_numpy(pos[start : start + self.batch_size])
                j = torch.from_numpy(neg[start : start + self.batch_size])
                x_ui, x_uj, p_u, q_i, q_j = net(u, i, j)
                ranking_loss = -F.logsigmoid(x_ui - x_uj).mean()
                penalty = (p_u.pow(2).sum() + q_i.pow(2).sum() + q_j.pow(2).sum()) / len(u)
                loss = ranking_loss + self.reg * penalty
                optimiser.zero_grad()
                loss.backward()
                optimiser.step()
                total_loss += float(loss.detach())
                n_batches += 1

            entry = {"epoch": epoch, "train_loss": round(total_loss / max(n_batches, 1), 6)}
            if epoch % self.eval_every == 0 or epoch == self.epochs:
                self._export(net)
                valid = evaluate_score_matrix(
                    self.user_factors @ self.item_factors.T,
                    self.dataset.ground_truth["valid"],
                    self.dataset.seen["valid"],
                    k=10,
                )
                entry["valid_recall@10"] = round(valid["recall@k"], 6)
                entry["valid_ndcg@10"] = round(valid["ndcg@k"], 6)
                if valid["recall@k"] > best_recall:
                    best_recall = valid["recall@k"]
                    best_factors = (self.user_factors.copy(), self.item_factors.copy())
                    stale = 0
                else:
                    stale += 1
            self.history.append(entry)
            if stale >= self.patience and best_factors is not None:
                break

        if best_factors is not None:
            self.user_factors, self.item_factors = best_factors
        else:
            self._export(net)
        self.stats = {
            **self._params(),
            "epochs_ran": len(self.history),
            "best_epoch": max(
                (h for h in self.history if "valid_recall@10" in h),
                key=lambda h: h["valid_recall@10"],
                default={},
            ).get("epoch"),
            "best_valid_recall@10": round(best_recall, 6),
            "final_train_loss": self.history[-1]["train_loss"] if self.history else None,
            "n_train_triples_per_epoch": int(len(self._positives) * self.negatives_per_positive),
            "fit_seconds": round(perf_counter() - t0, 2),
            "from_cache": False,
        }
        return self

    def _export(self, net: _BPRNet) -> None:
        net.eval()
        with torch.no_grad():
            self.user_factors = net.user.weight.detach().cpu().numpy().astype(np.float32)
            self.item_factors = net.item.weight.detach().cpu().numpy().astype(np.float32)

    # ----------------------------------------------------------------- scores
    def score(self, user_idx: int, item_idx: int) -> float:
        return float(self.user_factors[int(user_idx)] @ self.item_factors[int(item_idx)])

    def user_embedding(self) -> np.ndarray:
        return self.user_factors

    def item_embedding(self) -> np.ndarray:
        return self.item_factors


class BPRRecall(RecallModel):
    """Serve BPR-MF as a recall source: one dense user x item score matrix."""

    name = "bpr"
    implemented = True

    def __init__(self, dataset, **overrides):
        self.dataset = dataset
        self.model = BPRMF(dataset, **overrides)
        self.stats = self.model.stats
        # seen["valid"] is exactly the train-history map produced by prepare.py
        self._train_users = set(dataset.seen.get("valid", {}))
        self._scores: np.ndarray | None = None

    @property
    def scores(self) -> np.ndarray:
        if self._scores is None:
            self._scores = self.model.user_factors @ self.model.item_factors.T
        return self._scores

    def recall(self, user_idx: int, k: int, exclude: set[int]) -> list[Candidate]:
        user_idx = int(user_idx)
        if user_idx < 0 or user_idx not in self._train_users:
            return []        # no train history -> popularity fallback covers it
        row = self.scores[user_idx].copy()
        blocked = set(exclude) | self.dataset.seen.get("valid", {}).get(user_idx, set())
        if blocked:
            row[np.asarray(sorted(blocked), dtype=np.int64)] = -np.inf
        k = min(k, self.dataset.n_items)
        top = np.argpartition(-row, kth=k - 1)[:k]
        top = top[np.argsort(-row[top], kind="stable")]
        top = [int(i) for i in top if np.isfinite(row[i])]
        if not top:
            return []
        values = row[top]
        lo, hi = float(values.min()), float(values.max())
        norm = (values - lo) / (hi - lo) if hi - lo > 1e-12 else np.ones_like(values)
        return [
            Candidate(item_idx=i, score=float(s), source=self.name)
            for i, s in zip(top, norm)
        ]
