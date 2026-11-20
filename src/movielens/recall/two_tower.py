"""Two-tower representation learning and FAISS retrieval.

User tower: user-id embedding + mean-pooled embeddings of the user's recent
TRAIN history. Item tower: item-id embedding + genre multi-hot + release year.
The two sides only meet through a dot product, so every item can be embedded
offline and retrieved with an ANN index.

Training uses sampled softmax / InfoNCE over in-batch positives plus extra
uniform/popularity-sampled negatives. Items already seen by a user are masked,
including a positive that happens to occur in another row of the same batch.
For a training interaction at position t, the user history is strictly the
prefix before t; evaluation uses the full train history. This prevents both
future leakage and the trivial "target is already in the profile" shortcut.

Only train-window items are sampled as negatives. The index nevertheless
contains the whole catalogue: unseen items can still be represented by their
id/genre/year tower, which is exactly where content features help coverage.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn

from ..config import load_config
from ..eval.metrics import evaluate_score_matrix
from ..faiss_index import FaissItemIndex
from ..schema import Candidate, RecallModel


class _TwoTowerNet(nn.Module):
    def __init__(
        self,
        n_users: int,
        n_items: int,
        n_item_features: int,
        n_user_features: int,
        dim: int,
        hidden_dim: int,
        dropout: float,
    ):
        super().__init__()
        self.user_id = nn.Embedding(n_users, dim)
        # Shared with the user-history pooling and the item tower.
        self.item_id = nn.Embedding(n_items, dim)
        nn.init.normal_(self.user_id.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.item_id.weight, mean=0.0, std=0.02)
        self.user_mlp = nn.Sequential(
            nn.Linear(2 * dim + n_user_features, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
        )
        self.item_mlp = nn.Sequential(
            nn.Linear(dim + n_item_features, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
        )

    def user_tower(self, users, history_items, history_mask, user_features):
        history_emb = self.item_id(history_items)
        weights = history_mask.unsqueeze(-1)
        pooled = (history_emb * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)
        inputs = torch.cat([self.user_id(users), pooled, user_features], dim=-1)
        return F.normalize(self.user_mlp(inputs), p=2, dim=-1)

    def item_tower(self, items, item_features):
        inputs = torch.cat([self.item_id(items), item_features], dim=-1)
        return F.normalize(self.item_mlp(inputs), p=2, dim=-1)


def info_nce_loss(
    user_vectors: torch.Tensor,
    positive_vectors: torch.Tensor,
    users: torch.Tensor,
    positive_items: torch.Tensor,
    *,
    seen_bool: torch.Tensor,
    temperature: float,
    negative_vectors: torch.Tensor | None = None,
    negative_items: torch.Tensor | None = None,
) -> torch.Tensor:
    """InfoNCE with explicit false-negative masking.

    Column j is the positive for row j. Every other in-batch item already seen
    by that row's user is masked, rather than incorrectly treated as a dislike.
    Extra negatives have shape ``(batch, n_negatives, dim)`` and receive the
    same treatment. The diagonal is always restored, even though positives are
    naturally present in the user's train-history seen set.
    """
    batch = len(users)
    labels = torch.arange(batch, device=users.device)
    logits = user_vectors @ positive_vectors.T / temperature
    in_batch_seen = seen_bool[users][:, positive_items]
    in_batch_seen[labels, labels] = False
    logits = logits.masked_fill(in_batch_seen, -1e4)

    if negative_vectors is not None:
        if negative_items is None:
            raise ValueError("negative_items is required with negative_vectors")
        extra_logits = torch.einsum("bd,bnd->bn", user_vectors, negative_vectors) / temperature
        extra_seen = seen_bool[users[:, None], negative_items] | (
            negative_items == positive_items[:, None]
        )
        extra_logits = extra_logits.masked_fill(extra_seen, -1e4)
        logits = torch.cat([logits, extra_logits], dim=1)
    return F.cross_entropy(logits, labels)


class TwoTowerModel:
    """Fit and cache user/item embeddings; FAISS serving lives below."""

    def __init__(
        self,
        dataset,
        *,
        dim: int | None = None,
        hidden_dim: int | None = None,
        lr: float | None = None,
        weight_decay: float | None = None,
        epochs: int | None = None,
        batch_size: int | None = None,
        extra_negatives: int | None = None,
        hard_negative_ratio: float | None = None,
        temperature: float | None = None,
        history_size: int | None = None,
        dropout: float | None = None,
        eval_every: int | None = None,
        patience: int | None = None,
        seed: int | None = None,
        use_cache: bool | None = None,
    ):
        base = load_config()
        cfg = base.raw.get("recall", {}).get("two_tower", {})
        self.dataset = dataset
        self.dim = int(cfg.get("dim", 64) if dim is None else dim)
        self.hidden_dim = int(cfg.get("hidden_dim", 128) if hidden_dim is None else hidden_dim)
        self.lr = float(cfg.get("lr", 1e-3) if lr is None else lr)
        self.weight_decay = float(
            cfg.get("weight_decay", 1e-4) if weight_decay is None else weight_decay
        )
        self.epochs = int(cfg.get("epochs", 50) if epochs is None else epochs)
        self.batch_size = int(cfg.get("batch_size", 1024) if batch_size is None else batch_size)
        self.extra_negatives = int(
            cfg.get("extra_negatives", 32) if extra_negatives is None else extra_negatives
        )
        self.hard_negative_ratio = float(
            cfg.get("hard_negative_ratio", 0.5)
            if hard_negative_ratio is None
            else hard_negative_ratio
        )
        self.temperature = float(cfg.get("temperature", 0.1) if temperature is None else temperature)
        self.history_size = int(cfg.get("history_size", 32) if history_size is None else history_size)
        self.dropout = float(cfg.get("dropout", 0.1) if dropout is None else dropout)
        self.eval_every = int(cfg.get("eval_every", 5) if eval_every is None else eval_every)
        self.patience = int(cfg.get("patience", 4) if patience is None else patience)
        self.seed = int(base.project.seed if seed is None else seed)
        self.use_cache = bool(cfg.get("cache", True) if use_cache is None else use_cache)
        if self.dim <= 0 or self.history_size <= 0 or self.temperature <= 0:
            raise ValueError("dim, history_size and temperature must be positive")

        self.user_vectors = np.zeros((dataset.n_users, self.dim), dtype=np.float32)
        self.item_vectors = np.zeros((dataset.n_items, self.dim), dtype=np.float32)
        self.history: list[dict] = []
        self.stats: dict = {}
        self._prepare()
        self.fingerprint = self._fingerprint()
        self._load_or_fit()

    def _params(self) -> dict:
        return {
            "dim": self.dim,
            "hidden_dim": self.hidden_dim,
            "lr": self.lr,
            "weight_decay": self.weight_decay,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "extra_negatives": self.extra_negatives,
            "hard_negative_ratio": self.hard_negative_ratio,
            "temperature": self.temperature,
            "history_size": self.history_size,
            "dropout": self.dropout,
            "eval_every": self.eval_every,
            "patience": self.patience,
            "seed": self.seed,
            "architecture": "id+mean_history|id+genre+year-v1",
        }

    def _prepare(self) -> None:
        train = self.dataset.frame("train")
        positives = train[train["is_positive"]] if "is_positive" in train else train
        positives = positives.sort_values(["user_idx", "timestamp"], kind="stable")
        self._positives = positives[["user_idx", "item_idx"]].to_numpy(dtype=np.int64)
        self._positions = positives.groupby("user_idx").cumcount().to_numpy(dtype=np.int64)
        self._histories = [
            np.asarray(group["item_idx"].to_numpy(), dtype=np.int64)
            for _, group in positives.groupby("user_idx", sort=True)
        ]
        # groupby above omits users without a positive; restore continuous ids.
        by_user = {int(u): h for u, h in zip(sorted(positives["user_idx"].unique()), self._histories)}
        self._histories = [by_user.get(u, np.empty(0, dtype=np.int64)) for u in range(self.dataset.n_users)]
        self._train_items = np.unique(self._positives[:, 1])
        popularity = np.bincount(self._positives[:, 1], minlength=self.dataset.n_items).astype(float)
        weights = popularity[self._train_items] ** 0.75
        self._neg_weights = weights / weights.sum()
        self._seen_bool = np.zeros((self.dataset.n_users, self.dataset.n_items), dtype=bool)
        self._seen_bool[train["user_idx"], train["item_idx"]] = True

        years = self.dataset.movies["year"].to_numpy(dtype=np.float32)
        known = years[years > 0]
        mean = float(known.mean()) if len(known) else 0.0
        std = float(known.std()) if len(known) else 1.0
        normalised_year = np.where(years > 0, (years - mean) / max(std, 1.0), 0.0)
        self._item_features = np.concatenate(
            [self.dataset.genre_matrix.astype(np.float32), normalised_year[:, None]], axis=1
        ).astype(np.float32)
        self._user_features = self._make_user_features()

    def _make_user_features(self) -> np.ndarray:
        """Optional ml-1m demographics; zip is deliberately not used as a feature."""
        frame = self.dataset.user_features
        columns = [] if frame is None else [c for c in ("gender", "age", "occupation") if c in frame]
        if not columns:
            return np.zeros((self.dataset.n_users, 0), dtype=np.float32)
        parts = []
        for col in columns:
            values = frame[col].astype(str)
            categories = sorted(values.unique())
            one_hot = pd.get_dummies(values, dtype=float).reindex(columns=categories, fill_value=0.0)
            aligned = one_hot.reindex(frame.index).fillna(0.0).to_numpy(dtype=np.float32)
            full = np.zeros((self.dataset.n_users, len(categories)), dtype=np.float32)
            full[frame.index.to_numpy(dtype=int)] = aligned
            parts.append(full)
        return np.concatenate(parts, axis=1)

    def history_batch(self, users: np.ndarray, positions: np.ndarray | None = None):
        """Return padded item ids and mask; ``positions`` selects a strict prefix."""
        items = np.zeros((len(users), self.history_size), dtype=np.int64)
        mask = np.zeros_like(items, dtype=np.float32)
        for row, user in enumerate(users):
            history = self._histories[int(user)]
            end = len(history) if positions is None else int(positions[row])
            recent = history[max(0, end - self.history_size) : end]
            if len(recent):
                items[row, -len(recent) :] = recent
                mask[row, -len(recent) :] = 1.0
        return items, mask

    def sample_extra_negatives(self, users: np.ndarray, positives: np.ndarray, rng):
        """Sample train-window negatives; seen items are masked by InfoNCE."""
        shape = (len(users), self.extra_negatives)
        hard = rng.random(shape) < self.hard_negative_ratio
        negatives = np.where(
            hard,
            rng.choice(self._train_items, size=shape, p=self._neg_weights),
            rng.choice(self._train_items, size=shape),
        )
        return negatives.astype(np.int64)

    def _fingerprint(self) -> str:
        digest = hashlib.blake2b(digest_size=16)
        digest.update(self._positives.tobytes())
        digest.update(self._positions.tobytes())
        digest.update(self._item_features.tobytes())
        digest.update(self._user_features.tobytes())
        digest.update(json.dumps(self._params(), sort_keys=True).encode())
        return digest.hexdigest()

    def _cache_dir(self) -> Path:
        return load_config().artifacts_dir() / "models" / "two_tower"

    def _cache_paths(self) -> tuple[Path, Path]:
        base = self._cache_dir()
        return base / f"embeddings_{self.fingerprint}.npz", base / f"embeddings_{self.fingerprint}.json"

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
                    self.user_vectors, self.item_vectors = user, item
                    payload = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
                    self.history = payload.pop("history", [])
                    self.stats = {**payload, "from_cache": True, "fingerprint": self.fingerprint}
                    return
        self.fit()
        if self.use_cache:
            npz_path, meta_path = self._cache_paths()
            npz_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(npz_path, user=self.user_vectors, item=self.item_vectors)
            meta_path.write_text(
                json.dumps({**self.stats, "history": self.history}, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )

    def _new_net(self) -> _TwoTowerNet:
        return _TwoTowerNet(
            self.dataset.n_users,
            self.dataset.n_items,
            self._item_features.shape[1],
            self._user_features.shape[1],
            self.dim,
            self.hidden_dim,
            self.dropout,
        )

    def fit(self) -> "TwoTowerModel":
        t0 = perf_counter()
        torch.manual_seed(self.seed)
        rng = np.random.default_rng(self.seed)
        net = self._new_net()
        optimiser = torch.optim.AdamW(net.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        item_features = torch.from_numpy(self._item_features)
        user_features = torch.from_numpy(self._user_features)
        seen_bool = torch.from_numpy(self._seen_bool)

        best_recall, best_vectors, stale = -1.0, None, 0
        for epoch in range(1, self.epochs + 1):
            order = rng.permutation(len(self._positives))
            net.train()
            total_loss, n_batches = 0.0, 0
            for start in range(0, len(order), self.batch_size):
                rows = order[start : start + self.batch_size]
                users_np = self._positives[rows, 0]
                positives_np = self._positives[rows, 1]
                history_np, mask_np = self.history_batch(users_np, self._positions[rows])
                negatives_np = self.sample_extra_negatives(users_np, positives_np, rng)
                users = torch.from_numpy(users_np)
                positives = torch.from_numpy(positives_np)
                negatives = torch.from_numpy(negatives_np)
                user_vec = net.user_tower(
                    users,
                    torch.from_numpy(history_np),
                    torch.from_numpy(mask_np),
                    user_features[users],
                )
                positive_vec = net.item_tower(positives, item_features[positives])
                if self.extra_negatives > 0:
                    flat_negatives = negatives.reshape(-1)
                    negative_vec = net.item_tower(flat_negatives, item_features[flat_negatives]).reshape(
                        len(users), self.extra_negatives, self.dim
                    )
                else:
                    negative_vec = None
                loss = info_nce_loss(
                    user_vec,
                    positive_vec,
                    users,
                    positives,
                    seen_bool=seen_bool,
                    temperature=self.temperature,
                    negative_vectors=negative_vec,
                    negative_items=negatives if negative_vec is not None else None,
                )
                optimiser.zero_grad()
                loss.backward()
                optimiser.step()
                total_loss += float(loss.detach())
                n_batches += 1

            entry = {"epoch": epoch, "train_loss": round(total_loss / max(n_batches, 1), 6)}
            if epoch % self.eval_every == 0 or epoch == self.epochs:
                self._export(net, user_features)
                valid = evaluate_score_matrix(
                    self.user_vectors @ self.item_vectors.T,
                    self.dataset.ground_truth["valid"],
                    self.dataset.seen["valid"],
                    k=10,
                )
                entry["valid_recall@10"] = round(valid["recall@k"], 6)
                entry["valid_ndcg@10"] = round(valid["ndcg@k"], 6)
                if valid["recall@k"] > best_recall:
                    best_recall = valid["recall@k"]
                    best_vectors = (self.user_vectors.copy(), self.item_vectors.copy())
                    stale = 0
                else:
                    stale += 1
            self.history.append(entry)
            if stale >= self.patience and best_vectors is not None:
                break

        if best_vectors is not None:
            self.user_vectors, self.item_vectors = best_vectors
        else:
            net.eval()
            self._export(net, user_features)
        self.stats = {
            **self._params(),
            "fingerprint": self.fingerprint,
            "epochs_ran": len(self.history),
            "best_epoch": max(
                (h for h in self.history if "valid_recall@10" in h),
                key=lambda h: h["valid_recall@10"],
                default={},
            ).get("epoch"),
            "best_valid_recall@10": round(best_recall, 6),
            "final_train_loss": self.history[-1]["train_loss"] if self.history else None,
            "n_train_positives": int(len(self._positives)),
            "n_user_feature_dims": int(self._user_features.shape[1]),
            "fit_seconds": round(perf_counter() - t0, 2),
            "from_cache": False,
        }
        return self

    def _export(self, net: _TwoTowerNet, user_features: torch.Tensor) -> None:
        net.eval()
        with torch.no_grad():
            items = torch.arange(self.dataset.n_items)
            self.item_vectors = net.item_tower(items, torch.from_numpy(self._item_features)).numpy()
            users = torch.arange(self.dataset.n_users)
            history, mask = self.history_batch(users.numpy())
            self.user_vectors = net.user_tower(
                users,
                torch.from_numpy(history),
                torch.from_numpy(mask),
                user_features,
            ).numpy()
        self.user_vectors = self.user_vectors.astype(np.float32)
        self.item_vectors = self.item_vectors.astype(np.float32)

    def score(self, user_idx: int, item_idx: int) -> float:
        return float(self.user_vectors[int(user_idx)] @ self.item_vectors[int(item_idx)])


class TwoTowerRecall(RecallModel):
    """Serve fixed (T+1) user vectors with exact or approximate retrieval.

    ``backend=auto`` uses FAISS everywhere except macOS, where the current
    torch/faiss binary wheels each ship an incompatible OpenMP runtime. There
    NumPy performs the same exact inner-product search as IndexFlatIP. The
    standalone ``movielens.faiss_index`` component and compare_index.py still
    exercise real Flat/HNSW indexes without importing torch.
    """

    name = "two_tower"
    implemented = True

    def __init__(
        self,
        dataset,
        index_type: str | None = None,
        backend: str | None = None,
        **overrides,
    ):
        self.dataset = dataset
        base = load_config()
        self.cfg = base.raw.get("recall", {}).get("two_tower", {})
        self.model = TwoTowerModel(dataset, **overrides)
        configured = backend or self.cfg.get("backend", "auto")
        self.backend = ("numpy" if sys.platform == "darwin" else "faiss") if configured == "auto" else configured
        if self.backend not in {"numpy", "faiss"}:
            raise ValueError("backend must be auto, numpy or faiss")
        self.index_type = (index_type or self.cfg.get("index_type", "flat")).lower()
        self._train_users = set(dataset.seen.get("valid", {}))
        self._faiss = None
        if self.backend == "faiss":
            self.build_index(self.index_type)
        elif self.index_type != "flat":
            raise ValueError("NumPy backend only supports exact flat retrieval")
        self.stats = {
            **self.model.stats,
            "backend": self.backend,
            "index_type": self.index_type,
            "index_ntotal": self.dataset.n_items,
        }

    @property
    def index(self):
        return self._faiss.index if self._faiss is not None else None

    def build_index(self, index_type: str = "flat") -> None:
        self._faiss = FaissItemIndex(
            self.model.item_vectors,
            cache_dir=self.model._cache_dir(),
            fingerprint=self.model.fingerprint,
            index_type=index_type,
            hnsw_m=int(self.cfg.get("hnsw_m", 32)),
            hnsw_ef_construction=int(self.cfg.get("hnsw_ef_construction", 80)),
            hnsw_ef_search=int(self.cfg.get("hnsw_ef_search", 64)),
            use_cache=bool(self.cfg.get("cache", True)),
        )
        self.backend, self.index_type = "faiss", index_type

    def save(self, path: str) -> None:
        if self._faiss is None:
            raise RuntimeError("No FAISS index on the NumPy backend")
        self._faiss.save(path)

    def _numpy_recall(self, user_idx: int, k: int, blocked: set[int]):
        scores = self.model.item_vectors @ self.model.user_vectors[user_idx]
        if blocked:
            scores[np.asarray(sorted(blocked), dtype=np.int64)] = -np.inf
        k = min(k, self.dataset.n_items)
        top = np.argpartition(-scores, kth=k - 1)[:k]
        top = top[np.argsort(-scores[top], kind="stable")]
        return [(int(i), float(scores[i])) for i in top if np.isfinite(scores[i])]

    def recall(self, user_idx: int, k: int, exclude: set[int]) -> list[Candidate]:
        user_idx = int(user_idx)
        if user_idx < 0 or user_idx not in self._train_users:
            return []
        blocked = set(exclude) | self.dataset.seen.get("valid", {}).get(user_idx, set())
        if self.backend == "numpy":
            pairs = self._numpy_recall(user_idx, k, blocked)
        else:
            fetch = min(self.dataset.n_items, max(k + len(blocked) + 16, k * 4))
            query = self.model.user_vectors[user_idx][None, :]
            scores, ids = self._faiss.search(query, fetch)
            pairs = [
                (int(item), float(score))
                for item, score in zip(ids[0], scores[0])
                if item >= 0 and item not in blocked
            ][:k]
        return [
            Candidate(item_idx=item, score=(score + 1.0) / 2.0, source=self.name)
            for item, score in pairs
        ]
