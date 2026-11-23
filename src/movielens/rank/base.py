"""Versioned rank checkpoints and the shared prediction contract."""

import hashlib
import json
from dataclasses import asdict, replace
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np
import pandas as pd
import torch
from torch import nn

from ..schema import RankModel

DENSE_FEATURES = 11


def checkpoint_path(name, dataset, cfg):
    settings = {
        "version": 1,
        "model": name,
        "rank": asdict(replace(cfg.rank, model="none")),
        "seed": cfg.project.seed,
        "dataset_shape": (dataset.n_users, dataset.n_items),
        "recall": asdict(cfg.recall),
        "recall_raw": cfg.raw.get("recall", {}),
        "mf": cfg.raw.get("mf", {}),
        "rerank": asdict(cfg.rerank),
        "eval_k": cfg.eval.k,
    }
    digest = hashlib.blake2b(digest_size=16)
    digest.update(json.dumps(settings, sort_keys=True).encode())
    # VALID affects the selected checkpoint. TEST never participates.
    frame = dataset.interactions[dataset.interactions["split"].isin(["train", "valid"])]
    digest.update(pd.util.hash_pandas_object(frame, index=False).to_numpy().tobytes())
    digest.update(dataset.genre_matrix.tobytes())
    digest.update(dataset.movies["year"].to_numpy(dtype=np.int32).tobytes())
    if dataset.user_features is not None:
        digest.update(pd.util.hash_pandas_object(dataset.user_features).to_numpy().tobytes())
    return cfg.artifacts_dir() / "models" / "rank" / f"{name}_{digest.hexdigest()}.pt"


def save_checkpoint(path, model, stats):
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(dir=path.parent, suffix=".pt", delete=False) as temp:
        staging = Path(temp.name)
    try:
        torch.save({"state_dict": model.state_dict(), "stats": stats}, staging)
        staging.replace(path)
    finally:
        staging.unlink(missing_ok=True)


class RankNetwork(nn.Module):
    def __init__(self, dataset, cfg):
        super().__init__()
        dim = cfg.rank.dim
        self.user = nn.Embedding(dataset.n_users + 1, dim, padding_idx=0)
        self.item = nn.Embedding(dataset.n_items + 1, dim, padding_idx=0)
        for embedding in (self.user, self.item):
            nn.init.normal_(embedding.weight[1:], std=0.05)
        self.register_buffer("centre", torch.zeros(DENSE_FEATURES))
        self.register_buffer("scale", torch.ones(DENSE_FEATURES))

    def normalize(self, dense):
        return ((dense - self.centre) / self.scale).clamp(-8.0, 8.0)


class TrainedRanker(RankModel):
    def __init__(self, dataset, cfg):
        from .features import RankFeatures

        path = checkpoint_path(self.name, dataset, cfg)
        if not path.exists():
            raise FileNotFoundError(
                f"Missing {self.name} rank checkpoint. Run: python scripts/train_rank.py"
            )
        self.features = RankFeatures(dataset, cfg)
        self.model = self.network(dataset, cfg)
        payload = torch.load(path, map_location="cpu", weights_only=True)
        self.model.load_state_dict(payload["state_dict"])
        self.model.eval()
        self.stats = payload["stats"]

    @torch.inference_mode()
    def predict(self, user_idx, candidates):
        if (not candidates or not 0 <= user_idx < len(self.features.user_count)
                or not self.features.user_count[user_idx]):
            return candidates
        scores = self.model(*self.features.encode(user_idx, candidates)).sigmoid().tolist()
        return [replace(candidate, rank_score=score) for candidate, score in zip(candidates, scores)]
