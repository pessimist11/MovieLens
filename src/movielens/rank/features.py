"""Features shared by snapshot training, validation and API prediction."""

import numpy as np
import torch

RECALL_SOURCES = ("itemcf", "bpr", "two_tower")


class RankFeatures:
    def __init__(self, dataset, cfg):
        frame = dataset.frame("train").sort_values("timestamp", kind="stable")
        users = frame["user_idx"].to_numpy()
        items = frame["item_idx"].to_numpy()
        ratings = frame["rating"].to_numpy()
        age_days = (frame["timestamp"].max() - frame["timestamp"].to_numpy()) / 86400
        decay = np.exp2(-age_days / cfg.rank.popularity_half_life_days)
        self.user_count = np.bincount(users, minlength=dataset.n_users)
        self.user_mean = np.bincount(users, weights=ratings, minlength=dataset.n_users)
        self.user_mean /= self.user_count.clip(min=1)
        self.popularity = np.log1p(np.bincount(items, minlength=dataset.n_items))
        self.decayed_popularity = np.log1p(np.bincount(
            items, weights=decay, minlength=dataset.n_items
        ))
        self.genres = dataset.genre_matrix.astype(np.float32)
        self.history = np.zeros((dataset.n_users, cfg.rank.history_size), dtype=np.int64)
        for user, events in frame.groupby("user_idx", sort=False):
            recent = events["item_idx"].to_numpy()[-cfg.rank.history_size:]
            self.history[user, -len(recent):] = recent + 1

    def encode(self, user, candidates):
        items = np.array([c.item_idx for c in candidates], dtype=np.int64)
        count = len(items)
        recall = np.array([
            [c.recall_scores.get(source, 0.0) for source in RECALL_SOURCES]
            for c in candidates
        ], dtype=np.float32)
        present = np.array([
            [source in c.recall_scores for source in RECALL_SOURCES] for c in candidates
        ], dtype=np.float32)
        dense = np.column_stack((
            [60 * c.score for c in candidates],
            np.sign(recall) * np.log1p(np.abs(recall)),
            present,
            self.popularity[items],
            self.decayed_popularity[items],
            np.full(count, np.log1p(self.user_count[user])),
            np.full(count, self.user_mean[user]),
        )).astype(np.float32)
        return (
            torch.full((count,), user + 1, dtype=torch.long),
            torch.from_numpy(items + 1),
            torch.from_numpy(self.genres[items]),
            torch.from_numpy(dense),
            torch.from_numpy(np.tile(self.history[user], (count, 1))),
        )
