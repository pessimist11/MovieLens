"""DIN with candidate-conditioned activation over recent behavior."""

import torch
from torch import nn

from .base import DENSE_FEATURES, RankNetwork, TrainedRanker


class DIN(RankNetwork):
    def __init__(self, dataset, cfg):
        super().__init__(dataset, cfg)
        dim, hidden = cfg.rank.dim, cfg.rank.hidden_dim
        self.activation = nn.Sequential(
            nn.Linear(4 * dim, dim), nn.PReLU(), nn.Linear(dim, 1)
        )
        self.mlp = nn.Sequential(
            nn.Linear(3 * dim + dataset.genre_matrix.shape[1] + DENSE_FEATURES, hidden),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden, 1),
        )

    def forward(self, users, items, genres, dense, history):
        target = self.item(items)
        keys = self.item(history)
        queries = target.unsqueeze(1).expand_as(keys)
        interaction = torch.cat((queries, keys, queries - keys, queries * keys), dim=-1)
        mask = history.ne(0)
        weights = self.activation(interaction).squeeze(-1).sigmoid() * mask
        context = (weights.unsqueeze(-1) * keys).sum(dim=1)
        context = context / mask.sum(dim=1, keepdim=True).clamp_min(1).sqrt()
        features = torch.cat((self.user(users), target, context, genres, self.normalize(dense)), dim=-1)
        return self.mlp(features).squeeze(-1)


class DINRanker(TrainedRanker):
    name = "din"
    implemented = True
    network = DIN
