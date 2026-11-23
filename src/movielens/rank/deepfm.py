"""DeepFM with shared fields for the factorization machine and MLP."""

import torch
from torch import nn

from .base import DENSE_FEATURES, RankNetwork, TrainedRanker


class DeepFM(RankNetwork):
    def __init__(self, dataset, cfg):
        super().__init__(dataset, cfg)
        dim, hidden = cfg.rank.dim, cfg.rank.hidden_dim
        n_genres = dataset.genre_matrix.shape[1]
        self.genre = nn.Linear(n_genres, dim, bias=False)
        self.numeric = nn.Parameter(torch.empty(DENSE_FEATURES, dim))
        nn.init.normal_(self.numeric, std=0.05)
        self.wide = nn.Linear(DENSE_FEATURES + n_genres, 1)
        self.wide_user = nn.Embedding(dataset.n_users + 1, 1, padding_idx=0)
        self.wide_item = nn.Embedding(dataset.n_items + 1, 1, padding_idx=0)
        nn.init.zeros_(self.wide_user.weight)
        nn.init.zeros_(self.wide_item.weight)
        self.mlp = nn.Sequential(
            nn.Linear((3 + DENSE_FEATURES) * dim, hidden),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden, 1),
        )

    def forward(self, users, items, genres, dense, history):
        dense = self.normalize(dense)
        fields = torch.cat((
            torch.stack((self.user(users), self.item(items), self.genre(genres)), dim=1),
            dense.unsqueeze(-1) * self.numeric,
        ), dim=1)
        fm = 0.5 * (fields.sum(dim=1).square() - fields.square().sum(dim=1)).sum(dim=1)
        wide = self.wide(torch.cat((genres, dense), dim=1))
        wide = wide + self.wide_user(users) + self.wide_item(items)
        return fm + (wide + self.mlp(fields.flatten(1))).squeeze(-1)


class DeepFMRanker(TrainedRanker):
    name = "deepfm"
    implemented = True
    network = DeepFM
