"""The processed dataset object shared by training, evaluation and serving.

Everything lives in continuous-index space (``user_idx`` / ``item_idx``).
Raw ``userId`` / ``movieId`` are only re-attached at the API boundary.
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass
class Dataset:
    interactions: pd.DataFrame
    movies: pd.DataFrame
    samples: pd.DataFrame
    user2idx: dict[int, int]
    item2idx: dict[int, int]
    idx2user: dict[int, int]
    idx2movie: dict[int, int]
    seen: dict[str, dict[int, set[int]]]
    ground_truth: dict[str, dict[int, set[int]]]
    genre_matrix: np.ndarray
    genre_list: list[str]
    item_popularity: np.ndarray
    tail_items: set[int]
    n_users: int
    n_items: int
    split_meta: dict
    user_features: pd.DataFrame | None = None

    def frame(self, split: str | None = None) -> pd.DataFrame:
        if split is None:
            return self.interactions
        return self.interactions[self.interactions["split"] == split]

    def seen_for(self, user_idx: int, split: str = "all") -> set[int]:
        return self.seen.get(split, {}).get(user_idx, set())

    def genres_of(self, item_idx: int) -> list[str]:
        raw = str(self.movies.iloc[item_idx]["genres"])
        return [] if raw in ("(no genres listed)", "nan", "") else raw.split("|")

    def movie_row(self, item_idx: int) -> dict:
        row = self.movies.iloc[item_idx]
        return {
            "movie_id": int(row["movie_id"]),
            "title": str(row["title"]),
            "genres": self.genres_of(item_idx),
            "year": int(row.get("year", 0) or 0),
        }

    def user_idx_of(self, user_id: int) -> int | None:
        return self.user2idx.get(int(user_id))

    @property
    def tail_threshold(self) -> float:
        return float(self.split_meta.get("tail_threshold", 0.0))


def load_dataset(processed_dir: str | Path | None = None) -> Dataset:
    """Load all processed artifacts produced by ``data.prepare``."""
    from ..config import load_config

    cfg = load_config()
    base = Path(processed_dir) if processed_dir else cfg.processed_dir()
    required = ["interactions.parquet", "movies.parquet", "samples.parquet", "meta.pkl"]
    missing = [name for name in required if not (base / name).exists()]
    if missing:
        raise FileNotFoundError(
            f"Missing {missing} in {base}. Run: python -m movielens.data.prepare"
        )

    interactions = pd.read_parquet(base / "interactions.parquet")
    movies = pd.read_parquet(base / "movies.parquet")
    samples = pd.read_parquet(base / "samples.parquet")
    with (base / "meta.pkl").open("rb") as fh:
        meta = pickle.load(fh)

    return Dataset(
        interactions=interactions,
        movies=movies,
        samples=samples,
        user2idx=meta["user2idx"],
        item2idx=meta["item2idx"],
        idx2user=meta["idx2user"],
        idx2movie=meta["idx2movie"],
        seen=meta["seen"],
        ground_truth=meta["ground_truth"],
        genre_matrix=meta["genre_matrix"],
        genre_list=meta["genre_list"],
        item_popularity=meta["item_popularity"],
        tail_items=meta["tail_items"],
        n_users=int(meta["n_users"]),
        n_items=int(meta["n_items"]),
        split_meta=meta["split_meta"],
        user_features=meta.get("user_features"),
    )
