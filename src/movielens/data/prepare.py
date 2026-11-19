"""Time-aware dataset preparation.

Responsibilities:
  1. read ml-latest-small (csv) or ml-1m (dat),
  2. build stable continuous indices for users / items,
  3. split by TIME (never randomly) into train / valid / test,
  4. build seen sets and ground truth with strict no-leakage assertions,
  5. sample pairwise negatives from items that existed in the training window,
  6. build the genre matrix used by MMR and the popularity stats used by eval.

Run:  python -m movielens.data.prepare
"""

from __future__ import annotations

import argparse
import json
import pickle
import re
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import Config, load_config
from .download import download_dataset

YEAR_RE = re.compile(r"\((\d{4})\)\s*$")
NO_GENRES = "(no genres listed)"
RATING_COLS = ["userId", "movieId", "rating", "timestamp"]
MOVIE_COLS = ["movieId", "title", "genres"]


def _read_raw(raw_path: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame | None]:
    if (raw_path / "ratings.csv").exists():
        ratings = pd.read_csv(raw_path / "ratings.csv")
        movies = pd.read_csv(raw_path / "movies.csv")
        users = None
    elif (raw_path / "ratings.dat").exists():
        ratings = pd.read_csv(
            raw_path / "ratings.dat", sep="::", engine="python", names=RATING_COLS
        )
        movies = pd.read_csv(
            raw_path / "movies.dat",
            sep="::",
            engine="python",
            names=MOVIE_COLS,
            encoding="latin-1",
        )
        users_path = raw_path / "users.dat"
        users = (
            pd.read_csv(
                users_path,
                sep="::",
                engine="python",
                names=["userId", "gender", "age", "occupation", "zip"],
            )
            if users_path.exists()
            else None
        )
    else:
        raise FileNotFoundError(f"No ratings.csv / ratings.dat under {raw_path}")
    return ratings[RATING_COLS], movies[MOVIE_COLS], users


def _filter_sparse(df: pd.DataFrame, min_u: int, min_i: int) -> pd.DataFrame:
    """Iteratively drop users / items below the interaction threshold."""
    for _ in range(5):
        u_counts = df["userId"].map(df["userId"].value_counts())
        i_counts = df["movieId"].map(df["movieId"].value_counts())
        keep = (u_counts >= min_u) & (i_counts >= min_i)
        if keep.all():
            break
        df = df.loc[keep].copy()
    return df


def _assign_split(df: pd.DataFrame, cfg: Config) -> tuple[pd.DataFrame, dict]:
    strategy = cfg.split.strategy
    if strategy == "global":
        q1 = float(np.quantile(df["timestamp"].to_numpy(), cfg.split.train_ratio))
        q2 = float(
            np.quantile(
                df["timestamp"].to_numpy(),
                cfg.split.train_ratio + cfg.split.valid_ratio,
            )
        )
        ts = df["timestamp"].to_numpy()
        split = np.where(ts < q1, "train", np.where(ts < q2, "valid", "test"))
        meta = {"strategy": strategy, "t1": q1, "t2": q2}
    elif strategy == "leave_last_out":
        df = df.sort_values(["userId", "timestamp"]).copy()
        rank_from_last = df.groupby("userId").cumcount(ascending=False).to_numpy()
        split = np.where(
            rank_from_last == 0, "test", np.where(rank_from_last == 1, "valid", "train")
        )
        meta = {"strategy": strategy, "t1": None, "t2": None}
    else:
        raise ValueError(f"Unknown split strategy: {strategy}")
    out = df.copy()
    out["split"] = split
    return out, meta


def _genre_matrix(movies: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    genre_list = sorted(
        {
            g
            for raw in movies["genres"].astype(str)
            for g in raw.split("|")
            if g and g != NO_GENRES
        }
    )
    index = {g: j for j, g in enumerate(genre_list)}
    matrix = np.zeros((len(movies), len(genre_list)), dtype=np.float32)
    for i, raw in enumerate(movies["genres"].astype(str)):
        for g in raw.split("|"):
            if g in index:
                matrix[i, index[g]] = 1.0
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    np.divide(matrix, norms, out=matrix, where=norms > 0)
    return matrix, genre_list


def _seen_map(frame: pd.DataFrame) -> dict[int, set[int]]:
    return {
        int(u): set(int(x) for x in group["item_idx"].to_numpy())
        for u, group in frame.groupby("user_idx")
    }


def _truth_map(frame: pd.DataFrame) -> dict[int, set[int]]:
    positives = frame[frame["is_positive"]]
    return {
        int(u): set(int(x) for x in group["item_idx"].to_numpy())
        for u, group in positives.groupby("user_idx")
        if len(group) > 0
    }


def _sample_pairs(train: pd.DataFrame, cfg: Config, n_items: int, seen: np.ndarray):
    """Vectorised pairwise negatives.

    Negatives are drawn only from items that existed in the train window,
    excluding everything the user has already seen. A mixture of uniform and
    popularity-biased (hard) negatives is controlled by ``hard_negative_ratio``.
    """
    rng = np.random.default_rng(cfg.project.seed)
    positives = train[train["is_positive"]][["user_idx", "item_idx"]].to_numpy()
    if len(positives) == 0:
        raise ValueError("No positive interactions in train; lower positive_threshold")

    n_neg = max(1, cfg.split.negatives_per_positive)
    u = np.repeat(positives[:, 0], n_neg)
    p = np.repeat(positives[:, 1], n_neg)
    is_hard = rng.random(len(u)) < cfg.split.hard_negative_ratio

    train_items = np.unique(train["item_idx"].to_numpy())
    popularity = np.bincount(train["item_idx"].to_numpy(), minlength=n_items).astype(float)
    weights = popularity[train_items] ** 0.75
    weights = weights / weights.sum()

    neg = np.where(
        is_hard,
        rng.choice(train_items, size=len(u), p=weights),
        rng.choice(train_items, size=len(u)),
    )
    for _ in range(12):
        bad = seen[u, neg] | (neg == p)
        if not bad.any():
            break
        replacement = rng.choice(train_items, size=int(bad.sum()), p=weights)
        neg[bad] = replacement
    good = ~(seen[u, neg] | (neg == p))
    return pd.DataFrame(
        {
            "user_idx": u[good].astype(np.int32),
            "pos_item_idx": p[good].astype(np.int32),
            "neg_item_idx": neg[good].astype(np.int32),
            "is_hard": is_hard[good],
        }
    )


def prepare(cfg: Config | None = None, raw_dir: Path | None = None) -> dict:
    cfg = cfg or load_config()
    raw_root = download_dataset(cfg.data.dataset, raw_dir=raw_dir)
    ratings, movies, users = _read_raw(raw_root)

    n_dupes = int(ratings.duplicated(["userId", "movieId"]).sum())
    ratings = ratings.sort_values("timestamp").drop_duplicates(
        ["userId", "movieId"], keep="last"
    )
    ratings = _filter_sparse(
        ratings, cfg.data.min_user_interactions, cfg.data.min_item_interactions
    )
    ratings, split_meta = _assign_split(ratings, cfg)

    # Item id space = the full movie catalogue, including movies never rated.
    # This matters for coverage / long-tail metrics.
    movie_ids = np.sort(movies["movieId"].unique())
    user_ids = np.sort(ratings["userId"].unique())
    item2idx = {int(mid): i for i, mid in enumerate(movie_ids)}
    user2idx = {int(uid): i for i, uid in enumerate(user_ids)}
    idx2movie = {i: int(mid) for mid, i in item2idx.items()}
    idx2user = {i: int(uid) for uid, i in user2idx.items()}
    n_users, n_items = len(user_ids), len(movie_ids)

    movies = movies.assign(item_idx=movies["movieId"].map(item2idx))
    movies = movies.sort_values("item_idx").reset_index(drop=True)
    movies["year"] = (
        movies["title"].astype(str).str.extract(YEAR_RE, expand=False).fillna("0").astype(int)
    )
    movies = movies.rename(columns={"movieId": "movie_id"})[
        ["item_idx", "movie_id", "title", "genres", "year"]
    ]

    interactions = ratings.assign(
        user_idx=ratings["userId"].map(user2idx).astype(np.int32),
        item_idx=ratings["movieId"].map(item2idx).astype(np.int32),
        is_positive=ratings["rating"].to_numpy() >= cfg.data.positive_threshold,
    )[
        [
            "user_idx",
            "item_idx",
            "rating",
            "timestamp",
            "split",
            "is_positive",
            "userId",
            "movieId",
        ]
    ].reset_index(drop=True)

    train = interactions[interactions["split"] == "train"]
    valid = interactions[interactions["split"] == "valid"]
    test = interactions[interactions["split"] == "test"]
    for name, frame in (("train", train), ("valid", valid), ("test", test)):
        if frame.empty:
            raise ValueError(f"Split {name!r} is empty; adjust split ratios")

    # Hard no-leakage guarantee: no (user, item) pair may appear in two splits.
    pair_sets = {
        name: set(zip(frame["user_idx"].to_numpy(), frame["item_idx"].to_numpy()))
        for name, frame in (("train", train), ("valid", valid), ("test", test))
    }
    for a, b in (("train", "valid"), ("train", "test"), ("valid", "test")):
        overlap = pair_sets[a] & pair_sets[b]
        if overlap:
            raise AssertionError(f"Time leakage: {a}/{b} share {len(overlap)} pairs")

    seen = {
        "valid": _seen_map(train),
        "test": _seen_map(pd.concat([train, valid])),
        "all": _seen_map(interactions),
    }
    ground_truth = {"valid": _truth_map(valid), "test": _truth_map(test)}
    seen_train = np.zeros((n_users, n_items), dtype=bool)
    seen_train[train["user_idx"].to_numpy(), train["item_idx"].to_numpy()] = True
    samples = _sample_pairs(train, cfg, n_items, seen_train)

    genre_matrix, genre_list = _genre_matrix(movies)
    item_popularity = np.bincount(
        train["item_idx"].to_numpy(), minlength=n_items
    ).astype(np.int64)
    # "Long tail" = the least popular ``tail_quantile`` share of the catalogue.
    tail_threshold = float(np.quantile(item_popularity, cfg.eval.tail_quantile))
    tail_items = set(int(i) for i in np.where(item_popularity <= tail_threshold)[0])

    user_features = None
    if users is not None:
        user_features = users.assign(user_idx=users["userId"].map(user2idx))
        user_features = user_features.dropna(subset=["user_idx"])
        user_features["user_idx"] = user_features["user_idx"].astype(int)
        user_features = user_features.set_index("user_idx").drop(columns=["userId"])

    split_meta.update(
        {
            "dataset": cfg.data.dataset,
            "positive_threshold": cfg.data.positive_threshold,
            "min_user_interactions": cfg.data.min_user_interactions,
            "dropped_duplicate_ratings": n_dupes,
            "tail_threshold": tail_threshold,
            "n_tail_items": len(tail_items),
        }
    )
    meta = {
        "user2idx": user2idx,
        "item2idx": item2idx,
        "idx2user": idx2user,
        "idx2movie": idx2movie,
        "seen": seen,
        "ground_truth": ground_truth,
        "genre_matrix": genre_matrix,
        "genre_list": genre_list,
        "item_popularity": item_popularity,
        "tail_items": tail_items,
        "n_users": n_users,
        "n_items": n_items,
        "split_meta": split_meta,
        "user_features": user_features,
    }

    out_dir = cfg.processed_dir()
    out_dir.mkdir(parents=True, exist_ok=True)
    interactions.to_parquet(out_dir / "interactions.parquet", index=False)
    movies.to_parquet(out_dir / "movies.parquet", index=False)
    samples.to_parquet(out_dir / "samples.parquet", index=False)
    with (out_dir / "meta.pkl").open("wb") as fh:
        pickle.dump(meta, fh, protocol=pickle.HIGHEST_PROTOCOL)

    summary = {
        **{k: v for k, v in split_meta.items() if isinstance(v, (int, float, str, type(None)))},
        "n_users": n_users,
        "n_items": n_items,
        "n_genres": len(genre_list),
        "n_interactions": int(len(interactions)),
        "split_counts": {s: int(len(f)) for s, f in (("train", train), ("valid", valid), ("test", test))},
        "n_train_positives": int(train["is_positive"].sum()),
        "n_pairs": int(len(samples)),
        "eval_users_valid": len(ground_truth["valid"]),
        "eval_users_test": len(ground_truth["test"]),
        "cold_start_test_users": int(
            len(set(ground_truth["test"]) - set(seen["test"]))
        ),
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print("[prepare] summary:", json.dumps(summary, ensure_ascii=False))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare a time-split MovieLens dataset")
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--strategy", choices=["global", "leave_last_out"], default=None)
    args = parser.parse_args()
    cfg = load_config()
    if args.dataset or args.strategy:
        from dataclasses import replace

        cfg = replace(
            cfg,
            data=replace(cfg.data, dataset=args.dataset) if args.dataset else cfg.data,
            split=replace(cfg.split, strategy=args.strategy) if args.strategy else cfg.split,
        )
    prepare(cfg)


if __name__ == "__main__":
    main()
