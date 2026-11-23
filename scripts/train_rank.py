"""Train and cache the stage 5 rankers from time-sliced recall candidates."""

import argparse

import torch

from movielens.config import load_config
from movielens.data import load_dataset
from movielens.rank import RANK_REGISTRY
from movielens.rank.train import train_rankers


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=[cls.name for cls in RANK_REGISTRY])
    parser.add_argument("--force", action="store_true", help="replace existing matching checkpoints")
    args = parser.parse_args()
    cfg = load_config()
    torch.set_num_threads(min(torch.get_num_threads(), 4))
    classes = [cls for cls in RANK_REGISTRY if args.model is None or cls.name == args.model]
    classes.sort(key=lambda cls: cls.name)
    for name, path in train_rankers(load_dataset(), cfg, classes, force=args.force).items():
        print(f"[rank] {name}: {path}")


if __name__ == "__main__":
    main()
