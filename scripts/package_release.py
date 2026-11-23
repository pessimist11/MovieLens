"""Export source, processed data and active model caches into a release directory."""

import argparse
from pathlib import Path
from shutil import copy2, copytree

from movielens.config import ROOT
from movielens.pipeline import Recommender
from movielens.rank.base import checkpoint_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path, help="path to a new release directory")
    parser.add_argument("--lock", type=Path, help="resolved runtime requirements for the target platform")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output directory already exists")

    rec = Recommender()
    cfg = rec.cfg
    files = [
        ROOT / "configs" / "default.yaml",
        ROOT / "requirements.txt",
        ROOT / "scripts" / "smoke_deploy.py",
    ]
    files.extend(cfg.processed_dir() / name for name in (
        "interactions.parquet", "movies.parquet", "samples.parquet", "meta.pkl", "summary.json"
    ))
    files.extend(sorted((cfg.artifacts_dir() / "metrics").glob("*.json")))
    raw_dataset = cfg.raw_dir() / cfg.data.dataset
    files.extend(path for path in (raw_dataset / "README.txt", raw_dataset / "README")
                 if path.is_file())
    if not (cfg.artifacts_dir() / "metrics" / "baseline.json").is_file():
        parser.error("run scripts/evaluate.py before packaging")

    for recall in rec.recall_models:
        if recall.name == "popular":
            continue
        model = recall.model if recall.name in {"bpr", "two_tower"} else recall
        if not model.use_cache:
            parser.error(f"enable {recall.name} caching before creating a release")
        files.extend(model._cache_paths())
        if recall.name == "two_tower":
            index = model._cache_dir() / f"{recall.index_type}_{model.fingerprint}.faiss"
            if index.is_file():
                files.append(index)
    if rec.ranker is not None:
        files.append(checkpoint_path(rec.ranker.name, rec.dataset, cfg))

    for source in [*files, *([args.lock] if args.lock else [])]:
        if not source.is_file():
            parser.error(f"release input is missing: {source}")
    args.output.mkdir(parents=True)
    copytree(ROOT / "src", args.output / "src", ignore=lambda _, names: {
        name for name in names if name in {"__pycache__", ".DS_Store"} or name.endswith(".egg-info")
    })
    for source in files:
        destination = args.output / source.relative_to(ROOT)
        destination.parent.mkdir(parents=True, exist_ok=True)
        copy2(source, destination)
    if args.lock:
        copy2(args.lock, args.output / "requirements.lock.txt")
    print(f"Prepared release: {args.output.resolve()}")


if __name__ == "__main__":
    main()
