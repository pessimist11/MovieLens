"""Download and extract a MovieLens dataset.

Supported datasets:
  - ml-latest-small : ~100k ratings, 610 users, 9.7k movies (fast iteration)
  - ml-1m           : ~1M ratings, 6k users, demographics available ( richer )
"""

from __future__ import annotations

import argparse
import shutil
import zipfile
from pathlib import Path

from ..config import load_config

BASE_URL = "https://files.grouplens.org/datasets/movielens"
DATASETS = {
    "ml-latest-small": f"{BASE_URL}/ml-latest-small.zip",
    "ml-1m": f"{BASE_URL}/ml-1m.zip",
}


def download_dataset(dataset: str | None = None, raw_dir: Path | None = None, force: bool = False) -> Path:
    """Download the zip, extract it, and return the extracted directory."""
    cfg = load_config()
    dataset = dataset or cfg.data.dataset
    if dataset not in DATASETS:
        raise ValueError(f"Unknown dataset {dataset!r}; choose from {sorted(DATASETS)}")

    raw_dir = Path(raw_dir) if raw_dir else cfg.raw_dir()
    raw_dir.mkdir(parents=True, exist_ok=True)
    target = raw_dir / dataset
    if target.exists() and not force:
        print(f"[download] already present: {target}")
        return target

    zip_path = raw_dir / f"{dataset}.zip"
    if not zip_path.exists() or force:
        import ssl
        import urllib.request

        import certifi

        url = DATASETS[dataset]
        print(f"[download] fetching {url}")
        # Standalone Python builds (e.g. uv-managed) do not ship system CA
        # certificates; certifi's bundle keeps TLS verification enabled.
        context = ssl.create_default_context(cafile=certifi.where())
        with urllib.request.urlopen(url, timeout=120, context=context) as resp, zip_path.open(
            "wb"
        ) as out:
            shutil.copyfileobj(resp, out)
        print(f"[download] saved {zip_path} ({zip_path.stat().st_size / 1e6:.1f} MB)")

    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(raw_dir)
    print(f"[download] extracted to {target}")
    return target


def main() -> None:
    parser = argparse.ArgumentParser(description="Download a MovieLens dataset")
    parser.add_argument("--dataset", choices=sorted(DATASETS), default=None)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    download_dataset(args.dataset, force=args.force)


if __name__ == "__main__":
    main()
