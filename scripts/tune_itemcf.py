"""ItemCF hyper-parameter sweep, evaluated on the VALID split only.

Tuning on test would leak the evaluation set into model selection, so every
number here is a valid number; the test set is only touched by evaluate.py
after the configuration is frozen.

Run:  .venv/bin/python scripts/tune_itemcf.py
Output: artifacts/metrics/itemcf_tuning.json
"""

from __future__ import annotations

import copy
import json

from movielens.config import load_config
from movielens.pipeline import Recommender
from movielens.recall.itemcf import ItemCFRecall

SWEEPS = {
    "alpha": [0.0, 0.25, 0.5, 0.75, 1.0],
    "top_n_neighbours": [20, 50, 80, 160, 320],
    "iuf": [True, False],
    "shrink": [0.0, 5.0, 20.0, 50.0],
    "history_decay": [0.0, 0.01, 0.05, 0.1, 0.2],
}


def main() -> None:
    cfg = load_config()
    base = Recommender()
    rows = []
    for param, values in SWEEPS.items():
        for value in values:
            model = ItemCFRecall(base.dataset, use_cache=False, **{param: value})
            rec = copy.copy(base)
            rec.recall_models = [model]
            rec.reranker = None
            metrics = rec.evaluate("valid").as_dict()
            row = {
                "param": param,
                "value": value,
                "split": "valid",
                "sim_nnz": model.stats["sim_nnz"],
                "mean_neighbours": model.stats["mean_neighbours"],
                **metrics,
            }
            rows.append(row)
            print(
                f"[tune] {param}={str(value):6s} recall@10={metrics['recall@k']:.4f}"
                f"  ndcg@10={metrics['ndcg@k']:.4f}  coverage={metrics['coverage']:.4f}"
                f"  tail_share={metrics['long_tail_share']:.4f}  ILD={metrics['ild']:.4f}"
            )
    out = cfg.artifacts_dir() / "metrics" / "itemcf_tuning.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[tune] wrote {out}")


if __name__ == "__main__":
    main()
