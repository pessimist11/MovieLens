"""BPR-MF hyper-parameter sweep, evaluated on the VALID split only.

Run:  .venv/bin/python scripts/tune_bpr.py
Output: artifacts/metrics/bpr_tuning.json
"""

from __future__ import annotations

import json

from movielens.config import load_config
from movielens.data import load_dataset
from movielens.eval.metrics import evaluate_score_matrix
from movielens.mf.bpr import BPRMF

SWEEPS = {
    "dim": [16, 32, 64, 128],
    "lr": [0.001, 0.005, 0.01],
    "reg": [0.0001, 0.001, 0.01],
    "negatives_per_positive": [1, 4, 8],
    "hard_negative_ratio": [0.0, 0.5, 1.0],
}


def main() -> None:
    cfg = load_config()
    dataset = load_dataset()
    rows = []
    for param, values in SWEEPS.items():
        for value in values:
            model = BPRMF(dataset, use_cache=False, **{param: value})
            valid = evaluate_score_matrix(
                model.user_factors @ model.item_factors.T,
                dataset.ground_truth["valid"],
                dataset.seen["valid"],
                k=10,
            )
            row = {
                "param": param,
                "value": value,
                "split": "valid",
                "best_epoch": model.stats["best_epoch"],
                "epochs_ran": model.stats["epochs_ran"],
                "final_train_loss": model.stats["final_train_loss"],
                "fit_seconds": model.stats["fit_seconds"],
                "recall@k": round(valid["recall@k"], 6),
                "ndcg@k": round(valid["ndcg@k"], 6),
            }
            rows.append(row)
            print(
                f"[tune-bpr] {param}={str(value):7s} recall@10={row['recall@k']:.4f}"
                f"  ndcg@10={row['ndcg@k']:.4f}  best_epoch={row['best_epoch']}"
                f"  loss={row['final_train_loss']:.4f}  fit={row['fit_seconds']}s"
            )
    out = cfg.artifacts_dir() / "metrics" / "bpr_tuning.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[tune-bpr] wrote {out}")


if __name__ == "__main__":
    main()
