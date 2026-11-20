"""Small VALID-only sweep for two-tower temperature / history length.

All runs use the same time split, seed and early-stopping policy. TEST is never
read here; the final configuration is judged only by valid Recall@10/NDCG@10.
"""

from __future__ import annotations

import json

from movielens.config import load_config
from movielens.data import load_dataset
from movielens.eval.metrics import evaluate_score_matrix
from movielens.recall.two_tower import TwoTowerModel

GRID = [
    {"temperature": 0.05, "history_size": 32},
    {"temperature": 0.10, "history_size": 32},
    {"temperature": 0.20, "history_size": 32},
    {"temperature": 0.10, "history_size": 64},
]


def main() -> None:
    ds = load_dataset()
    rows = []
    for override in GRID:
        model = TwoTowerModel(
            ds,
            epochs=30,
            eval_every=5,
            patience=3,
            use_cache=False,
            **override,
        )
        valid = evaluate_score_matrix(
            model.user_vectors @ model.item_vectors.T,
            ds.ground_truth["valid"],
            ds.seen["valid"],
            k=10,
        )
        row = {
            **override,
            **valid,
            "best_epoch": model.stats["best_epoch"],
            "epochs_ran": model.stats["epochs_ran"],
            "fit_seconds": model.stats["fit_seconds"],
        }
        rows.append(row)
        print(
            f"[tune] tau={override['temperature']:<4} "
            f"history={override['history_size']:<3} "
            f"valid recall@10={valid['recall@k']:.4f} ndcg@10={valid['ndcg@k']:.4f} "
            f"best_epoch={row['best_epoch']} fit={row['fit_seconds']}s"
        )
    rows.sort(key=lambda r: (r["recall@k"], r["ndcg@k"]), reverse=True)
    path = load_config().artifacts_dir() / "metrics" / "two_tower_tuning.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[tune] best on valid: {rows[0]}")
    print(f"[tune] wrote {path}")


if __name__ == "__main__":
    main()
