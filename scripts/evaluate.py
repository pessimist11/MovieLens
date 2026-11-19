"""Full-pipeline offline evaluation + MMR relevance/diversity trade-off scan.

Run:  .venv/bin/python scripts/evaluate.py
Output: artifacts/metrics/baseline.json, artifacts/metrics/mmr_tradeoff.json
"""

from __future__ import annotations

import json

from movielens.config import load_config
from movielens.eval.metrics import evaluate_all
from movielens.pipeline import Recommender
from movielens.rerank import MMRReranker


def lambda_sweep(rec: Recommender, split: str = "valid", max_users: int = 100) -> list[dict]:
    truth_all = rec.dataset.ground_truth[split]
    users = list(truth_all)[:max_users]
    truth = {u: truth_all[u] for u in users}
    rows = []
    for lam in (0.5, 0.7, 0.9, 1.0):
        reranker = MMRReranker(rec.dataset, lam)
        recommendations = {}
        for user_idx in users:
            candidates, _ = rec.recommend_raw(
                user_idx,
                rec.cfg.eval.k,
                rec.dataset.seen[split].get(user_idx, set()),
                reranker=reranker,
            )
            recommendations[user_idx] = [c.item_idx for c in candidates]
        metrics = evaluate_all(
            recommendations,
            truth,
            n_items=rec.dataset.n_items,
            tail_items=rec.dataset.tail_items,
            item_popularity=rec.dataset.item_popularity,
            genre_matrix=rec.dataset.genre_matrix,
            k=rec.cfg.eval.k,
        )
        rows.append({"lambda": lam, **metrics.as_dict()})
    return rows


def main() -> None:
    cfg = load_config()
    rec = Recommender()
    valid = rec.evaluate("valid").as_dict()
    test = rec.evaluate("test").as_dict()
    snapshot = {
        "model": "framework-baseline (popular recall + MMR)",
        "dataset": cfg.data.dataset,
        "split_strategy": cfg.split.strategy,
        "rerank_lambda": cfg.rerank.lambda_,
        "valid": valid,
        "test": test,
    }
    out_dir = cfg.artifacts_dir() / "metrics"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "baseline.json").write_text(
        json.dumps(snapshot, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    tradeoff = lambda_sweep(rec)
    (out_dir / "mmr_tradeoff.json").write_text(
        json.dumps(tradeoff, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print("[evaluate] test:", json.dumps(test, ensure_ascii=False))
    print("[evaluate] MMR lambda sweep (valid subset):")
    for row in tradeoff:
        print(
            f"  lambda={row['lambda']:.2f}  recall@10={row['recall@k']:.4f}"
            f"  ndcg@10={row['ndcg@k']:.4f}  ILD={row['ild']:.4f}"
        )
    print(f"[evaluate] wrote {out_dir}/baseline.json and mmr_tradeoff.json")


if __name__ == "__main__":
    main()
