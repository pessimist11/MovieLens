"""Full-pipeline offline evaluation: ablation table + MMR trade-off scan.

Every variant is evaluated on the SAME test set, with the SAME k and the SAME
seen filtering, so the numbers are comparable across layers.

Run:  .venv/bin/python scripts/evaluate.py [--rank] [--no-sweep]
Output: artifacts/metrics/{baseline.json, ablation.json, mmr_tradeoff.json}
"""

from __future__ import annotations

import argparse
import copy
import json

from movielens.config import load_config
from movielens.eval.metrics import evaluate_all
from movielens.pipeline import MMR_CANDIDATE_LIMIT, Recommender
from movielens.rank import RANK_REGISTRY
from movielens.rerank import MMRReranker

# The incremental story of the project. The last two rows are the interesting
# comparison: popularity as a backfill vs popularity inside equal-weight RRF.
DEFAULT_VARIANT = "itemcf+bpr+two_tower+popular_fallback+mmr"
DEFAULT_SOURCES = ["itemcf", "bpr", "two_tower", "popular"]

ABLATIONS = [
    {"label": "popular", "sources": ["popular"], "mmr": False},
    {"label": "popular+mmr", "sources": ["popular"], "mmr": True},
    {"label": "itemcf", "sources": ["itemcf"], "mmr": False},
    {"label": "itemcf+mmr", "sources": ["itemcf"], "mmr": True},
    {"label": "bpr", "sources": ["bpr"], "mmr": False},
    {"label": "bpr+mmr", "sources": ["bpr"], "mmr": True},
    {"label": "two_tower", "sources": ["two_tower"], "mmr": False},
    {"label": "two_tower+mmr", "sources": ["two_tower"], "mmr": True},
    {"label": "itemcf+bpr+mmr", "sources": ["itemcf", "bpr"], "mmr": True},
    {
        "label": "itemcf+bpr+two_tower+mmr",
        "sources": ["itemcf", "bpr", "two_tower"],
        "mmr": True,
    },
    {
        "label": DEFAULT_VARIANT,
        "sources": DEFAULT_SOURCES,
        "mmr": True,
    },
    {"label": "itemcf+popular_fallback+mmr", "sources": ["itemcf", "popular"], "mmr": True},
    {
        "label": "rrf_equal(popular,itemcf)+mmr",
        "sources": ["popular", "itemcf"],
        "mmr": True,
        "fallback": False,
    },
]


def build_variant(
    base: Recommender,
    sources: list[str],
    use_mmr: bool,
    lam: float | None = None,
    fallback: bool = True,
    ranker=None,
):
    """Shallow-copy the pipeline and choose each variant's stages explicitly."""
    rec = copy.copy(base)
    available = {m.name: m for m in base.recall_models}
    missing = [s for s in sources if s not in available]
    if missing:
        raise ValueError(f"Recall sources not implemented/loaded: {missing}")
    models = []
    for source in sources:
        model = copy.copy(available[source])
        if source == "popular":
            model.fallback_only = fallback
        models.append(model)
    rec.recall_models = models
    rec.ranker = ranker
    if use_mmr:
        rec.reranker = MMRReranker(base.dataset, lam if lam is not None else base.cfg.rerank.lambda_)
    else:
        rec.reranker = None
    return rec


def lambda_sweep(rec: Recommender, max_users: int | None = 100) -> list[dict]:
    split = "valid"
    truth_all = rec.dataset.ground_truth[split]
    users = list(truth_all)[:max_users]
    truth = {u: truth_all[u] for u in users}
    unreranked = {
        u: rec.recommend_raw(
            u, MMR_CANDIDATE_LIMIT, rec.dataset.seen_for(u, split), use_rerank=False
        )[0]
        for u in users
    }
    rows = []
    for lam in (0.5, 0.7, 0.9, 1.0):
        reranker = MMRReranker(rec.dataset, lam)
        recommendations = {
            u: [c.item_idx for c in reranker.rerank(candidates, rec.cfg.eval.k)]
            for u, candidates in unreranked.items()
        }
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
    parser = argparse.ArgumentParser(description="Offline ablation evaluation")
    parser.add_argument("--no-sweep", action="store_true", help="skip the MMR lambda sweep")
    parser.add_argument("--rank", action="store_true", help="include trained DIN and DeepFM")
    args = parser.parse_args()

    cfg = load_config()
    base = Recommender()
    rankers = {}
    if args.rank:
        rankers = {cls.name: base.load_ranker(cls.name) for cls in RANK_REGISTRY if cls.implemented}
    elif base.ranker is not None:
        rankers[base.ranker.name] = base.ranker
    variants = ABLATIONS + [
        {
            "label": f"itemcf+bpr+two_tower+popular_fallback+{name}" + ("+mmr" if mmr else ""),
            "sources": DEFAULT_SOURCES, "mmr": mmr, "rank": name,
        }
        for name in rankers for mmr in (False, True)
    ]
    out_dir = cfg.artifacts_dir() / "metrics"
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for spec in variants:
        rec = build_variant(
            base, spec["sources"], spec["mmr"], fallback=spec.get("fallback", True),
            ranker=rankers.get(spec.get("rank")),
        )
        row = {
            "variant": spec["label"],
            "sources": spec["sources"],
            "rank": spec.get("rank", "none"),
            "mmr": spec["mmr"],
            "popular_as_fallback": spec.get("fallback", True),
            "rerank_lambda": cfg.rerank.lambda_ if spec["mmr"] else None,
            "valid": rec.evaluate("valid").as_dict(),
            "test": rec.evaluate("test").as_dict(),
        }
        rows.append(row)
        t = row["test"]
        print(
            f"[ablation] {spec['label']:28s} recall@10={t['recall@k']:.4f}"
            f"  ndcg@10={t['ndcg@k']:.4f}  coverage={t['coverage']:.4f}"
            f"  tail_cov={t['long_tail_coverage']:.4f}  tail_share={t['long_tail_share']:.4f}"
            f"  ILD={t['ild']:.4f}"
        )
    (out_dir / "ablation.json").write_text(
        json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    default = next(
        r for r in rows if r["sources"] == DEFAULT_SOURCES
        and r["rank"] == cfg.rank.model and r["mmr"] == cfg.rerank.enabled
    ) if cfg.rank.model != "none" else next(r for r in rows if r["variant"] == DEFAULT_VARIANT)
    snapshot = {
        "model": default["variant"],
        "dataset": cfg.data.dataset,
        "split_strategy": cfg.split.strategy,
        "rerank_lambda": cfg.rerank.lambda_,
        "recall_sources": [m.name for m in base.recall_models],
        "model_stats": {m.name: getattr(m, "stats", {}) for m in base.recall_models},
        "rank_stats": {name: ranker.stats for name, ranker in rankers.items()},
        "valid": default["valid"],
        "test": default["test"],
    }
    (out_dir / "baseline.json").write_text(
        json.dumps(snapshot, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    if not args.no_sweep:
        if rankers:
            tradeoff = [
                {"rank": name, **row}
                for name, ranker in [("none", None), *rankers.items()]
                for row in lambda_sweep(
                    build_variant(base, DEFAULT_SOURCES, True, ranker=ranker),
                    max_users=None,
                )
            ]
        else:
            tradeoff = lambda_sweep(base)
        (out_dir / "mmr_tradeoff.json").write_text(
            json.dumps(tradeoff, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print("[evaluate] MMR lambda sweep (valid):")
        for row in tradeoff:
            print(
                f"  {row.get('rank', 'none')} lambda={row['lambda']:.2f}  recall@10={row['recall@k']:.4f}"
                f"  ndcg@10={row['ndcg@k']:.4f}  ILD={row['ild']:.4f}"
            )
    print(f"[evaluate] wrote ablation.json / baseline.json"
          f"{'' if args.no_sweep else ' / mmr_tradeoff.json'} in {out_dir}")


if __name__ == "__main__":
    main()
