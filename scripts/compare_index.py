"""Compare exact IndexFlatIP and approximate HNSW on the same embeddings.

Run after two-tower training has populated its cache:
  .venv/bin/python scripts/compare_index.py

A fresh process is intentional on macOS: torch and faiss wheels each bundle an
OpenMP runtime, and HNSW native initialisation after torch computation can abort
the process. Loading cached embeddings performs no torch computation.
"""

from __future__ import annotations

import json
from time import perf_counter

import numpy as np

from movielens.config import load_config
from movielens.data import load_dataset
from movielens.eval.metrics import evaluate_all
from movielens.faiss_index import FaissItemIndex


def retrieve(index: FaissItemIndex, user_vectors: np.ndarray, ds, users: list[int], split: str, k: int):
    recommendations, latencies = {}, []
    for user in users:
        seen = ds.seen[split].get(user, set())
        fetch = min(ds.n_items, max(k + len(seen) + 16, k * 4))
        t0 = perf_counter()
        scores, ids = index.search(user_vectors[user][None, :], fetch)
        latencies.append((perf_counter() - t0) * 1000)
        recommendations[user] = [int(i) for i in ids[0] if i >= 0 and i not in seen][:k]
    return recommendations, np.asarray(latencies)


def main() -> None:
    ds = load_dataset()
    artifacts = load_config().artifacts_dir()
    cache = artifacts / "models" / "two_tower"
    baseline_path = artifacts / "metrics" / "baseline.json"
    fingerprint = None
    if baseline_path.exists():
        baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
        fingerprint = baseline.get("model_stats", {}).get("two_tower", {}).get("fingerprint")
    paths = sorted(cache.glob("embeddings_*.npz"), key=lambda p: p.stat().st_mtime)
    if fingerprint:
        path = cache / f"embeddings_{fingerprint}.npz"
        if not path.exists():
            raise FileNotFoundError(f"Active embedding cache not found: {path}")
    elif paths:
        path = paths[-1]  # compatibility with snapshots written before fingerprints
    else:
        raise RuntimeError("Train two-tower first, then rerun this script")
    with np.load(path) as data:
        user_vectors, item_vectors = data["user"], data["item"]
    fingerprint = path.stem.removeprefix("embeddings_")
    users = list(ds.ground_truth["test"])
    k = 200
    flat = FaissItemIndex(item_vectors, cache_dir=cache, fingerprint=fingerprint, index_type="flat")
    exact, flat_ms = retrieve(flat, user_vectors, ds, users, "test", k)
    numpy_recommendations = {}
    for user in users:
        scores = item_vectors @ user_vectors[user]
        seen = ds.seen["test"].get(user, set())
        if seen:
            scores[np.asarray(sorted(seen), dtype=np.int64)] = -np.inf
        top = np.argpartition(-scores, kth=k - 1)[:k]
        top = top[np.argsort(-scores[top], kind="stable")]
        numpy_recommendations[user] = top.tolist()
    hnsw = FaissItemIndex(item_vectors, cache_dir=cache, fingerprint=fingerprint, index_type="hnsw")
    approx, hnsw_ms = retrieve(hnsw, user_vectors, ds, users, "test", k)

    overlaps = [
        len(set(exact[u][:10]) & set(approx[u][:10])) / 10
        for u in users
    ]
    numpy_overlaps = [
        len(set(exact[u][:10]) & set(numpy_recommendations[u][:10])) / 10
        for u in users
    ]
    candidate_overlaps = [
        len(set(exact[u]) & set(approx[u])) / max(1, len(exact[u]))
        for u in users
    ]
    payload = {"index_type": "flat", "k": k}
    results = {}
    for name, recommendations, latencies in (
        ("flat", exact, flat_ms),
        ("hnsw", approx, hnsw_ms),
    ):
        metrics = evaluate_all(
            {u: rec[:10] for u, rec in recommendations.items()},
            ds.ground_truth["test"],
            n_items=ds.n_items,
            tail_items=ds.tail_items,
            item_popularity=ds.item_popularity,
            genre_matrix=ds.genre_matrix,
            k=10,
        ).as_dict()
        results[name] = {
            **metrics,
            "latency_ms_mean": round(float(latencies.mean()), 4),
            "latency_ms_p95": round(float(np.percentile(latencies, 95)), 4),
        }
    payload.update(
        {
            "results": results,
            "top10_overlap": round(float(np.mean(overlaps)), 6),
            "candidate_overlap@200": round(float(np.mean(candidate_overlaps)), 6),
            "numpy_flat_top10_overlap": round(float(np.mean(numpy_overlaps)), 6),
            "numpy_flat_exact_list_match": round(
                float(np.mean([numpy_recommendations[u] == exact[u] for u in users])), 6
            ),
            "n_users": len(users),
        }
    )
    out = load_config().artifacts_dir() / "metrics" / "faiss_index.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    print(f"[compare_index] wrote {out}")


if __name__ == "__main__":
    main()
