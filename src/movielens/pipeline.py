"""The single pipeline used by BOTH offline evaluation and online serving.

Keeping one code path here is deliberate: most train/serve skew bugs come from
having a separate "research" pipeline and a separate "production" pipeline.
"""

from __future__ import annotations

from collections import defaultdict
from time import perf_counter

from .config import Config, load_config
from .data import Dataset, load_dataset
from .eval.metrics import evaluate_all
from .rank import RANK_REGISTRY
from .recall import RECALL_REGISTRY
from .rerank import MMRReranker
from .schema import Candidate, Metrics, Recommendation

RRF_CONSTANT = 60.0


class Recommender:
    def __init__(self, dataset: Dataset | None = None, cfg: Config | None = None):
        self.cfg = cfg or load_config()
        self.dataset = dataset or load_dataset()
        self.recall_models = [
            cls(self.dataset) for cls in RECALL_REGISTRY if cls.implemented
        ]
        self.ranker = next(
            (cls(self.dataset) for cls in RANK_REGISTRY if cls.implemented), None
        )
        self.reranker = (
            MMRReranker(self.dataset, self.cfg.rerank.lambda_)
            if self.cfg.rerank.enabled
            else None
        )
        self._genre_sets = [
            set() if str(g) in ("(no genres listed)", "nan", "")
            else set(str(g).split("|"))
            for g in self.dataset.movies["genres"].tolist()
        ]

    def stages(self) -> dict:
        """Report which stages are wired in, for /health and the README roadmap."""
        return {
            "recall": [
                {
                    "name": cls.name,
                    "implemented": cls.implemented,
                    "role": next(
                        (
                            "fallback" if m.fallback_only else "primary"
                            for m in self.recall_models
                            if m.name == cls.name
                        ),
                        None,
                    ),
                }
                for cls in RECALL_REGISTRY
            ],
            "rank": [
                {"name": cls.name, "implemented": cls.implemented}
                for cls in RANK_REGISTRY
            ],
            "rerank": {
                "name": MMRReranker.name,
                "implemented": self.reranker is not None,
                "lambda": self.cfg.rerank.lambda_,
            },
        }

    def _rrf(self, models, user_idx: int, exclude: set[int]) -> list[Candidate]:
        """Reciprocal Rank Fusion over the given sources.

        RRF avoids comparing raw scores from models with different scales;
        each source contributes ``weight / (60 + rank)`` per candidate.
        """
        weights = self.cfg.raw.get("recall", {}).get("source_weights", {}) or {}
        fused: dict[int, float] = defaultdict(float)
        sources: dict[int, set[str]] = defaultdict(set)
        for model in models:
            weight = float(weights.get(model.name, 1.0))
            for rank, cand in enumerate(
                model.recall(user_idx, self.cfg.recall.topk_per_source, exclude)
            ):
                fused[cand.item_idx] += weight / (RRF_CONSTANT + rank + 1)
                sources[cand.item_idx].add(cand.source)
        merged = [
            Candidate(
                item_idx=i,
                score=score,
                source="+".join(sorted(sources[i])),
            )
            for i, score in fused.items()
        ]
        merged.sort(key=lambda c: c.score, reverse=True)
        return merged

    def recall(self, user_idx: int, exclude: set[int]) -> list[Candidate]:
        """Fuse personalised sources, then backfill with fallback sources.

        Equal-weight fusion with a popularity source measurably hurts both
        coverage and recall (see artifacts/metrics/ablation.json): blockbusters
        always sit at rank 0 of the popular list, so they outrank personalised
        candidates. Popularity therefore only fills the tail of the candidate
        pool, or the whole pool for cold-start users.
        """
        primary = [m for m in self.recall_models if not m.fallback_only]
        fallback = [m for m in self.recall_models if m.fallback_only]
        merged = self._rrf(primary, user_idx, exclude)
        if not fallback:
            return merged
        if not merged:
            return self._rrf(fallback, user_idx, exclude)
        if len(merged) >= self.cfg.recall.min_primary_candidates:
            return merged

        # Backfill BELOW every personalised candidate: fallback items must not
        # outrank them, only guarantee enough material for ranking / MMR.
        present = {c.item_idx for c in merged}
        floor = min(c.score for c in merged) * 0.5
        extras = [
            Candidate(item_idx=c.item_idx, score=floor * (0.99**rank), source=c.source)
            for rank, c in enumerate(self._rrf(fallback, user_idx, exclude))
            if c.item_idx not in present
        ]
        return merged + extras

    def recommend_raw(
        self,
        user_idx: int,
        k: int,
        seen: set[int],
        use_rerank: bool = True,
        reranker=None,
    ) -> tuple[list[Candidate], dict]:
        """Return candidates + per-stage latency. Shared by evaluation and API."""
        t0 = perf_counter()
        candidates = self.recall(user_idx, seen)
        t1 = perf_counter()

        if self.ranker is not None:
            candidates = self.ranker.predict(user_idx, candidates)
            candidates.sort(
                key=lambda c: c.rank_score if c.rank_score is not None else c.score,
                reverse=True,
            )
        t2 = perf_counter()

        active_reranker = reranker if reranker is not None else self.reranker
        if use_rerank and active_reranker is not None:
            final = active_reranker.rerank(candidates[:50], k)
        else:
            final = candidates[:k]
        t3 = perf_counter()

        latency = {
            "recall_ms": round((t1 - t0) * 1000, 2),
            "rank_ms": round((t2 - t1) * 1000, 2),
            "rerank_ms": round((t3 - t2) * 1000, 2),
            "total_ms": round((t3 - t0) * 1000, 2),
            "n_candidates": len(candidates),
        }
        return final, latency

    def _reason(self, user_idx: int | None, item_idx: int, source: str = "") -> str:
        if user_idx is None or user_idx < 0:
            return "新用户兜底：当前热门"
        # Collaborative filtering can point at the exact neighbour that fired,
        # which is a much stronger explanation than a genre overlap guess.
        if "itemcf" in source:
            for model in self.recall_models:
                if model.name == "itemcf" and hasattr(model, "explain"):
                    found = model.explain(user_idx, item_idx)
                    if found is not None:
                        history_item, _ = found
                        title = self.dataset.movie_row(history_item)["title"]
                        return f"与你喜欢的《{title}》常被一起观看"
        history = self.dataset.seen_for(user_idx, "all")
        target = self._genre_sets[item_idx]
        if not history or not target:
            return "当前热门"
        best_item, best_overlap = None, 0
        for seen_item in history:
            overlap = len(target & self._genre_sets[seen_item])
            if overlap > best_overlap:
                best_item, best_overlap = seen_item, overlap
        if best_item is None:
            return "当前热门"
        return f"因为你喜欢《{self.dataset.movie_row(best_item)['title']}》"

    def recommend(
        self,
        user_id: int,
        k: int = 10,
        use_rerank: bool = True,
        lam: float | None = None,
    ) -> dict:
        """Public API: raw userId in, human-readable recommendation list out."""
        user_idx = self.dataset.user_idx_of(user_id)
        cold_start = user_idx is None
        seen = set() if cold_start else self.dataset.seen_for(user_idx, "all")
        reranker = (
            MMRReranker(self.dataset, lam)
            if lam is not None and self.reranker is not None
            else None
        )
        candidates, latency = self.recommend_raw(
            user_idx if user_idx is not None else -1,
            k,
            seen,
            use_rerank,
            reranker=reranker,
        )

        items: list[Recommendation] = []
        for cand in candidates:
            row = self.dataset.movie_row(cand.item_idx)
            items.append(
                Recommendation(
                    movie_id=row["movie_id"],
                    title=row["title"],
                    genres=row["genres"],
                    score=round(cand.score, 6),
                    source=cand.source,
                    rank_score=cand.rank_score,
                    reason=self._reason(user_idx, cand.item_idx, cand.source),
                )
            )
        return {
            "user_id": int(user_id),
            "cold_start": cold_start,
            "k": k,
            "items": [vars(i) for i in items],
            "latency_ms": latency,
            "sources": [m.name for m in self.recall_models],
        }

    def evaluate(self, split: str = "test", k: int | None = None) -> Metrics:
        """Full-pipeline offline evaluation on the given time split."""
        k = k or self.cfg.eval.k
        truth = self.dataset.ground_truth[split]
        seen_by_split = self.dataset.seen[split]
        recommendations: dict[int, list[int]] = {}
        for user_idx in truth:
            candidates, _ = self.recommend_raw(
                user_idx, k, seen_by_split.get(user_idx, set()), use_rerank=True
            )
            recommendations[user_idx] = [c.item_idx for c in candidates]
        return evaluate_all(
            recommendations,
            truth,
            n_items=self.dataset.n_items,
            tail_items=self.dataset.tail_items,
            item_popularity=self.dataset.item_popularity,
            genre_matrix=self.dataset.genre_matrix,
            k=k,
        )
