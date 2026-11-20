"""API contract tests. The recommender dependency is overridden, so no dataset is needed."""

from types import SimpleNamespace

import numpy as np
import pytest
from fastapi.testclient import TestClient

from movielens.api import main
from movielens.api.main import app, get_recommender


class FakeRecommender:
    cfg = SimpleNamespace(rerank=SimpleNamespace(lambda_=0.7), eval=SimpleNamespace(k=10))
    dataset = SimpleNamespace(
        n_users=10,
        n_items=3,
        idx2user={i: i + 1 for i in range(10)},
        item2idx={101: 0, 102: 1},
        item_popularity=np.array([5, 2, 1]),
        tail_items={1, 2},
        movie_row=lambda item_idx: {
            "movie_id": 101 + item_idx,
            "title": f"Movie {item_idx}",
            "genres": ["Drama"],
            "year": 2001,
        },
    )

    def stages(self):
        return {"recall": [{"name": "popular", "implemented": True}], "rank": []}

    def recommend(self, user_id, k=10, use_rerank=True, lam=None):
        return {
            "user_id": user_id,
            "cold_start": False,
            "k": k,
            "items": [
                {
                    "movie_id": 101 + i,
                    "title": f"Movie {i}",
                    "genres": ["Drama"],
                    "score": 0.9 - i * 0.1,
                    "source": "popular",
                    "rank_score": None,
                    "reason": "当前热门",
                }
                for i in range(k)
            ],
            "latency_ms": {
                "recall_ms": 1.0,
                "rank_ms": 0.0,
                "rerank_ms": 0.5,
                "total_ms": 1.5,
                "n_candidates": 20,
            },
            "sources": ["popular"],
        }


@pytest.fixture
def client(monkeypatch):
    # Avoid constructing the real pipeline (and training models) in lifespan.
    monkeypatch.setattr(main, "Recommender", lambda: FakeRecommender())
    app.dependency_overrides[get_recommender] = lambda: FakeRecommender()
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()


def test_health_and_index(client):
    assert client.get("/health").json()["status"] == "ok"
    assert "MovieLens" in client.get("/").text


def test_recommend_contract(client):
    response = client.get("/recommend/1?k=3&lam=0.5")
    assert response.status_code == 200
    body = response.json()
    assert len(body["items"]) == 3
    assert body["cold_start"] is False
    assert body["rerank_lambda"] == 0.5
    assert body["latency_ms"]["total_ms"] >= 0


def test_validation_and_misc_routes(client):
    assert client.get("/recommend/1?k=100").status_code == 422
    assert client.get("/recommend/not-a-number").status_code == 422
    assert client.get("/users?limit=5").json()["user_ids"] == [1, 2, 3, 4, 5]
    movie = client.get("/movies/101").json()
    assert movie["train_popularity"] == 5 and movie["is_long_tail"] is False
    assert client.get("/movies/999").status_code == 404
    assert client.get("/metrics").status_code == 200
