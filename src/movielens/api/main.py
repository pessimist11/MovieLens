"""FastAPI entry point.

Run:  .venv/bin/python -m uvicorn movielens.api.main:app --reload
"""

from __future__ import annotations

import json
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse

from ..config import ROOT, load_config
from ..pipeline import Recommender

WEB_DIR = Path(__file__).resolve().parents[1] / "web"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the dataset, models and index once at startup."""
    try:
        app.state.recommender = Recommender()
        app.state.load_error = None
    except Exception as exc:  # noqa: BLE001 - keep the service alive with a clear hint
        app.state.recommender = None
        app.state.load_error = str(exc)
    yield


app = FastAPI(
    title="MovieLens recommender",
    version="0.1.0",
    description="Recall -> rank -> rerank pipeline with FastAPI",
    lifespan=lifespan,
)


def get_recommender(request: Request) -> Recommender:
    rec = getattr(request.app.state, "recommender", None)
    if rec is None:
        raise HTTPException(
            status_code=503,
            detail=(
                "Dataset not prepared. Run: "
                ".venv/bin/python -m movielens.data.prepare "
                f"(startup error: {getattr(request.app.state, 'load_error', None)})"
            ),
        )
    return rec


@app.get("/")
def index():
    return FileResponse(WEB_DIR / "index.html")


@app.get("/health")
def health(request: Request):
    rec = getattr(request.app.state, "recommender", None)
    return {
        "status": "ok",
        "ready": rec is not None,
        "dataset": load_config().data.dataset,
        "stages": rec.stages() if rec is not None else None,
        "n_users": rec.dataset.n_users if rec is not None else None,
        "n_items": rec.dataset.n_items if rec is not None else None,
        "load_error": getattr(request.app.state, "load_error", None),
    }


@app.get("/recommend/{user_id}")
def recommend(
    user_id: int,
    k: int = Query(10, ge=1, le=50),
    use_rerank: bool = Query(True),
    lam: float | None = Query(None, ge=0.0, le=1.0, description="MMR relevance weight"),
    rec: Recommender = Depends(get_recommender),
):
    result = rec.recommend(user_id, k=k, use_rerank=use_rerank, lam=lam)
    result["rerank_lambda"] = lam if lam is not None else rec.cfg.rerank.lambda_
    result["use_rerank"] = use_rerank
    return result


@app.get("/users")
def users(limit: int = Query(20, ge=1, le=200), rec: Recommender = Depends(get_recommender)):
    """Sample raw userIds for the demo dropdown."""
    ids = [rec.dataset.idx2user[i] for i in range(min(limit, rec.dataset.n_users))]
    return {"user_ids": ids}


@app.get("/movies/{movie_id}")
def movie(movie_id: int, rec: Recommender = Depends(get_recommender)):
    ds = rec.dataset
    item_idx = ds.item2idx.get(int(movie_id))
    if item_idx is None:
        raise HTTPException(status_code=404, detail=f"movieId {movie_id} not found")
    return {
        **ds.movie_row(item_idx),
        "item_idx": item_idx,
        "train_popularity": int(ds.item_popularity[item_idx]),
        "is_long_tail": item_idx in ds.tail_items,
    }


@app.get("/metrics")
def metrics():
    path = ROOT / "artifacts" / "metrics" / "baseline.json"
    if not path.exists():
        return {
            "status": "not_evaluated",
            "hint": "Run: .venv/bin/python scripts/evaluate.py",
        }
    return json.loads(path.read_text(encoding="utf-8"))
