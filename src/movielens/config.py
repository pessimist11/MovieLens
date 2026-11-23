"""Typed configuration loading.

All paths are resolved relative to the repository root, so scripts work no
matter which directory they are launched from.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = ROOT / "configs" / "default.yaml"


@dataclass(frozen=True)
class ProjectCfg:
    seed: int = 42


@dataclass(frozen=True)
class DataCfg:
    dataset: str = "ml-latest-small"
    raw_dir: str = "data/raw"
    processed_dir: str = "data/processed"
    min_user_interactions: int = 5
    min_item_interactions: int = 1
    positive_threshold: float = 0.0


@dataclass(frozen=True)
class SplitCfg:
    strategy: str = "global"
    train_ratio: float = 0.8
    valid_ratio: float = 0.1
    negatives_per_positive: int = 4
    hard_negative_ratio: float = 0.5


@dataclass(frozen=True)
class EvalCfg:
    k: int = 10
    tail_quantile: float = 0.8


@dataclass(frozen=True)
class RecallCfg:
    topk_per_source: int = 200
    popularity_decay: float = 0.0
    min_primary_candidates: int = 50


@dataclass(frozen=True)
class RankCfg:
    model: str = "none"
    snapshot_fraction: float = 0.75
    snapshot_bpr_epochs: int = 15
    snapshot_two_tower_epochs: int = 20
    negatives_per_user: int = 96
    dim: int = 16
    hidden_dim: int = 64
    history_size: int = 20
    popularity_half_life_days: float = 180.0
    epochs: int = 12
    patience: int = 4
    batch_size: int = 1024
    lr: float = 0.001


@dataclass(frozen=True)
class RerankCfg:
    enabled: bool = True
    lambda_: float = 0.7


@dataclass(frozen=True)
class ApiCfg:
    host: str = "127.0.0.1"
    port: int = 8000


def _build(cls: type, values: dict[str, Any]):
    """Instantiate a dataclass from a dict, ignoring unknown keys."""
    known = {f.name for f in dataclasses.fields(cls)}
    kwargs: dict[str, Any] = {}
    for key, value in values.items():
        # YAML uses ``lambda`` (a Python keyword); map it to ``lambda_``.
        key = "lambda_" if key == "lambda" else key
        if key in known:
            kwargs[key] = value
    return cls(**kwargs)


@dataclass(frozen=True)
class Config:
    project: ProjectCfg = field(default_factory=ProjectCfg)
    data: DataCfg = field(default_factory=DataCfg)
    split: SplitCfg = field(default_factory=SplitCfg)
    eval: EvalCfg = field(default_factory=EvalCfg)
    recall: RecallCfg = field(default_factory=RecallCfg)
    rank: RankCfg = field(default_factory=RankCfg)
    rerank: RerankCfg = field(default_factory=RerankCfg)
    api: ApiCfg = field(default_factory=ApiCfg)
    raw: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_yaml(cls, path: str | Path | None = None) -> "Config":
        path = Path(path) if path else DEFAULT_CONFIG_PATH
        with path.open("r", encoding="utf-8") as fh:
            values = yaml.safe_load(fh) or {}
        return cls(
            project=_build(ProjectCfg, values.get("project", {})),
            data=_build(DataCfg, values.get("data", {})),
            split=_build(SplitCfg, values.get("split", {})),
            eval=_build(EvalCfg, values.get("eval", {})),
            recall=_build(RecallCfg, values.get("recall", {})),
            rank=_build(RankCfg, values.get("rank", {})),
            rerank=_build(RerankCfg, values.get("rerank", {})),
            api=_build(ApiCfg, values.get("api", {})),
            raw=values,
        )

    @property
    def root(self) -> Path:
        return ROOT

    def raw_dir(self) -> Path:
        return ROOT / self.data.raw_dir

    def processed_dir(self) -> Path:
        return ROOT / self.data.processed_dir

    def artifacts_dir(self) -> Path:
        return ROOT / "artifacts"


def load_config(path: str | Path | None = None) -> Config:
    return Config.from_yaml(path)
