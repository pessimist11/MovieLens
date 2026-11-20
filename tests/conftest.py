"""Shared test fixtures.

ItemCF and BPR persist fitted artifacts under ``artifacts/models``. Tests must
not write into the repository, so each model's cache root is redirected to a tmp
dir while keeping the real YAML hyper-parameters.
"""

from types import SimpleNamespace

import pytest

import movielens.mf.bpr as bpr_module
import movielens.recall.itemcf as itemcf_module
import movielens.recall.two_tower as two_tower_module
from movielens.config import load_config as real_load_config


@pytest.fixture(autouse=True)
def isolate_model_caches(tmp_path, monkeypatch):
    def fake_load_config():
        cfg = real_load_config()
        return SimpleNamespace(
            raw=cfg.raw, artifacts_dir=lambda: tmp_path, project=cfg.project
        )

    monkeypatch.setattr(itemcf_module, "load_config", fake_load_config)
    monkeypatch.setattr(bpr_module, "load_config", fake_load_config)
    monkeypatch.setattr(two_tower_module, "load_config", fake_load_config)
    yield
