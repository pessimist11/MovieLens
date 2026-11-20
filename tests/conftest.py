"""Shared test fixtures.

ItemCF persists its similarity matrix under ``artifacts/models``. Tests must not
write into the repository, so the cache root is redirected to a tmp dir while
keeping the real YAML hyper-parameters.
"""

from types import SimpleNamespace

import pytest

import movielens.recall.itemcf as itemcf_module
from movielens.config import load_config as real_load_config


@pytest.fixture(autouse=True)
def isolate_itemcf_cache(tmp_path, monkeypatch):
    def fake_load_config():
        cfg = real_load_config()
        return SimpleNamespace(raw=cfg.raw, artifacts_dir=lambda: tmp_path)

    monkeypatch.setattr(itemcf_module, "load_config", fake_load_config)
    yield
