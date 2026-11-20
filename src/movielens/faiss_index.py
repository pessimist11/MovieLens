"""FAISS-only item index, deliberately independent from torch.

Keeping this module free of torch imports lets exact/approximate index tests and
benchmarks run in a clean native process. It also makes the retrieval component
reusable by future embedding models.
"""

from __future__ import annotations

from pathlib import Path

import faiss
import numpy as np


class FaissItemIndex:
    def __init__(
        self,
        vectors: np.ndarray,
        *,
        cache_dir: str | Path,
        fingerprint: str,
        index_type: str = "flat",
        hnsw_m: int = 32,
        hnsw_ef_construction: int = 80,
        hnsw_ef_search: int = 64,
        use_cache: bool = True,
    ):
        self.vectors = np.ascontiguousarray(vectors, dtype=np.float32)
        self.cache_dir = Path(cache_dir)
        self.fingerprint = fingerprint
        self.hnsw_m = int(hnsw_m)
        self.hnsw_ef_construction = int(hnsw_ef_construction)
        self.hnsw_ef_search = int(hnsw_ef_search)
        self.use_cache = bool(use_cache)
        self.index = None
        self.index_type = ""
        self.build(index_type)

    def _path(self, index_type: str) -> Path:
        return self.cache_dir / f"{index_type}_{self.fingerprint}.faiss"

    def build(self, index_type: str = "flat") -> None:
        index_type = index_type.lower()
        path = self._path(index_type)
        if self.use_cache and path.exists():
            index = faiss.read_index(str(path))
        else:
            if index_type == "flat":
                base = faiss.IndexFlatIP(self.vectors.shape[1])
            elif index_type == "hnsw":
                base = faiss.IndexHNSWFlat(
                    self.vectors.shape[1], self.hnsw_m, faiss.METRIC_INNER_PRODUCT
                )
                base.hnsw.efConstruction = self.hnsw_ef_construction
            else:
                raise ValueError("index_type must be 'flat' or 'hnsw'")
            # FAISS result id == continuous item_idx, including after reloading.
            index = faiss.IndexIDMap2(base)
            index.add_with_ids(
                self.vectors, np.arange(len(self.vectors), dtype=np.int64)
            )
            if self.use_cache:
                path.parent.mkdir(parents=True, exist_ok=True)
                faiss.write_index(index, str(path))
        if index_type == "hnsw":
            faiss.downcast_index(index.index).hnsw.efSearch = self.hnsw_ef_search
        self.index, self.index_type = index, index_type

    def search(self, query: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        query = np.ascontiguousarray(query, dtype=np.float32)
        return self.index.search(query, k)

    def save(self, path: str | Path) -> None:
        faiss.write_index(self.index, str(path))

    @property
    def ntotal(self) -> int:
        return int(self.index.ntotal)
