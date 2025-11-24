"""Minimal loader to read attribution arrays from a given folder.

This loader assumes `numpy_dir` directly contains .npy files named as:
  <prefix>_<METHOD>_obj_trajs.npy
  <prefix>_<METHOD>_map_polylines.npy

It selects files by `scenario_id` (preferred) or `batch_id`, using wildcards
for the method part. No metadata.json is used.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import numpy as np


@dataclass
class AttributionData:
    """Container for attribution arrays."""

    obj_attr: Optional[np.ndarray]
    map_attr: Optional[np.ndarray]
    prefix: str
    scenario_id: Optional[str]
    batch_id: Optional[int]


class AttributionLoader:
    """Load attribution `.npy` files from a single folder based on prefix."""

    def __init__(self, numpy_dir: Path) -> None:
        # Folder that directly contains the .npy arrays
        self.numpy_dir = numpy_dir
        self._cache: Dict[str, AttributionData] = {}

    def _extract_prefix_from_stem(self, stem: str) -> str:
        # Expect <prefix>_<method>_obj_trajs or <prefix>_<method>_map_polylines
        parts = stem.split("_")
        if len(parts) >= 3:
            return "_".join(parts[:-3])
        return stem

    def resolve_prefix(
        self,
        scenario_id: Optional[str] = None,
        batch_id: Optional[int] = None,
        element_idx: int = 0,
    ) -> Optional[str]:
        if scenario_id:
            pattern = f"scene_{scenario_id}_*_obj_trajs.npy"
            for path in self.numpy_dir.glob(pattern):
                return self._extract_prefix_from_stem(path.stem)

        if batch_id is not None:
            pattern = f"batch_{batch_id}*_*_obj_trajs.npy"
            for path in self.numpy_dir.glob(pattern):
                return self._extract_prefix_from_stem(path.stem)

        return None

    def load(
        self,
        scenario_id: Optional[str] = None,  # 场景ID，可选参数
        batch_id: Optional[int] = None,     # 批次ID，可选参数
        element_idx: int = 0,               # 元素索引，默认为0
    ) -> Optional[AttributionData]:        # 返回类型为AttributionData或None
        prefix = self.resolve_prefix(scenario_id, batch_id, element_idx)
        if not prefix:
            return None

        cache_key = prefix
        if cache_key in self._cache:
            return self._cache[cache_key]

        obj_candidates = list(self.numpy_dir.glob(f"{prefix}_*_obj_trajs.npy"))
        map_candidates = list(self.numpy_dir.glob(f"{prefix}_*_map_polylines.npy"))
        obj_file = obj_candidates[0] if obj_candidates else Path("/nonexistent")
        map_file = map_candidates[0] if map_candidates else Path("/nonexistent")

        obj_attr = self._safe_load(obj_file)
        map_attr = self._safe_load(map_file)

        data = AttributionData(
            obj_attr=obj_attr,
            map_attr=map_attr,
            prefix=prefix,
            scenario_id=scenario_id,
            batch_id=batch_id,
        )
        self._cache[cache_key] = data
        return data

    @staticmethod
    def _safe_load(path: Path) -> Optional[np.ndarray]:
        """Load a NumPy array and surface any issues instead of silently failing."""
        if not path.exists():
            raise FileNotFoundError(f"Attribution file not found: {path}")
        # Let numpy raise any errors so that they are visible to the caller.
        return np.load(path)
