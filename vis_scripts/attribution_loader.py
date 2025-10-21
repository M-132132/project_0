"""Utilities for loading attribution results and metadata."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

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
    """Load attribution `.npy` files with scenario-aware mapping."""

    def __init__(self, numpy_dir: Path, method: str) -> None:
        self.method = method

        if numpy_dir.name == "numpy":
            self.numpy_dir = numpy_dir
            self.attr_dir = numpy_dir.parent
        else:
            self.attr_dir = numpy_dir
            self.numpy_dir = numpy_dir / "numpy"

        self._cache: Dict[str, AttributionData] = {}
        self._scenario_to_prefix: Dict[str, str] = {}
        self._batch_to_prefix: Dict[Tuple[int, int], str] = {}

        self._load_metadata()

    def _load_metadata(self) -> None:
        metadata_file = self.attr_dir / "metadata.json"
        if not metadata_file.exists():
            return

        try:
            records = json.loads(metadata_file.read_text(encoding="utf-8"))
        except Exception:
            return

        if not isinstance(records, list):
            return

        for record in records:
            if not isinstance(record, dict):
                continue

            batch_id = record.get("batch_id")
            prefixes = record.get("file_prefixes") or []
            scenarios = record.get("scenario_ids") or []

            for idx, prefix in enumerate(prefixes):
                if not prefix:
                    continue

                if batch_id is not None:
                    self._batch_to_prefix[(batch_id, idx)] = prefix

                if idx < len(scenarios):
                    scenario_id = scenarios[idx]
                    if scenario_id:
                        self._scenario_to_prefix[str(scenario_id)] = prefix

    def resolve_prefix(
        self,
        scenario_id: Optional[str] = None,
        batch_id: Optional[int] = None,
        element_idx: int = 0,
    ) -> Optional[str]:
        if scenario_id and scenario_id in self._scenario_to_prefix:
            return self._scenario_to_prefix[scenario_id]

        if batch_id is not None:
            key = (batch_id, element_idx)
            if key in self._batch_to_prefix:
                return self._batch_to_prefix[key]

        if scenario_id:
            candidate = self.numpy_dir.glob(f"scene_{scenario_id}_{self.method}_obj_trajs.npy")
            try:
                path = next(candidate)
                return path.stem.replace(f"_{self.method}_obj_trajs", "")
            except StopIteration:
                pass

        if batch_id is not None:
            candidate = self.numpy_dir.glob(f"batch_{batch_id}*_{self.method}_obj_trajs.npy")
            try:
                path = next(candidate)
                return path.stem.replace(f"_{self.method}_obj_trajs", "")
            except StopIteration:
                pass

        return None

    def load(
        self,
        scenario_id: Optional[str] = None,
        batch_id: Optional[int] = None,
        element_idx: int = 0,
    ) -> Optional[AttributionData]:
        prefix = self.resolve_prefix(scenario_id, batch_id, element_idx)
        if not prefix:
            return None

        cache_key = f"{prefix}:{self.method}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        obj_file = self.numpy_dir / f"{prefix}_{self.method}_obj_trajs.npy"
        map_file = self.numpy_dir / f"{prefix}_{self.method}_map_polylines.npy"

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
        if not path.exists():
            return None
        try:
            return np.load(path)
        except Exception:
            return None
