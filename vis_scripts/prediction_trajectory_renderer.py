from pathlib import Path
from typing import Dict, Iterable, List, Optional

import json
import numpy as np
from metadrive.utils.utils import import_pygame

pygame, _ = import_pygame()


class PredictionTrajectoryRenderer:
    def __init__(
        self,
        prediction_path: Optional[str] = None,
        max_modes_per_object: Optional[int] = None,
        min_probability: float = 0.0,
    ) -> None:
        self.prediction_path = Path(prediction_path) if prediction_path else None
        self.max_modes_per_object = max_modes_per_object
        self.min_probability = float(min_probability)

        self._prediction_index: Dict[str, List[Dict]] = {}
        self._current_predictions: List[Dict] = []
        self.current_frame = 0

        if self.prediction_path:
            self._load_prediction_data()

        self.config = {
            "line_width": 2,
            "dash_length": 12,
            "dash_gap": 6,
            "colors": [
                (255, 120, 120),
                (255, 200, 120),
                (120, 200, 255),
                (180, 130, 255),
            ],
        }

    def reset(self, env, target_track_id: Optional[str] = None) -> None:
        self.current_frame = 0
        self._current_predictions = []

        if not self._prediction_index:
            return

        scenario_id = self._resolve_scenario_id(env)
        if scenario_id is None:
            return

        candidates = self._prediction_index.get(scenario_id, [])
        if not candidates:
            return

        target_id = self._normalise_id(target_track_id)
        filtered = [
            item
            for item in candidates
            if self._match_object_id(item.get("object_id"), target_id)
        ]

        if not filtered:
            return

        self._current_predictions = self._limit_prediction_modes(filtered)

    def update_frame(self) -> None:
        self.current_frame += 1

    def draw_trajectory(self, env) -> None:
        if not self._current_predictions:
            return

        renderer = getattr(env.engine, "top_down_renderer", None)
        if not renderer or not hasattr(renderer, "screen_canvas"):
            return

        screen_surface = renderer.screen_canvas
        frame_surface = renderer._frame_canvas
        offset = self._calculate_camera_offset(renderer, screen_surface, frame_surface)

        for idx, prediction in enumerate(self._current_predictions):
            trajectory = prediction.get("trajectory")
            if not trajectory or len(trajectory) < 2:
                continue

            screen_points = [
                self._world_to_screen(point, frame_surface, offset)
                for point in trajectory
            ]

            color = self.config["colors"][idx % len(self.config["colors"])]
            self._draw_dashed_polyline(screen_surface, screen_points, color)

            start_point = screen_points[0]
            pygame.draw.circle(screen_surface, color, start_point, 4, 1)

    def _load_prediction_data(self) -> None:
        if not self.prediction_path or not self.prediction_path.exists():
            self._prediction_index.clear()
            return

        try:
            raw = json.loads(self.prediction_path.read_text(encoding="utf-8"))
        except Exception:
            self._prediction_index.clear()
            return

        items = raw.get("predictions") if isinstance(raw, dict) else None
        if not isinstance(items, list):
            self._prediction_index.clear()
            return

        index: Dict[str, List[Dict]] = {}
        for entry in items:
            for flattened in self._flatten_entry(entry):
                scenario_key = flattened.get("scenario_id")
                if scenario_key is None:
                    continue
                index.setdefault(str(scenario_key), []).append(flattened)

        self._prediction_index = index

    def _flatten_entry(self, entry: Dict) -> Iterable[Dict]:
        scenario_id = entry.get("scenario_id")
        object_id = self._normalise_id(entry.get("object_id") or entry.get("track_id"))
        if not scenario_id or not object_id:
            return []

        trajectories = (
            entry.get("predicted_trajectory")
            or entry.get("trajectories")
            or []
        )
        probabilities = (
            entry.get("predicted_probability")
            or entry.get("probabilities")
            or []
        )

        results: List[Dict] = []
        for mode_idx, trajectory in enumerate(trajectories):
            if not trajectory:
                continue

            try:
                trajectory_2d = [
                    [float(point[0]), float(point[1])]
                    for point in trajectory
                    if isinstance(point, (list, tuple)) and len(point) >= 2
                ]
            except (TypeError, ValueError):
                continue

            if len(trajectory_2d) < 2:
                continue

            probability = None
            if mode_idx < len(probabilities):
                try:
                    probability = float(probabilities[mode_idx])
                except (TypeError, ValueError):
                    probability = None

            if probability is not None and probability < self.min_probability:
                continue

            results.append(
                {
                    "scenario_id": str(scenario_id),
                    "object_id": object_id,
                    "mode_index": mode_idx,
                    "probability": probability,
                    "trajectory": trajectory_2d,
                }
            )

        return results

    def _limit_prediction_modes(self, predictions: List[Dict]) -> List[Dict]:
        if self.max_modes_per_object is None:
            return predictions

        grouped: Dict[str, List[Dict]] = {}
        for item in predictions:
            grouped.setdefault(item.get("object_id"), []).append(item)

        limited: List[Dict] = []
        for items in grouped.values():
            items.sort(key=lambda i: (i.get("probability") or 0.0), reverse=True)
            limited.extend(items[: self.max_modes_per_object])
        return limited

    def _resolve_scenario_id(self, env) -> Optional[str]:
        data_manager = getattr(env.engine, "data_manager", None)
        scenario = getattr(data_manager, "current_scenario", None)
        if isinstance(scenario, dict):
            metadata = scenario.get("metadata", {})
            scenario_id = metadata.get("scenario_id") or metadata.get("id")
            if scenario_id is not None:
                return str(scenario_id)
            value = scenario.get("id")
            return str(value) if value is not None else None
        return None

    def _normalise_id(self, value) -> Optional[str]:
        if value is None:
            return None
        if isinstance(value, str):
            return value
        try:
            return str(value)
        except Exception:
            return None

    def _match_object_id(self, candidate: Optional[str], target: Optional[str]) -> bool:
        if target is None:
            return False
        candidate = self._normalise_id(candidate)
        if candidate is None:
            return False
        return candidate == target

    def _draw_dashed_polyline(self, surface, points, color):
        dash_length = self.config["dash_length"]
        dash_gap = self.config["dash_gap"]
        width = self.config["line_width"]

        for start, end in zip(points[:-1], points[1:]):
            start_vec = np.array(start, dtype=float)
            end_vec = np.array(end, dtype=float)
            segment = end_vec - start_vec
            total_length = np.linalg.norm(segment)
            if total_length <= 0:
                continue

            direction = segment / total_length
            distance = 0.0
            while distance < total_length:
                dash_start = start_vec + direction * distance
                dash_end = start_vec + direction * min(distance + dash_length, total_length)
                pygame.draw.line(
                    surface,
                    color,
                    (int(dash_start[0]), int(dash_start[1])),
                    (int(dash_end[0]), int(dash_end[1])),
                    width,
                )
                distance += dash_length + dash_gap

    def _calculate_camera_offset(self, renderer, screen_surface, frame_surface):
        ego_vehicle = getattr(renderer, "current_track_agent", None)
        if ego_vehicle is None:
            return (0, 0)
        frame_position = frame_surface.pos2pix(*ego_vehicle.position)
        width, height = screen_surface.get_size()
        return (frame_position[0] - width / 2, frame_position[1] - height / 2)

    def _world_to_screen(self, world_pos, frame_surface, offset):
        frame_pixel = frame_surface.pos2pix(world_pos[0], world_pos[1])
        return (
            int(frame_pixel[0] - offset[0]),
            int(frame_pixel[1] - offset[1]),
        )
