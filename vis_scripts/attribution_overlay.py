"""Attribution overlay for visualization.

Draws overlays using attribution arrays loaded from a folder provided
by YAML (attribution.numpy_dir). No method selection or metadata is used.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pygame

from attribution_loader import AttributionLoader

logger = logging.getLogger(__name__)


class AttributionOverlay:
    """Lightweight attribution overlay."""

    def __init__(self, config: Dict[str, Any]):
        self.enabled = config.get("enabled", False)
        self.opacity = float(config.get("opacity", 0.5))
        self.threshold = float(config.get("threshold", 0.02))

        self.obj_attr: Optional[np.ndarray] = None
        self.map_attr: Optional[np.ndarray] = None
        self.current_key: Optional[str] = None

        # Simple perceptually ordered colormap
        self.colormap = [
            (68, 1, 84),
            (59, 82, 139),
            (33, 144, 140),
            (92, 200, 99),
            (253, 231, 37),
        ]

        # If attribution is disabled in config, keep overlay inert.
        if not self.enabled:
            self.loader = None
            return

        numpy_dir = config.get("numpy_dir")
        if not numpy_dir:
            raise ValueError("Attribution enabled but 'numpy_dir' is missing in config")

        self.loader = AttributionLoader(Path(numpy_dir))
        logger.info("Attribution overlay enabled: dir=%s", numpy_dir)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def reset(
        self,
        env,
        scenario_id: Optional[str] = None,
        batch_idx: Optional[int] = None,
        element_idx: int = 0,
    ) -> None:
        if not self.enabled:
            raise RuntimeError(
                "AttributionOverlay.reset called but overlay is disabled (enabled=False)"
            )
        if self.loader is None:
            raise RuntimeError(
                "AttributionOverlay.reset called but loader is not initialized"
            )

        key = scenario_id or (f"batch-{batch_idx}" if batch_idx is not None else None)
        if key and key == self.current_key:
            return

        data = self.loader.load(
            scenario_id=scenario_id, batch_id=batch_idx, element_idx=element_idx
        )
        if data is None:
            raise RuntimeError(
                f"No attribution found for scenario={scenario_id!r}, batch={batch_idx!r}"
            )

        self.obj_attr = data.obj_attr
        self.map_attr = data.map_attr
        self.current_key = key
        logger.info(
            "Attribution loaded: prefix=%s scenario=%s batch=%s",
            data.prefix,
            data.scenario_id,
            data.batch_id,
        )

    # ------------------------------------------------------------------
    # Drawing
    # ------------------------------------------------------------------

    def draw(self, env) -> None:
        if not self.enabled:
            raise RuntimeError(
                "AttributionOverlay.draw called but overlay is disabled (enabled=False)"
            )
        if not self._has_data():
            raise RuntimeError(
                "AttributionOverlay.draw called but no attribution data is loaded"
            )

        renderer = getattr(getattr(env, "engine", None), "top_down_renderer", None)
        target_surface = (
            getattr(renderer, "screen_canvas", None) if renderer is not None else None
        )
        if target_surface is None:
            target_surface = pygame.display.get_surface()
        if target_surface is None:
            raise RuntimeError("No target surface found for attribution overlay drawing")

        overlay = pygame.Surface(target_surface.get_size(), pygame.SRCALPHA)

        if self.obj_attr is not None:
            self._draw_agent_attribution(env, overlay)

        if self.map_attr is not None:
            self._draw_map_attribution(env, overlay)

        target_surface.blit(overlay, (0, 0))

    def _has_data(self) -> bool:
        return self.obj_attr is not None or self.map_attr is not None

    # ------------------------------------------------------------------
    # Agent attribution
    # ------------------------------------------------------------------

    def _draw_agent_attribution(self, env, surface) -> None:
        agent_scores = np.abs(self.obj_attr).sum(axis=(1, 2))
        max_score = agent_scores.max() if agent_scores.max() > 0 else 1.0
        normalized_scores = agent_scores / max_score

        for agent_id, score in enumerate(normalized_scores):
            if score < self.threshold:
                continue

            screen_pos = self._get_agent_screen_position(env, agent_id)
            if screen_pos is None:
                continue

            self._draw_importance_circle(surface, screen_pos, score)

    def _get_agent_screen_position(
        self, env, agent_id: int
    ) -> Optional[Tuple[int, int]]:
        """Return screen position for the given agent_id using scenario tracks.

        We avoid env.engine.get_objects() here, because its keys are engine-internal
        object names which do not necessarily match dataset track_ids. Instead, we read
        the agent trajectory directly from scenario['tracks'] and project the current
        position to screen coordinates.
        """
        manager = getattr(env.engine, "data_manager", None)
        scenario = getattr(manager, "current_scenario", None)
        if scenario is None:
            return None

        # Get tracks dictionary
        if isinstance(scenario, dict):
            all_tracks = scenario.get("tracks", {}) or {}
        else:
            all_tracks = getattr(scenario, "tracks", {}) or {}

        if not all_tracks or agent_id < 0 or agent_id >= len(all_tracks):
            return None

        track_ids = list(all_tracks.keys())
        track_id = track_ids[agent_id]
        track = all_tracks[track_id]

        # Extract state.position from track
        if isinstance(track, dict):
            state = track.get("state")
        else:
            state = getattr(track, "state", None)
        if not isinstance(state, Dict):
            return None

        positions = state.get("position")
        if not isinstance(positions, (list, tuple, np.ndarray)) or len(positions) == 0:
            return None

        # Choose a timestep based on current episode step
        step = getattr(env, "episode_step", 0)
        if not isinstance(step, (int, np.integer)):
            step = 0
        t = max(0, min(int(step), len(positions) - 1))

        pos = positions[t]
        if not (isinstance(pos, (list, tuple, np.ndarray)) and len(pos) >= 2):
            return None

        world_pos = (float(pos[0]), float(pos[1]))
        return self._world_to_screen(env, world_pos)

    # ------------------------------------------------------------------
    # Map attribution
    # ------------------------------------------------------------------

    def _draw_map_attribution(self, env, surface) -> None:
        map_scores = np.abs(self.map_attr).sum(axis=2)
        max_score = map_scores.max() if map_scores.max() > 0 else 1.0
        flat_scores = map_scores.flatten()
        top_indices = flat_scores.argsort()[-50:]

        for idx in top_indices:
            seg_idx, pt_idx = np.unravel_index(idx, map_scores.shape)
            score = map_scores[seg_idx, pt_idx] / max_score

            if score < self.threshold:
                continue

            screen_pos = self._get_map_point_screen_position(env, seg_idx, pt_idx)
            if screen_pos is None:
                continue

            self._draw_importance_point(surface, screen_pos, score)

    def _get_map_point_screen_position(
        self, env, seg_idx: int, pt_idx: int
    ) -> Optional[Tuple[int, int]]:
        manager = getattr(env.engine, "data_manager", None)
        scenario = getattr(manager, "current_scenario", None)

        if scenario is not None:
            map_features = getattr(scenario, "map_features", None)
            if map_features and seg_idx < len(map_features):
                polyline = getattr(map_features[seg_idx], "polyline", None)
                if polyline is not None and pt_idx < len(polyline):
                    point = polyline[pt_idx]
                    world_pos = (point[0], point[1])
                    return self._world_to_screen(env, world_pos)

        return None

    # ------------------------------------------------------------------
    # Coordinate transforms and drawing primitives
    # ------------------------------------------------------------------

    def _world_to_screen(
        self, env, world_pos: Tuple[float, float]
    ) -> Optional[Tuple[int, int]]:
        renderer = getattr(getattr(env, "engine", None), "top_down_renderer", None)
        if renderer is not None:
            surface = getattr(renderer, "_frame_canvas", None)
            screen_canvas = getattr(renderer, "screen_canvas", None)

            if surface is not None and screen_canvas is not None:
                px, py = surface.pos2pix(world_pos[0], world_pos[1])

                if getattr(renderer, "target_agent_heading_up", False):
                    return int(px), int(py)

                field_w, field_h = screen_canvas.get_size()

                cam_pos = getattr(renderer, "position", None)
                if cam_pos is None:
                    track_agent = getattr(renderer, "current_track_agent", None)
                    if track_agent is not None and hasattr(track_agent, "position"):
                        cam_pos = track_agent.position

                if cam_pos is not None:
                    cam_px, cam_py = surface.pos2pix(cam_pos[0], cam_pos[1])
                    off_x = cam_px - field_w / 2
                    off_y = cam_py - field_h / 2
                else:
                    off_x = off_y = 0

                return int(px - off_x), int(py - off_y)

        screen = pygame.display.get_surface()
        if screen:
            screen_center = (screen.get_width() // 2, screen.get_height() // 2)
            scale = 5.0
            screen_x = int(screen_center[0] + world_pos[0] * scale)
            screen_y = int(screen_center[1] - world_pos[1] * scale)
            return screen_x, screen_y

        return None

    def _draw_importance_circle(
        self, surface, pos: Tuple[int, int], importance: float
    ) -> None:
        color = self._get_importance_color(importance)
        radius = int(15 + importance * 20)
        alpha = int(255 * self.opacity)

        circle_size = radius * 2
        circle_surface = pygame.Surface((circle_size, circle_size), pygame.SRCALPHA)
        pygame.draw.circle(circle_surface, (*color, alpha), (radius, radius), radius)
        surface.blit(circle_surface, (pos[0] - radius, pos[1] - radius))

    def _draw_importance_point(
        self, surface, pos: Tuple[int, int], importance: float
    ) -> None:
        color = self._get_importance_color(importance)
        radius = int(3 + importance * 5)
        alpha = int(255 * self.opacity)

        point_size = radius * 2
        point_surface = pygame.Surface((point_size, point_size), pygame.SRCALPHA)
        pygame.draw.circle(point_surface, (*color, alpha), (radius, radius), radius)
        surface.blit(point_surface, (pos[0] - radius, pos[1] - radius))

    def _get_importance_color(self, importance: float) -> Tuple[int, int, int]:
        importance = max(0.0, min(1.0, float(importance)))
        color_idx = importance * (len(self.colormap) - 1)
        lower_idx = int(color_idx)
        upper_idx = min(lower_idx + 1, len(self.colormap) - 1)

        if lower_idx == upper_idx:
            return self.colormap[lower_idx]

        t = color_idx - lower_idx
        lower_color = self.colormap[lower_idx]
        upper_color = self.colormap[upper_idx]
        return tuple(
            int(lower_color[i] * (1 - t) + upper_color[i] * t) for i in range(3)
        )

