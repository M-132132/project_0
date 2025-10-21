"""归因可视化叠加层，与轨迹渲染器并行工作。"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pygame

from attribution_loader import AttributionLoader

logger = logging.getLogger(__name__)


class AttributionOverlay:
    """归因可视化叠加模块。保持原有绘制效果，增强数据加载与场景匹配。"""

    def __init__(self, config: Dict[str, Any]):
        self.enabled = config.get('enabled', False)
        self.opacity = float(config.get('opacity', 0.5))
        self.threshold = float(config.get('threshold', 0.02))

        self.obj_attr: Optional[np.ndarray] = None
        self.map_attr: Optional[np.ndarray] = None
        self.current_key: Optional[str] = None

        self.colormap = [
            (68, 1, 84),
            (59, 82, 139),
            (33, 144, 140),
            (92, 200, 99),
            (253, 231, 37),
        ]

        if not self.enabled:
            self.loader: Optional[AttributionLoader] = None
            self.method = config.get('method', 'IntegratedGradients')
            return

        numpy_dir = config.get('numpy_dir')
        method = config.get('method', 'IntegratedGradients')
        if not numpy_dir:
            logger.warning('归因叠加启用但未提供 numpy_dir，功能已禁用。')
            self.enabled = False
            self.loader = None
            self.method = method
            return

        self.loader = AttributionLoader(Path(numpy_dir), method)
        self.method = method
        logger.info('归因叠加层已启用: %s', method)

    def reset(
        self,
        env,
        scenario_id: Optional[str] = None,
        batch_idx: Optional[int] = None,
        element_idx: int = 0,
    ) -> None:
        if not self.enabled or self.loader is None:
            return

        key = scenario_id or (f"batch-{batch_idx}" if batch_idx is not None else None)
        if key and key == self.current_key:
            return

        data = self.loader.load(scenario_id=scenario_id, batch_id=batch_idx, element_idx=element_idx)
        if data is None:
            self.obj_attr = None
            self.map_attr = None
            self.current_key = None
            logger.warning('未找到归因数据: scenario=%s batch=%s', scenario_id, batch_idx)
            return

        self.obj_attr = data.obj_attr
        self.map_attr = data.map_attr
        self.current_key = key
        logger.info('归因数据已加载: prefix=%s scenario=%s batch=%s', data.prefix, data.scenario_id, data.batch_id)

    def draw(self, env) -> None:
        if not self.enabled or not self._has_data():
            return

        try:
            screen = pygame.display.get_surface()
            if screen is None:
                return

            overlay = pygame.Surface(screen.get_size(), pygame.SRCALPHA)

            if self.obj_attr is not None:
                self._draw_agent_attribution(env, overlay)

            if self.map_attr is not None:
                self._draw_map_attribution(env, overlay)

            screen.blit(overlay, (0, 0))

        except Exception as exc:
            logger.error('绘制归因叠加层失败: %s', exc)

    # ----- 原有绘制逻辑保持不变 -----
    def _has_data(self) -> bool:
        return self.obj_attr is not None or self.map_attr is not None

    def _draw_agent_attribution(self, env, surface):
        try:
            agent_scores = np.abs(self.obj_attr).sum(axis=(1, 2))
            max_score = agent_scores.max() if agent_scores.max() > 0 else 1
            normalized_scores = agent_scores / max_score

            for agent_id, score in enumerate(normalized_scores):
                if score < self.threshold:
                    continue

                screen_pos = self._get_agent_screen_position(env, agent_id)
                if screen_pos is None:
                    continue

                self._draw_importance_circle(surface, screen_pos, score, 'agent')

        except Exception as exc:
            logger.error('绘制智能体归因失败: %s', exc)

    def _draw_map_attribution(self, env, surface):
        try:
            map_scores = np.abs(self.map_attr).sum(axis=2)
            max_score = map_scores.max() if map_scores.max() > 0 else 1
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

                self._draw_importance_point(surface, screen_pos, score, 'map')

        except Exception as exc:
            logger.error('绘制地图归因失败: %s', exc)

    def _get_agent_screen_position(self, env, agent_id: int) -> Optional[Tuple[int, int]]:
        try:
            manager = getattr(env.engine, 'data_manager', None)
            scenario = getattr(manager, 'current_scenario', None)
            if scenario is not None and hasattr(scenario, 'dynamic_map_states'):
                current_frame = getattr(env, 'episode_step', 0)
                states = getattr(scenario, 'dynamic_map_states', [])
                if current_frame < len(states):
                    frame_data = states[current_frame]
                    tracks = getattr(frame_data, 'tracks', [])
                    if agent_id < len(tracks):
                        track = tracks[agent_id]
                        state = getattr(track, 'get_state', None)
                        if callable(state):
                            value = state()
                            world_pos = (value.position[0], value.position[1])
                            return self._world_to_screen(env, world_pos)

            if hasattr(env, 'engine') and hasattr(env.engine, 'get_objects'):
                vehicles = [obj for obj in env.engine.get_objects().values() if hasattr(obj, 'position')]
                if agent_id < len(vehicles):
                    vehicle = vehicles[agent_id]
                    world_pos = (vehicle.position[0], vehicle.position[1])
                    return self._world_to_screen(env, world_pos)

            return None

        except Exception as exc:
            logger.debug('获取智能体%s位置失败: %s', agent_id, exc)
            return None

    def _get_map_point_screen_position(self, env, seg_idx: int, pt_idx: int) -> Optional[Tuple[int, int]]:
        try:
            manager = getattr(env.engine, 'data_manager', None)
            scenario = getattr(manager, 'current_scenario', None)

            if scenario is not None:
                map_features = getattr(scenario, 'map_features', None)
                if map_features and seg_idx < len(map_features):
                    polyline = getattr(map_features[seg_idx], 'polyline', None)
                    if polyline is not None and pt_idx < len(polyline):
                        point = polyline[pt_idx]
                        world_pos = (point[0], point[1])
                        return self._world_to_screen(env, world_pos)

            return None

        except Exception as exc:
            logger.debug('获取地图点[%s,%s]位置失败: %s', seg_idx, pt_idx, exc)
            return None

    def _world_to_screen(self, env, world_pos: Tuple[float, float]) -> Optional[Tuple[int, int]]:
        try:
            renderer = getattr(getattr(env, 'engine', None), 'top_down_renderer', None)
            if renderer is not None:
                surface = getattr(renderer, '_frame_canvas', None)
                screen_canvas = getattr(renderer, 'screen_canvas', None)

                if surface is not None and screen_canvas is not None:
                    px, py = surface.pos2pix(world_pos[0], world_pos[1])

                    if getattr(renderer, 'target_agent_heading_up', False):
                        return int(px), int(py)

                    field_w, field_h = screen_canvas.get_size()

                    cam_pos = getattr(renderer, 'position', None)
                    if cam_pos is None:
                        track_agent = getattr(renderer, 'current_track_agent', None)
                        if track_agent is not None and hasattr(track_agent, 'position'):
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

        except Exception as exc:
            logger.debug('坐标转换失败: %s', exc)
            return None

    def _draw_importance_circle(self, surface, pos: Tuple[int, int], importance: float, type_: str):
        color = self._get_importance_color(importance)
        radius = int(15 + importance * 20)
        alpha = int(255 * self.opacity)

        circle_size = radius * 2
        circle_surface = pygame.Surface((circle_size, circle_size), pygame.SRCALPHA)
        pygame.draw.circle(circle_surface, (*color, alpha), (radius, radius), radius)
        surface.blit(circle_surface, (pos[0] - radius, pos[1] - radius))

    def _draw_importance_point(self, surface, pos: Tuple[int, int], importance: float, type_: str):
        color = self._get_importance_color(importance)
        radius = int(3 + importance * 5)
        alpha = int(255 * self.opacity)

        point_size = radius * 2
        point_surface = pygame.Surface((point_size, point_size), pygame.SRCALPHA)
        pygame.draw.circle(point_surface, (*color, alpha), (radius, radius), radius)
        surface.blit(point_surface, (pos[0] - radius, pos[1] - radius))

    def _get_importance_color(self, importance: float) -> Tuple[int, int, int]:
        importance = max(0, min(1, importance))
        color_idx = importance * (len(self.colormap) - 1)
        lower_idx = int(color_idx)
        upper_idx = min(lower_idx + 1, len(self.colormap) - 1)

        if lower_idx == upper_idx:
            return self.colormap[lower_idx]

        t = color_idx - lower_idx
        lower_color = self.colormap[lower_idx]
        upper_color = self.colormap[upper_idx]
        return tuple(int(lower_color[i] * (1 - t) + upper_color[i] * t) for i in range(3))
