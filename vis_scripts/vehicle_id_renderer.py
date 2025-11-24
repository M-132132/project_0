import logging
from typing import Dict, Any, Optional, Tuple

import pygame


class VehicleIdRenderer:
    """在俯视图中为每个车辆绘制其ID。"""

    def __init__(self, config: Dict[str, Any]):
        self.enabled = config.get("enabled", False)
        self.color_mode = config.get("color_mode", "hash")
        self.font_min_px = int(config.get("font_min_px", 12))
        self.font_scale_h = float(config.get("font_scale_h", 0.012))
        self.bg_opacity = float(config.get("bg_opacity", 0.6))

        self._font: Optional[pygame.font.Font] = None
        self._font_size: int = 0
        self._color_cache: Dict[str, Tuple[int, int, int]] = {}
        self._log_once = False

    def draw(self, env) -> None:
        """在每个车辆对象上绘制ID标签。"""
        if not self.enabled:
            return

        renderer = getattr(env.engine, "top_down_renderer", None)
        if not renderer:
            return

        # Prefer the renderer's composed screen canvas; fall back to display surface
        screen_surface = getattr(renderer, "screen_canvas", None)
        if screen_surface is None:
            screen_surface = pygame.display.get_surface()
        if screen_surface is None:
            return

        screen_surface = renderer.screen_canvas
        self._update_font(screen_surface)
        if not self._font:
            return

        width, height = screen_surface.get_size()
        margin = 20

        # 核心逻辑：从引擎获取当前所有活动对象，这与从 scenario.tracks 获取数据源的逻辑一致
        vehicles = [
            obj for obj in env.engine.get_objects().values() if hasattr(obj, "position")
        ]

        if not self._log_once:
            logging.info(
                "VehicleIdRenderer.draw called: renderer=%s surface=%s vehicles=%d font_ready=%s",
                type(renderer).__name__,
                type(screen_surface).__name__ if screen_surface else None,
                len(vehicles),
                bool(self._font),
            )
            self._log_once = True

        for vehicle in vehicles:
            # 检查对象是否在屏幕内
            # 将车辆的当前世界坐标转换为屏幕坐标
            screen_pos = self._world_to_screen(renderer, vehicle.position)
            if not screen_pos:
                continue

            x, y = screen_pos
            if (
                x < -margin
                or y < -margin
                or x > width + margin
                or y > height + margin
            ):
                continue

            # 将车辆的 ID (vehicle.name) 绘制到屏幕上
            self._draw_id_label(screen_surface, (x, y), str(vehicle.name))

    def _update_font(self, surface: pygame.Surface) -> None:
        target_font_size = max(
            self.font_min_px, int(surface.get_height() * self.font_scale_h)
        )
        if self._font is None or self._font_size != target_font_size:
            # try:
                # 确保 pygame 和 font 模块已经初始化
            if not pygame.get_init():
                    pygame.init()
            if not pygame.font.get_init():
                    pygame.font.init()

            self._font = pygame.font.Font(None, target_font_size)
            self._font_size = target_font_size


    def _get_color(self, text: str) -> Tuple[int, int, int]:
        """根据ID获取稳定或统一的颜色。"""
        if self.color_mode == "white":
            return (255, 255, 255)

        if text not in self._color_cache:
            # 基于ID的哈希值生成一个稳定的颜色
            h = hash(text)
            r = 150 + (h & 0xFF) % 106
            g = 150 + ((h >> 8) & 0xFF) % 106
            b = 150 + ((h >> 16) & 0xFF) % 106
            self._color_cache[text] = (r, g, b)
        return self._color_cache[text]

    def _draw_id_label(
        self, surface: pygame.Surface, pos: Tuple[int, int], text: str
    ) -> None:
        """在指定位置绘制带背景的ID文本。"""
        if not self._font:
            return

        color = self._get_color(text)
        text_surface = self._font.render(text, True, color)
        text_rect = text_surface.get_rect()

        # 创建一个半透明的背景
        bg_surface = pygame.Surface(
            (text_rect.width + 6, text_rect.height + 4), pygame.SRCALPHA
        )
        bg_color = (0, 0, 0, int(255 * self.bg_opacity))
        bg_surface.fill(bg_color)

        # 将文本绘制到背景上
        bg_surface.blit(text_surface, (3, 2))

        # 将带背景的标签绘制到主屏幕上，位于车辆位置的右上方
        label_pos = (pos[0] + 8, pos[1] - text_rect.height - 8)
        surface.blit(bg_surface, label_pos)

    @staticmethod
    def _world_to_screen(
        renderer, world_pos: Tuple[float, float]
    ) -> Optional[Tuple[int, int]]:
        """将世界坐标转换为屏幕坐标。"""
        try:
            screen_canvas = getattr(renderer, "screen_canvas", None)
            frame_canvas = getattr(renderer, "_frame_canvas", None)

            if frame_canvas is not None and screen_canvas is not None:
                px, py = frame_canvas.pos2pix(world_pos[0], world_pos[1])

                cam_pos = getattr(renderer, "position", None)
                if cam_pos is None:
                    track_agent = getattr(renderer, "current_track_agent", None)
                    if track_agent is not None and hasattr(track_agent, "position"):
                        cam_pos = track_agent.position

                if cam_pos is not None:
                    field_w, field_h = screen_canvas.get_size()
                    cam_px, cam_py = frame_canvas.pos2pix(cam_pos[0], cam_pos[1])
                    off_x = cam_px - field_w / 2
                    off_y = cam_py - field_h / 2
                    return int(px - off_x), int(py - off_y)

            return None
        except Exception:
            return None