import numpy as np
from metadrive.utils.utils import import_pygame

pygame, _ = import_pygame()


class GroundTruthTrajectoryRenderer:
    def __init__(self):
        self.planned_trajectory = []
        self.current_frame = 0
        self.target_track_id = None
        self.config = {
            "completed_color": (80, 200, 80),
            "current_color": (255, 255, 0),
            "future_color": (100, 150, 255),
            "line_width": 3,
            "current_radius": 10,
        }

    def reset(self, env):
        self.planned_trajectory = []
        self.current_frame = 0
        self.target_track_id = None
        self.planned_trajectory = self._extract_target_trajectory(env)

    def update_frame(self):
        self.current_frame += 1

    def draw_trajectory(self, env):
        if len(self.planned_trajectory) < 2:
            return

        renderer = getattr(env.engine, "top_down_renderer", None)
        if not renderer or not hasattr(renderer, "screen_canvas"):
            return

        screen_surface = renderer.screen_canvas
        frame_surface = renderer._frame_canvas
        offset = self._calculate_camera_offset(renderer, screen_surface, frame_surface)

        screen_points = []
        for point in self.planned_trajectory:
            screen_point = self._world_to_screen(point, frame_surface, offset)
            if self._is_point_visible(screen_point, screen_surface):
                screen_points.append(screen_point)

        if len(screen_points) < 2:
            return

        self._draw_segments(screen_surface, screen_points)
        self._draw_current_marker(screen_surface, screen_points)

    def _extract_target_trajectory(self, env):
        manager = getattr(env.engine, "data_manager", None)
        scenario = getattr(manager, "current_scenario", None)
        if scenario is None:
            return []

        tracks = self._get_tracks_dict(scenario)
        if not tracks:
            return []

        tracks_to_predict = self._get_tracks_to_predict(scenario)
        track = self._select_target_track(tracks, tracks_to_predict)
        if not track:
            return []

        return self._extract_positions(track)

    def _get_tracks_dict(self, scenario):
        if hasattr(scenario, "tracks"):
            return dict(getattr(scenario, "tracks", {}) or {})
        if isinstance(scenario, dict):
            return dict(scenario.get("tracks", {}) or {})
        return {}

    def _get_tracks_to_predict(self, scenario):
        if hasattr(scenario, "tracks_to_predict"):
            value = getattr(scenario, "tracks_to_predict")
            if value:
                return value
        if isinstance(scenario, dict):
            value = scenario.get("tracks_to_predict")
            if value:
                return value
            metadata = scenario.get("metadata", {})
            if isinstance(metadata, dict):
                return metadata.get("tracks_to_predict")
        metadata = getattr(scenario, "metadata", None)
        if isinstance(metadata, dict):
            return metadata.get("tracks_to_predict")
        return None

    def _select_target_track(self, tracks, tracks_to_predict):
        candidate_ids = []
        candidate_indices = []

        def add_id(value):
            if value is None:
                return
            try:
                candidate_ids.append(str(value))
            except Exception:
                pass

        def add_index(value):
            if value is None:
                return
            if isinstance(value, (list, tuple, set)):
                for item in value:
                    add_index(item)
                return
            try:
                candidate_indices.append(int(value))
            except Exception:
                pass

        def collect(data, key_hint=None):
            if isinstance(key_hint, str):
                add_id(key_hint)

            if isinstance(data, dict):
                add_id(data.get("track_id"))
                add_id(data.get("object_id"))
                add_id(data.get("id"))
                add_index(data.get("track_index"))
                add_index(data.get("track_indexes"))
                add_index(data.get("track_indices"))
                add_index(data.get("track_ids"))
            elif isinstance(data, (list, tuple, set)):
                for item in data:
                    collect(item)
            elif isinstance(data, str):
                add_id(data)
            else:
                add_index(data)

        if isinstance(tracks_to_predict, dict):
            flat_keys = {"track_index", "track_indexes", "track_indices", "track_ids"}
            if flat_keys.intersection(tracks_to_predict.keys()):
                collect(tracks_to_predict)
            else:
                for key, value in tracks_to_predict.items():
                    collect(value, key_hint=key)
        elif tracks_to_predict is not None:
            collect(tracks_to_predict)

        for candidate in dict.fromkeys(candidate_ids):
            if candidate in tracks:
                self.target_track_id = candidate
                return tracks[candidate]

        if candidate_indices:
            ordered_ids = list(tracks.keys())
            for index in dict.fromkeys(candidate_indices):
                if 0 <= index < len(ordered_ids):
                    candidate = ordered_ids[index]
                    self.target_track_id = candidate
                    return tracks[candidate]

        return None

    def _extract_positions(self, track_data):
        if not isinstance(track_data, dict):
            return []

        state = track_data.get("state")
        if not isinstance(state, dict):
            return []

        positions = state.get("position")
        if not isinstance(positions, (list, tuple, np.ndarray)):
            return []

        trajectory = []
        for pos in positions:
            if isinstance(pos, (list, tuple, np.ndarray)) and len(pos) >= 2:
                trajectory.append([float(pos[0]), float(pos[1])])
        return trajectory

    def _draw_segments(self, surface, points):
        for idx in range(len(points) - 1):
            color = (
                self.config["completed_color"]
                if idx < self.current_frame
                else self.config["future_color"]
            )
            pygame.draw.line(surface, color, points[idx], points[idx + 1], self.config["line_width"])

    def _draw_current_marker(self, surface, points):
        if self.current_frame >= len(points):
            return

        current = points[self.current_frame]
        pygame.draw.circle(surface, self.config["current_color"], current, self.config["current_radius"])
        pygame.draw.circle(surface, (0, 0, 0), current, self.config["current_radius"], 2)

    def _calculate_camera_offset(self, renderer, screen_surface, frame_surface):
        ego = getattr(renderer, "current_track_agent", None)
        if ego is None:
            return (0, 0)

        frame_position = frame_surface.pos2pix(*ego.position)
        width, height = screen_surface.get_size()
        return (frame_position[0] - width / 2, frame_position[1] - height / 2)

    def _world_to_screen(self, world_pos, frame_surface, offset):
        frame_pixel = frame_surface.pos2pix(world_pos[0], world_pos[1])
        return (
            int(frame_pixel[0] - offset[0]),
            int(frame_pixel[1] - offset[1]),
        )

    def _is_point_visible(self, screen_pos, screen_surface):
        return (
            0 <= screen_pos[0] < screen_surface.get_width()
            and 0 <= screen_pos[1] < screen_surface.get_height()
        )
