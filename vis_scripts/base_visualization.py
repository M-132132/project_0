import logging
import os
from pathlib import Path

from metadrive.envs.scenario_env import ScenarioEnv
from metadrive.policy.replay_policy import ReplayEgoCarPolicy
from metadrive.scenario.utils import get_number_of_scenarios
from metadrive.utils.utils import import_pygame

from ground_truth_trajectory_renderer import GroundTruthTrajectoryRenderer
from prediction_trajectory_renderer import PredictionTrajectoryRenderer

try:
    import yaml
except ImportError:  # pragma: no cover - should exist in runtime env
    yaml = None

pygame, _ = import_pygame()

DEFAULT_CONFIG_PATH = Path(__file__).with_name("visualization_defaults.yaml")
PROJECT_ROOT = DEFAULT_CONFIG_PATH.parent.parent


def _load_settings() -> dict:
    if yaml is None:
        raise RuntimeError("PyYAML is required to load visualization defaults.")

    if not DEFAULT_CONFIG_PATH.is_file():
        raise FileNotFoundError(
            f"Default config missing: {DEFAULT_CONFIG_PATH}. Please create it before running."
        )

    try:
        data = yaml.safe_load(DEFAULT_CONFIG_PATH.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ValueError(f"Failed to parse {DEFAULT_CONFIG_PATH}: {exc}") from exc

    if not isinstance(data, dict):
        raise ValueError(f"Configuration in {DEFAULT_CONFIG_PATH} must be a mapping.")

    return data


def _resolve_path(path_value):
    if not path_value:
        return None

    path_obj = Path(path_value)
    if not path_obj.is_absolute():
        path_obj = PROJECT_ROOT / path_obj

    return path_obj.resolve()


class BaseVisualizationEnv(ScenarioEnv):
    def __init__(self, config):
        config = dict(config)
        prediction_kwargs = config.pop("prediction_renderer_kwargs", {}) or {}
        super().__init__(config)

        self.ground_truth_renderer = GroundTruthTrajectoryRenderer()
        self.prediction_renderer = (
            PredictionTrajectoryRenderer(**prediction_kwargs)
            if prediction_kwargs
            else None
        )

    def reset(self, seed=None):
        result = super().reset(seed)
        self.ground_truth_renderer.reset(self)
        target_track_id = self.ground_truth_renderer.target_track_id

        if self.prediction_renderer:
            self.prediction_renderer.reset(self, target_track_id)

        return result

    def step(self, action):
        result = super().step(action)
        self.ground_truth_renderer.update_frame()

        if self.prediction_renderer:
            self.prediction_renderer.update_frame()

        return result

    def _render_topdown(self, text, *args, **kwargs):
        if self.engine.top_down_renderer is None:
            from metadrive.engine.top_down_renderer import TopDownRenderer

            self.engine.top_down_renderer = TopDownRenderer(*args, **kwargs)

        original_update = pygame.display.update

        def enhanced_update(*update_args, **update_kwargs):
            self.ground_truth_renderer.draw_trajectory(self)

            if self.prediction_renderer:
                self.prediction_renderer.draw_trajectory(self)

            return original_update(*update_args, **update_kwargs)

        pygame.display.update = enhanced_update
        try:
            return super()._render_topdown(text, *args, **kwargs)
        finally:
            pygame.display.update = original_update


def main():
    settings = _load_settings()

    database_path = _resolve_path(settings.get("database_path"))
    if database_path is None:
        raise ValueError("'database_path' must be set in visualization_defaults.yaml")

    database_path = os.fspath(database_path)
    num_scenarios = get_number_of_scenarios(database_path)

    scenario_index = settings.get("scenario_index")
    if scenario_index is not None:
        try:
            scenario_index = int(scenario_index)
        except (TypeError, ValueError) as exc:
            raise ValueError("'scenario_index' must be null or an integer") from exc
        if scenario_index < 0 or scenario_index >= num_scenarios:
            raise ValueError("'scenario_index' out of range for dataset")

    prediction_kwargs = {}
    prediction_path = _resolve_path(settings.get("prediction_path"))
    if prediction_path:
        prediction_kwargs = {
            "prediction_path": os.fspath(prediction_path),
            "max_modes_per_object": settings.get("max_prediction_modes"),
            "min_probability": settings.get("min_prediction_probability", 0.0),
        }

    env_config = {
        "use_render": False,
        "agent_policy": ReplayEgoCarPolicy,
        "manual_control": False,
        "show_interface": False,
        "show_logo": False,
        "show_fps": False,
        "log_level": logging.CRITICAL,
        "num_scenarios": num_scenarios,
        "data_directory": database_path,
        "prediction_renderer_kwargs": prediction_kwargs,
    }

    env = BaseVisualizationEnv(env_config)
    scenario_range = (
        range(num_scenarios)
        if scenario_index is None
        else [scenario_index]
    )

    for scenario_id in scenario_range:
        env.reset(seed=scenario_id)

        while env.episode_step < env.engine.data_manager.current_scenario_length:
            env.step([0, 0])
            env.render(
                film_size=(3000, 3000),
                semantic_map=False,
                target_vehicle_heading_up=False,
                mode="top_down",
            )


if __name__ == "__main__":
    main()
