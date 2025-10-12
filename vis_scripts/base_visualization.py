# python vis_scripts/base_visualization.py

import logging
import os
from pathlib import Path
from typing import Optional

from metadrive.envs.scenario_env import ScenarioEnv
from metadrive.policy.replay_policy import ReplayEgoCarPolicy
from metadrive.scenario.utils import get_number_of_scenarios
from metadrive.utils.utils import import_pygame

from ground_truth_trajectory_renderer import GroundTruthTrajectoryRenderer
from prediction_trajectory_renderer import PredictionTrajectoryRenderer
from attribution_overlay import AttributionOverlay

try:
    import yaml
except ImportError:  # pragma: no cover - should exist in runtime env
    yaml = None

pygame, _ = import_pygame()

DEFAULT_CONFIG_PATH = Path(__file__).with_name("visualization_defaults.yaml")
PROJECT_ROOT = DEFAULT_CONFIG_PATH.parent.parent.resolve()

os.chdir(PROJECT_ROOT)
print("当前工作目录:", os.getcwd())
print("当前文件路径:", __file__)


def _load_settings() -> dict:
    """
    加载默认配置文件
    该函数用于加载YAML格式的默认配置文件，并进行必要的错误检查。
    如果配置文件不存在或格式不正确，将抛出相应的异常。
    返回:
        dict: 包含配置信息的字典
    异常:
        RuntimeError: 当PyYAML库未安装时抛出
        FileNotFoundError: 当默认配置文件不存在时抛出
        ValueError: 当配置文件内容不是有效的字典格式时抛出
    """
    # 检查是否安装了PyYAML库
    if yaml is None:
        raise RuntimeError("PyYAML is required to load visualization defaults.")

    # 检查默认配置文件是否存在
    if not DEFAULT_CONFIG_PATH.is_file():
        raise FileNotFoundError(
            f"Default config missing: {DEFAULT_CONFIG_PATH}. Please create it before running."
        )

    try:
        # 尝试读取并解析YAML配置文件
        data = yaml.safe_load(DEFAULT_CONFIG_PATH.read_text(encoding="utf-8"))
    except Exception as exc:
        # 如果解析过程中发生错误，抛出包含详细信息的异常
        raise ValueError(f"Failed to parse {DEFAULT_CONFIG_PATH}: {exc}") from exc

    # 验证配置文件内容是否为字典类型
    if not isinstance(data, dict):
        raise ValueError(f"Configuration in {DEFAULT_CONFIG_PATH} must be a mapping.")

    return data


def _resolve_path(path_value):

    """
    解析并规范化路径

    参数:
        path_value (str): 需要解析的路径字符串

    返回:
        Path: 解析后的绝对路径对象，如果输入为空则返回None

    功能:
        1. 检查输入是否为空
        2. 将路径字符串转换为Path对象
        3. 如果是相对路径，则将其转换为相对于PROJECT_ROOT的绝对路径
        4. 返回解析后的绝对路径
    """
    if not path_value:  # 检查路径值是否为空
        return None

    path_obj = Path(path_value)  # 创建Path对象
    if not path_obj.is_absolute():  # 检查是否为绝对路径
        path_obj = PROJECT_ROOT / path_obj  # 如果是相对路径，则与PROJECT_ROOT拼接

    return path_obj.resolve()  # 返回解析后的绝对路径


class BaseVisualizationEnv(ScenarioEnv):
    def __init__(self, config):
        config = dict(config)
        prediction_kwargs = config.pop("prediction_renderer_kwargs", {}) or {}
        attribution_kwargs = config.pop("attribution_config", {}) or {}
        super().__init__(config)

        self.ground_truth_renderer = GroundTruthTrajectoryRenderer()
        self.prediction_renderer = (
            PredictionTrajectoryRenderer(**prediction_kwargs)
            if prediction_kwargs
            else None
        )
        self.attribution_overlay = (
            AttributionOverlay(attribution_kwargs)
            if attribution_kwargs.get('enabled', False)
            else None
        )

    def reset(self, seed=None):
        result = super().reset(seed)
        self.ground_truth_renderer.reset(self)
        target_track_id = self.ground_truth_renderer.target_track_id

        if self.prediction_renderer:
            self.prediction_renderer.reset(self, target_track_id)

        if self.attribution_overlay:
            scenario_id = self._get_current_scenario_id()
            batch_idx = seed if isinstance(seed, int) else None
            self.attribution_overlay.reset(self, scenario_id=scenario_id, batch_idx=batch_idx)

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

            if self.attribution_overlay:
                self.attribution_overlay.draw(self)

            return original_update(*update_args, **update_kwargs)

        pygame.display.update = enhanced_update
        try:
            return super()._render_topdown(text, *args, **kwargs)
        finally:
            pygame.display.update = original_update

    def _get_current_scenario_id(self) -> Optional[str]:
        manager = getattr(self.engine, "data_manager", None)
        scenario = getattr(manager, "current_scenario", None)

        if scenario is None:
            return None

        # Scenario may be dict-like
        if isinstance(scenario, dict):
            metadata = scenario.get("metadata", {})
            if isinstance(metadata, dict):
                value = metadata.get("scenario_id") or metadata.get("id")
                if value:
                    return str(value)
            value = scenario.get("scenario_id") or scenario.get("id")
            if value:
                return str(value)
            return None

        metadata = getattr(scenario, "metadata", None)
        if isinstance(metadata, dict):
            value = metadata.get("scenario_id") or metadata.get("id")
            if value:
                return str(value)

        for attr in ("scenario_id", "id", "ID"):
            if hasattr(scenario, attr):
                value = getattr(scenario, attr)
                if isinstance(value, dict):
                    nested = value.get("scenario_id") or value.get("id")
                    if nested:
                        return str(nested)
                elif value:
                    return str(value)

        return None


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

    # 归因配置
    attribution_config = settings.get("attribution", {})
    
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
        "attribution_config": attribution_config,
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
