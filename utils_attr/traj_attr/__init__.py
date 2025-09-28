"""
轨迹预测模型归因计算框架

该框架提供了用于计算轨迹预测模型输入归因的统一接口，支持：
- AutoBot, Wayformer, MTR, SMART等轨迹预测模型
- Dirichlet, Integrated Gradients, DeepLift等归因方法
- 灵活的距离度量和基线生成策略
"""

from .base.traj_attr_base import TrajAttrBase
from .base.distance_metrics import DistanceMetrics
from .methods.dirichlet_attr import DirichletAttribution
from .methods.captum_attr import CaptumAttribution
from .utils_traj_attr.baseline_generator import BaselineGenerator

__all__ = [
    'TrajAttrBase',
    'DistanceMetrics', 
    'DirichletAttribution',
    'CaptumAttribution',
    'BaselineGenerator'
]