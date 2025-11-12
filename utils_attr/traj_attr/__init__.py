"""
Trajectory attribution framework.

Provides a unified interface for computing input attributions in trajectory forecasting models. It supports:
- AutoBot, Wayformer, MTR, SMART and related models
- Captum-integrated attribution methods (e.g., Integrated Gradients, DeepLift)
- Flexible distance metrics and baseline generation utilities
"""

from .base.traj_attr_base import TrajAttrBase
from .base.distance_metrics import DistanceMetrics
from .methods.captum_attr import CaptumAttribution
from .utils_traj_attr.baseline_generator import BaselineGenerator

__all__ = [
    'TrajAttrBase',
    'DistanceMetrics',
    'CaptumAttribution',
    'BaselineGenerator',
]
