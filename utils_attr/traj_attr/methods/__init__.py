"""
Attribution method integrations for trajectory attribution.

Includes Captum-based implementations and optional custom methods.
"""

from .captum_attr import CaptumAttribution

__all__ = ['CaptumAttribution']
