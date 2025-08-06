"""Pause detection plugins."""

from .webrtc import WebRTCPauseDetector
from .energy import EnergyPauseDetector
from .base import BasePauseDetector

__all__ = [
    "WebRTCPauseDetector",
    "EnergyPauseDetector",
    "BasePauseDetector"
]