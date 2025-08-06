"""Plugins for the conversation library."""

from .pause_detection import WebRTCPauseDetector, EnergyPauseDetector
from .response_generation import EchoResponseGenerator, TemplateResponseGenerator
from .middleware import LoggingMiddleware, TimingMiddleware, AuthenticationMiddleware, AnalyticsMiddleware

__all__ = [
    "WebRTCPauseDetector",
    "EnergyPauseDetector", 
    "EchoResponseGenerator",
    "TemplateResponseGenerator",
    "LoggingMiddleware",
    "TimingMiddleware",
    "AuthenticationMiddleware",
    "AnalyticsMiddleware"
]