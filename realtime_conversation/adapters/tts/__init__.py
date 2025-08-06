"""TTS engine adapters for the conversation library."""

from .chatterbox import ChatterboxTTSEngine
from .base import BaseTTSEngine

__all__ = [
    "ChatterboxTTSEngine",
    "BaseTTSEngine"
]