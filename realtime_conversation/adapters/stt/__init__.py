"""STT engine adapters for the conversation library."""

from .whisper import WhisperSTTEngine
from .base import BaseSTTEngine

__all__ = [
    "WhisperSTTEngine",
    "BaseSTTEngine"
]