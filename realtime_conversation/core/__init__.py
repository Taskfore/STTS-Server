"""Core components of the real-time conversation library."""

from .interfaces import (
    AudioData,
    TranscriptionResult,
    SynthesisResult,
    ConversationContext,
    STTEngine,
    TTSEngine,
    PauseDetector,
    ResponseGenerator,
    WebSocketAdapter,
    ConversationMiddleware,
    ConfigurationProvider
)
from .engine import ConversationEngine

__all__ = [
    "AudioData",
    "TranscriptionResult",
    "SynthesisResult", 
    "ConversationContext",
    "STTEngine",
    "TTSEngine",
    "PauseDetector",
    "ResponseGenerator",
    "WebSocketAdapter",
    "ConversationMiddleware",
    "ConfigurationProvider",
    "ConversationEngine"
]