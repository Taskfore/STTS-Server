"""
Real-time WebSocket Conversation Library

A framework-agnostic library for building real-time audio conversation systems
that integrate Speech-to-Text (STT), Text-to-Speech (TTS), and response generation.
"""

from .core.engine import ConversationEngine
from .core.interfaces import (
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

__version__ = "0.1.0"
__author__ = "Career Link Team"
__description__ = "Framework-agnostic real-time conversation library"

__all__ = [
    "ConversationEngine",
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
    "ConfigurationProvider"
]