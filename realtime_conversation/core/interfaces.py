"""
Core interfaces and data models for the real-time conversation library.

This module defines the protocol interfaces that enable dependency injection
and framework-agnostic operation of the conversation system.
"""

from typing import Protocol, List, Dict, Any, Optional, AsyncGenerator, runtime_checkable
from abc import abstractmethod
import asyncio
from dataclasses import dataclass
from enum import Enum


class ConversationState(Enum):
    """Enumeration of conversation states."""
    LISTENING = "listening"
    PROCESSING = "processing"
    SPEAKING = "speaking"
    ERROR = "error"


@dataclass
class AudioData:
    """Standardized audio data container."""
    
    data: bytes
    sample_rate: int
    channels: int = 1
    format: str = "pcm"
    
    @property
    def duration_seconds(self) -> float:
        """Calculate audio duration in seconds."""
        # Assume 16-bit samples (2 bytes per sample)
        samples = len(self.data) // (2 * self.channels)
        return samples / self.sample_rate
    
    @property
    def duration_ms(self) -> int:
        """Calculate audio duration in milliseconds."""
        return int(self.duration_seconds * 1000)


@dataclass
class TranscriptionSegment:
    """Individual transcription segment with timing information."""
    
    text: str
    start: float
    end: float
    confidence: Optional[float] = None


@dataclass
class TranscriptionResult:
    """STT transcription result with timing information."""
    
    text: str
    language: str
    segments: List[TranscriptionSegment]
    confidence: Optional[float] = None
    partial: bool = False
    
    @property
    def duration(self) -> float:
        """Get total duration of transcription."""
        if not self.segments:
            return 0.0
        return self.segments[-1].end - self.segments[0].start


@dataclass
class SynthesisResult:
    """TTS synthesis result with audio data."""
    
    audio_data: AudioData
    text: str
    voice_id: str
    synthesis_time: Optional[float] = None


@dataclass
class ConversationContext:
    """Context object passed through the conversation pipeline."""
    
    # Input data
    audio_input: Optional[AudioData] = None
    transcription: Optional[TranscriptionResult] = None
    
    # Processing results
    response_text: Optional[str] = None
    synthesis_result: Optional[SynthesisResult] = None
    
    # State and metadata
    state: ConversationState = ConversationState.LISTENING
    metadata: Dict[str, Any] = None
    user_data: Dict[str, Any] = None
    error: Optional[Exception] = None
    
    def __post_init__(self):
        """Initialize default values."""
        if self.metadata is None:
            self.metadata = {}
        if self.user_data is None:
            self.user_data = {}


# Protocol definitions for dependency injection

@runtime_checkable
class STTEngine(Protocol):
    """Speech-to-Text engine interface."""
    
    @abstractmethod
    async def transcribe(
        self, 
        audio: AudioData, 
        language: Optional[str] = None
    ) -> Optional[TranscriptionResult]:
        """
        Transcribe audio data to text with timing information.
        
        Args:
            audio: Audio data to transcribe
            language: Target language (None for auto-detection)
            
        Returns:
            Transcription result with timing information, or None on failure
        """
        ...
    
    @abstractmethod
    async def is_available(self) -> bool:
        """Check if the STT engine is ready for transcription."""
        ...
    
    @property
    @abstractmethod
    def model_loaded(self) -> bool:
        """Check if the STT model is loaded."""
        ...


@runtime_checkable
class TTSEngine(Protocol):
    """Text-to-Speech engine interface."""
    
    @abstractmethod
    async def synthesize(
        self, 
        text: str, 
        voice_config: Dict[str, Any]
    ) -> Optional[SynthesisResult]:
        """
        Synthesize text to speech audio.
        
        Args:
            text: Text to synthesize
            voice_config: Voice configuration (voice_id, temperature, speed, etc.)
            
        Returns:
            Synthesis result with audio data, or None on failure
        """
        ...
    
    @abstractmethod
    async def list_voices(self) -> List[Dict[str, Any]]:
        """List available voices."""
        ...
    
    @property
    @abstractmethod
    def model_loaded(self) -> bool:
        """Check if the TTS model is loaded."""
        ...


@runtime_checkable
class PauseDetector(Protocol):
    """Voice Activity Detection interface."""
    
    @abstractmethod
    async def process_chunk(self, audio: AudioData) -> Dict[str, Any]:
        """
        Process audio chunk and return VAD results.
        
        Args:
            audio: Audio chunk to process
            
        Returns:
            VAD result containing events, state, and confidence information
        """
        ...
    
    @abstractmethod
    def reset(self) -> None:
        """Reset detector state."""
        ...
    
    @property
    @abstractmethod
    def is_speaking(self) -> bool:
        """Check if currently detecting speech."""
        ...


@runtime_checkable
class ResponseGenerator(Protocol):
    """Response generation interface."""
    
    @abstractmethod
    async def generate_response(
        self, 
        transcription: TranscriptionResult, 
        context: ConversationContext
    ) -> str:
        """
        Generate a text response based on input transcription.
        
        Args:
            transcription: Input transcription result
            context: Conversation context with history and metadata
            
        Returns:
            Generated response text
        """
        ...
    
    @abstractmethod
    def set_response_mode(self, mode: str) -> None:
        """Set the response generation mode."""
        ...
    
    @abstractmethod
    def clear_history(self) -> None:
        """Clear conversation history."""
        ...


@runtime_checkable
class WebSocketAdapter(Protocol):
    """WebSocket framework adapter interface."""
    
    @abstractmethod
    async def accept_connection(self) -> None:
        """Accept the WebSocket connection."""
        ...
    
    @abstractmethod
    async def receive_audio(self) -> Optional[AudioData]:
        """Receive audio data from the client."""
        ...
    
    @abstractmethod
    async def receive_command(self) -> Optional[Dict[str, Any]]:
        """Receive command data from the client."""
        ...
    
    @abstractmethod
    async def send_json(self, data: Dict[str, Any]) -> None:
        """Send JSON data to the client."""
        ...
    
    @abstractmethod
    async def send_audio(self, audio: AudioData) -> None:
        """Send audio data to the client."""
        ...
    
    @abstractmethod
    async def close(self, code: int = 1000) -> None:
        """Close the WebSocket connection."""
        ...
    
    @property
    @abstractmethod
    def is_connected(self) -> bool:
        """Check if WebSocket is connected."""
        ...


@runtime_checkable
class ConversationMiddleware(Protocol):
    """Middleware interface for processing conversation context."""
    
    @abstractmethod
    async def process(
        self, 
        context: ConversationContext, 
        next_middleware
    ) -> ConversationContext:
        """
        Process conversation context and call next middleware.
        
        Args:
            context: Current conversation context
            next_middleware: Next middleware function in the chain
            
        Returns:
            Modified conversation context
        """
        ...


@runtime_checkable
class ConfigurationProvider(Protocol):
    """Configuration provider interface."""
    
    @abstractmethod
    def get_stt_config(self) -> Dict[str, Any]:
        """Get STT engine configuration."""
        ...
    
    @abstractmethod
    def get_tts_config(self) -> Dict[str, Any]:
        """Get TTS engine configuration."""
        ...
    
    @abstractmethod
    def get_conversation_config(self) -> Dict[str, Any]:
        """Get conversation engine configuration."""
        ...
    
    @abstractmethod
    def get_pause_detection_config(self) -> Dict[str, Any]:
        """Get pause detection configuration."""
        ...
    
    @abstractmethod
    def get_audio_config(self) -> Dict[str, Any]:
        """Get audio processing configuration."""
        ...


# Event system interfaces

class ConversationEvent:
    """Base class for conversation events."""
    
    def __init__(self, event_type: str, data: Dict[str, Any] = None):
        self.event_type = event_type
        self.data = data or {}
        self.timestamp = asyncio.get_event_loop().time()


@runtime_checkable
class EventHandler(Protocol):
    """Interface for handling conversation events."""
    
    @abstractmethod
    async def handle(self, event: ConversationEvent) -> None:
        """Handle a conversation event."""
        ...


@runtime_checkable
class EventBus(Protocol):
    """Event bus interface for pub/sub messaging."""
    
    @abstractmethod
    async def publish(self, event: ConversationEvent) -> None:
        """Publish an event to all subscribers."""
        ...
    
    @abstractmethod
    def subscribe(self, event_type: str, handler: EventHandler) -> None:
        """Subscribe to events of a specific type."""
        ...
    
    @abstractmethod
    def unsubscribe(self, event_type: str, handler: EventHandler) -> None:
        """Unsubscribe from events of a specific type."""
        ...