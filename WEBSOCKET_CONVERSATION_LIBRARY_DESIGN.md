# WebSocket Conversation Library Design Document

## Executive Summary

This document analyzes the current WebSocket conversation router implementation in the STTS server and proposes a comprehensive architecture for transforming it into a standalone, framework-agnostic library. The current implementation demonstrates excellent functionality but suffers from tight coupling to FastAPI and application-specific components, limiting its reusability.

## Current Implementation Analysis

### Architecture Overview

The WebSocket conversation router (`routers/websocket_conversation.py`) implements a sophisticated real-time audio conversation system with the following flow:

1. **Audio Input**: Receives PCM audio streams via WebSocket
2. **Pause Detection**: Uses WebRTC VAD or energy-based fallback to detect speech boundaries  
3. **Speech-to-Text**: Transcribes audio using OpenAI Whisper
4. **Response Generation**: Generates contextual responses (echo/template modes)
5. **Text-to-Speech**: Synthesizes responses using Chatterbox TTS
6. **Audio Output**: Streams synthesized audio back to client

### Key Components

#### ConversationProcessor (Lines 49-478)
- Core conversation flow orchestrator
- Handles audio buffering, pause detection, STT/TTS coordination
- Manages conversation state transitions ("listening", "processing", "speaking")

#### WebSocket Handler (Lines 480-697)
- FastAPI-specific WebSocket endpoint
- Audio chunk processing and command handling  
- Error management and connection lifecycle

#### Supporting Infrastructure
- **OptimizedAudioBuffer**: Thread-safe audio buffering with deque-based implementation
- **PCMAudioDecoder**: PCM-to-numpy conversion utilities
- **PauseDetector**: WebRTC VAD integration with fallback energy detection
- **ConversationResponseGenerator**: Template and echo-based response generation

### Performance Characteristics

- **Thread Pool**: Dedicated 3-worker ThreadPoolExecutor for conversation processing
- **Audio Processing**: Optimized buffer management with configurable duration limits
- **Real-time Capability**: Sub-second latency for STT→TTS pipeline
- **Concurrent Support**: FastAPI handles multiple simultaneous conversations

## Critical Architectural Gaps

### 1. Framework Coupling

**Problem**: Hard-coded FastAPI dependencies throughout the codebase.

```python
# Lines 12, 480-495 - FastAPI-specific imports and decorators
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Query, Request
@router.websocket("/conversation")
async def websocket_conversation(websocket: WebSocket, ...):
```

**Impact**: Cannot be used with Django Channels, Socket.IO, or other WebSocket frameworks.

### 2. Global State Dependencies

**Problem**: Relies on application-specific global variables and app state.

```python
# Lines 513-522 - Direct access to FastAPI app state
request = websocket.scope.get("app")
stt_engine = request.state.stt_engine

# Lines 382, 536 - Global engine state
if not engine.MODEL_LOADED:
```

**Impact**: Prevents clean instantiation and testing in different environments.

### 3. Configuration System Lock-in

**Problem**: Tightly coupled to application's configuration system.

```python
# Lines 27-37 - Hard-coded config imports
from config import (
    get_default_voice_id,
    get_predefined_voices_path,
    get_reference_audio_path,
    # ... 8 more config functions
)
```

**Impact**: Cannot adapt to different configuration sources or formats.

### 4. Concrete Implementation Dependencies

**Problem**: Direct imports instead of interface-based design.

```python
# Lines 16, 18, 24 - Concrete class imports
from stt_engine import STTEngine
import engine  # Global TTS engine
from conversation_engine import ConversationResponseGenerator
```

**Impact**: Difficult to swap implementations or mock for testing.

### 5. Audio Processing Pipeline Rigidity

**Problem**: Audio codec and processing logic embedded within conversation flow.

```python
# Lines 434-457 - Embedded audio encoding logic
encoded_audio = utils.encode_audio(
    audio_array=audio_np,
    sample_rate=sample_rate,
    output_format="wav",
    target_sample_rate=target_sample_rate,
)
```

**Impact**: Cannot easily support different audio formats or processing chains.

### 6. Event System Limitations

**Problem**: No extensible event system for middleware or plugins.

**Current Flow**: Fixed STT→Response→TTS pipeline with no insertion points.

**Impact**: Cannot add custom processing steps, logging, analytics, or business logic.

## Proposed Standalone Library Architecture

### Design Principles

1. **Framework Agnostic**: Support multiple WebSocket frameworks through adapters
2. **Dependency Injection**: Protocol-based interfaces for all major components
3. **Event-Driven**: Extensible pipeline through middleware and events
4. **Configuration Flexible**: Support various configuration sources and formats
5. **Type-Safe**: Comprehensive typing with Python protocols
6. **Testing-Friendly**: Easy mocking and component isolation

### Library Structure

```
realtime-conversation/
├── core/
│   ├── __init__.py
│   ├── interfaces.py          # Protocol definitions
│   ├── engine.py             # Core ConversationEngine
│   ├── pipeline.py           # Event-driven processing pipeline
│   ├── context.py            # Conversation context management
│   └── events.py             # Event system and types
├── adapters/
│   ├── __init__.py
│   ├── websocket/
│   │   ├── __init__.py
│   │   ├── fastapi.py        # FastAPI WebSocket adapter
│   │   ├── django.py         # Django Channels adapter
│   │   ├── socketio.py       # Socket.IO adapter
│   │   └── base.py           # Base WebSocket adapter
│   ├── stt/
│   │   ├── __init__.py
│   │   ├── whisper.py        # OpenAI Whisper implementation
│   │   ├── azure.py          # Azure Speech Services
│   │   ├── google.py         # Google Speech-to-Text
│   │   └── base.py           # STT base classes
│   ├── tts/
│   │   ├── __init__.py
│   │   ├── chatterbox.py     # Chatterbox TTS implementation
│   │   ├── elevenlabs.py     # ElevenLabs adapter
│   │   ├── azure.py          # Azure Text-to-Speech
│   │   └── base.py           # TTS base classes
│   └── config/
│       ├── __init__.py
│       ├── yaml.py           # YAML configuration provider
│       ├── env.py            # Environment variable provider
│       ├── dict.py           # Dictionary-based provider
│       └── base.py           # Configuration base classes
├── audio/
│   ├── __init__.py
│   ├── codecs.py             # Audio format handling (WAV, MP3, Opus)
│   ├── processing.py         # Audio processing utilities
│   ├── buffers.py           # Audio buffer implementations
│   └── formats.py           # Audio format definitions
├── plugins/
│   ├── __init__.py
│   ├── pause_detection/
│   │   ├── __init__.py
│   │   ├── webrtc.py        # WebRTC VAD implementation
│   │   ├── energy.py        # Energy-based detection
│   │   └── base.py          # Pause detection interfaces
│   ├── response_generation/
│   │   ├── __init__.py
│   │   ├── template.py      # Template-based responses
│   │   ├── llm.py          # LLM integration (OpenAI, etc.)
│   │   └── base.py         # Response generation interfaces
│   └── middleware/
│       ├── __init__.py
│       ├── logging.py       # Conversation logging
│       ├── analytics.py     # Usage analytics
│       ├── auth.py         # Authentication middleware
│       └── base.py         # Middleware base classes
├── utils/
│   ├── __init__.py
│   ├── threading.py         # Thread pool utilities
│   ├── timing.py           # Performance monitoring
│   └── validation.py       # Input validation utilities
└── examples/
    ├── __init__.py
    ├── fastapi_basic.py     # Basic FastAPI integration
    ├── django_channels.py   # Django Channels example
    └── custom_pipeline.py   # Custom middleware example
```

### Core Interface Definitions

```python
# core/interfaces.py
from typing import Protocol, AsyncGenerator, Dict, Any, Optional, List
from abc import abstractmethod
import asyncio

class AudioData:
    """Standardized audio data container."""
    def __init__(self, data: bytes, sample_rate: int, channels: int = 1):
        self.data = data
        self.sample_rate = sample_rate
        self.channels = channels
        self.duration = len(data) / (sample_rate * channels * 2)  # 16-bit samples

class TranscriptionResult:
    """STT transcription result with timing information."""
    def __init__(self, text: str, language: str, segments: List[Dict], confidence: float):
        self.text = text
        self.language = language
        self.segments = segments
        self.confidence = confidence

class SynthesisResult:
    """TTS synthesis result with audio data."""
    def __init__(self, audio_data: AudioData, text: str, voice_id: str):
        self.audio_data = audio_data
        self.text = text
        self.voice_id = voice_id

class ConversationContext:
    """Context object passed through the conversation pipeline."""
    def __init__(self):
        self.audio_input: Optional[AudioData] = None
        self.transcription: Optional[TranscriptionResult] = None
        self.response_text: Optional[str] = None
        self.synthesis_result: Optional[SynthesisResult] = None
        self.metadata: Dict[str, Any] = {}
        self.user_data: Dict[str, Any] = {}

# Protocol definitions for dependency injection
class STTEngine(Protocol):
    """Speech-to-Text engine interface."""
    
    @abstractmethod
    async def transcribe(self, audio: AudioData, language: Optional[str] = None) -> TranscriptionResult:
        """Transcribe audio data to text with timing information."""
        ...
    
    @abstractmethod
    async def is_available(self) -> bool:
        """Check if the STT engine is ready for transcription."""
        ...

class TTSEngine(Protocol):
    """Text-to-Speech engine interface."""
    
    @abstractmethod
    async def synthesize(self, text: str, voice_config: Dict[str, Any]) -> SynthesisResult:
        """Synthesize text to speech audio."""
        ...
    
    @abstractmethod
    async def list_voices(self) -> List[Dict[str, Any]]:
        """List available voices."""
        ...

class PauseDetector(Protocol):
    """Voice Activity Detection interface."""
    
    @abstractmethod
    async def process_chunk(self, audio: AudioData) -> Dict[str, Any]:
        """Process audio chunk and return VAD results."""
        ...
    
    @abstractmethod  
    def reset(self) -> None:
        """Reset detector state."""
        ...

class ResponseGenerator(Protocol):
    """Response generation interface."""
    
    @abstractmethod
    async def generate_response(self, transcription: TranscriptionResult, context: ConversationContext) -> str:
        """Generate a text response based on input transcription."""
        ...

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

class ConversationMiddleware(Protocol):
    """Middleware interface for processing conversation context."""
    
    @abstractmethod
    async def process(self, context: ConversationContext, next_middleware) -> ConversationContext:
        """Process conversation context and call next middleware."""
        ...

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
```

### Core Conversation Engine

```python
# core/engine.py
from typing import List, Optional, Dict, Any
import asyncio
import logging
from .interfaces import (
    STTEngine, TTSEngine, PauseDetector, ResponseGenerator,
    WebSocketAdapter, ConversationMiddleware, ConfigurationProvider,
    ConversationContext, AudioData
)
from .pipeline import ConversationPipeline
from .events import ConversationEventBus

logger = logging.getLogger(__name__)

class ConversationEngine:
    """
    Core conversation engine that orchestrates STT, TTS, and response generation
    through a configurable middleware pipeline.
    """
    
    def __init__(self, config_provider: Optional[ConfigurationProvider] = None):
        self.config_provider = config_provider
        self.stt_engine: Optional[STTEngine] = None
        self.tts_engine: Optional[TTSEngine] = None
        self.pause_detector: Optional[PauseDetector] = None
        self.response_generator: Optional[ResponseGenerator] = None
        self.pipeline = ConversationPipeline()
        self.event_bus = ConversationEventBus()
        self.middleware: List[ConversationMiddleware] = []
        
        # Conversation state
        self.is_listening = False
        self.is_processing = False
        
    def configure_stt(self, stt_engine: STTEngine) -> None:
        """Configure the Speech-to-Text engine."""
        self.stt_engine = stt_engine
        logger.info(f"STT engine configured: {type(stt_engine).__name__}")
        
    def configure_tts(self, tts_engine: TTSEngine) -> None:
        """Configure the Text-to-Speech engine."""
        self.tts_engine = tts_engine
        logger.info(f"TTS engine configured: {type(tts_engine).__name__}")
        
    def configure_pause_detection(self, pause_detector: PauseDetector) -> None:
        """Configure the pause detection system."""
        self.pause_detector = pause_detector
        logger.info(f"Pause detector configured: {type(pause_detector).__name__}")
        
    def configure_response_generation(self, response_generator: ResponseGenerator) -> None:
        """Configure the response generation system."""
        self.response_generator = response_generator
        logger.info(f"Response generator configured: {type(response_generator).__name__}")
        
    def add_middleware(self, middleware: ConversationMiddleware) -> None:
        """Add middleware to the conversation pipeline."""
        self.middleware.append(middleware)
        logger.info(f"Middleware added: {type(middleware).__name__}")
        
    async def handle_conversation(self, websocket_adapter: WebSocketAdapter) -> None:
        """
        Main conversation handling loop.
        
        Args:
            websocket_adapter: Framework-specific WebSocket adapter
        """
        if not self._validate_configuration():
            raise ValueError("ConversationEngine not properly configured")
            
        await websocket_adapter.accept_connection()
        logger.info("Conversation session started")
        
        try:
            await websocket_adapter.send_json({
                "type": "ready",
                "message": "Conversation engine ready"
            })
            
            async for context in self._process_audio_stream(websocket_adapter):
                if context.synthesis_result:
                    await websocket_adapter.send_audio(context.synthesis_result.audio_data)
                    
                # Send status updates
                await websocket_adapter.send_json({
                    "type": "transcription",
                    "text": context.transcription.text if context.transcription else "",
                    "response": context.response_text or ""
                })
                
        except Exception as e:
            logger.error(f"Conversation error: {e}", exc_info=True)
            await websocket_adapter.send_json({
                "type": "error",
                "message": str(e)
            })
        finally:
            logger.info("Conversation session ended")
            await websocket_adapter.close()
            
    async def _process_audio_stream(self, websocket_adapter: WebSocketAdapter):
        """Process incoming audio stream through the conversation pipeline."""
        
        while True:
            audio_data = await websocket_adapter.receive_audio()
            if audio_data is None:
                break
                
            # Create conversation context
            context = ConversationContext()
            context.audio_input = audio_data
            
            # Process through pause detection
            if self.pause_detector:
                vad_result = await self.pause_detector.process_chunk(audio_data)
                
                # Check for speech end event
                if "speech_end" in vad_result.get("events", []):
                    # Process complete utterance through pipeline
                    try:
                        context = await self._run_pipeline(context)
                        yield context
                    except Exception as e:
                        logger.error(f"Pipeline processing error: {e}")
                        
    async def _run_pipeline(self, context: ConversationContext) -> ConversationContext:
        """Run the conversation context through the middleware pipeline."""
        
        # Build pipeline with middleware
        async def pipeline_chain(ctx: ConversationContext) -> ConversationContext:
            # STT step
            if self.stt_engine and ctx.audio_input:
                ctx.transcription = await self.stt_engine.transcribe(ctx.audio_input)
                
            # Response generation step
            if self.response_generator and ctx.transcription:
                ctx.response_text = await self.response_generator.generate_response(
                    ctx.transcription, ctx
                )
                
            # TTS step
            if self.tts_engine and ctx.response_text:
                tts_config = self.config_provider.get_tts_config() if self.config_provider else {}
                ctx.synthesis_result = await self.tts_engine.synthesize(
                    ctx.response_text, tts_config
                )
                
            return ctx
        
        # Execute middleware chain
        current_func = pipeline_chain
        for middleware in reversed(self.middleware):
            current_middleware = middleware
            current_func = lambda ctx, func=current_func, mw=current_middleware: mw.process(ctx, func)
            
        return await current_func(context)
        
    def _validate_configuration(self) -> bool:
        """Validate that required components are configured."""
        required_components = [
            ("STT Engine", self.stt_engine),
            ("TTS Engine", self.tts_engine),
            ("Response Generator", self.response_generator)
        ]
        
        for name, component in required_components:
            if component is None:
                logger.error(f"{name} not configured")
                return False
                
        return True
```

### Framework Adapters

```python
# adapters/websocket/fastapi.py
from typing import Optional, Dict, Any
import json
import base64
from fastapi import WebSocket, WebSocketDisconnect
from ...core.interfaces import WebSocketAdapter, AudioData
from ...audio.codecs import decode_pcm, encode_wav

class FastAPIWebSocketAdapter(WebSocketAdapter):
    """FastAPI WebSocket adapter implementation."""
    
    def __init__(self, websocket: WebSocket):
        self.websocket = websocket
        
    async def accept_connection(self) -> None:
        await self.websocket.accept()
        
    async def receive_audio(self) -> Optional[AudioData]:
        try:
            data = await self.websocket.receive()
            
            if data["type"] == "websocket.disconnect":
                return None
                
            if data["type"] == "websocket.receive":
                if "bytes" in data:
                    # Convert PCM bytes to AudioData
                    audio_np = decode_pcm(data["bytes"])
                    return AudioData(data["bytes"], sample_rate=16000)
                elif "text" in data:
                    # Handle commands
                    command = json.loads(data["text"])
                    if command.get("action") == "stop":
                        return None
                        
        except WebSocketDisconnect:
            return None
        except Exception as e:
            logger.error(f"Error receiving audio: {e}")
            return None
            
    async def send_json(self, data: Dict[str, Any]) -> None:
        await self.websocket.send_json(data)
        
    async def send_audio(self, audio: AudioData) -> None:
        # Encode audio as base64 for JSON transmission
        wav_data = encode_wav(audio.data, audio.sample_rate)
        audio_b64 = base64.b64encode(wav_data).decode()
        
        await self.websocket.send_json({
            "type": "audio_response",
            "audio_data": audio_b64,
            "sample_rate": audio.sample_rate,
            "duration_ms": int(audio.duration * 1000)
        })
        
    async def close(self, code: int = 1000) -> None:
        await self.websocket.close(code)

# adapters/websocket/django.py - Similar implementation for Django Channels
# adapters/websocket/socketio.py - Similar implementation for Socket.IO
```

### Configuration System

```python
# adapters/config/yaml.py
from typing import Dict, Any
import yaml
from pathlib import Path
from ...core.interfaces import ConfigurationProvider

class YAMLConfigurationProvider(ConfigurationProvider):
    """YAML-based configuration provider."""
    
    def __init__(self, config_path: Path):
        self.config_path = config_path
        self._config = self._load_config()
        
    def _load_config(self) -> Dict[str, Any]:
        with open(self.config_path, 'r') as f:
            return yaml.safe_load(f)
            
    def get_stt_config(self) -> Dict[str, Any]:
        return self._config.get("stt_engine", {})
        
    def get_tts_config(self) -> Dict[str, Any]:
        return self._config.get("tts_engine", {})
        
    def get_conversation_config(self) -> Dict[str, Any]:
        return self._config.get("conversation", {})

# adapters/config/env.py - Environment variable provider
# adapters/config/dict.py - Dictionary-based provider for testing
```

### Usage Examples

#### Basic FastAPI Integration

```python
# examples/fastapi_basic.py
from fastapi import FastAPI, WebSocket
from realtime_conversation import ConversationEngine
from realtime_conversation.adapters.websocket import FastAPIWebSocketAdapter
from realtime_conversation.adapters.stt import WhisperSTTEngine
from realtime_conversation.adapters.tts import ChatterboxTTSEngine
from realtime_conversation.adapters.config import YAMLConfigurationProvider
from realtime_conversation.plugins.pause_detection import WebRTCPauseDetector
from realtime_conversation.plugins.response_generation import TemplateResponseGenerator

app = FastAPI()

# Initialize configuration
config = YAMLConfigurationProvider(Path("config.yaml"))

# Create and configure conversation engine
engine = ConversationEngine(config)
engine.configure_stt(WhisperSTTEngine(model_size="base"))
engine.configure_tts(ChatterboxTTSEngine(model_path="/path/to/model"))
engine.configure_pause_detection(WebRTCPauseDetector(aggressiveness=2))
engine.configure_response_generation(TemplateResponseGenerator())

@app.websocket("/conversation")
async def websocket_endpoint(websocket: WebSocket):
    adapter = FastAPIWebSocketAdapter(websocket)
    await engine.handle_conversation(adapter)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

#### Custom Middleware Example

```python
# examples/custom_pipeline.py
from realtime_conversation.core.interfaces import ConversationMiddleware, ConversationContext

class LoggingMiddleware(ConversationMiddleware):
    """Middleware that logs all conversation interactions."""
    
    async def process(self, context: ConversationContext, next_middleware) -> ConversationContext:
        # Pre-processing
        if context.transcription:
            logger.info(f"User said: {context.transcription.text}")
            
        # Call next middleware in chain
        context = await next_middleware(context)
        
        # Post-processing
        if context.response_text:
            logger.info(f"Assistant replied: {context.response_text}")
            
        return context

class AuthenticationMiddleware(ConversationMiddleware):
    """Middleware that validates user authentication."""
    
    def __init__(self, valid_tokens: set):
        self.valid_tokens = valid_tokens
        
    async def process(self, context: ConversationContext, next_middleware) -> ConversationContext:
        token = context.user_data.get("auth_token")
        if token not in self.valid_tokens:
            raise PermissionError("Invalid authentication token")
            
        return await next_middleware(context)

# Usage
engine.add_middleware(AuthenticationMiddleware({"valid_token_123"}))
engine.add_middleware(LoggingMiddleware())
```

## Implementation Roadmap

### Phase 1: Core Infrastructure (2-3 weeks)
- [ ] Define core interfaces and protocols
- [ ] Implement base ConversationEngine class
- [ ] Create FastAPI WebSocket adapter
- [ ] Extract and adapt current audio processing utilities
- [ ] Basic configuration system

### Phase 2: Component Adaptation (2-3 weeks)  
- [ ] Migrate current STT/TTS engines to new interfaces
- [ ] Implement pause detection plugins
- [ ] Create response generation adapters
- [ ] Audio codec abstraction layer
- [ ] Comprehensive test suite

### Phase 3: Extended Framework Support (2-3 weeks)
- [ ] Django Channels adapter
- [ ] Socket.IO adapter
- [ ] Alternative STT/TTS engine implementations
- [ ] Enhanced configuration providers
- [ ] Documentation and examples

### Phase 4: Advanced Features (3-4 weeks)
- [ ] Middleware system and plugins
- [ ] Event system and hooks  
- [ ] Performance monitoring and analytics
- [ ] Production deployment guides
- [ ] Community examples and tutorials

## Benefits of Proposed Architecture

### For Developers
- **Framework Freedom**: Use with any WebSocket framework
- **Component Flexibility**: Easily swap STT/TTS providers
- **Testing Simplicity**: Mock components using protocols
- **Extension Points**: Add custom middleware and plugins

### For Applications
- **Reduced Coupling**: Clean separation of concerns
- **Improved Maintainability**: Modular architecture
- **Enhanced Scalability**: Framework-agnostic deployment
- **Better Performance**: Optimized for specific use cases

### For Ecosystem
- **Reusability**: Single library supporting multiple frameworks
- **Standardization**: Common interfaces for conversation systems
- **Innovation**: Easy experimentation with new components
- **Community**: Shared plugins and extensions

## Migration Strategy

### Backward Compatibility
1. Keep existing FastAPI router as legacy option
2. Provide migration guide with examples
3. Gradual deprecation timeline with clear communication
4. Support both approaches during transition period

### Risk Mitigation
1. Comprehensive test coverage for all components
2. Performance benchmarking against current implementation
3. Staged rollout with feature flags
4. Fallback mechanisms for critical components

## Conclusion

The proposed standalone library architecture transforms the current monolithic WebSocket conversation router into a flexible, reusable component that can serve diverse applications and frameworks. While requiring significant refactoring effort, the long-term benefits of modularity, testability, and extensibility make this a worthwhile investment.

The phased implementation approach ensures minimal disruption to current functionality while systematically addressing architectural limitations. The resulting library will enable broader adoption and innovation in real-time conversation systems.