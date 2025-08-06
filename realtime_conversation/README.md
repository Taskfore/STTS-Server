# Real-time WebSocket Conversation Library

A framework-agnostic Python library for building real-time audio conversation systems that integrate Speech-to-Text (STT), Text-to-Speech (TTS), and response generation through WebSocket connections.

## Features

- **Framework Agnostic**: Support for FastAPI, Django Channels, Socket.IO, and custom WebSocket implementations
- **Modular Architecture**: Pluggable STT engines, TTS engines, pause detection, and response generation
- **Middleware Pipeline**: Extensible middleware system for logging, authentication, analytics, and custom processing
- **Protocol-Based Design**: Type-safe interfaces using Python protocols for easy testing and mocking
- **Audio Processing**: Built-in audio codec support with PCM, WAV, and extensible format handling
- **Configuration Flexible**: YAML, environment variable, and dictionary-based configuration providers

## Quick Start

### Basic FastAPI Integration

```python
from fastapi import FastAPI, WebSocket
from realtime_conversation import ConversationEngine
from realtime_conversation.adapters.websocket import FastAPIWebSocketAdapter
from realtime_conversation.adapters.stt import WhisperSTTEngine
from realtime_conversation.adapters.tts import ChatterboxTTSEngine
from realtime_conversation.plugins.pause_detection import WebRTCPauseDetector
from realtime_conversation.plugins.response_generation import EchoResponseGenerator

app = FastAPI()

# Create and configure conversation engine
engine = ConversationEngine()
engine.configure_stt(WhisperSTTEngine(model_size="base"))
engine.configure_tts(ChatterboxTTSEngine())
engine.configure_pause_detection(WebRTCPauseDetector())
engine.configure_response_generation(EchoResponseGenerator())

@app.websocket("/ws/conversation")
async def websocket_endpoint(websocket: WebSocket):
    adapter = FastAPIWebSocketAdapter(websocket)
    await engine.handle_conversation(adapter)
```

### With Configuration and Middleware

```python
from pathlib import Path
from realtime_conversation.adapters.config import YAMLConfigurationProvider
from realtime_conversation.plugins.middleware import LoggingMiddleware, TimingMiddleware

# Load configuration
config = YAMLConfigurationProvider(Path("config.yaml"))

# Create engine with middleware
engine = ConversationEngine(config)
engine.add_middleware(TimingMiddleware())
engine.add_middleware(LoggingMiddleware())

# Configure components from config
stt_config = config.get_stt_config()
engine.configure_stt(WhisperSTTEngine(**stt_config))
```

## Library Structure

```
realtime_conversation/
├── core/                     # Core engine and interfaces
│   ├── interfaces.py         # Protocol definitions
│   ├── engine.py            # Main ConversationEngine
│   └── pipeline.py          # Middleware pipeline
├── adapters/                # Framework and service adapters
│   ├── websocket/           # WebSocket framework adapters
│   │   ├── fastapi.py       # FastAPI WebSocket adapter
│   │   └── base.py          # Base WebSocket adapter
│   ├── stt/                 # STT engine adapters
│   │   ├── whisper.py       # OpenAI Whisper adapter
│   │   └── base.py          # Base STT adapter
│   ├── tts/                 # TTS engine adapters
│   │   ├── chatterbox.py    # Chatterbox TTS adapter
│   │   └── base.py          # Base TTS adapter
│   └── config/              # Configuration providers
│       ├── yaml.py          # YAML configuration
│       ├── env.py           # Environment variables
│       └── dict.py          # Dictionary configuration
├── plugins/                 # Pluggable components
│   ├── pause_detection/     # Voice activity detection
│   │   ├── webrtc.py        # WebRTC VAD
│   │   └── energy.py        # Energy-based detection
│   ├── response_generation/ # Response generators
│   │   ├── echo.py          # Echo responses
│   │   └── template.py      # Template-based responses
│   └── middleware/          # Pipeline middleware
│       ├── logging.py       # Conversation logging
│       ├── timing.py        # Performance monitoring
│       ├── auth.py          # Authentication
│       └── analytics.py     # Usage analytics
├── audio/                   # Audio processing utilities
│   ├── codecs.py           # Audio format handling
│   ├── buffers.py          # Audio buffer management
│   └── processing.py       # Audio processing functions
└── examples/               # Usage examples
    ├── fastapi_basic.py    # Basic FastAPI integration
    └── custom_pipeline.py  # Advanced custom pipeline
```

## Core Concepts

### ConversationEngine

The main orchestrator that coordinates STT, TTS, response generation, and pause detection through a configurable middleware pipeline.

```python
engine = ConversationEngine(config_provider)
engine.configure_stt(stt_engine)
engine.configure_tts(tts_engine)
engine.configure_pause_detection(pause_detector)
engine.configure_response_generation(response_generator)
engine.add_middleware(middleware)
```

### Protocol-Based Architecture

All major components implement Python protocols for type safety and easy testing:

```python
class STTEngine(Protocol):
    async def transcribe(self, audio: AudioData, language: Optional[str] = None) -> Optional[TranscriptionResult]:
        ...
    
    @property
    def model_loaded(self) -> bool:
        ...

class TTSEngine(Protocol):
    async def synthesize(self, text: str, voice_config: Dict[str, Any]) -> Optional[SynthesisResult]:
        ...
```

### Middleware Pipeline

Extensible processing pipeline with middleware for cross-cutting concerns:

```python
class LoggingMiddleware(ConversationMiddleware):
    async def process(self, context: ConversationContext, next_middleware) -> ConversationContext:
        # Pre-processing
        logger.info(f"Processing: {context.transcription.text}")
        
        # Call next middleware
        result = await next_middleware(context)
        
        # Post-processing
        logger.info(f"Response: {result.response_text}")
        return result
```

### Configuration System

Flexible configuration with multiple providers:

```python
# YAML configuration
config = YAMLConfigurationProvider("config.yaml")

# Environment variables
config = EnvConfigurationProvider(prefix="CONV_")

# Dictionary (for testing)
config = DictConfigurationProvider({
    "stt_engine": {"model_size": "base"},
    "tts_engine": {"temperature": 0.7}
})
```

## Components

### STT Engines

- **WhisperSTTEngine**: OpenAI Whisper with device auto-detection and model size selection
- **BaseSTTEngine**: Base class for custom STT implementations

### TTS Engines

- **ChatterboxTTSEngine**: Chatterbox TTS with voice cloning support
- **BaseTTSEngine**: Base class for custom TTS implementations

### Pause Detection

- **WebRTCPauseDetector**: WebRTC VAD with configurable aggressiveness
- **EnergyPauseDetector**: Energy-based detection with adaptive thresholds

### Response Generation

- **EchoResponseGenerator**: Simple echo responses for testing
- **TemplateResponseGenerator**: Pattern-matching with variable substitution

### Middleware

- **LoggingMiddleware**: Comprehensive conversation logging
- **TimingMiddleware**: Performance monitoring and statistics
- **AuthenticationMiddleware**: Token-based authentication
- **AnalyticsMiddleware**: Usage analytics and metrics collection

## WebSocket Adapters

### FastAPI

```python
from realtime_conversation.adapters.websocket import FastAPIWebSocketAdapter

@app.websocket("/ws/conversation")
async def websocket_endpoint(websocket: WebSocket):
    adapter = FastAPIWebSocketAdapter(websocket)
    await engine.handle_conversation(adapter)
```

### Custom Adapter

```python
from realtime_conversation.core.interfaces import WebSocketAdapter

class CustomWebSocketAdapter(WebSocketAdapter):
    async def receive_audio(self) -> Optional[AudioData]:
        # Custom audio reception logic
        ...
    
    async def send_audio(self, audio: AudioData) -> None:
        # Custom audio transmission logic
        ...
```

## Configuration Examples

### YAML Configuration

```yaml
stt_engine:
  model_size: "base"
  device: "auto"
  language: "en"

tts_engine:
  device: "cuda"
  temperature: 0.7
  speed_factor: 1.0

conversation:
  response_mode: "template"
  max_history_length: 50

pause_detection:
  aggressiveness: 2
  min_speech_frames: 3
  min_pause_frames: 10

audio:
  sample_rate: 16000
  channels: 1
  buffer_duration: 10.0
```

### Environment Variables

```bash
CONV_STT_ENGINE_MODEL_SIZE=base
CONV_STT_ENGINE_DEVICE=auto
CONV_TTS_ENGINE_DEVICE=cuda
CONV_CONVERSATION_RESPONSE_MODE=template
CONV_PAUSE_DETECTION_AGGRESSIVENESS=2
```

## Advanced Usage

### Custom Middleware

```python
class CustomMiddleware(ConversationMiddleware):
    async def process(self, context: ConversationContext, next_middleware) -> ConversationContext:
        # Add custom logic before processing
        start_time = time.time()
        
        # Process through pipeline
        result = await next_middleware(context)
        
        # Add custom logic after processing
        processing_time = time.time() - start_time
        result.metadata["custom_timing"] = processing_time
        
        return result

engine.add_middleware(CustomMiddleware())
```

### Authentication with Permissions

```python
from realtime_conversation.plugins.middleware.auth import TokenAuthenticationMiddleware

auth_middleware = TokenAuthenticationMiddleware(
    valid_tokens={"token1", "token2", "token3"},
    allow_anonymous=False
)
auth_middleware.set_required_permissions({"conversation", "audio_processing"})
engine.add_middleware(auth_middleware)
```

### Analytics and Monitoring

```python
from realtime_conversation.plugins.middleware import AnalyticsMiddleware

analytics = AnalyticsMiddleware(
    track_usage=True,
    track_performance=True,
    track_errors=True,
    retention_days=30
)
engine.add_middleware(analytics)

# Get statistics
stats = analytics.get_usage_statistics()
performance = analytics.get_performance_statistics()
```

## Migration from Legacy System

The library includes a compatibility layer for migrating from the original tightly-coupled implementation:

```python
# Legacy endpoint (maintains same interface)
@router.websocket("/ws/conversation")
async def legacy_conversation(websocket: WebSocket, ...):
    # Uses new library internally
    await websocket_conversation_v2(websocket, ...)

# New endpoint (full features)
@router.websocket("/ws/conversation/v2")
async def modern_conversation(websocket: WebSocket, ...):
    engine = await get_conversation_engine()
    adapter = FastAPIWebSocketAdapter(websocket)
    await engine.handle_conversation(adapter)
```

## Testing

The protocol-based design makes testing straightforward:

```python
class MockSTTEngine:
    async def transcribe(self, audio: AudioData, language: Optional[str] = None) -> Optional[TranscriptionResult]:
        return TranscriptionResult(
            text="Mock transcription",
            language="en",
            segments=[],
            confidence=1.0
        )
    
    @property
    def model_loaded(self) -> bool:
        return True

# Use in tests
engine = ConversationEngine()
engine.configure_stt(MockSTTEngine())
```

## Error Handling

The library provides comprehensive error handling with proper logging and graceful degradation:

```python
try:
    await engine.handle_conversation(adapter)
except Exception as e:
    logger.error(f"Conversation error: {e}")
    await adapter.send_json({"type": "error", "message": str(e)})
```

## Performance Considerations

- **Model Loading**: Models are loaded once and cached for reuse
- **Thread Pool Processing**: CPU-intensive operations run in thread pools
- **Memory Management**: Automatic cleanup of audio buffers and temporary data
- **Device Detection**: Automatic GPU/CPU selection with fallback logic

## Dependencies

- `fastapi` - For FastAPI WebSocket adapter
- `numpy` - Audio processing
- `torch` - ML model support
- `openai-whisper` - STT engine
- `webrtcvad` - Voice activity detection (optional)
- `pyyaml` - YAML configuration support
- `scipy` - Audio processing utilities

## License

This library is part of the Career Link STTS Server project.

## Contributing

1. Follow the protocol-based architecture
2. Add comprehensive type hints
3. Include proper error handling and logging
4. Write tests for new components
5. Update documentation

For more examples, see the `examples/` directory and the existing FastAPI router integration.