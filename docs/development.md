# Development Guide Documentation

## Overview

This guide provides comprehensive information for developers working with the STTS Server library integration system. It covers development environment setup, coding standards, testing practices, and extension patterns.

## Development Environment Setup

### Prerequisites

- Python 3.8 or higher
- Git for version control
- Code editor with Python support (VS Code, PyCharm, etc.)

### Local Development Setup

```bash
# Clone repository
git clone <repository-url>
cd stts-server

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/macOS
# or
venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt

# Install development dependencies
pip install -r requirements-dev.txt  # If available

# Create development configuration
cp config.yaml config-dev.yaml
```

### Development Configuration

Create a development-specific configuration:

```yaml
# config-dev.yaml
server:
  debug: true
  reload: true
  port: 8004

logging:
  level: "DEBUG"
  format: "detailed"

middleware:
  timing:
    enabled: true
    enable_historical_tracking: false  # Lighter for development
  logging:
    enabled: true
    log_level: "DEBUG"
  analytics:
    enabled: false  # Disabled for development

features:
  use_adapters: true
  library_integration: true
  enhanced_monitoring: false  # Lighter for development

# Development-specific paths
paths:
  reference_audio: "./dev_reference_audio"
  outputs: "./dev_outputs"
```

### Running Development Server

```bash
# Run with auto-reload for development
python server_v2.py --config config-dev.yaml --reload

# Or with uvicorn directly
uvicorn server_v2:app --reload --port 8004
```

## Project Structure

### Understanding the Codebase

```
stts-server/
├── server_v2.py                    # Main application entry point
├── config.py                       # Configuration management
├── models.py                       # Pydantic data models
├── utils.py                        # Shared utilities
├── engine.py                       # Legacy TTS engine
├── stt_engine.py                   # Legacy STT engine
├── adapters/
│   └── legacy_engines.py          # Adapter implementations
├── middleware/
│   └── base.py                     # Middleware framework
├── routers/
│   ├── core/                       # Core business logic
│   │   ├── tts.py                 # TTS endpoints
│   │   ├── stt.py                 # STT endpoints
│   │   └── conversation.py        # Conversation pipeline
│   ├── management/                 # System management
│   │   ├── config.py              # Configuration endpoints
│   │   └── files.py               # File management
│   └── websocket/                  # WebSocket endpoints
│       ├── websocket_stt.py
│       └── websocket_conversation_v2.py
├── tests/                          # Test files
├── docs/                           # Documentation
└── ui/                            # Web interface
```

### Core Components

#### 1. **Adapters** (`adapters/legacy_engines.py`)
Bridge legacy engines to library interfaces:

```python
class LegacySTTEngineAdapter:
    """Adapts legacy STT engine to library interface."""
    
    def __init__(self, legacy_engine: STTEngine):
        self.legacy_engine = legacy_engine
    
    async def transcribe(self, audio: AudioData, language: Optional[str] = None) -> Optional[TranscriptionResult]:
        """Implement library interface."""
        # Convert library format to legacy format
        # Call legacy engine
        # Convert result back to library format
        pass
```

#### 2. **Middleware** (`middleware/base.py`)
Request processing pipeline:

```python
class BaseMiddleware(Protocol):
    async def process(
        self, 
        context: RequestContext, 
        next_processor: Callable[[RequestContext], Awaitable[RequestContext]]
    ) -> RequestContext:
        """Process request through middleware."""
        pass
```

#### 3. **Routers** (`routers/`)
Organized endpoint handlers with dependency injection:

```python
@router.post("/tts")
async def synthesize_speech(
    request: CustomTTSRequest,
    tts_adapter: LegacyTTSEngineAdapter = Depends(get_tts_adapter)
):
    """TTS endpoint with adapter dependency."""
    pass
```

## Coding Standards

### Python Code Style

Follow PEP 8 with these specific guidelines:

```python
# Good: Clear function names and type hints
async def transcribe_audio_file(
    audio_file: UploadFile,
    language: Optional[str] = None,
    stt_adapter: LegacySTTEngineAdapter = Depends(get_stt_adapter)
) -> Dict[str, Any]:
    """Transcribe uploaded audio file to text.
    
    Args:
        audio_file: Uploaded audio file
        language: Optional language code
        stt_adapter: STT engine adapter
        
    Returns:
        Dictionary containing transcription result
    """
    pass

# Good: Descriptive variable names
request_context = RequestContext(
    request_id=generate_correlation_id(),
    request_type="stt",
    input_data={"filename": audio_file.filename}
)

# Good: Error handling with context
try:
    result = await stt_adapter.transcribe(audio_data, language)
except Exception as e:
    logger.error(f"STT transcription failed: {e}", exc_info=True)
    raise HTTPException(500, "Transcription failed")
```

### Type Hints

Use comprehensive type hints:

```python
from typing import Dict, List, Optional, Union, Any, Callable, Awaitable
from dataclasses import dataclass

@dataclass
class RequestContext:
    request_id: str
    request_type: str
    start_time: float = field(default_factory=time.time)
    input_data: Dict[str, Any] = field(default_factory=dict)
    output_data: Dict[str, Any] = field(default_factory=dict)
    metrics: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    status: str = "processing"

# Function type hints
async def process_with_middleware(
    context: RequestContext,
    processor: Callable[[RequestContext], Awaitable[RequestContext]]
) -> RequestContext:
    """Process context through middleware pipeline."""
    pass
```

### Documentation Standards

Use comprehensive docstrings:

```python
class LegacyTTSEngineAdapter:
    """Adapter bridging legacy TTS engine to library interface.
    
    This adapter provides a clean interface between the legacy engine.py
    global functions and the modern library-based TTS engine interface.
    It handles data conversion, error translation, and resource management.
    
    Attributes:
        default_temperature: Default synthesis temperature
        default_speed_factor: Default speech speed multiplier
        
    Example:
        >>> adapter = LegacyTTSEngineAdapter()
        >>> result = await adapter.synthesize("Hello world", voice_config)
        >>> print(f"Generated {len(result.audio_data)} bytes of audio")
    """
    
    def __init__(self, default_temperature: float = 0.8):
        """Initialize TTS adapter.
        
        Args:
            default_temperature: Default synthesis temperature (0.0-2.0)
        """
        self.default_temperature = default_temperature
    
    async def synthesize(
        self, 
        text: str, 
        voice_config: Dict[str, Any]
    ) -> Optional[SynthesisResult]:
        """Synthesize text to speech using legacy engine.
        
        Converts the provided text to speech using the configured voice
        and parameters. Handles text chunking for long inputs and provides
        proper error handling and resource cleanup.
        
        Args:
            text: Text to synthesize (max 5000 characters)
            voice_config: Voice configuration dictionary containing:
                - voice_path: Path to reference voice file
                - temperature: Synthesis temperature (0.0-2.0)
                - speed_factor: Speech speed multiplier (0.5-2.0)
                
        Returns:
            SynthesisResult containing audio data and metadata, or None on failure
            
        Raises:
            ValueError: If text is too long or voice_config is invalid
            SynthesisError: If synthesis fails
            
        Example:
            >>> voice_config = {
            ...     "voice_path": "/path/to/voice.wav",
            ...     "temperature": 0.8,
            ...     "speed_factor": 1.0
            ... }
            >>> result = await adapter.synthesize("Hello world", voice_config)
            >>> if result:
            ...     print(f"Synthesis successful: {result.duration}s audio")
        """
        pass
```

## Testing Practices

### Test Structure

Organize tests by component and type:

```
tests/
├── unit/                           # Unit tests
│   ├── test_adapters.py
│   ├── test_middleware.py
│   └── test_routers.py
├── integration/                    # Integration tests
│   ├── test_api_endpoints.py
│   ├── test_websocket.py
│   └── test_library_integration.py
├── performance/                    # Performance tests
│   ├── test_load.py
│   └── test_memory.py
└── fixtures/                       # Test data and fixtures
    ├── audio_samples/
    └── config_samples/
```

### Unit Testing

Use pytest with comprehensive fixtures:

```python
# tests/unit/test_adapters.py
import pytest
from unittest.mock import Mock, AsyncMock
from adapters.legacy_engines import LegacySTTEngineAdapter
from models import AudioData, TranscriptionResult

@pytest.fixture
def mock_legacy_engine():
    """Mock legacy STT engine."""
    engine = Mock()
    engine.model_loaded = True
    engine.transcribe_numpy_with_timing = Mock()
    return engine

@pytest.fixture
def stt_adapter(mock_legacy_engine):
    """STT adapter with mocked legacy engine."""
    return LegacySTTEngineAdapter(mock_legacy_engine)

@pytest.fixture
def sample_audio_data():
    """Sample audio data for testing."""
    return AudioData(
        data=b"fake_audio_data",
        sample_rate=16000,
        channels=1,
        format="wav"
    )

class TestLegacySTTEngineAdapter:
    """Test STT engine adapter functionality."""
    
    async def test_transcribe_success(self, stt_adapter, sample_audio_data, mock_legacy_engine):
        """Test successful transcription."""
        # Setup mock response
        mock_result = Mock()
        mock_result.text = "Hello world"
        mock_result.language = "en"
        mock_result.segments = []
        mock_legacy_engine.transcribe_numpy_with_timing.return_value = mock_result
        
        # Test transcription
        result = await stt_adapter.transcribe(sample_audio_data, "en")
        
        # Assertions
        assert result is not None
        assert result.text == "Hello world"
        assert result.language == "en"
        mock_legacy_engine.transcribe_numpy_with_timing.assert_called_once()
    
    async def test_transcribe_failure(self, stt_adapter, sample_audio_data, mock_legacy_engine):
        """Test transcription failure handling."""
        # Setup mock to raise exception
        mock_legacy_engine.transcribe_numpy_with_timing.side_effect = Exception("Transcription failed")
        
        # Test transcription
        result = await stt_adapter.transcribe(sample_audio_data, "en")
        
        # Should return None on failure
        assert result is None
    
    async def test_is_available(self, stt_adapter, mock_legacy_engine):
        """Test adapter availability check."""
        # Test when model loaded
        mock_legacy_engine.model_loaded = True
        assert await stt_adapter.is_available() == True
        
        # Test when model not loaded
        mock_legacy_engine.model_loaded = False
        assert await stt_adapter.is_available() == False
```

### Integration Testing

Test complete request flows:

```python
# tests/integration/test_api_endpoints.py
import pytest
from fastapi.testclient import TestClient
from server_v2 import app

@pytest.fixture
def client():
    """Test client for API testing."""
    return TestClient(app)

@pytest.fixture
def sample_tts_request():
    """Sample TTS request data."""
    return {
        "text": "Hello world",
        "voice_id": "test_voice",
        "temperature": 0.8
    }

class TestTTSEndpoints:
    """Test TTS API endpoints."""
    
    def test_tts_synthesis(self, client, sample_tts_request):
        """Test TTS synthesis endpoint."""
        response = client.post("/tts", json=sample_tts_request)
        
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("audio/")
        assert len(response.content) > 0
    
    def test_tts_voices_list(self, client):
        """Test TTS voices listing."""
        response = client.get("/tts/voices")
        
        assert response.status_code == 200
        data = response.json()
        assert "data" in data
        assert "voices" in data["data"]
    
    def test_tts_statistics(self, client):
        """Test TTS statistics endpoint."""
        response = client.get("/tts/statistics")
        
        assert response.status_code == 200
        data = response.json()
        assert "system" in data["data"]
        assert "middleware" in data["data"]
```

### Performance Testing

Monitor performance during development:

```python
# tests/performance/test_load.py
import asyncio
import time
import statistics
from concurrent.futures import ThreadPoolExecutor
import requests

class PerformanceTest:
    """Performance testing for API endpoints."""
    
    def __init__(self, base_url: str = "http://localhost:8004"):
        self.base_url = base_url
    
    async def test_tts_performance(self, num_requests: int = 100):
        """Test TTS endpoint performance under load."""
        print(f"Testing TTS performance with {num_requests} requests...")
        
        response_times = []
        errors = 0
        
        async def make_request():
            start_time = time.time()
            try:
                response = requests.post(
                    f"{self.base_url}/tts",
                    json={"text": "Performance test", "voice_id": "test"}
                )
                response_time = time.time() - start_time
                response_times.append(response_time)
                
                if response.status_code != 200:
                    errors += 1
                    
            except Exception:
                errors += 1
                
        # Run concurrent requests
        tasks = [make_request() for _ in range(num_requests)]
        await asyncio.gather(*tasks)
        
        # Calculate statistics
        if response_times:
            avg_time = statistics.mean(response_times)
            median_time = statistics.median(response_times)
            p95_time = sorted(response_times)[int(len(response_times) * 0.95)]
            
            print(f"Results:")
            print(f"  Average response time: {avg_time:.3f}s")
            print(f"  Median response time: {median_time:.3f}s")
            print(f"  95th percentile: {p95_time:.3f}s")
            print(f"  Error rate: {errors/num_requests:.3%}")
            
            # Performance assertions
            assert avg_time < 5.0, f"Average response time too high: {avg_time:.3f}s"
            assert errors/num_requests < 0.05, f"Error rate too high: {errors/num_requests:.3%}"

if __name__ == "__main__":
    test = PerformanceTest()
    asyncio.run(test.test_tts_performance(50))
```

## Extension Patterns

### Adding New Middleware

Create custom middleware following the base pattern:

```python
# middleware/custom_middleware.py
import time
from typing import Dict, Any, Callable, Awaitable
from middleware.base import RequestContext

class CustomMiddleware:
    """Custom middleware for specific functionality."""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.enabled = self.config.get("enabled", True)
    
    async def process(
        self,
        context: RequestContext,
        next_processor: Callable[[RequestContext], Awaitable[RequestContext]]
    ) -> RequestContext:
        """Process request through custom middleware."""
        if not self.enabled:
            return await next_processor(context)
        
        # Pre-processing
        context.metadata["custom_start"] = time.time()
        
        try:
            # Call next middleware/processor
            result_context = await next_processor(context)
            
            # Post-processing
            processing_time = time.time() - context.metadata["custom_start"]
            result_context.metadata["custom_processing_time"] = processing_time
            
            return result_context
            
        except Exception as e:
            # Error handling
            context.metadata["custom_error"] = str(e)
            raise

# Register middleware in pipeline creation
def create_middleware_pipeline() -> MiddlewarePipeline:
    pipeline = MiddlewarePipeline()
    
    # Add built-in middleware
    pipeline.add_middleware(TimingMiddleware())
    pipeline.add_middleware(LoggingMiddleware())
    
    # Add custom middleware
    custom_config = config_manager.get_dict("middleware.custom", {})
    if custom_config.get("enabled", False):
        pipeline.add_middleware(CustomMiddleware(custom_config))
    
    return pipeline
```

### Adding New Adapters

Create adapters for new engines or libraries:

```python
# adapters/new_engine_adapter.py
from typing import Optional, Dict, Any
from models import AudioData, SynthesisResult
from .new_engine import NewTTSEngine  # Your new engine

class NewTTSEngineAdapter:
    """Adapter for new TTS engine."""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.engine = NewTTSEngine(self.config)
    
    async def synthesize(
        self, 
        text: str, 
        voice_config: Dict[str, Any]
    ) -> Optional[SynthesisResult]:
        """Synthesize text using new engine."""
        try:
            # Convert voice_config to engine-specific format
            engine_config = self._convert_voice_config(voice_config)
            
            # Call new engine
            audio_data = await self.engine.generate_speech(text, engine_config)
            
            # Convert result to library format
            return SynthesisResult(
                audio_data=AudioData(
                    data=audio_data,
                    sample_rate=22050,
                    channels=1,
                    format="wav"
                ),
                text=text,
                voice_id=voice_config.get("voice_id", "default"),
                duration=len(audio_data) / 22050  # Calculate duration
            )
            
        except Exception as e:
            logger.error(f"New TTS engine synthesis failed: {e}")
            return None
    
    def _convert_voice_config(self, voice_config: Dict[str, Any]) -> Dict[str, Any]:
        """Convert library voice config to engine-specific format."""
        return {
            "voice_file": voice_config.get("voice_path"),
            "temperature": voice_config.get("temperature", 0.8),
            "speed": voice_config.get("speed_factor", 1.0)
        }
    
    @property
    def model_loaded(self) -> bool:
        """Check if engine model is loaded."""
        return self.engine.is_ready()

# Add factory function
def create_new_tts_adapter() -> NewTTSEngineAdapter:
    """Create new TTS adapter with configuration."""
    config = config_manager.get_dict("new_tts_engine", {})
    return NewTTSEngineAdapter(config)

# Update dependency injection
def get_tts_adapter() -> Union[LegacyTTSEngineAdapter, NewTTSEngineAdapter]:
    """Get TTS adapter based on configuration."""
    adapter_type = config_manager.get_string("tts_engine.adapter_type", "legacy")
    
    if adapter_type == "new":
        return create_new_tts_adapter()
    else:
        return create_legacy_tts_adapter()
```

### Adding New Routers

Create focused routers for new functionality:

```python
# routers/new_feature/new_endpoints.py
from fastapi import APIRouter, Depends, HTTPException
from typing import Dict, Any
from models import NewFeatureRequest, NewFeatureResponse

router = APIRouter(prefix="/new-feature", tags=["new-feature"])

@router.post("/process", response_model=NewFeatureResponse)
async def process_new_feature(
    request: NewFeatureRequest,
    # Add dependencies as needed
    config_adapter: ConfigurationAdapter = Depends(get_config_adapter)
) -> NewFeatureResponse:
    """Process new feature request."""
    try:
        # Validate request
        if not request.input_data:
            raise HTTPException(400, "Input data required")
        
        # Process request
        result = await process_new_feature_logic(request, config_adapter)
        
        return NewFeatureResponse(
            status="success",
            result=result,
            processing_time=time.time() - start_time
        )
        
    except Exception as e:
        logger.error(f"New feature processing failed: {e}")
        raise HTTPException(500, "Processing failed")

@router.get("/status")
async def get_new_feature_status() -> Dict[str, Any]:
    """Get new feature system status."""
    return {
        "status": "ready",
        "version": "1.0.0",
        "features_available": True
    }

# Register router in main application
# In server_v2.py:
from routers.new_feature import new_endpoints
app.include_router(new_endpoints.router)
```

### Adding Configuration Options

Extend configuration system for new features:

```python
# In config.py - add to DEFAULT_CONFIG
DEFAULT_CONFIG = {
    # Existing configuration...
    
    "new_feature": {
        "enabled": True,
        "option1": "default_value",
        "option2": 42,
        "advanced_settings": {
            "timeout": 30,
            "retry_count": 3
        }
    }
}

# Add accessor functions
def get_new_feature_enabled() -> bool:
    return config_manager.get_bool("new_feature.enabled", True)

def get_new_feature_option1() -> str:
    return config_manager.get_string("new_feature.option1", "default_value")

def get_new_feature_advanced_settings() -> Dict[str, Any]:
    return config_manager.get_dict("new_feature.advanced_settings", {})

# Add validation rules
def validate_new_feature_config(config_data: Dict[str, Any]) -> List[str]:
    """Validate new feature configuration."""
    errors = []
    new_feature = config_data.get("new_feature", {})
    
    # Validate option ranges
    option2 = new_feature.get("option2", 42)
    if not 1 <= option2 <= 100:
        errors.append("new_feature.option2 must be between 1 and 100")
    
    # Validate advanced settings
    advanced = new_feature.get("advanced_settings", {})
    timeout = advanced.get("timeout", 30)
    if timeout <= 0:
        errors.append("new_feature.advanced_settings.timeout must be positive")
    
    return errors

# Register validation function
CONFIG_VALIDATORS.append(validate_new_feature_config)
```

## Debugging and Profiling

### Debug Configuration

Use debug-friendly settings:

```yaml
# config-debug.yaml
server:
  debug: true
  reload: true

logging:
  level: "DEBUG"
  format: "detailed"

middleware:
  timing:
    enabled: true
    enable_historical_tracking: true
  logging:
    enabled: true
    include_request_data: true
    include_response_data: true

features:
  enhanced_monitoring: true
```

### Debug Tools

```python
# debug_tools.py
import logging
import traceback
from typing import Any, Dict

def setup_debug_logging():
    """Setup comprehensive debug logging."""
    logging.basicConfig(
        level=logging.DEBUG,
        format='[%(asctime)s] %(levelname)s [%(name)s:%(lineno)d] %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('debug.log')
        ]
    )

def debug_request_context(context: RequestContext):
    """Debug request context information."""
    print(f"🔍 Request Context Debug:")
    print(f"  ID: {context.request_id}")
    print(f"  Type: {context.request_type}")
    print(f"  Status: {context.status}")
    print(f"  Input keys: {list(context.input_data.keys())}")
    print(f"  Output keys: {list(context.output_data.keys())}")
    print(f"  Metrics: {context.metrics}")
    print(f"  Metadata: {context.metadata}")

def debug_middleware_pipeline(pipeline: MiddlewarePipeline):
    """Debug middleware pipeline information."""
    print(f"🔍 Middleware Pipeline Debug:")
    print(f"  Middleware count: {len(pipeline.middleware)}")
    for i, middleware in enumerate(pipeline.middleware):
        print(f"  {i+1}. {type(middleware).__name__}")

def profile_function(func):
    """Decorator for profiling function execution."""
    def wrapper(*args, **kwargs):
        import cProfile
        import pstats
        
        profiler = cProfile.Profile()
        profiler.enable()
        
        try:
            result = func(*args, **kwargs)
        finally:
            profiler.disable()
            
            # Print profiling results
            stats = pstats.Stats(profiler)
            stats.sort_stats('cumulative')
            stats.print_stats(10)  # Top 10 functions
        
        return result
    return wrapper

# Usage example
@profile_function
def expensive_operation():
    """Function to profile."""
    pass
```

### Performance Profiling

```python
# performance_profiler.py
import asyncio
import time
import psutil
import gc
from typing import List, Dict, Any

class PerformanceProfiler:
    """Profile application performance."""
    
    def __init__(self):
        self.metrics = []
        self.start_time = time.time()
    
    def start_profiling(self):
        """Start performance profiling."""
        self.start_time = time.time()
        self.metrics = []
    
    def record_metric(self, name: str, value: float, metadata: Dict[str, Any] = None):
        """Record a performance metric."""
        self.metrics.append({
            'name': name,
            'value': value,
            'timestamp': time.time(),
            'metadata': metadata or {}
        })
    
    def get_memory_usage(self) -> Dict[str, float]:
        """Get current memory usage."""
        process = psutil.Process()
        memory_info = process.memory_info()
        
        return {
            'rss_mb': memory_info.rss / 1024 / 1024,
            'vms_mb': memory_info.vms / 1024 / 1024,
            'percent': process.memory_percent()
        }
    
    def get_cpu_usage(self) -> float:
        """Get current CPU usage."""
        return psutil.cpu_percent(interval=1)
    
    def profile_endpoint(self, endpoint_func):
        """Decorator to profile endpoint performance."""
        async def wrapper(*args, **kwargs):
            start_time = time.time()
            start_memory = self.get_memory_usage()
            
            try:
                result = await endpoint_func(*args, **kwargs)
                
                end_time = time.time()
                end_memory = self.get_memory_usage()
                
                # Record metrics
                self.record_metric('response_time', end_time - start_time, {
                    'endpoint': endpoint_func.__name__,
                    'success': True
                })
                
                self.record_metric('memory_usage', end_memory['rss_mb'], {
                    'endpoint': endpoint_func.__name__,
                    'memory_delta': end_memory['rss_mb'] - start_memory['rss_mb']
                })
                
                return result
                
            except Exception as e:
                end_time = time.time()
                self.record_metric('response_time', end_time - start_time, {
                    'endpoint': endpoint_func.__name__,
                    'success': False,
                    'error': str(e)
                })
                raise
                
        return wrapper

# Global profiler instance
profiler = PerformanceProfiler()

# Usage in routers
@profiler.profile_endpoint
async def synthesize_speech(request: TTSRequest):
    """Profiled TTS endpoint."""
    pass
```

## Best Practices

### 1. **Code Organization**
- Keep routers focused on single responsibilities
- Use dependency injection for testability
- Separate business logic from HTTP handling

### 2. **Error Handling**
- Use specific exception types
- Provide meaningful error messages
- Log errors with context

### 3. **Performance**
- Profile new features during development
- Use async/await for I/O operations
- Monitor memory usage in long-running processes

### 4. **Testing**
- Write tests before implementing features
- Use comprehensive fixtures
- Test error conditions

### 5. **Documentation**
- Document all public APIs
- Include examples in docstrings
- Keep documentation updated with code changes

### 6. **Configuration**
- Make features configurable
- Validate configuration at startup
- Use environment-specific configurations

### 7. **Monitoring**
- Add metrics to new features
- Use correlation IDs for request tracking
- Monitor performance impact of changes