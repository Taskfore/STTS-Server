# Adapter Pattern Documentation

## Overview

The adapter pattern provides a clean bridge between the existing legacy engines and the `realtime_conversation` library interfaces. This allows for gradual migration while maintaining full backward compatibility and introducing modern architectural patterns.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Library Interfaces                       │
│              (realtime_conversation)                        │
├─────────────────────┬─────────────────────┬─────────────────┤
│   STTEngine         │    TTSEngine        │ ConfigProvider │
│   Protocol          │    Protocol         │ Protocol        │
└─────────────────────┼─────────────────────┼─────────────────┘
                      │                     │
                      ▼                     ▼
┌─────────────────────────────────────────────────────────────┐
│                     Adapter Layer                           │
├─────────────────────┬─────────────────────┬─────────────────┤
│ LegacySTTEngine     │ LegacyTTSEngine     │ Configuration   │
│ Adapter             │ Adapter             │ Adapter         │
└─────────────────────┼─────────────────────┼─────────────────┘
                      │                     │
                      ▼                     ▼
┌─────────────────────────────────────────────────────────────┐
│                    Legacy Engines                           │
├─────────────────────┬─────────────────────┬─────────────────┤
│   stt_engine.py     │    engine.py        │   config.py     │
│   (STTEngine)       │  (Global functions) │ (ConfigManager) │
└─────────────────────┴─────────────────────┴─────────────────┘
```

## Adapter Implementation

### Base Adapter Concepts

All adapters implement the following principles:

1. **Protocol Compliance**: Implement library interfaces exactly
2. **Error Translation**: Convert legacy errors to library-compatible formats
3. **Data Transformation**: Convert between legacy and library data formats
4. **Resource Management**: Proper cleanup and resource handling

### STT Adapter

#### Interface Implementation

```python
class LegacySTTEngineAdapter:
    """Bridges legacy STTEngine to library STTEngine protocol."""
    
    async def transcribe(
        self, 
        audio: AudioData, 
        language: Optional[str] = None
    ) -> Optional[TranscriptionResult]:
        """Convert AudioData → legacy format → library result."""
        
    async def is_available(self) -> bool:
        """Check if the STT engine is ready."""
        
    @property
    def model_loaded(self) -> bool:
        """Check if the STT model is loaded."""
```

#### Data Transformation

```python
def _convert_legacy_transcription_result(self, legacy_result) -> TranscriptionResult:
    """Convert legacy result to library format."""
    segments = []
    for seg in legacy_result.segments:
        segment = TranscriptionSegment(
            text=seg.text.strip(),
            start=float(seg.start),
            end=float(seg.end),
            confidence=getattr(seg, 'confidence', None)
        )
        segments.append(segment)
    
    return TranscriptionResult(
        text=legacy_result.text.strip(),
        language=legacy_result.language,
        segments=segments,
        confidence=None,
        partial=False
    )
```

#### Usage Example

```python
# Create adapter
stt_adapter = LegacySTTEngineAdapter(legacy_stt_engine)

# Use library interface
audio_data = AudioData(data=audio_bytes, sample_rate=16000, channels=1)
result = await stt_adapter.transcribe(audio_data, language="en")

print(f"Transcribed: {result.text}")
print(f"Language: {result.language}")
print(f"Segments: {len(result.segments)}")
```

### TTS Adapter

#### Interface Implementation

```python
class LegacyTTSEngineAdapter:
    """Bridges legacy engine.py to library TTSEngine protocol."""
    
    async def synthesize(
        self, 
        text: str, 
        voice_config: Dict[str, Any]
    ) -> Optional[SynthesisResult]:
        """Convert voice_config → legacy call → library result."""
        
    async def list_voices(self) -> List[Dict[str, Any]]:
        """List available voices using existing utilities."""
        
    @property
    def model_loaded(self) -> bool:
        """Check if the TTS model is loaded."""
```

#### Voice Configuration Mapping

```python
# Library interface expects voice_config dict
voice_config = {
    "voice_path": "/path/to/voice.wav",
    "voice_id": "female_voice_01",
    "temperature": 0.8,
    "exaggeration": 0.5,
    "cfg_weight": 0.5,
    "seed": 0,
    "speed_factor": 1.0
}

# Adapter maps to legacy engine.synthesize() parameters
audio_tensor, sample_rate = engine.synthesize(
    text=text,
    audio_prompt_path=voice_config["voice_path"],
    temperature=voice_config["temperature"],
    exaggeration=voice_config["exaggeration"],
    cfg_weight=voice_config["cfg_weight"],
    seed=voice_config["seed"]
)
```

#### Audio Data Conversion

```python
def _create_audio_data(self, audio_np: np.ndarray, sample_rate: int) -> AudioData:
    """Convert numpy array to library AudioData format."""
    # Convert to 16-bit PCM bytes
    audio_int16 = (audio_np * 32767).astype(np.int16)
    audio_bytes = audio_int16.tobytes()
    
    return AudioData(
        data=audio_bytes,
        sample_rate=sample_rate,
        channels=1,
        format="pcm"
    )
```

### Configuration Adapter

#### Interface Implementation

```python
class ConfigurationAdapter:
    """Bridges legacy config.py to library ConfigurationProvider."""
    
    def get_stt_config(self) -> Dict[str, Any]:
        """Get STT configuration from legacy config system."""
        return {
            "model_size": self.config_manager.get_string("stt_engine.model_size", "base"),
            "device": self.config_manager.get_string("stt_engine.device", "auto"),
            "language": self.config_manager.get_string("stt_engine.language", "auto")
        }
    
    def get_tts_config(self) -> Dict[str, Any]:
        """Get TTS configuration from legacy config system."""
        return {
            "device": self.config_manager.get_string("tts_engine.device", "auto"),
            "temperature": self.config_manager.get_float("gen.default_temperature", 0.8),
            "speed_factor": self.config_manager.get_float("gen.default_speed_factor", 1.0)
        }
```

## Factory Functions

### Adapter Creation

```python
def create_legacy_stt_adapter() -> LegacySTTEngineAdapter:
    """Create STT adapter with existing engine."""
    legacy_engine = STTEngine()
    return LegacySTTEngineAdapter(legacy_engine)

def create_legacy_tts_adapter() -> LegacyTTSEngineAdapter:
    """Create TTS adapter."""
    return LegacyTTSEngineAdapter()

def create_config_adapter() -> ConfigurationAdapter:
    """Create configuration adapter."""
    return ConfigurationAdapter()
```

### Dependency Injection

```python
# In router dependencies
def get_stt_adapter() -> LegacySTTEngineAdapter:
    """Dependency injection for STT adapter."""
    global _stt_adapter_instance
    
    if _stt_adapter_instance is None:
        _stt_adapter_instance = create_legacy_stt_adapter()
    
    return _stt_adapter_instance

# Usage in endpoint
@router.post("/transcribe")
async def transcribe_audio(
    audio_file: UploadFile,
    stt_adapter: LegacySTTEngineAdapter = Depends(get_stt_adapter)
):
    # Use adapter with library interface
    audio_data = AudioData(...)
    result = await stt_adapter.transcribe(audio_data)
    return {"text": result.text}
```

## Error Handling

### Exception Translation

```python
async def transcribe(self, audio: AudioData, language: Optional[str] = None) -> Optional[TranscriptionResult]:
    try:
        # Legacy engine call
        legacy_result = self.legacy_engine.transcribe_numpy_with_timing(audio_np, language)
        return self._convert_legacy_result(legacy_result)
        
    except LegacyEngineError as e:
        logger.error(f"Legacy engine error: {e}")
        return None  # Library expects None on failure
        
    except Exception as e:
        logger.error(f"Adapter error: {e}", exc_info=True)
        return None
```

### Graceful Degradation

```python
async def is_available(self) -> bool:
    """Check adapter availability with fallback."""
    try:
        return self.legacy_engine.model_loaded
    except Exception as e:
        logger.warning(f"Error checking engine availability: {e}")
        return False  # Safe default
```

## Performance Considerations

### Minimal Overhead

- **Direct Calls**: Adapters make direct calls to legacy engines
- **Efficient Conversion**: Minimal data transformation overhead
- **Caching**: Adapter instances are cached and reused
- **Lazy Loading**: Components loaded only when needed

### Memory Management

```python
def __del__(self):
    """Cleanup adapter resources."""
    if hasattr(self, 'legacy_engine'):
        # Perform any necessary cleanup
        pass
```

## Testing Adapters

### Unit Tests

```python
import pytest
from adapters.legacy_engines import LegacySTTEngineAdapter

@pytest.fixture
def mock_legacy_engine():
    """Mock legacy engine for testing."""
    engine = Mock()
    engine.model_loaded = True
    return engine

@pytest.fixture
def stt_adapter(mock_legacy_engine):
    """STT adapter with mocked engine."""
    return LegacySTTEngineAdapter(mock_legacy_engine)

async def test_transcribe_success(stt_adapter):
    """Test successful transcription."""
    # Mock legacy result
    mock_result = Mock()
    mock_result.text = "Hello world"
    mock_result.language = "en"
    mock_result.segments = []
    
    stt_adapter.legacy_engine.transcribe_numpy_with_timing.return_value = mock_result
    
    # Test adapter
    audio_data = AudioData(data=b"audio", sample_rate=16000, channels=1)
    result = await stt_adapter.transcribe(audio_data)
    
    assert result.text == "Hello world"
    assert result.language == "en"
```

### Integration Tests

```python
async def test_adapter_with_real_engine():
    """Test adapter with real legacy engine."""
    # Create adapter with real engine
    stt_adapter = create_legacy_stt_adapter()
    
    # Test with real audio data
    audio_data = AudioData(...)
    result = await stt_adapter.transcribe(audio_data)
    
    assert result is not None
    assert isinstance(result.text, str)
```

## Migration Strategy

### Gradual Adapter Adoption

1. **Phase 1**: Create adapters alongside legacy code
2. **Phase 2**: Use adapters in new endpoints
3. **Phase 3**: Migrate existing endpoints to use adapters
4. **Phase 4**: Replace adapters with native library engines

### Feature Flags

```python
# Enable/disable adapter usage
USE_ADAPTERS = config_manager.get_bool("features.use_adapters", True)

def get_stt_engine():
    if USE_ADAPTERS:
        return create_legacy_stt_adapter()
    else:
        return legacy_stt_engine  # Direct legacy usage
```

## Best Practices

### Adapter Design

1. **Single Responsibility**: Each adapter handles one engine type
2. **Error Resilience**: Graceful handling of legacy engine errors
3. **Type Safety**: Proper type hints and validation
4. **Documentation**: Clear documentation of transformation logic

### Performance Optimization

1. **Instance Caching**: Reuse adapter instances across requests
2. **Efficient Conversion**: Minimize data transformation overhead
3. **Resource Sharing**: Share underlying engines when safe
4. **Memory Management**: Proper cleanup of resources

### Monitoring Integration

```python
async def transcribe(self, audio: AudioData, language: Optional[str] = None) -> Optional[TranscriptionResult]:
    """Transcribe with monitoring."""
    start_time = time.time()
    
    try:
        result = await self._transcribe_impl(audio, language)
        
        # Record success metrics
        duration = time.time() - start_time
        logger.info(f"STT adapter success in {duration:.3f}s")
        
        return result
        
    except Exception as e:
        # Record error metrics
        duration = time.time() - start_time
        logger.error(f"STT adapter error after {duration:.3f}s: {e}")
        return None
```

## Troubleshooting

### Common Issues

1. **Type Mismatches**: Ensure proper data conversion between formats
2. **Resource Leaks**: Verify proper cleanup of adapter resources
3. **Performance Issues**: Check for inefficient conversions or caching problems
4. **Error Propagation**: Ensure errors are properly translated and logged

### Debug Information

```python
def get_adapter_info(self) -> Dict[str, Any]:
    """Get adapter debugging information."""
    return {
        "adapter_type": type(self).__name__,
        "legacy_engine_type": type(self.legacy_engine).__name__,
        "model_loaded": self.model_loaded,
        "last_error": getattr(self, '_last_error', None)
    }
```

## Future Enhancements

### Native Library Migration

- Replace adapters with native `realtime_conversation` engines
- Maintain compatibility during transition
- Performance improvements with native implementations

### Advanced Features

- Connection pooling for multiple engine instances
- Load balancing across multiple adapters
- Circuit breaker pattern for resilience
- Metrics collection and reporting