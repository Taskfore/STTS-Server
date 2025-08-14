# Router Organization Documentation

## Overview

The router system has been completely reorganized to provide clean separation of concerns, focused responsibilities, and enhanced functionality. The new structure follows domain-driven design principles with clear boundaries between different system areas.

## Router Architecture

```
routers/
├── core/                    # Core business logic
│   ├── tts.py              # Text-to-speech with middleware
│   ├── stt.py              # Speech-to-text with adapters  
│   └── conversation.py     # STT→TTS pipeline with library
├── management/             # System management
│   ├── config.py           # Enhanced configuration
│   └── files.py            # File management with analytics
├── websocket/              # Real-time endpoints
│   ├── websocket_stt.py    # Real-time STT
│   ├── websocket_conversation.py     # Legacy conversation
│   └── websocket_conversation_v2.py  # Library-integrated conversation
└── ui.py                   # Web UI endpoints (kept separate)
```

## Design Principles

### 1. **Domain Separation**
- **Core**: Business logic and primary functionality
- **Management**: System administration and configuration
- **WebSocket**: Real-time communication
- **UI**: User interface endpoints

### 2. **Single Responsibility**
Each router handles one specific domain:
- TTS router: Only text-to-speech functionality
- STT router: Only speech-to-text functionality
- Config router: Only configuration management
- Files router: Only file operations

### 3. **Clean Dependencies**
- No circular dependencies between routers
- Clear dependency injection patterns
- Minimal coupling between components

### 4. **Consistent Patterns**
- Standard error handling across all routers
- Consistent response formats
- Uniform logging and monitoring

## Core Routers

### TTS Router (`routers/core/tts.py`)

Handles text-to-speech synthesis with full middleware integration.

#### Key Features
- **Middleware Pipeline**: Complete request processing with timing, logging, analytics
- **Adapter Pattern**: Clean interface to legacy TTS engine
- **Enhanced Monitoring**: Statistics, performance tracking, usage analytics
- **Background Processing**: Non-blocking audio generation

#### Endpoints

```python
# Core TTS synthesis
POST /tts
{
    "text": "Hello world",
    "voice_path": "/path/to/voice.wav",
    "temperature": 0.8,
    "speed_factor": 1.0
}

# Voice management
GET /tts/voices                    # List available voices
POST /tts/voices/validate          # Validate voice file

# System monitoring
GET /tts/statistics               # System statistics
GET /tts/middleware/status        # Middleware status
POST /tts/middleware/reload       # Reload middleware pipeline
```

#### Architecture Pattern

```python
@router.post("/tts")
async def synthesize_speech(
    request: CustomTTSRequest,
    background_tasks: BackgroundTasks,
    tts_adapter: LegacyTTSEngineAdapter = Depends(get_tts_adapter),
    middleware_pipeline: MiddlewarePipeline = Depends(get_middleware_pipeline)
):
    # Create request context
    context = RequestContext(
        request_id=str(uuid.uuid4())[:8],
        request_type="tts",
        input_data=request.dict()
    )
    
    # Process through middleware pipeline
    async def core_tts_processing(ctx: RequestContext) -> RequestContext:
        # Core TTS logic with adapter
        result = await tts_adapter.synthesize(text, voice_config)
        ctx.output_data = {"audio_data": result.audio_data}
        return ctx
    
    result_context = await middleware_pipeline.process(context, core_tts_processing)
    
    # Return streaming response
    return StreamingResponse(...)
```

### STT Router (`routers/core/stt.py`)

Handles speech-to-text transcription with adapter integration.

#### Key Features
- **Adapter Integration**: Bridge to legacy STT engine via adapters
- **Enhanced Endpoints**: Status, models, configuration management
- **File Validation**: Comprehensive audio file validation
- **Error Handling**: Detailed error reporting and recovery

#### Endpoints

```python
# Core STT transcription
POST /stt
# File upload with language detection

# System management
GET /stt/status                   # Engine status and availability
GET /stt/models                   # Available models information
POST /stt/reload                  # Reload STT engine
GET /stt/config                   # Current configuration
```

#### Architecture Pattern

```python
@router.post("/stt")
async def transcribe_audio(
    audio_file: UploadFile = File(...),
    language: Optional[str] = Form(None),
    stt_adapter: LegacySTTEngineAdapter = Depends(get_stt_adapter)
):
    # File validation
    if not audio_file.filename.lower().endswith(('.wav', '.mp3', '.m4a', '.flac')):
        raise HTTPException(400, "Unsupported audio format")
    
    # Create AudioData from upload
    audio_data = AudioData(
        data=await audio_file.read(),
        sample_rate=16000,  # Default
        channels=1,
        format="wav"
    )
    
    # Process with adapter
    transcription_result = await stt_adapter.transcribe(audio_data, language)
    
    if transcription_result is None:
        raise HTTPException(500, "Transcription failed")
    
    return {
        "text": transcription_result.text,
        "language": transcription_result.language,
        "segments": transcription_result.segments
    }
```

### Conversation Router (`routers/core/conversation.py`)

Handles the STT→TTS pipeline with library integration.

#### Key Features
- **Library Integration**: ConversationEngine from realtime_conversation library
- **Adapter Fallback**: Falls back to individual adapters if library unavailable
- **Pipeline Processing**: Complete speech-to-speech conversation flow
- **Status Monitoring**: System health and component availability

#### Endpoints

```python
# Core conversation pipeline
POST /conversation
# Audio upload → transcription → synthesis → audio response

# System status
GET /conversation/status          # Component availability and system type
```

#### Architecture Pattern

```python
@router.post("/conversation")
async def process_conversation(
    audio_file: UploadFile = File(...),
    voice_path: Optional[str] = Form(None),
    temperature: float = Form(0.8),
    language: Optional[str] = Form(None)
):
    # Try library integration first
    conversation_engine = await get_conversation_system()
    
    if conversation_engine:
        # Use integrated conversation engine
        result = await conversation_engine.process_conversation(
            audio_data, voice_config, language
        )
    else:
        # Fallback to individual adapters
        stt_result = await stt_adapter.transcribe(audio_data, language)
        tts_result = await tts_adapter.synthesize(stt_result.text, voice_config)
        result = {"transcription": stt_result.text, "audio_data": tts_result.audio_data}
    
    return StreamingResponse(...)
```

## Management Routers

### Configuration Router (`routers/management/config.py`)

Enhanced configuration management with validation and schema support.

#### Key Features
- **Schema Validation**: Comprehensive configuration validation
- **Schema Export**: API access to configuration schema
- **Environment Integration**: Support for environment variables
- **Hot Reload**: Dynamic configuration updates

#### Endpoints

```python
# Configuration management
GET /config                       # Current configuration
POST /save_settings              # Update configuration (legacy compatible)
POST /config/update              # Enhanced configuration update

# Validation and schema
GET /config/validation           # Validate current configuration
GET /config/schema               # Get configuration schema
POST /config/validate            # Validate provided configuration

# Environment and system
GET /config/environment          # Environment-specific settings
GET /config/defaults             # Default configuration values
```

#### Enhanced Features

```python
@router.get("/config/validation")
async def validate_config():
    """Comprehensive configuration validation."""
    validation_result = config_manager.validate_config()
    
    return {
        "valid": validation_result.is_valid,
        "errors": validation_result.errors,
        "warnings": validation_result.warnings,
        "schema_version": validation_result.schema_version
    }

@router.get("/config/schema")
async def get_config_schema():
    """Get the complete configuration schema."""
    return {
        "schema": config_manager.get_schema(),
        "version": config_manager.schema_version,
        "documentation": config_manager.get_schema_docs()
    }
```

### Files Router (`routers/management/files.py`)

Enhanced file management with analytics and validation.

#### Key Features
- **Storage Analytics**: Detailed storage usage tracking
- **File Validation**: Comprehensive file validation and sanitization
- **Cleanup Utilities**: Automated cleanup and maintenance
- **Usage Tracking**: File access patterns and usage statistics

#### Endpoints

```python
# File operations
POST /upload_reference           # Upload reference audio (legacy compatible)
GET /get_reference_files         # List reference files (legacy compatible)
DELETE /files/reference/{filename}  # Delete reference file

# Storage management
GET /files/storage/usage         # Storage usage statistics
POST /files/storage/cleanup      # Cleanup temporary files
GET /files/storage/health        # Storage health check

# File validation
POST /files/validate             # Validate uploaded files
GET /files/supported_formats     # List supported file formats
```

#### Analytics Integration

```python
@router.get("/files/storage/usage")
async def get_storage_usage():
    """Get detailed storage usage analytics."""
    usage_stats = await file_manager.get_storage_statistics()
    
    return {
        "total": {
            "size_mb": usage_stats.total_size_mb,
            "file_count": usage_stats.total_files
        },
        "by_type": {
            "reference_audio": usage_stats.reference_audio_stats,
            "outputs": usage_stats.outputs_stats,
            "temporary": usage_stats.temporary_stats
        },
        "trends": usage_stats.usage_trends,
        "cleanup_recommendations": usage_stats.cleanup_suggestions
    }
```

## WebSocket Routers

### Real-time STT Router (`routers/websocket/websocket_stt.py`)

Real-time speech-to-text via WebSocket with timing information.

#### Key Features
- **Real-time Processing**: Continuous audio stream processing
- **Timing Information**: Precise timing data for overlap detection
- **Enhanced Response**: Detailed transcription results with segments
- **Connection Management**: Robust WebSocket connection handling

```python
@router.websocket("/ws/transcribe")
async def websocket_transcribe_endpoint(websocket: WebSocket):
    await websocket.accept()
    
    try:
        while True:
            # Receive PCM audio data
            audio_data = await websocket.receive_bytes()
            
            # Process with timing
            result = stt_engine.transcribe_numpy_with_timing(audio_np)
            
            # Send enhanced response
            response = {
                "type": "transcription",
                "text": result.text,
                "language": result.language,
                "timing": {
                    "start": result.start_time,
                    "end": result.end_time,
                    "duration": result.duration
                },
                "segments": result.segments
            }
            
            await websocket.send_json(response)
            
    except WebSocketDisconnect:
        logger.info("WebSocket disconnected")
```

### Conversation WebSocket Routers

Two versions for compatibility and enhancement:

#### Legacy Conversation (`websocket_conversation.py`)
- Original WebSocket conversation implementation
- Maintained for backward compatibility
- Direct engine integration

#### Library-Integrated Conversation (`websocket_conversation_v2.py`)
- Enhanced with library integration
- ConversationEngine support
- Advanced features and monitoring

```python
@router.websocket("/ws/conversation/v2")
async def websocket_conversation_v2(websocket: WebSocket):
    conversation_engine = await get_conversation_system()
    
    if not conversation_engine:
        await websocket.close(code=1011, reason="Conversation engine unavailable")
        return
    
    await websocket.accept()
    
    try:
        async for message in websocket_message_handler(websocket):
            if message["type"] == "audio":
                result = await conversation_engine.process_streaming_audio(
                    message["data"], message.get("config", {})
                )
                await websocket.send_json(result)
    except Exception as e:
        logger.error(f"Conversation WebSocket error: {e}")
        await websocket.close(code=1011, reason="Processing error")
```

## Router Integration Patterns

### Dependency Injection

All routers use consistent dependency injection patterns:

```python
# Adapter dependencies
def get_tts_adapter() -> LegacyTTSEngineAdapter:
    global _tts_adapter_instance
    if _tts_adapter_instance is None:
        _tts_adapter_instance = create_legacy_tts_adapter()
    return _tts_adapter_instance

def get_stt_adapter() -> LegacySTTEngineAdapter:
    global _stt_adapter_instance
    if _stt_adapter_instance is None:
        _stt_adapter_instance = create_legacy_stt_adapter()
    return _stt_adapter_instance

# Middleware dependencies
def get_middleware_pipeline() -> MiddlewarePipeline:
    global _pipeline_instance
    if _pipeline_instance is None:
        _pipeline_instance = create_middleware_pipeline()
    return _pipeline_instance
```

### Error Handling

Consistent error handling across all routers:

```python
# Standard error response format
def create_error_response(status_code: int, message: str, details: str = None):
    response_data = {
        "error": message,
        "status_code": status_code
    }
    if details:
        response_data["details"] = details
    
    return JSONResponse(
        status_code=status_code,
        content=response_data
    )

# Usage in routers
try:
    result = await process_request(request_data)
    return result
except ValidationError as e:
    return create_error_response(400, "Validation failed", str(e))
except ProcessingError as e:
    return create_error_response(500, "Processing failed", str(e))
except Exception as e:
    logger.error(f"Unexpected error: {e}", exc_info=True)
    return create_error_response(500, "Internal server error")
```

### Response Formats

Standardized response formats across routers:

```python
# Success response
{
    "status": "success",
    "data": { ... },
    "metadata": {
        "request_id": "abc123",
        "processing_time": 1.23,
        "timestamp": "2024-01-01T12:00:00Z"
    }
}

# Error response
{
    "status": "error",
    "error": "Error message",
    "details": "Detailed error information",
    "metadata": {
        "request_id": "abc123",
        "error_code": "PROCESSING_FAILED",
        "timestamp": "2024-01-01T12:00:00Z"
    }
}
```

## Legacy Compatibility

### Backward Compatibility

All legacy endpoints are maintained:

```python
# Legacy endpoint redirects
@app.get("/get_reference_files")
async def legacy_get_reference_files():
    """Legacy endpoint - redirects to new router."""
    return RedirectResponse(url="/files/reference", status_code=307)

@app.post("/save_settings")
async def legacy_save_settings(request: dict):
    """Legacy endpoint - forwards to new router."""
    return await config_router.save_settings(request)
```

### Migration Support

Gradual migration support for endpoints:

```python
# Feature flag for new vs legacy behavior
USE_NEW_ROUTERS = config_manager.get_bool("features.use_new_routers", True)

if USE_NEW_ROUTERS:
    app.include_router(core_tts_router, prefix="/tts", tags=["tts"])
else:
    app.include_router(legacy_tts_router, prefix="/tts", tags=["tts"])
```

## Testing Router Organization

### Unit Testing

```python
import pytest
from fastapi.testclient import TestClient
from routers.core.tts import router

@pytest.fixture
def tts_client():
    from fastapi import FastAPI
    app = FastAPI()
    app.include_router(router)
    return TestClient(app)

def test_tts_synthesis(tts_client):
    response = tts_client.post("/tts", json={
        "text": "Hello world",
        "voice_path": "test_voice.wav"
    })
    assert response.status_code == 200
```

### Integration Testing

```python
async def test_router_integration():
    """Test router integration with dependencies."""
    # Test TTS with middleware
    tts_response = await test_client.post("/tts", json=tts_request)
    assert "audio_data" in tts_response.json()
    
    # Test STT with adapter
    stt_response = await test_client.post("/stt", files=audio_file)
    assert "text" in stt_response.json()
    
    # Test conversation pipeline
    conv_response = await test_client.post("/conversation", files=audio_file)
    assert "transcription" in conv_response.json()
```

## Performance Considerations

### Router Overhead

- **Minimal Routing**: Direct path to appropriate handlers
- **Efficient Dependencies**: Cached dependency instances
- **Resource Sharing**: Shared adapters and pipelines across requests
- **Background Processing**: Non-blocking operations where possible

### Memory Management

```python
# Efficient resource management
@router.on_event("startup")
async def startup_event():
    # Initialize shared resources
    global _shared_tts_adapter
    _shared_tts_adapter = create_legacy_tts_adapter()

@router.on_event("shutdown") 
async def shutdown_event():
    # Cleanup resources
    if _shared_tts_adapter:
        await _shared_tts_adapter.cleanup()
```

## Configuration

### Router-Specific Configuration

```yaml
routers:
  core:
    tts:
      enable_middleware: true
      enable_statistics: true
      max_text_length: 5000
    stt:
      enable_adapters: true
      max_file_size_mb: 50
      supported_formats: ["wav", "mp3", "m4a", "flac"]
    conversation:
      enable_library_integration: true
      fallback_to_adapters: true
  
  management:
    config:
      enable_validation: true
      enable_schema_export: true
    files:
      enable_analytics: true
      enable_cleanup: true
      cleanup_interval_hours: 24
  
  websocket:
    enable_timing_info: true
    max_connections: 100
    heartbeat_interval: 30
```

## Best Practices

### 1. Router Design

- **Single Responsibility**: Each router handles one domain
- **Clean Dependencies**: Use dependency injection consistently
- **Error Handling**: Implement comprehensive error handling
- **Documentation**: Document all endpoints and parameters

### 2. Integration Patterns

- **Adapter Usage**: Use adapters for legacy engine integration
- **Middleware**: Apply middleware for cross-cutting concerns
- **Validation**: Validate all inputs using Pydantic models
- **Monitoring**: Include monitoring and analytics where appropriate

### 3. Performance

- **Resource Sharing**: Share expensive resources across requests
- **Background Tasks**: Use background tasks for non-critical operations
- **Caching**: Cache expensive operations and computations
- **Streaming**: Use streaming responses for large data

### 4. Maintainability

- **Consistent Patterns**: Follow consistent patterns across routers
- **Clear Boundaries**: Maintain clear boundaries between domains
- **Testability**: Design for easy testing and mocking
- **Documentation**: Keep documentation up-to-date with code changes