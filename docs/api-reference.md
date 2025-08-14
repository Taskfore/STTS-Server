# API Reference Documentation

## Overview

This document provides comprehensive API reference for the STTS Server library integration system. All endpoints support both legacy and library-integrated functionality with enhanced features, monitoring, and analytics.

## Base URL

```
http://localhost:8004
```

## Authentication

Currently, no authentication is required. Future versions may include API key authentication.

## Response Format

All API responses follow a consistent format:

### Success Response
```json
{
  "status": "success",
  "data": { ... },
  "metadata": {
    "request_id": "abc123",
    "processing_time": 1.23,
    "timestamp": "2024-01-01T12:00:00Z"
  }
}
```

### Error Response
```json
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

## Core APIs

### Text-to-Speech (TTS)

#### Synthesize Speech

**POST** `/tts`

Convert text to speech using the Chatterbox TTS engine with middleware pipeline processing.

**Headers:**
- `Content-Type: application/json`

**Request Body:**
```json
{
  "text": "Hello, this is a test message",
  "voice_path": "/path/to/voice.wav",
  "voice_id": "female_voice_01",
  "temperature": 0.8,
  "exaggeration": 0.5,
  "cfg_weight": 0.5,
  "seed": 0,
  "speed_factor": 1.0,
  "remove_silence": true,
  "format": "wav"
}
```

**Parameters:**
- `text` (string, required): Text to synthesize (max 5000 characters)
- `voice_path` (string, optional): Path to reference voice file
- `voice_id` (string, optional): ID of predefined voice
- `temperature` (float, optional): Synthesis temperature (0.0-2.0, default: 0.8)
- `exaggeration` (float, optional): Voice exaggeration (0.0-1.0, default: 0.5)
- `cfg_weight` (float, optional): CFG weight (0.0-1.0, default: 0.5)
- `seed` (integer, optional): Random seed for reproducibility (default: 0)
- `speed_factor` (float, optional): Speech speed multiplier (0.5-2.0, default: 1.0)
- `remove_silence` (boolean, optional): Remove silence from output (default: true)
- `format` (string, optional): Output format - "wav", "opus", "mp3" (default: "wav")

**Response:**
- **200 OK**: Streaming audio response
- **400 Bad Request**: Invalid parameters
- **500 Internal Server Error**: Processing failed

**Response Headers:**
- `Content-Type: audio/wav` (or appropriate format)
- `X-Processing-Time: 2.34`
- `X-Request-ID: abc123`

**Example:**
```bash
curl -X POST http://localhost:8004/tts \
  -H "Content-Type: application/json" \
  -d '{"text": "Hello world", "voice_id": "female_voice_01"}' \
  --output speech.wav
```

#### List Voices

**GET** `/tts/voices`

Get list of available predefined voices.

**Response:**
```json
{
  "status": "success",
  "data": {
    "voices": [
      {
        "id": "female_voice_01",
        "name": "Female Voice 1",
        "path": "/path/to/female_voice_01.wav",
        "description": "Clear female voice",
        "duration": 5.2,
        "sample_rate": 22050
      }
    ],
    "voice_count": 1
  }
}
```

#### Validate Voice

**POST** `/tts/voices/validate`

Validate a voice file for compatibility.

**Request Body:**
```json
{
  "voice_path": "/path/to/voice.wav"
}
```

**Response:**
```json
{
  "status": "success",
  "data": {
    "valid": true,
    "duration": 5.2,
    "sample_rate": 22050,
    "channels": 1,
    "format": "wav",
    "issues": []
  }
}
```

#### System Statistics

**GET** `/tts/statistics`

Get comprehensive TTS system statistics and analytics.

**Response:**
```json
{
  "status": "success",
  "data": {
    "system": {
      "adapter_type": "LegacyTTSEngineAdapter",
      "library_available": true,
      "middleware_enabled": true,
      "model_loaded": true
    },
    "middleware": {
      "enabled_count": 3,
      "total_requests": 150,
      "average_duration": 2.34,
      "error_rate": 0.02,
      "slow_requests": 5
    },
    "analytics": {
      "popular_voices": ["female_voice_01", "male_voice_02"],
      "format_preferences": {"wav": 60, "opus": 30, "mp3": 10},
      "usage_trends": {...},
      "performance_metrics": {...}
    }
  }
}
```

#### Middleware Status

**GET** `/tts/middleware/status`

Get middleware pipeline status and configuration.

**Response:**
```json
{
  "status": "success",
  "data": {
    "total_middleware": 3,
    "enabled_middleware": 3,
    "middleware_names": [
      "TimingMiddleware",
      "LoggingMiddleware",
      "AnalyticsMiddleware"
    ],
    "pipeline_status": "active",
    "configuration": {...}
  }
}
```

#### Reload Middleware

**POST** `/tts/middleware/reload`

Reload the middleware pipeline with current configuration.

**Response:**
```json
{
  "status": "success",
  "data": {
    "message": "Middleware pipeline reloaded",
    "middleware_count": 3,
    "reloaded_at": "2024-01-01T12:00:00Z"
  }
}
```

### Speech-to-Text (STT)

#### Transcribe Audio

**POST** `/stt`

Transcribe audio file to text using the STT adapter.

**Headers:**
- `Content-Type: multipart/form-data`

**Form Data:**
- `audio_file` (file, required): Audio file to transcribe
- `language` (string, optional): Language code (e.g., "en", "es", "auto")

**Supported Formats:** WAV, MP3, M4A, FLAC

**Response:**
```json
{
  "status": "success",
  "data": {
    "text": "Hello, this is the transcribed text",
    "language": "en",
    "confidence": 0.95,
    "segments": [
      {
        "text": "Hello, this is the transcribed text",
        "start": 0.0,
        "end": 3.2,
        "confidence": 0.95
      }
    ],
    "processing_time": 1.23
  }
}
```

**Example:**
```bash
curl -X POST http://localhost:8004/stt \
  -F "audio_file=@speech.wav" \
  -F "language=en"
```

#### STT Status

**GET** `/stt/status`

Get STT engine status and availability.

**Response:**
```json
{
  "status": "success",
  "data": {
    "available": true,
    "model_loaded": true,
    "adapter_type": "LegacySTTEngineAdapter",
    "library_integration": true,
    "current_model": "openai/whisper-base",
    "supported_languages": ["en", "es", "fr", "..."]
  }
}
```

#### STT Models

**GET** `/stt/models`

Get information about available STT models.

**Response:**
```json
{
  "status": "success",
  "data": {
    "current_model": "openai/whisper-base",
    "available_models": ["base", "small", "medium", "large"],
    "adapter_type": "LegacySTTEngineAdapter",
    "model_info": {
      "name": "base",
      "size": "39M",
      "languages": 99,
      "description": "Fastest model with good accuracy"
    }
  }
}
```

#### Reload STT Engine

**POST** `/stt/reload`

Reload the STT engine with current configuration.

**Response:**
```json
{
  "status": "success",
  "data": {
    "message": "STT engine reloaded successfully",
    "model": "openai/whisper-base",
    "reloaded_at": "2024-01-01T12:00:00Z"
  }
}
```

#### STT Configuration

**GET** `/stt/config`

Get current STT configuration.

**Response:**
```json
{
  "status": "success",
  "data": {
    "model_size": "base",
    "device": "cuda",
    "language": "auto",
    "adapter_config": {...},
    "performance_settings": {...}
  }
}
```

### Conversation Pipeline

#### Process Conversation

**POST** `/conversation`

Process audio through STT→TTS pipeline for speech-to-speech conversation.

**Headers:**
- `Content-Type: multipart/form-data`

**Form Data:**
- `audio_file` (file, required): Input audio file
- `voice_path` (string, optional): Voice for TTS response
- `voice_id` (string, optional): Predefined voice ID
- `temperature` (float, optional): TTS temperature (default: 0.8)
- `speed_factor` (float, optional): TTS speed (default: 1.0)
- `language` (string, optional): STT language (default: "auto")

**Response:**
- **200 OK**: Streaming audio response
- **400 Bad Request**: Invalid input
- **500 Internal Server Error**: Processing failed

**Response Headers:**
- `Content-Type: audio/wav`
- `X-Transcription: "Hello world"`
- `X-Processing-Time: 3.45`
- `X-Request-ID: abc123`

**Example:**
```bash
curl -X POST http://localhost:8004/conversation \
  -F "audio_file=@input.wav" \
  -F "voice_id=female_voice_01" \
  --output response.wav
```

#### Conversation Status

**GET** `/conversation/status`

Get conversation system status and component availability.

**Response:**
```json
{
  "status": "success",
  "data": {
    "system_type": "library_integrated",
    "stt_available": true,
    "tts_available": true,
    "library_available": true,
    "conversation_engine": "ConversationEngine",
    "fallback_mode": "adapters",
    "components": {
      "stt_adapter": "LegacySTTEngineAdapter",
      "tts_adapter": "LegacyTTSEngineAdapter"
    }
  }
}
```

## Management APIs

### Configuration Management

#### Get Configuration

**GET** `/config`

Get current system configuration.

**Response:**
```json
{
  "status": "success",
  "data": {
    "tts_engine": {
      "device": "auto",
      "temperature": 0.8
    },
    "stt_engine": {
      "model_size": "base",
      "device": "auto"
    },
    "middleware": {
      "timing": {"enabled": true},
      "logging": {"enabled": true},
      "analytics": {"enabled": true}
    }
  }
}
```

#### Update Configuration

**POST** `/config/update`

Update system configuration with validation.

**Request Body:**
```json
{
  "tts_engine": {
    "temperature": 0.9
  },
  "middleware": {
    "analytics": {"enabled": false}
  }
}
```

**Response:**
```json
{
  "status": "success",
  "data": {
    "message": "Configuration updated successfully",
    "updated_keys": ["tts_engine.temperature", "middleware.analytics.enabled"],
    "restart_required": false
  }
}
```

#### Save Settings (Legacy)

**POST** `/save_settings`

Legacy endpoint for saving configuration (backward compatibility).

**Request Body:**
```json
{
  "gen": {
    "default_temperature": 0.8,
    "default_speed_factor": 1.0
  }
}
```

#### Validate Configuration

**GET** `/config/validation`

Validate current configuration against schema.

**Response:**
```json
{
  "status": "success",
  "data": {
    "valid": true,
    "errors": [],
    "warnings": [
      "tts_engine.temperature is higher than recommended (0.9 > 0.8)"
    ],
    "schema_version": "1.0.0"
  }
}
```

#### Configuration Schema

**GET** `/config/schema`

Get the complete configuration schema.

**Response:**
```json
{
  "status": "success",
  "data": {
    "schema": {
      "type": "object",
      "properties": {
        "tts_engine": {
          "type": "object",
          "properties": {
            "device": {"type": "string", "default": "auto"},
            "temperature": {"type": "number", "minimum": 0.0, "maximum": 2.0}
          }
        }
      }
    },
    "version": "1.0.0",
    "documentation": {...}
  }
}
```

#### Environment Settings

**GET** `/config/environment`

Get environment-specific configuration.

**Response:**
```json
{
  "status": "success",
  "data": {
    "environment": "development",
    "config_source": "config.yaml",
    "environment_overrides": {
      "STTS_LOG_LEVEL": "DEBUG",
      "STTS_DEVICE": "cpu"
    },
    "active_profiles": ["development", "local"]
  }
}
```

#### Default Configuration

**GET** `/config/defaults`

Get default configuration values.

**Response:**
```json
{
  "status": "success",
  "data": {
    "defaults": {
      "tts_engine": {"device": "auto", "temperature": 0.8},
      "stt_engine": {"model_size": "base", "device": "auto"}
    },
    "source": "DEFAULT_CONFIG",
    "version": "1.0.0"
  }
}
```

### File Management

#### Upload Reference Audio

**POST** `/upload_reference`

Upload reference audio file for voice cloning.

**Headers:**
- `Content-Type: multipart/form-data`

**Form Data:**
- `file` (file, required): Audio file to upload
- `filename` (string, optional): Custom filename

**Response:**
```json
{
  "status": "success",
  "data": {
    "message": "File uploaded successfully",
    "filename": "custom_voice.wav",
    "path": "/path/to/reference_audio/custom_voice.wav",
    "size": 1048576,
    "duration": 5.2,
    "validation": {
      "format": "wav",
      "sample_rate": 22050,
      "channels": 1,
      "quality": "good"
    }
  }
}
```

#### Get Reference Files

**GET** `/get_reference_files`

List available reference audio files.

**Response:**
```json
{
  "status": "success",
  "data": {
    "files": [
      {
        "filename": "custom_voice.wav",
        "path": "/path/to/reference_audio/custom_voice.wav",
        "size": 1048576,
        "duration": 5.2,
        "uploaded_at": "2024-01-01T12:00:00Z",
        "last_used": "2024-01-01T12:30:00Z",
        "usage_count": 5
      }
    ],
    "total_files": 1,
    "total_size": 1048576
  }
}
```

#### Delete Reference File

**DELETE** `/files/reference/{filename}`

Delete a reference audio file.

**Response:**
```json
{
  "status": "success",
  "data": {
    "message": "File deleted successfully",
    "filename": "custom_voice.wav",
    "deleted_at": "2024-01-01T12:00:00Z"
  }
}
```

#### Storage Usage

**GET** `/files/storage/usage`

Get detailed storage usage statistics.

**Response:**
```json
{
  "status": "success",
  "data": {
    "total": {
      "size_mb": 150.5,
      "file_count": 25
    },
    "by_type": {
      "reference_audio": {
        "size_mb": 50.2,
        "file_count": 10,
        "average_file_size_mb": 5.02
      },
      "outputs": {
        "size_mb": 75.3,
        "file_count": 12,
        "cleanup_eligible": 8
      },
      "temporary": {
        "size_mb": 25.0,
        "file_count": 3,
        "oldest_file_age_hours": 2
      }
    },
    "trends": {
      "daily_growth_mb": 5.2,
      "weekly_usage": {...}
    },
    "cleanup_recommendations": [
      "Delete 8 old output files to free 20MB",
      "Clear temporary files older than 1 hour"
    ]
  }
}
```

#### Storage Cleanup

**POST** `/files/storage/cleanup`

Perform storage cleanup operations.

**Request Body:**
```json
{
  "cleanup_temporary": true,
  "cleanup_old_outputs": true,
  "max_age_hours": 24,
  "dry_run": false
}
```

**Response:**
```json
{
  "status": "success",
  "data": {
    "cleaned_files": 8,
    "freed_space_mb": 25.5,
    "operations": [
      "Deleted 5 temporary files (15MB)",
      "Deleted 3 old output files (10.5MB)"
    ],
    "cleanup_at": "2024-01-01T12:00:00Z"
  }
}
```

#### Storage Health

**GET** `/files/storage/health`

Check storage system health.

**Response:**
```json
{
  "status": "success",
  "data": {
    "healthy": true,
    "total_space_gb": 100.0,
    "used_space_gb": 15.5,
    "available_space_gb": 84.5,
    "usage_percentage": 15.5,
    "warnings": [],
    "recommendations": [
      "Storage usage is healthy",
      "Consider cleanup if usage exceeds 80%"
    ]
  }
}
```

#### Validate Files

**POST** `/files/validate`

Validate uploaded files for compatibility.

**Headers:**
- `Content-Type: multipart/form-data`

**Form Data:**
- `files` (file[], required): Files to validate

**Response:**
```json
{
  "status": "success",
  "data": {
    "valid_files": [
      {
        "filename": "voice1.wav",
        "valid": true,
        "format": "wav",
        "duration": 5.2,
        "issues": []
      }
    ],
    "invalid_files": [
      {
        "filename": "voice2.mp3",
        "valid": false,
        "issues": ["Sample rate too low", "Multiple channels detected"]
      }
    ],
    "summary": {
      "total": 2,
      "valid": 1,
      "invalid": 1
    }
  }
}
```

#### Supported Formats

**GET** `/files/supported_formats`

Get list of supported file formats.

**Response:**
```json
{
  "status": "success",
  "data": {
    "formats": [
      {
        "extension": "wav",
        "mime_type": "audio/wav",
        "description": "Uncompressed audio",
        "recommended": true
      },
      {
        "extension": "mp3",
        "mime_type": "audio/mpeg",
        "description": "Compressed audio",
        "recommended": false
      }
    ],
    "recommendations": {
      "best_quality": ["wav", "flac"],
      "compatibility": ["wav", "mp3"],
      "upload_size": ["mp3", "opus"]
    }
  }
}
```

## WebSocket APIs

### Real-time STT

**WebSocket** `/ws/transcribe`

Real-time speech-to-text transcription via WebSocket.

**Connection:**
```javascript
const ws = new WebSocket('ws://localhost:8004/ws/transcribe');
```

**Send Audio Data:**
```javascript
// Send PCM audio data (16-bit, 16kHz, mono)
ws.send(pcmAudioData);
```

**Receive Transcription:**
```javascript
ws.onmessage = (event) => {
  const response = JSON.parse(event.data);
  console.log(response);
};
```

**Response Format:**
```json
{
  "type": "transcription",
  "text": "Hello world",
  "language": "en",
  "partial": false,
  "timing": {
    "start": 1.5,
    "end": 3.2,
    "duration": 1.7
  },
  "segments": [
    {
      "text": "Hello world",
      "start": 1.5,
      "end": 3.2,
      "confidence": 0.95
    }
  ]
}
```

### WebSocket Conversation

**WebSocket** `/ws/conversation`

Real-time conversation via WebSocket (legacy).

**WebSocket** `/ws/conversation/v2`

Enhanced real-time conversation with library integration.

**Connection:**
```javascript
const ws = new WebSocket('ws://localhost:8004/ws/conversation/v2');
```

**Send Message:**
```javascript
ws.send(JSON.stringify({
  "type": "audio",
  "data": audioData,
  "config": {
    "voice_id": "female_voice_01",
    "temperature": 0.8
  }
}));
```

**Response Format:**
```json
{
  "type": "conversation_response",
  "transcription": "Hello there",
  "response_text": "Hi, how can I help you?",
  "audio_data": "base64_encoded_audio",
  "timing": {
    "transcription_time": 1.2,
    "synthesis_time": 2.1,
    "total_time": 3.3
  }
}
```

#### WebSocket Status

**GET** `/ws/conversation/status`

Get WebSocket conversation system status.

**Response:**
```json
{
  "status": "success",
  "data": {
    "status": "ready",
    "active_connections": 3,
    "max_connections": 100,
    "library_integration": true,
    "conversation_engine": "ConversationEngine",
    "supported_features": [
      "real_time_transcription",
      "conversation_pipeline",
      "timing_information"
    ]
  }
}
```

## System APIs

### Health Check

**GET** `/health`

System health check with library integration status.

**Response:**
```json
{
  "status": "success",
  "data": {
    "status": "healthy",
    "version": "2.1.0",
    "architecture": "library_integrated",
    "uptime": 3600,
    "features": {
      "adapter_pattern": true,
      "middleware_pipeline": true,
      "library_integration": true,
      "websocket_support": true
    },
    "components": {
      "tts_engine": "available",
      "stt_engine": "available",
      "conversation_engine": "available",
      "middleware": "active"
    }
  }
}
```

### System Information

**GET** `/info`

Detailed system information.

**Response:**
```json
{
  "status": "success",
  "data": {
    "version": "2.1.0",
    "architecture": "library_integrated",
    "build_info": {
      "build_date": "2024-01-01",
      "commit_hash": "abc123",
      "python_version": "3.9.0"
    },
    "system": {
      "platform": "linux",
      "cpu_count": 8,
      "memory_gb": 16,
      "gpu_available": true,
      "gpu_type": "NVIDIA RTX 3080"
    },
    "features": {
      "adapters": true,
      "middleware": true,
      "library_integration": true,
      "analytics": true
    }
  }
}
```

## Error Codes

### HTTP Status Codes

- `200 OK`: Success
- `400 Bad Request`: Invalid input or parameters
- `401 Unauthorized`: Authentication required (future)
- `403 Forbidden`: Access denied (future)
- `404 Not Found`: Resource not found
- `413 Payload Too Large`: File too large
- `415 Unsupported Media Type`: Invalid file format
- `422 Unprocessable Entity`: Validation failed
- `500 Internal Server Error`: Processing failed
- `503 Service Unavailable`: Service temporarily unavailable

### Application Error Codes

- `VALIDATION_FAILED`: Input validation failed
- `PROCESSING_FAILED`: Core processing failed
- `MODEL_NOT_LOADED`: AI model not available
- `FILE_TOO_LARGE`: Uploaded file exceeds size limit
- `UNSUPPORTED_FORMAT`: File format not supported
- `CONFIGURATION_ERROR`: Configuration invalid
- `ADAPTER_ERROR`: Adapter processing failed
- `MIDDLEWARE_ERROR`: Middleware processing failed
- `STORAGE_ERROR`: File storage operation failed

## Rate Limiting

Currently, no rate limiting is implemented. Future versions may include:

- Request rate limits per IP
- File upload size limits
- Concurrent connection limits
- Processing time limits

## API Versioning

The API uses URL path versioning:

- `/v1/audio/speech` - OpenAI-compatible TTS endpoint
- `/ws/conversation/v2` - Enhanced WebSocket conversation

## SDKs and Examples

### Python Example

```python
import requests
import json

# TTS example
response = requests.post(
    "http://localhost:8004/tts",
    json={
        "text": "Hello world",
        "voice_id": "female_voice_01"
    }
)

with open("output.wav", "wb") as f:
    f.write(response.content)

# STT example
with open("input.wav", "rb") as f:
    response = requests.post(
        "http://localhost:8004/stt",
        files={"audio_file": f}
    )

result = response.json()
print(f"Transcription: {result['data']['text']}")
```

### JavaScript Example

```javascript
// TTS example
const ttsResponse = await fetch('/tts', {
  method: 'POST',
  headers: {'Content-Type': 'application/json'},
  body: JSON.stringify({
    text: 'Hello world',
    voice_id: 'female_voice_01'
  })
});

const audioBlob = await ttsResponse.blob();
const audioUrl = URL.createObjectURL(audioBlob);

// STT example
const formData = new FormData();
formData.append('audio_file', audioFile);

const sttResponse = await fetch('/stt', {
  method: 'POST',
  body: formData
});

const result = await sttResponse.json();
console.log('Transcription:', result.data.text);
```

### WebSocket Example

```javascript
// Real-time STT
const ws = new WebSocket('ws://localhost:8004/ws/transcribe');

ws.onopen = () => {
  console.log('Connected to STT WebSocket');
};

ws.onmessage = (event) => {
  const response = JSON.parse(event.data);
  console.log('Transcription:', response.text);
  console.log('Duration:', response.timing.duration);
};

// Send PCM audio data
navigator.mediaDevices.getUserMedia({audio: true})
  .then(stream => {
    const audioContext = new AudioContext();
    const source = audioContext.createMediaStreamSource(stream);
    const processor = audioContext.createScriptProcessor(1024, 1, 1);
    
    processor.onaudioprocess = (e) => {
      const inputData = e.inputBuffer.getChannelData(0);
      const pcmData = new Int16Array(inputData.length);
      
      for (let i = 0; i < inputData.length; i++) {
        pcmData[i] = inputData[i] * 32767;
      }
      
      ws.send(pcmData.buffer);
    };
    
    source.connect(processor);
    processor.connect(audioContext.destination);
  });
```