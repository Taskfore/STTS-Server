# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Common Development Commands

### Environment Setup
```bash
# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # Linux/macOS
# .\venv\Scripts\activate  # Windows

# Install dependencies based on hardware
pip install -r requirements.txt              # CPU-only
pip install -r requirements-nvidia.txt       # NVIDIA GPU (CUDA)
pip install -r requirements-rocm.txt         # AMD GPU (ROCm)
```

### Running the Server
```bash
# Start the TTS server (loads models automatically on first run)
python server.py

# Server will be available at http://localhost:8004 (default port)
# API documentation at http://localhost:8004/docs
```

### Docker Deployment
```bash
# NVIDIA GPU
docker compose up -d --build

# AMD ROCm GPU  
docker compose -f docker-compose-rocm.yml up -d --build

# CPU-only
docker compose -f docker-compose-cpu.yml up -d --build
```

### Testing
No formal test suite exists. Test manually via:
- Web UI at http://localhost:8004
- API endpoints at http://localhost:8004/docs (Swagger UI)
- Direct API calls to `/tts` and `/v1/audio/speech`

## High-Level Architecture

### Core Components

**Chatterbox TTS Server** is a FastAPI-based speech synthesis server that wraps the Chatterbox TTS model with STT capabilities via OpenAI Whisper.

### Engine Architecture Pattern
The codebase uses two different state management patterns:

1. **TTS Engine (Legacy)**: Uses global variables in `engine.py`
   ```python
   # Global state (existing pattern)
   chatterbox_model: Optional[ChatterboxTTS] = None
   MODEL_LOADED: bool = False
   ```

2. **STT Engine (New)**: Uses FastAPI app state with dependency injection
   ```python
   # Clean app state pattern
   app.state.stt_engine = STTEngine()
   stt_engine: STTEngine = Depends(get_stt_engine)
   ```

**When adding new engines**: Follow the STT pattern (app state + dependency injection) rather than globals.

### Configuration System
- **Single source of truth**: `config.yaml` (auto-generated from `config.py` defaults)
- **Thread-safe**: `YamlConfigManager` class with locking
- **Device auto-detection**: Automatically detects CUDA/MPS/CPU with fallback logic
- **Hot-reload**: Some settings require server restart, others apply immediately

### Request Flow Architecture

**TTS Pipeline (`/tts` endpoint):**
```
Client Request → FastAPI Validation → Text Chunking (utils.py) → 
TTS Engine (engine.py) → Audio Processing (utils.py) → 
Encoding (utils.py) → Streaming Response
```

**STT Pipeline (`/stt` endpoint):**
```
Audio Upload → Temporary File Storage → STT Engine (stt_engine.py) → 
Transcription → Cleanup → JSON Response
```

### Key Architectural Decisions

1. **Text Chunking**: Large texts are intelligently split by sentences to prevent TTS engine overload
2. **Dual Audio Processing**: Both real-time streaming and file-based processing supported
3. **Multi-platform GPU Support**: Automatic device detection with graceful fallbacks
4. **Voice Management**: Separate directories for predefined voices (`voices/`) vs reference audio for cloning (`reference_audio/`)
5. **Configuration Persistence**: UI state is automatically saved to `config.yaml` for session persistence

### Directory Structure Logic
```
├── engine.py              # TTS model loading & synthesis (global state)
├── stt_engine.py          # STT model with clean OOP pattern
├── server.py              # FastAPI app with all endpoints
├── config.py              # Configuration management with defaults
├── utils.py               # Shared audio/text processing utilities
├── models.py              # Pydantic request/response models
├── ui/                    # Static web interface files
├── voices/                # Predefined voice samples
├── reference_audio/       # User-uploaded voice cloning samples
└── outputs/               # Generated audio files
```

### API Endpoints Architecture

**Primary Endpoints:**
- `POST /tts` - Full-featured TTS with all parameters
- `POST /stt` - Speech-to-text transcription  
- `POST /v1/audio/speech` - OpenAI-compatible TTS endpoint

**Management Endpoints:**
- `POST /save_settings` - Update config.yaml
- `POST /upload_reference` - Upload reference audio for voice cloning
- `GET /api/ui/initial-data` - Bootstrap data for web UI

### Device Management Strategy
The server implements robust device detection:
1. **Auto-detection**: Tests actual device functionality, not just availability
2. **Graceful fallback**: CUDA → MPS → CPU chain
3. **Per-engine configuration**: TTS and STT engines can use different devices
4. **Apple Silicon support**: Special installation sequence required for MPS

### Audio Processing Pipeline
1. **Chunking**: Text split by sentences with configurable chunk size
2. **Synthesis**: Per-chunk generation with concatenation
3. **Post-processing**: Optional silence trimming, speed adjustment
4. **Encoding**: Support for WAV, Opus, MP3 output formats
5. **Streaming**: Direct audio streaming without intermediate file storage

### Configuration Extension Pattern
When adding new configuration sections:
```python
# 1. Add to DEFAULT_CONFIG in config.py
"new_engine": {
    "device": "auto",
    "model_size": "base",
}

# 2. Add accessor functions
def get_new_engine_device() -> str:
    return config_manager.get_string("new_engine.device", "auto")

# 3. Use in modules
from config import get_new_engine_device
```

### Error Handling Philosophy
- **Graceful degradation**: If optional features fail, core functionality continues
- **Detailed logging**: All errors logged with context for debugging
- **User-friendly messages**: API returns clear error descriptions
- **Resource cleanup**: Temporary files automatically cleaned up on errors

### Performance Considerations
- **Model loading**: Models loaded once at startup, cached in memory
- **Concurrent requests**: FastAPI handles concurrent TTS/STT requests
- **Memory management**: Temporary audio files cleaned up after processing
- **GPU memory**: Chunking prevents GPU OOM on large texts

## Development Guidelines

### When adding new features:
1. Follow the STT engine pattern (OOP + app state) for new engines
2. Add configuration to `config.py` DEFAULT_CONFIG first
3. Use dependency injection for clean testing
4. Add Pydantic models for request validation
5. Include proper error handling and logging
6. Clean up temporary resources in finally blocks

### When modifying existing code:
1. Don't break the existing TTS global pattern - it works
2. Maintain backward compatibility for API endpoints
3. Update config.yaml structure carefully (add, don't remove)
4. Test across different hardware configurations if changing device logic

### Integration Testing:
Test the full pipeline manually via the web UI and verify:
- Model loading on different devices (CPU/CUDA/MPS)
- Text chunking with various input sizes
- Voice cloning vs predefined voices
- Configuration persistence across restarts
- Docker deployment functionality