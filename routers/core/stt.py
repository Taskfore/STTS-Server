# File: routers/core/stt.py
# Speech-to-Text API endpoints using library adapters and realtime_conversation integration

import logging
import uuid
import asyncio
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, HTTPException, UploadFile, File, Form, Depends, Request

# Library imports (with fallback to adapters)
try:
    from realtime_conversation.adapters.stt import WhisperSTTEngine
    from realtime_conversation.core.interfaces import AudioData
    LIBRARY_AVAILABLE = True
except ImportError:
    LIBRARY_AVAILABLE = False
    WhisperSTTEngine = None
    AudioData = None

# Fallback to legacy adapter
from adapters.legacy_engines import LegacySTTEngineAdapter, create_legacy_stt_adapter

from models import STTResponse, ErrorResponse
from config import get_output_path, config_manager

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/stt", tags=["Speech-to-Text"])

# Global adapter instance (will be replaced with proper DI later)
_stt_adapter_instance: Optional[LegacySTTEngineAdapter] = None


def get_stt_adapter() -> LegacySTTEngineAdapter:
    """
    Dependency to get STT adapter.
    
    This uses the adapter pattern to bridge between legacy engines and the new library interface.
    When the library is fully integrated, this will switch to native library engines.
    """
    global _stt_adapter_instance
    
    if _stt_adapter_instance is None:
        if LIBRARY_AVAILABLE:
            # TODO: Use native WhisperSTTEngine when fully integrated
            logger.info("Library available but using legacy adapter for compatibility")
            _stt_adapter_instance = create_legacy_stt_adapter()
        else:
            logger.info("Using legacy STT adapter")
            _stt_adapter_instance = create_legacy_stt_adapter()
    
    return _stt_adapter_instance


def get_legacy_stt_engine(request: Request):
    """Legacy dependency for backward compatibility."""
    if not hasattr(request.app.state, 'stt_engine'):
        raise HTTPException(status_code=503, detail="STT engine not initialized")
    return request.app.state.stt_engine


@router.post(
    "",
    response_model=STTResponse,
    summary="Transcribe speech to text using library adapter",
    responses={
        200: {
            "model": STTResponse,
            "description": "Successful speech-to-text transcription.",
        },
        400: {
            "model": ErrorResponse,
            "description": "Invalid request parameters or audio file.",
        },
        503: {
            "model": ErrorResponse,
            "description": "STT engine not available or model not loaded.",
        },
    },
)
async def transcribe_audio(
    audio_file: UploadFile = File(..., description="Audio file to transcribe (.wav, .mp3, .m4a, .flac)"),
    language: Optional[str] = Form(None, description="Language code or None for auto-detection"),
    stt_adapter: LegacySTTEngineAdapter = Depends(get_stt_adapter)
):
    """
    Transcribes speech from an uploaded audio file to text using the library adapter.
    This provides a clean interface while maintaining compatibility with the existing engine.
    """
    if not await stt_adapter.is_available():
        logger.error("STT request failed: Model not available.")
        raise HTTPException(
            status_code=503,
            detail="STT engine model is not currently loaded or available.",
        )
    
    if not audio_file.filename:
        raise HTTPException(status_code=400, detail="No audio file provided.")
    
    # Validate file extension
    allowed_extensions = {".wav", ".mp3", ".m4a", ".flac", ".ogg"}
    file_ext = Path(audio_file.filename).suffix.lower()
    if file_ext not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported audio format: {file_ext}. Supported: {', '.join(allowed_extensions)}"
        )
    
    logger.info(f"Received STT request for file: {audio_file.filename}")
    
    # Save uploaded file temporarily
    temp_audio_path = get_output_path() / f"temp_stt_{uuid.uuid4().hex[:8]}{file_ext}"
    
    try:
        # Save uploaded file
        with open(temp_audio_path, "wb") as buffer:
            import shutil
            shutil.copyfileobj(audio_file.file, buffer)
        
        # Read file and create AudioData
        with open(temp_audio_path, "rb") as f:
            audio_bytes = f.read()
        
        # Create AudioData object for library interface
        if LIBRARY_AVAILABLE and AudioData:
            audio_data = AudioData(
                data=audio_bytes,
                sample_rate=16000,  # Will be adjusted by the adapter
                channels=1,
                format="file"  # Indicates this is a file rather than raw PCM
            )
            
            # Use library interface
            transcription_result = await stt_adapter.transcribe(audio_data, language)
        else:
            # Fallback to direct legacy engine call for file transcription
            transcription_result = await _transcribe_file_legacy(temp_audio_path, language, stt_adapter)
        
        if transcription_result is None:
            raise HTTPException(
                status_code=500,
                detail="Transcription failed. Please check audio file format and content."
            )
        
        logger.info(f"STT transcription successful. Text length: {len(transcription_result.text)} characters")
        
        # Convert library result to API response
        return STTResponse(
            text=transcription_result.text,
            language=transcription_result.language if transcription_result.language != 'unknown' else language,
            duration=transcription_result.duration if hasattr(transcription_result, 'duration') else None
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error during STT processing: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"STT processing error: {str(e)}")
    finally:
        # Clean up temporary file
        if temp_audio_path.exists():
            temp_audio_path.unlink(missing_ok=True)
        await audio_file.close()


async def _transcribe_file_legacy(
    file_path: Path, 
    language: Optional[str], 
    stt_adapter: LegacySTTEngineAdapter
) -> Optional[object]:
    """
    Fallback method for file transcription using legacy engine directly.
    
    This is used when the library AudioData interface isn't available or
    when we need to handle file formats that require special processing.
    """
    try:
        # Use the legacy engine's file transcription method directly
        transcribed_text = stt_adapter.legacy_engine.transcribe_file(str(file_path), language)
        
        if transcribed_text is None:
            return None
        
        # Create a simple result object
        class SimpleTranscriptionResult:
            def __init__(self, text: str, language: str):
                self.text = text
                self.language = language or 'unknown'
                self.duration = None
        
        return SimpleTranscriptionResult(transcribed_text, language)
        
    except Exception as e:
        logger.error(f"Error in legacy file transcription: {e}")
        return None


@router.post("/stream", summary="Stream-based STT (Future Enhancement)")
async def transcribe_audio_stream():
    """
    Placeholder for stream-based STT transcription.
    
    This will be implemented when we integrate the full library pipeline
    with real-time audio processing capabilities.
    """
    raise HTTPException(
        status_code=501,
        detail="Stream-based transcription not yet implemented. Use /stt for file-based transcription."
    )


@router.get("/models", summary="List available STT models")
async def list_stt_models():
    """List available STT models and their configurations."""
    try:
        if LIBRARY_AVAILABLE:
            # When library is available, get models from WhisperSTTEngine
            models = ["tiny", "base", "small", "medium", "large"]
        else:
            # Fallback to basic model list
            models = ["base"]  # Current default
        
        current_model = config_manager.get_string("stt_engine.model_size", "base")
        current_device = config_manager.get_string("stt_engine.device", "auto")
        
        return {
            "available_models": models,
            "current_model": current_model,
            "current_device": current_device,
            "library_available": LIBRARY_AVAILABLE,
            "adapter_type": "legacy" if not LIBRARY_AVAILABLE else "library_ready"
        }
        
    except Exception as e:
        logger.error(f"Error listing STT models: {e}")
        raise HTTPException(status_code=500, detail="Failed to list STT models")


@router.get("/status", summary="STT engine status")
async def get_stt_status(stt_adapter: LegacySTTEngineAdapter = Depends(get_stt_adapter)):
    """Get detailed status of the STT engine."""
    try:
        is_available = await stt_adapter.is_available()
        model_loaded = stt_adapter.model_loaded
        
        status = {
            "available": is_available,
            "model_loaded": model_loaded,
            "library_integration": LIBRARY_AVAILABLE,
            "adapter_type": "LegacySTTEngineAdapter",
            "engine_type": type(stt_adapter.legacy_engine).__name__ if hasattr(stt_adapter, 'legacy_engine') else "Unknown"
        }
        
        # Add model info if available
        if hasattr(stt_adapter.legacy_engine, 'get_model_info'):
            try:
                model_info = stt_adapter.legacy_engine.get_model_info()
                status["model_info"] = model_info
            except Exception as e:
                status["model_info_error"] = str(e)
        
        return status
        
    except Exception as e:
        logger.error(f"Error getting STT status: {e}")
        raise HTTPException(status_code=500, detail="Failed to get STT status")


@router.post("/reload", summary="Reload STT model")
async def reload_stt_model():
    """
    Reload the STT model.
    
    This can be useful when changing model configurations or
    recovering from model loading failures.
    """
    global _stt_adapter_instance
    
    try:
        # Clear current adapter to force reload
        _stt_adapter_instance = None
        
        # Create new adapter (this will reload the model)
        new_adapter = get_stt_adapter()
        
        # Check if model loaded successfully
        if await new_adapter.is_available():
            return {
                "status": "success",
                "message": "STT model reloaded successfully",
                "model_loaded": new_adapter.model_loaded
            }
        else:
            return {
                "status": "error", 
                "message": "STT model reload failed",
                "model_loaded": False
            }
            
    except Exception as e:
        logger.error(f"Error reloading STT model: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to reload STT model: {str(e)}")


# Legacy compatibility endpoint
@router.post("/legacy", include_in_schema=False)
async def transcribe_audio_legacy(
    audio_file: UploadFile = File(...),
    language: Optional[str] = Form(None),
    legacy_engine=Depends(get_legacy_stt_engine)
):
    """
    Legacy compatibility endpoint that uses the original STT engine directly.
    
    This maintains 100% backward compatibility while the new system is being tested.
    """
    # This is essentially the original implementation from routers/stt.py
    if not legacy_engine.model_loaded:
        raise HTTPException(status_code=503, detail="STT engine not available")
    
    if not audio_file.filename:
        raise HTTPException(status_code=400, detail="No audio file provided")
    
    allowed_extensions = {".wav", ".mp3", ".m4a", ".flac", ".ogg"}
    file_ext = Path(audio_file.filename).suffix.lower()
    if file_ext not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported audio format: {file_ext}. Supported: {', '.join(allowed_extensions)}"
        )
    
    temp_audio_path = get_output_path() / f"temp_stt_legacy_{uuid.uuid4().hex[:8]}{file_ext}"
    
    try:
        with open(temp_audio_path, "wb") as buffer:
            import shutil
            shutil.copyfileobj(audio_file.file, buffer)
        
        transcribed_text = legacy_engine.transcribe_file(str(temp_audio_path), language)
        
        if transcribed_text is None:
            raise HTTPException(status_code=500, detail="Transcription failed")
        
        return STTResponse(
            text=transcribed_text,
            language=language,
            duration=None
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in legacy STT: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"STT processing error: {str(e)}")
    finally:
        if temp_audio_path.exists():
            temp_audio_path.unlink(missing_ok=True)
        await audio_file.close()