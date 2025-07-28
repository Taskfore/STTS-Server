# File: routers/stt.py
# Speech-to-Text API endpoints

import logging
import uuid
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, HTTPException, UploadFile, File, Form, Depends, Request

from stt_engine import STTEngine
from models import STTResponse, ErrorResponse
from config import get_output_path

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/stt", tags=["Speech-to-Text"])


def get_stt_engine(request: Request) -> STTEngine:
    """Dependency to get STT engine from app state."""
    if not hasattr(request.app.state, 'stt_engine'):
        raise HTTPException(status_code=503, detail="STT engine not initialized")
    return request.app.state.stt_engine


@router.post(
    "",
    response_model=STTResponse,
    summary="Transcribe speech to text",
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
    stt_engine: STTEngine = Depends(get_stt_engine)
):
    """
    Transcribes speech from an uploaded audio file to text.
    Supports common audio formats and multiple languages.
    """
    if not stt_engine.model_loaded:
        logger.error("STT request failed: Model not loaded.")
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
        with open(temp_audio_path, "wb") as buffer:
            import shutil
            shutil.copyfileobj(audio_file.file, buffer)
        
        # Transcribe the audio
        transcribed_text = stt_engine.transcribe_file(str(temp_audio_path), language)
        
        if transcribed_text is None:
            raise HTTPException(
                status_code=500,
                detail="Transcription failed. Please check audio file format and content."
            )
        
        logger.info(f"STT transcription successful. Text length: {len(transcribed_text)} characters")
        
        return STTResponse(
            text=transcribed_text,
            language=language,
            duration=None  # Could add duration calculation if needed
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