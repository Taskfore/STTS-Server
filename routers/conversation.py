# File: routers/conversation.py
# Conversation pipeline API endpoints (STT→TTS)

import io
import logging
import time
import uuid
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, HTTPException, UploadFile, File, Form, Depends, Request
from fastapi.responses import StreamingResponse

import engine
import utils
from stt_engine import STTEngine
from models import ErrorResponse
from config import (
    get_output_path, 
    get_predefined_voices_path, 
    get_reference_audio_path,
    get_default_voice_id,
    get_gen_default_temperature,
    get_gen_default_exaggeration,
    get_gen_default_cfg_weight,
    get_gen_default_seed,
    get_gen_default_speed_factor,
    get_audio_sample_rate,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/conversation", tags=["Conversation Pipeline"])


def get_stt_engine(request: Request) -> STTEngine:
    """Dependency to get STT engine from app state."""
    if not hasattr(request.app.state, 'stt_engine'):
        raise HTTPException(status_code=503, detail="STT engine not initialized")
    return request.app.state.stt_engine


@router.post(
    "",
    summary="Speech-to-speech conversation (STT→TTS)",
    responses={
        200: {
            "content": {"audio/wav": {}, "audio/opus": {}, "audio/mp3": {}},
            "description": "Successful conversation response with audio output.",
        },
        400: {
            "model": ErrorResponse,
            "description": "Invalid request parameters or audio file.",
        },
        503: {
            "model": ErrorResponse,
            "description": "STT or TTS engine not available.",
        },
    },
)
async def process_conversation(
    audio_file: UploadFile = File(..., description="Audio file containing speech to process"),
    voice_mode: str = Form("predefined", description="Voice mode for TTS response"),
    predefined_voice_id: Optional[str] = Form(None, description="Predefined voice for response"),
    reference_audio_filename: Optional[str] = Form(None, description="Reference audio for voice cloning"),
    output_format: str = Form("wav", description="Output audio format"),
    language: Optional[str] = Form(None, description="STT language (auto-detect if None)"),
    temperature: Optional[float] = Form(None, description="Temperature for TTS generation (uses default if None)"),
    exaggeration: Optional[float] = Form(None, description="Exaggeration for TTS generation (uses default if None)"),
    cfg_weight: Optional[float] = Form(None, description="CFG weight for TTS generation (uses default if None)"),
    seed: Optional[int] = Form(None, description="Seed for TTS generation (uses default if None)"),
    speed_factor: Optional[float] = Form(None, description="Speed factor for TTS generation (uses default if None)"),
    stt_engine: STTEngine = Depends(get_stt_engine)
):
    """
    Complete conversation pipeline: transcribes input speech and generates TTS response.
    Returns audio response directly as streaming content.
    """
    # Validate engines are loaded
    if not stt_engine.model_loaded:
        raise HTTPException(status_code=503, detail="STT engine not available")
    
    if not engine.MODEL_LOADED:
        raise HTTPException(status_code=503, detail="TTS engine not available")
    
    if not audio_file.filename:
        raise HTTPException(status_code=400, detail="No audio file provided")
    
    # Validate file extension for STT
    allowed_extensions = {".wav", ".mp3", ".m4a", ".flac", ".ogg"}
    file_ext = Path(audio_file.filename).suffix.lower()
    if file_ext not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported audio format: {file_ext}. Supported: {', '.join(allowed_extensions)}"
        )
    
    logger.info(f"Received conversation request: {audio_file.filename} -> {voice_mode}")
    
    # Save uploaded file temporarily
    temp_audio_path = get_output_path() / f"temp_conversation_{uuid.uuid4().hex[:8]}{file_ext}"
    
    try:
        # Step 1: Save uploaded audio
        with open(temp_audio_path, "wb") as buffer:
            import shutil
            shutil.copyfileobj(audio_file.file, buffer)
        
        # Step 2: STT - Transcribe audio to text
        transcribed_text = stt_engine.transcribe_file(str(temp_audio_path), language)
        if transcribed_text is None:
            raise HTTPException(
                status_code=500, 
                detail="Speech transcription failed. Please check audio quality and format."
            )
        
        logger.info(f"Transcribed text: '{transcribed_text[:100]}...' ({len(transcribed_text)} chars)")
        
        # Step 3: Determine voice for TTS response
        audio_prompt_path_for_engine: Optional[Path] = None
        
        if voice_mode == "predefined":
            if not predefined_voice_id:
                # Use default voice if none specified
                predefined_voice_id = get_default_voice_id()
            
            voices_dir = get_predefined_voices_path(ensure_absolute=True)
            potential_path = voices_dir / predefined_voice_id
            if not potential_path.is_file():
                raise HTTPException(
                    status_code=404,
                    detail=f"Predefined voice '{predefined_voice_id}' not found"
                )
            audio_prompt_path_for_engine = potential_path
            
        elif voice_mode == "clone":
            if not reference_audio_filename:
                raise HTTPException(
                    status_code=400,
                    detail="Reference audio filename required for voice cloning"
                )
            
            ref_dir = get_reference_audio_path(ensure_absolute=True)
            potential_path = ref_dir / reference_audio_filename
            if not potential_path.is_file():
                raise HTTPException(
                    status_code=404,
                    detail=f"Reference audio '{reference_audio_filename}' not found"
                )
            audio_prompt_path_for_engine = potential_path
        
        # Step 4: TTS - Convert transcribed text to speech
        audio_tensor, sample_rate = engine.synthesize(
            text=transcribed_text,
            audio_prompt_path=str(audio_prompt_path_for_engine) if audio_prompt_path_for_engine else None,
            temperature=temperature if temperature is not None else get_gen_default_temperature(),
            exaggeration=exaggeration if exaggeration is not None else get_gen_default_exaggeration(),
            cfg_weight=cfg_weight if cfg_weight is not None else get_gen_default_cfg_weight(),
            seed=seed if seed is not None else get_gen_default_seed(),
        )
        
        if audio_tensor is None or sample_rate is None:
            raise HTTPException(
                status_code=500,
                detail="TTS generation failed for transcribed text"
            )
        
        # Step 5: Process and encode audio
        audio_np = audio_tensor.cpu().numpy().squeeze()
        
        # Apply speed factor if configured
        final_speed_factor = speed_factor if speed_factor is not None else get_gen_default_speed_factor()
        if final_speed_factor != 1.0:
            audio_np, _ = utils.apply_speed_factor(audio_np, sample_rate, final_speed_factor)
        
        # Encode to requested format
        encoded_audio_bytes = utils.encode_audio(
            audio_array=audio_np,
            sample_rate=sample_rate,
            output_format=output_format,
            target_sample_rate=get_audio_sample_rate(),
        )
        
        if encoded_audio_bytes is None or len(encoded_audio_bytes) < 100:
            raise HTTPException(
                status_code=500,
                detail=f"Failed to encode conversation response to {output_format}"
            )
        
        # Step 6: Return streaming response
        media_type = f"audio/{output_format}"
        timestamp_str = time.strftime("%Y%m%d_%H%M%S")
        download_filename = utils.sanitize_filename(f"conversation_response_{timestamp_str}.{output_format}")
        headers = {"Content-Disposition": f'attachment; filename="{download_filename}"'}
        
        logger.info(f"Conversation complete: {len(transcribed_text)} chars -> {len(encoded_audio_bytes)} bytes audio")
        
        return StreamingResponse(
            io.BytesIO(encoded_audio_bytes), 
            media_type=media_type, 
            headers=headers
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in conversation pipeline: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Conversation pipeline error: {str(e)}")
    finally:
        # Clean up temporary file
        if temp_audio_path.exists():
            temp_audio_path.unlink(missing_ok=True)
        await audio_file.close()