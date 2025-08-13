# File: routers/core/tts.py
# Text-to-Speech generation endpoints using the realtime_conversation library

import io
import logging
import time
import numpy as np
from pathlib import Path
from typing import Optional, List, Dict, Any, Literal
from contextlib import asynccontextmanager

from fastapi import APIRouter, HTTPException, BackgroundTasks, Depends
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

# Import existing components for compatibility
import engine  # Will be replaced with library adapter later
import utils
from config import (
    get_predefined_voices_path,
    get_reference_audio_path,
    get_gen_default_temperature,
    get_gen_default_exaggeration,
    get_gen_default_cfg_weight,
    get_gen_default_seed,
    get_gen_default_speed_factor,
    get_audio_sample_rate,
    get_audio_output_format,
    config_manager,
)
from models import CustomTTSRequest, ErrorResponse

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/tts", tags=["Text-to-Speech"])


class OpenAISpeechRequest(BaseModel):
    """OpenAI-compatible speech synthesis request."""
    model: str
    input_: str = Field(..., alias="input")
    voice: str
    response_format: Literal["wav", "opus", "mp3"] = "wav"
    speed: float = 1.0
    seed: Optional[int] = None


@router.post(
    "",
    tags=["TTS Generation"],
    summary="Generate speech with custom parameters",
    responses={
        200: {
            "content": {"audio/wav": {}, "audio/opus": {}, "audio/mp3": {}},
            "description": "Successful audio generation.",
        },
        400: {
            "model": ErrorResponse,
            "description": "Invalid request parameters or input.",
        },
        404: {
            "model": ErrorResponse,
            "description": "Required resource not found (e.g., voice file).",
        },
        500: {
            "model": ErrorResponse,
            "description": "Internal server error during generation.",
        },
        503: {
            "model": ErrorResponse,
            "description": "TTS engine not available or model not loaded.",
        },
    },
)
async def synthesize_speech(
    request: CustomTTSRequest, background_tasks: BackgroundTasks
):
    """
    Generates speech audio from text using specified parameters.
    Handles various voice modes (predefined, clone) and audio processing options.
    Returns audio as a stream (WAV, Opus, or MP3).
    """
    perf_monitor = utils.PerformanceMonitor(
        enabled=config_manager.get_bool("server.enable_performance_monitor", False)
    )
    perf_monitor.record("TTS request received")

    if not engine.MODEL_LOADED:
        logger.error("TTS request failed: Model not loaded.")
        raise HTTPException(
            status_code=503,
            detail="TTS engine model is not currently loaded or available.",
        )

    logger.info(
        f"Received TTS request: mode='{request.voice_mode}', format='{request.output_format}'"
    )
    logger.debug(
        f"TTS params: seed={request.seed}, split={request.split_text}, chunk_size={request.chunk_size}"
    )
    logger.debug(f"Input text (first 100 chars): '{request.text[:100]}...'")

    try:
        # Resolve voice path
        audio_prompt_path_for_engine = _resolve_voice_path(request)
        perf_monitor.record("Voice path resolved")

        # Process text and generate audio
        audio_segments = await _process_text_chunks(request, audio_prompt_path_for_engine, perf_monitor)
        
        # Concatenate and post-process audio
        final_audio_np = _concatenate_and_process_audio(audio_segments, perf_monitor)
        
        # Encode and return response
        return _create_streaming_response(
            final_audio_np, 
            request.output_format or get_audio_output_format(),
            perf_monitor
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in TTS synthesis: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"TTS synthesis error: {str(e)}")


@router.post("/v1/audio/speech", tags=["OpenAI Compatible"])
async def openai_speech_endpoint(request: OpenAISpeechRequest):
    """
    OpenAI-compatible speech synthesis endpoint.
    Provides compatibility with OpenAI's text-to-speech API.
    """
    try:
        # Resolve voice path (check both predefined and reference directories)
        voice_path = _resolve_openai_voice_path(request.voice)
        
        if not voice_path:
            raise HTTPException(
                status_code=404, detail=f"Voice file '{request.voice}' not found."
            )

        # Check if TTS model is loaded
        if not engine.MODEL_LOADED:
            raise HTTPException(
                status_code=503,
                detail="TTS engine model is not currently loaded or available.",
            )

        # Synthesize audio using engine
        audio_tensor, sample_rate = engine.synthesize(
            text=request.input_,
            audio_prompt_path=str(voice_path),
            temperature=get_gen_default_temperature(),
            exaggeration=get_gen_default_exaggeration(),
            cfg_weight=get_gen_default_cfg_weight(),
            seed=request.seed if request.seed is not None else get_gen_default_seed(),
        )

        if audio_tensor is None or sample_rate is None:
            raise HTTPException(
                status_code=500, detail="TTS engine failed to synthesize audio."
            )

        # Apply speed factor
        if request.speed != 1.0:
            audio_tensor, _ = utils.apply_speed_factor(audio_tensor, sample_rate, request.speed)

        # Convert to numpy and encode
        audio_np = audio_tensor.cpu().numpy().squeeze()
        encoded_audio = utils.encode_audio(
            audio_array=audio_np,
            sample_rate=sample_rate,
            output_format=request.response_format,
            target_sample_rate=get_audio_sample_rate(),
        )

        if encoded_audio is None:
            raise HTTPException(status_code=500, detail="Failed to encode audio.")

        # Return streaming response
        media_type = f"audio/{request.response_format}"
        return StreamingResponse(io.BytesIO(encoded_audio), media_type=media_type)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in OpenAI speech endpoint: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# Helper functions

def _resolve_voice_path(request: CustomTTSRequest) -> Optional[Path]:
    """Resolve the voice file path based on request parameters."""
    if request.voice_mode == "predefined":
        if not request.predefined_voice_id:
            raise HTTPException(
                status_code=400,
                detail="Missing 'predefined_voice_id' for 'predefined' voice mode.",
            )
        voices_dir = get_predefined_voices_path(ensure_absolute=True)
        voice_path = voices_dir / request.predefined_voice_id
        if not voice_path.is_file():
            logger.error(f"Predefined voice file not found: {voice_path}")
            raise HTTPException(
                status_code=404,
                detail=f"Predefined voice file '{request.predefined_voice_id}' not found.",
            )
        logger.info(f"Using predefined voice: {request.predefined_voice_id}")
        return voice_path

    elif request.voice_mode == "clone":
        if not request.reference_audio_filename:
            raise HTTPException(
                status_code=400,
                detail="Missing 'reference_audio_filename' for 'clone' voice mode.",
            )
        ref_dir = get_reference_audio_path(ensure_absolute=True)
        voice_path = ref_dir / request.reference_audio_filename
        if not voice_path.is_file():
            logger.error(f"Reference audio file not found: {voice_path}")
            raise HTTPException(
                status_code=404,
                detail=f"Reference audio file '{request.reference_audio_filename}' not found.",
            )
        
        # Validate reference audio
        max_dur = config_manager.get_int("audio_output.max_reference_duration_sec", 30)
        is_valid, msg = utils.validate_reference_audio(voice_path, max_dur)
        if not is_valid:
            raise HTTPException(status_code=400, detail=f"Invalid reference audio: {msg}")
        
        logger.info(f"Using reference audio for cloning: {request.reference_audio_filename}")
        return voice_path

    return None


def _resolve_openai_voice_path(voice_name: str) -> Optional[Path]:
    """Resolve voice path for OpenAI-compatible endpoint."""
    # Check predefined voices first
    predefined_path = get_predefined_voices_path(ensure_absolute=True) / voice_name
    if predefined_path.is_file():
        return predefined_path
    
    # Check reference audio
    reference_path = get_reference_audio_path(ensure_absolute=True) / voice_name
    if reference_path.is_file():
        return reference_path
    
    return None


async def _process_text_chunks(
    request: CustomTTSRequest, 
    voice_path: Optional[Path], 
    perf_monitor: utils.PerformanceMonitor
) -> List[np.ndarray]:
    """Process text into chunks and synthesize each chunk."""
    
    # Determine if text should be split
    should_split = (
        request.split_text and 
        len(request.text) > (request.chunk_size * 1.5 if request.chunk_size else 120 * 1.5)
    )
    
    if should_split:
        chunk_size = request.chunk_size if request.chunk_size is not None else 120
        logger.info(f"Splitting text into chunks of size ~{chunk_size}.")
        text_chunks = utils.chunk_text_by_sentences(request.text, chunk_size)
        perf_monitor.record(f"Text split into {len(text_chunks)} chunks")
    else:
        text_chunks = [request.text]
        logger.info("Processing text as a single chunk")

    if not text_chunks:
        raise HTTPException(
            status_code=400, detail="Text processing resulted in no usable chunks."
        )

    audio_segments = []
    engine_sample_rate = None

    for i, chunk in enumerate(text_chunks):
        logger.info(f"Synthesizing chunk {i+1}/{len(text_chunks)}...")
        try:
            # Synthesize chunk
            chunk_audio_tensor, chunk_sr = engine.synthesize(
                text=chunk,
                audio_prompt_path=str(voice_path) if voice_path else None,
                temperature=request.temperature if request.temperature is not None else get_gen_default_temperature(),
                exaggeration=request.exaggeration if request.exaggeration is not None else get_gen_default_exaggeration(),
                cfg_weight=request.cfg_weight if request.cfg_weight is not None else get_gen_default_cfg_weight(),
                seed=request.seed if request.seed is not None else get_gen_default_seed(),
            )
            perf_monitor.record(f"Engine synthesized chunk {i+1}")

            if chunk_audio_tensor is None or chunk_sr is None:
                raise HTTPException(
                    status_code=500, 
                    detail=f"TTS engine failed to synthesize audio for chunk {i+1}."
                )

            if engine_sample_rate is None:
                engine_sample_rate = chunk_sr
            elif engine_sample_rate != chunk_sr:
                logger.warning(
                    f"Inconsistent sample rate: chunk {i+1} ({chunk_sr}Hz) differs from previous ({engine_sample_rate}Hz)"
                )

            # Apply speed factor if specified
            processed_audio = chunk_audio_tensor
            speed_factor = request.speed_factor if request.speed_factor is not None else get_gen_default_speed_factor()
            if speed_factor != 1.0:
                processed_audio, _ = utils.apply_speed_factor(processed_audio, chunk_sr, speed_factor)
                perf_monitor.record(f"Speed factor applied to chunk {i+1}")

            # Convert to numpy and add to segments
            audio_np = processed_audio.cpu().numpy().squeeze()
            audio_segments.append(audio_np)

        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error processing chunk {i+1}: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"Error processing audio chunk {i+1}: {str(e)}")

    return audio_segments


def _concatenate_and_process_audio(
    audio_segments: List[np.ndarray], 
    perf_monitor: utils.PerformanceMonitor
) -> np.ndarray:
    """Concatenate audio segments and apply global post-processing."""
    
    if not audio_segments:
        raise HTTPException(status_code=500, detail="No audio segments were generated.")

    # Concatenate segments
    final_audio_np = np.concatenate(audio_segments) if len(audio_segments) > 1 else audio_segments[0]
    perf_monitor.record("Audio segments concatenated")

    # Apply global audio processing
    sample_rate = 22050  # Chatterbox default sample rate
    
    if config_manager.get_bool("audio_processing.enable_silence_trimming", False):
        final_audio_np = utils.trim_lead_trail_silence(final_audio_np, sample_rate)
        perf_monitor.record("Silence trimming applied")

    if config_manager.get_bool("audio_processing.enable_internal_silence_fix", False):
        final_audio_np = utils.fix_internal_silence(final_audio_np, sample_rate)
        perf_monitor.record("Internal silence fix applied")

    if (config_manager.get_bool("audio_processing.enable_unvoiced_removal", False) 
        and utils.PARSELMOUTH_AVAILABLE):
        final_audio_np = utils.remove_long_unvoiced_segments(final_audio_np, sample_rate)
        perf_monitor.record("Unvoiced removal applied")

    return final_audio_np


def _create_streaming_response(
    audio_np: np.ndarray, 
    output_format: str, 
    perf_monitor: utils.PerformanceMonitor
) -> StreamingResponse:
    """Create streaming response with encoded audio."""
    
    sample_rate = 22050  # Chatterbox output sample rate
    target_sample_rate = get_audio_sample_rate()
    
    # Encode audio
    encoded_audio_bytes = utils.encode_audio(
        audio_array=audio_np,
        sample_rate=sample_rate,
        output_format=output_format,
        target_sample_rate=target_sample_rate,
    )
    perf_monitor.record(f"Audio encoded to {output_format}")

    if encoded_audio_bytes is None or len(encoded_audio_bytes) < 100:
        logger.error(f"Failed to encode audio to {output_format} or output too small")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to encode audio to {output_format} or generated invalid audio.",
        )

    # Create response
    media_type = f"audio/{output_format}"
    timestamp_str = time.strftime("%Y%m%d_%H%M%S")
    filename = utils.sanitize_filename(f"tts_output_{timestamp_str}.{output_format}")
    headers = {"Content-Disposition": f'attachment; filename="{filename}"'}

    logger.info(f"TTS synthesis completed: {filename}, {len(encoded_audio_bytes)} bytes")
    logger.debug(perf_monitor.report())

    return StreamingResponse(
        io.BytesIO(encoded_audio_bytes), 
        media_type=media_type, 
        headers=headers
    )


# Voice management endpoints

@router.get("/voices", response_model=List[Dict[str, str]], tags=["Voice Management"])
async def list_voices():
    """List available voices (both predefined and reference audio)."""
    try:
        voices = []
        
        # Add predefined voices
        predefined_voices = utils.get_predefined_voices()
        for voice in predefined_voices:
            voices.append({
                "id": voice["filename"],
                "name": voice["display_name"],
                "type": "predefined",
                "path": voice["filename"]
            })
        
        # Add reference audio files
        reference_files = utils.get_valid_reference_files()
        for ref_file in reference_files:
            voices.append({
                "id": ref_file,
                "name": ref_file.replace(".wav", "").replace(".mp3", ""),
                "type": "reference",
                "path": ref_file
            })
        
        return voices
        
    except Exception as e:
        logger.error(f"Error listing voices: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to list available voices")


@router.get("/voices/{voice_id}", tags=["Voice Management"])
async def get_voice_info(voice_id: str):
    """Get information about a specific voice."""
    try:
        # Check predefined voices
        predefined_path = get_predefined_voices_path(ensure_absolute=True) / voice_id
        if predefined_path.is_file():
            return {
                "id": voice_id,
                "type": "predefined",
                "path": str(predefined_path),
                "exists": True,
                "size_bytes": predefined_path.stat().st_size
            }
        
        # Check reference audio
        reference_path = get_reference_audio_path(ensure_absolute=True) / voice_id
        if reference_path.is_file():
            # Validate reference audio
            max_dur = config_manager.get_int("audio_output.max_reference_duration_sec", 30)
            is_valid, msg = utils.validate_reference_audio(reference_path, max_dur)
            
            return {
                "id": voice_id,
                "type": "reference", 
                "path": str(reference_path),
                "exists": True,
                "size_bytes": reference_path.stat().st_size,
                "validation": {
                    "is_valid": is_valid,
                    "message": msg
                }
            }
        
        raise HTTPException(status_code=404, detail=f"Voice '{voice_id}' not found")
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting voice info: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to get voice information")