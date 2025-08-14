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

# Library imports (with fallback to adapters)
try:
    from realtime_conversation.adapters.tts import ChatterboxTTSEngine
    LIBRARY_AVAILABLE = True
except ImportError:
    LIBRARY_AVAILABLE = False
    ChatterboxTTSEngine = None

# Import adapter and legacy engine
from adapters.legacy_engines import LegacyTTSEngineAdapter, create_legacy_tts_adapter
import engine  # Legacy engine (to be phased out)
import utils

# Import middleware system
from middleware.base import MiddlewarePipeline, RequestContext, TimingMiddleware, LoggingMiddleware, AnalyticsMiddleware
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

# Global adapter instance (will be replaced with proper DI later)
_tts_adapter_instance: Optional[LegacyTTSEngineAdapter] = None
_middleware_pipeline: Optional[MiddlewarePipeline] = None


def get_tts_adapter() -> LegacyTTSEngineAdapter:
    """
    Dependency to get TTS adapter.
    
    This uses the adapter pattern to bridge between legacy engines and the new library interface.
    When the library is fully integrated, this will switch to native library engines.
    """
    global _tts_adapter_instance
    
    if _tts_adapter_instance is None:
        if LIBRARY_AVAILABLE:
            # TODO: Use native ChatterboxTTSEngine when fully integrated
            logger.info("Library available but using legacy adapter for compatibility")
            _tts_adapter_instance = create_legacy_tts_adapter()
        else:
            logger.info("Using legacy TTS adapter")
            _tts_adapter_instance = create_legacy_tts_adapter()
    
    return _tts_adapter_instance


def get_middleware_pipeline() -> MiddlewarePipeline:
    """Get or create the middleware pipeline for TTS processing."""
    global _middleware_pipeline
    
    if _middleware_pipeline is None:
        _middleware_pipeline = MiddlewarePipeline()
        
        # Add built-in middleware based on configuration
        if config_manager.get_bool("server.enable_performance_monitor", False):
            _middleware_pipeline.add_middleware(TimingMiddleware())
        
        # Always add logging middleware
        log_level = logging.INFO if config_manager.get_string("server.log_level", "INFO") == "INFO" else logging.DEBUG
        _middleware_pipeline.add_middleware(LoggingMiddleware(log_level=log_level))
        
        # Add analytics if enabled
        if config_manager.get_bool("analytics.enable", False):
            _middleware_pipeline.add_middleware(AnalyticsMiddleware())
        
        logger.info(f"TTS middleware pipeline initialized with {_middleware_pipeline.enabled_middleware_count} middleware")
    
    return _middleware_pipeline


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
    request: CustomTTSRequest, 
    background_tasks: BackgroundTasks,
    tts_adapter: LegacyTTSEngineAdapter = Depends(get_tts_adapter),
    middleware_pipeline: MiddlewarePipeline = Depends(get_middleware_pipeline)
):
    """
    Generates speech audio from text using specified parameters.
    Handles various voice modes (predefined, clone) and audio processing options.
    Returns audio as a stream (WAV, Opus, or MP3).
    """
    # Create request context for middleware
    import uuid
    context = RequestContext(
        request_id=str(uuid.uuid4())[:8],
        request_type="tts"
    )
    
    # Add input data to context
    context.input_data.update({
        "text": request.text,
        "voice_mode": request.voice_mode,
        "voice_id": request.predefined_voice_id or request.reference_audio_filename,
        "output_format": request.output_format or get_audio_output_format(),
        "temperature": request.temperature,
        "exaggeration": request.exaggeration,
        "cfg_weight": request.cfg_weight,
        "seed": request.seed,
        "speed_factor": request.speed_factor,
        "split_text": request.split_text,
        "chunk_size": request.chunk_size
    })

    # Define core TTS processing function for middleware
    async def core_tts_processing(ctx: RequestContext) -> RequestContext:
        """Core TTS processing wrapped for middleware."""
        
        if not tts_adapter.model_loaded:
            raise HTTPException(
                status_code=503,
                detail="TTS engine model is not currently loaded or available.",
            )

        try:
            # Resolve voice path
            audio_prompt_path_for_engine = _resolve_voice_path(request)
            ctx.add_metadata("voice_path_resolved", True)

            # Process text and generate audio
            audio_segments = await _process_text_chunks(request, audio_prompt_path_for_engine, tts_adapter, ctx)
            
            # Concatenate and post-process audio
            final_audio_np = _concatenate_and_process_audio(audio_segments, ctx)
            
            # Add output data to context
            ctx.output_data.update({
                "audio_array": final_audio_np,
                "audio_size": len(final_audio_np.tobytes()) if hasattr(final_audio_np, 'tobytes') else 0,
                "sample_rate": 22050  # Chatterbox default
            })
            
            return ctx

        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Unexpected error in TTS synthesis: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"TTS synthesis error: {str(e)}")
    
    # Process through middleware pipeline
    try:
        result_context = await middleware_pipeline.process(context, core_tts_processing)
        
        if result_context.status == "error" or result_context.error:
            error = result_context.error or Exception("Unknown error in TTS processing")
            if isinstance(error, HTTPException):
                raise error
            else:
                raise HTTPException(status_code=500, detail=str(error))
        
        # Create streaming response from result
        final_audio_np = result_context.output_data["audio_array"]
        output_format = request.output_format or get_audio_output_format()
        
        return _create_streaming_response_from_context(final_audio_np, output_format, result_context)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Middleware processing error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"TTS processing error: {str(e)}")


@router.post("/v1/audio/speech", tags=["OpenAI Compatible"])
async def openai_speech_endpoint(
    request: OpenAISpeechRequest,
    tts_adapter: LegacyTTSEngineAdapter = Depends(get_tts_adapter)
):
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
        if not tts_adapter.model_loaded:
            raise HTTPException(
                status_code=503,
                detail="TTS engine model is not currently loaded or available.",
            )

        # Build voice configuration
        voice_config = {
            "voice_path": str(voice_path),
            "voice_id": voice_path.name if voice_path else None,
            "temperature": get_gen_default_temperature(),
            "exaggeration": get_gen_default_exaggeration(),
            "cfg_weight": get_gen_default_cfg_weight(),
            "seed": request.seed if request.seed is not None else get_gen_default_seed(),
            "speed_factor": request.speed,
        }

        # Synthesize audio using adapter
        synthesis_result = await tts_adapter.synthesize(request.input_, voice_config)

        if synthesis_result is None or synthesis_result.audio_data is None:
            raise HTTPException(
                status_code=500, detail="TTS adapter failed to synthesize audio."
            )

        # Get audio data
        audio_data = synthesis_result.audio_data.data
        sample_rate = synthesis_result.audio_data.sample_rate

        # Convert to numpy for encoding
        audio_np = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
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
    tts_adapter: LegacyTTSEngineAdapter,
    context: RequestContext
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
        context.add_metadata("text_chunks", len(text_chunks))
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
            # Build voice config for adapter
            voice_config = {
                "voice_path": str(voice_path) if voice_path else None,
                "voice_id": voice_path.name if voice_path else None,
                "temperature": request.temperature if request.temperature is not None else get_gen_default_temperature(),
                "exaggeration": request.exaggeration if request.exaggeration is not None else get_gen_default_exaggeration(),
                "cfg_weight": request.cfg_weight if request.cfg_weight is not None else get_gen_default_cfg_weight(),
                "seed": request.seed if request.seed is not None else get_gen_default_seed(),
                "speed_factor": request.speed_factor if request.speed_factor is not None else get_gen_default_speed_factor(),
            }
            
            # Synthesize chunk using adapter
            synthesis_result = await tts_adapter.synthesize(chunk, voice_config)
            context.add_metadata(f"chunk_{i+1}_synthesized", True)

            if synthesis_result is None or synthesis_result.audio_data is None:
                raise HTTPException(
                    status_code=500, 
                    detail=f"TTS adapter failed to synthesize audio for chunk {i+1}."
                )

            # Get audio data from synthesis result
            chunk_sr = synthesis_result.audio_data.sample_rate
            
            if engine_sample_rate is None:
                engine_sample_rate = chunk_sr
            elif engine_sample_rate != chunk_sr:
                logger.warning(
                    f"Inconsistent sample rate: chunk {i+1} ({chunk_sr}Hz) differs from previous ({engine_sample_rate}Hz)"
                )

            # Convert AudioData to numpy array
            # The adapter already applied speed factor during synthesis
            audio_bytes = synthesis_result.audio_data.data
            audio_np = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
            audio_segments.append(audio_np)

        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error processing chunk {i+1}: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"Error processing audio chunk {i+1}: {str(e)}")

    return audio_segments


def _concatenate_and_process_audio(
    audio_segments: List[np.ndarray], 
    context: RequestContext
) -> np.ndarray:
    """Concatenate audio segments and apply global post-processing."""
    
    if not audio_segments:
        raise HTTPException(status_code=500, detail="No audio segments were generated.")

    # Concatenate segments
    final_audio_np = np.concatenate(audio_segments) if len(audio_segments) > 1 else audio_segments[0]
    context.add_metadata("audio_segments_concatenated", len(audio_segments))

    # Apply global audio processing
    sample_rate = 22050  # Chatterbox default sample rate
    
    if config_manager.get_bool("audio_processing.enable_silence_trimming", False):
        final_audio_np = utils.trim_lead_trail_silence(final_audio_np, sample_rate)
        context.add_metadata("silence_trimming_applied", True)

    if config_manager.get_bool("audio_processing.enable_internal_silence_fix", False):
        final_audio_np = utils.fix_internal_silence(final_audio_np, sample_rate)
        context.add_metadata("internal_silence_fix_applied", True)

    if (config_manager.get_bool("audio_processing.enable_unvoiced_removal", False) 
        and utils.PARSELMOUTH_AVAILABLE):
        final_audio_np = utils.remove_long_unvoiced_segments(final_audio_np, sample_rate)
        context.add_metadata("unvoiced_removal_applied", True)

    return final_audio_np


def _create_streaming_response_from_context(
    audio_np: np.ndarray, 
    output_format: str, 
    context: RequestContext
) -> StreamingResponse:
    """Create streaming response with encoded audio using context data."""
    
    sample_rate = 22050  # Chatterbox output sample rate
    target_sample_rate = get_audio_sample_rate()
    
    # Encode audio
    encoded_audio_bytes = utils.encode_audio(
        audio_array=audio_np,
        sample_rate=sample_rate,
        output_format=output_format,
        target_sample_rate=target_sample_rate,
    )
    context.add_metadata("audio_encoded", True)

    if encoded_audio_bytes is None or len(encoded_audio_bytes) < 100:
        logger.error(f"Failed to encode audio to {output_format} or output too small")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to encode audio to {output_format} or generated invalid audio.",
        )

    # Add final output data to context
    context.output_data.update({
        "encoded_audio_size": len(encoded_audio_bytes),
        "output_format": output_format,
        "audio_duration_ms": len(audio_np) / sample_rate * 1000 if len(audio_np) > 0 else 0
    })

    # Create response
    media_type = f"audio/{output_format}"
    timestamp_str = time.strftime("%Y%m%d_%H%M%S")
    filename = utils.sanitize_filename(f"tts_output_{timestamp_str}.{output_format}")
    headers = {"Content-Disposition": f'attachment; filename="{filename}"'}

    logger.info(f"TTS synthesis completed via middleware: {filename}, {len(encoded_audio_bytes)} bytes, duration: {context.duration_ms:.2f}ms")

    return StreamingResponse(
        io.BytesIO(encoded_audio_bytes), 
        media_type=media_type, 
        headers=headers
    )


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


@router.get("/middleware/status", summary="TTS middleware status")
async def get_middleware_status(
    middleware_pipeline: MiddlewarePipeline = Depends(get_middleware_pipeline)
):
    """Get status and statistics for TTS middleware pipeline."""
    try:
        middleware_info = []
        
        for middleware in middleware_pipeline.middleware:
            info = {
                "name": middleware.name,
                "enabled": middleware.enabled,
                "type": type(middleware).__name__
            }
            
            # Get statistics if available
            if hasattr(middleware, 'get_statistics'):
                try:
                    info["statistics"] = middleware.get_statistics()
                except Exception as e:
                    info["statistics_error"] = str(e)
            
            middleware_info.append(info)
        
        return {
            "total_middleware": len(middleware_pipeline.middleware),
            "enabled_middleware": middleware_pipeline.enabled_middleware_count,
            "middleware_names": middleware_pipeline.get_middleware_names(),
            "middleware_details": middleware_info
        }
        
    except Exception as e:
        logger.error(f"Error getting middleware status: {e}")
        raise HTTPException(status_code=500, detail="Failed to get middleware status")


@router.post("/middleware/reload", summary="Reload TTS middleware")
async def reload_middleware():
    """Reload the TTS middleware pipeline."""
    global _middleware_pipeline
    
    try:
        # Clear current pipeline
        _middleware_pipeline = None
        
        # Create new pipeline (this will reinitialize with current config)
        new_pipeline = get_middleware_pipeline()
        
        return {
            "status": "success",
            "message": "TTS middleware pipeline reloaded",
            "enabled_middleware": new_pipeline.enabled_middleware_count,
            "middleware_names": new_pipeline.get_middleware_names()
        }
        
    except Exception as e:
        logger.error(f"Error reloading middleware: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to reload middleware: {str(e)}")


@router.get("/statistics", summary="TTS system statistics")
async def get_tts_statistics(
    middleware_pipeline: MiddlewarePipeline = Depends(get_middleware_pipeline),
    tts_adapter: LegacyTTSEngineAdapter = Depends(get_tts_adapter)
):
    """Get comprehensive TTS system statistics."""
    try:
        stats = {
            "system": {
                "adapter_type": type(tts_adapter).__name__,
                "model_loaded": tts_adapter.model_loaded,
                "library_available": LIBRARY_AVAILABLE
            },
            "middleware": {
                "total_count": len(middleware_pipeline.middleware),
                "enabled_count": middleware_pipeline.enabled_middleware_count,
                "middleware_names": middleware_pipeline.get_middleware_names()
            }
        }
        
        # Collect middleware statistics
        for middleware in middleware_pipeline.middleware:
            if hasattr(middleware, 'get_statistics'):
                try:
                    middleware_stats = middleware.get_statistics()
                    stats["middleware"][middleware.name] = middleware_stats
                except Exception as e:
                    stats["middleware"][f"{middleware.name}_error"] = str(e)
        
        return stats
        
    except Exception as e:
        logger.error(f"Error getting TTS statistics: {e}")
        raise HTTPException(status_code=500, detail="Failed to get TTS statistics")