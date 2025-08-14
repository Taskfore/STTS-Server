# File: routers/core/conversation.py  
# Conversation pipeline (STT→TTS) using library integration and adapters

import io
import logging
import time
import uuid
import asyncio
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, HTTPException, UploadFile, File, Form, Depends, Request
from fastapi.responses import StreamingResponse

# Library imports (with fallback)
try:
    from realtime_conversation import ConversationEngine
    from realtime_conversation.core.interfaces import AudioData, ConversationContext, ConversationState
    from realtime_conversation.plugins.response_generation import EchoResponseGenerator
    LIBRARY_AVAILABLE = True
except ImportError:
    LIBRARY_AVAILABLE = False
    ConversationEngine = None

# Fallback to adapters and legacy engines
from adapters.legacy_engines import (
    LegacySTTEngineAdapter, 
    LegacyTTSEngineAdapter,
    ConfigurationAdapter,
    create_legacy_stt_adapter, 
    create_legacy_tts_adapter,
    create_config_adapter
)

import engine
import utils
from models import ErrorResponse
from config import (
    get_output_path, 
    get_predefined_voices_path, 
    get_reference_audio_path,
    get_default_voice_id,
    get_audio_sample_rate,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/conversation", tags=["Conversation Pipeline"])

# Global instances (will be replaced with proper DI later)
_conversation_engine_instance: Optional[ConversationEngine] = None
_adapters: Optional[dict] = None


async def get_conversation_system():
    """
    Get or create conversation system components.
    
    This function provides either a full ConversationEngine (when library is available)
    or individual adapters (fallback mode) for conversation processing.
    """
    global _conversation_engine_instance, _adapters
    
    if LIBRARY_AVAILABLE and _conversation_engine_instance is None:
        # Create full library-based conversation engine
        try:
            config_adapter = create_config_adapter()
            
            # Create conversation engine with adapters
            conversation_engine = ConversationEngine(config_adapter)
            
            # Configure with legacy adapters for now
            stt_adapter = create_legacy_stt_adapter()
            tts_adapter = create_legacy_tts_adapter()
            
            conversation_engine.configure_stt(stt_adapter)
            conversation_engine.configure_tts(tts_adapter)
            
            # Add basic response generator
            response_generator = EchoResponseGenerator()
            conversation_engine.configure_response_generation(response_generator)
            
            _conversation_engine_instance = conversation_engine
            logger.info("ConversationEngine initialized with library")
            
        except Exception as e:
            logger.error(f"Failed to initialize ConversationEngine: {e}")
            # Fall back to adapters
            LIBRARY_AVAILABLE = False
    
    if not LIBRARY_AVAILABLE:
        # Use individual adapters
        if _adapters is None:
            _adapters = {
                "stt": create_legacy_stt_adapter(),
                "tts": create_legacy_tts_adapter(),
                "config": create_config_adapter()
            }
            logger.info("Conversation system using legacy adapters")
    
    return _conversation_engine_instance if LIBRARY_AVAILABLE else _adapters


def get_legacy_stt_engine(request: Request):
    """Legacy dependency for backward compatibility."""
    if not hasattr(request.app.state, 'stt_engine'):
        raise HTTPException(status_code=503, detail="STT engine not initialized")
    return request.app.state.stt_engine


@router.post(
    "",
    summary="Speech-to-speech conversation (STT→TTS) with library integration",
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
    temperature: Optional[float] = Form(None, description="Temperature for TTS generation"),
    exaggeration: Optional[float] = Form(None, description="Exaggeration for TTS generation"),
    cfg_weight: Optional[float] = Form(None, description="CFG weight for TTS generation"),
    seed: Optional[int] = Form(None, description="Seed for TTS generation"),
    speed_factor: Optional[float] = Form(None, description="Speed factor for TTS generation"),
):
    """
    Complete conversation pipeline using library integration.
    
    This endpoint demonstrates the new architecture:
    1. Uses adapters to bridge legacy engines with library interfaces
    2. Provides ConversationEngine when library is available
    3. Falls back to individual adapters when library isn't available
    4. Maintains full backward compatibility
    """
    # Get conversation system
    conversation_system = await get_conversation_system()
    
    if not audio_file.filename:
        raise HTTPException(status_code=400, detail="No audio file provided")
    
    # Validate file extension
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
        # Save uploaded audio
        with open(temp_audio_path, "wb") as buffer:
            import shutil
            shutil.copyfileobj(audio_file.file, buffer)
        
        if LIBRARY_AVAILABLE and isinstance(conversation_system, ConversationEngine):
            # Use full library pipeline
            result = await _process_with_conversation_engine(
                temp_audio_path, conversation_system, voice_mode, 
                predefined_voice_id, reference_audio_filename, 
                output_format, language, temperature, exaggeration, 
                cfg_weight, seed, speed_factor
            )
        else:
            # Use adapter-based processing
            result = await _process_with_adapters(
                temp_audio_path, conversation_system, voice_mode,
                predefined_voice_id, reference_audio_filename,
                output_format, language, temperature, exaggeration,
                cfg_weight, seed, speed_factor
            )
        
        return result
        
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


async def _process_with_conversation_engine(
    audio_path: Path,
    conversation_engine: ConversationEngine,
    voice_mode: str,
    predefined_voice_id: Optional[str],
    reference_audio_filename: Optional[str], 
    output_format: str,
    language: Optional[str],
    temperature: Optional[float],
    exaggeration: Optional[float],
    cfg_weight: Optional[float],
    seed: Optional[int],
    speed_factor: Optional[float]
) -> StreamingResponse:
    """Process conversation using the full ConversationEngine library."""
    
    try:
        # Create AudioData from file
        with open(audio_path, "rb") as f:
            audio_bytes = f.read()
        
        audio_data = AudioData(
            data=audio_bytes,
            sample_rate=16000,  # Will be handled by adapter
            channels=1,
            format="file"
        )
        
        # Create conversation context
        context = ConversationContext()
        context.audio_input = audio_data
        context.state = ConversationState.PROCESSING
        
        # Add voice configuration to context
        voice_config = _build_voice_config(
            voice_mode, predefined_voice_id, reference_audio_filename,
            temperature, exaggeration, cfg_weight, seed, speed_factor
        )
        context.user_data["voice_config"] = voice_config
        context.user_data["language"] = language
        
        # Process through conversation engine
        # Note: This is a simplified version - full integration would use WebSocket adapter
        # For now, we'll process the components individually but through the engine interfaces
        
        # STT
        transcription_result = await conversation_engine.stt_engine.transcribe(audio_data, language)
        if not transcription_result:
            raise HTTPException(status_code=500, detail="Speech transcription failed")
        
        context.transcription = transcription_result
        logger.info(f"Transcribed: '{transcription_result.text[:100]}...'")
        
        # Response Generation  
        response_text = await conversation_engine.response_generator.generate_response(
            transcription_result, context
        )
        context.response_text = response_text
        logger.info(f"Generated response: '{response_text}'")
        
        # TTS
        synthesis_result = await conversation_engine.tts_engine.synthesize(response_text, voice_config)
        if not synthesis_result:
            raise HTTPException(status_code=500, detail="TTS generation failed")
        
        context.synthesis_result = synthesis_result
        
        # Convert to streaming response
        return _create_audio_streaming_response(synthesis_result.audio_data, output_format)
        
    except Exception as e:
        logger.error(f"Error in ConversationEngine processing: {e}")
        raise


async def _process_with_adapters(
    audio_path: Path,
    adapters: dict,
    voice_mode: str,
    predefined_voice_id: Optional[str],
    reference_audio_filename: Optional[str],
    output_format: str, 
    language: Optional[str],
    temperature: Optional[float],
    exaggeration: Optional[float],
    cfg_weight: Optional[float],
    seed: Optional[int],
    speed_factor: Optional[float]
) -> StreamingResponse:
    """Process conversation using individual adapters (fallback mode)."""
    
    try:
        stt_adapter = adapters["stt"]
        tts_adapter = adapters["tts"]
        
        # Check adapters are available
        if not await stt_adapter.is_available():
            raise HTTPException(status_code=503, detail="STT engine not available")
        
        if not tts_adapter.model_loaded:
            raise HTTPException(status_code=503, detail="TTS engine not available")
        
        # Step 1: STT - Transcribe audio to text
        with open(audio_path, "rb") as f:
            audio_bytes = f.read()
        
        audio_data = AudioData(
            data=audio_bytes,
            sample_rate=16000,
            channels=1,
            format="file"
        )
        
        transcription_result = await stt_adapter.transcribe(audio_data, language)
        if not transcription_result:
            raise HTTPException(status_code=500, detail="Speech transcription failed")
        
        transcribed_text = transcription_result.text.strip()
        logger.info(f"Transcribed text: '{transcribed_text[:100]}...' ({len(transcribed_text)} chars)")
        
        # Step 2: Generate response text (simple echo for now)
        response_text = f"You said: {transcribed_text}"
        logger.info(f"Response text: '{response_text}'")
        
        # Step 3: TTS - Convert response to speech
        voice_config = _build_voice_config(
            voice_mode, predefined_voice_id, reference_audio_filename,
            temperature, exaggeration, cfg_weight, seed, speed_factor
        )
        
        synthesis_result = await tts_adapter.synthesize(response_text, voice_config)
        if not synthesis_result:
            raise HTTPException(status_code=500, detail="TTS generation failed")
        
        # Step 4: Return streaming response
        return _create_audio_streaming_response(synthesis_result.audio_data, output_format)
        
    except Exception as e:
        logger.error(f"Error in adapter-based processing: {e}")
        raise


def _build_voice_config(
    voice_mode: str,
    predefined_voice_id: Optional[str],
    reference_audio_filename: Optional[str],
    temperature: Optional[float],
    exaggeration: Optional[float], 
    cfg_weight: Optional[float],
    seed: Optional[int],
    speed_factor: Optional[float]
) -> dict:
    """Build voice configuration for TTS synthesis."""
    
    # Resolve voice path
    voice_path = None
    voice_id = None
    
    if voice_mode == "predefined":
        if not predefined_voice_id:
            predefined_voice_id = get_default_voice_id()
        
        voices_dir = get_predefined_voices_path(ensure_absolute=True)
        potential_path = voices_dir / predefined_voice_id
        if not potential_path.is_file():
            raise HTTPException(
                status_code=404,
                detail=f"Predefined voice '{predefined_voice_id}' not found"
            )
        voice_path = str(potential_path)
        voice_id = predefined_voice_id
        
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
        voice_path = str(potential_path)
        voice_id = reference_audio_filename
    
    # Build configuration
    from config import (
        get_gen_default_temperature, get_gen_default_exaggeration,
        get_gen_default_cfg_weight, get_gen_default_seed, get_gen_default_speed_factor
    )
    
    return {
        "voice_path": voice_path,
        "voice_id": voice_id,
        "temperature": temperature if temperature is not None else get_gen_default_temperature(),
        "exaggeration": exaggeration if exaggeration is not None else get_gen_default_exaggeration(),
        "cfg_weight": cfg_weight if cfg_weight is not None else get_gen_default_cfg_weight(),
        "seed": seed if seed is not None else get_gen_default_seed(),
        "speed_factor": speed_factor if speed_factor is not None else get_gen_default_speed_factor(),
    }


def _create_audio_streaming_response(audio_data: AudioData, output_format: str) -> StreamingResponse:
    """Create streaming response from AudioData."""
    
    # For now, assume audio_data.data is already in the correct format
    # In a full implementation, we'd convert between formats as needed
    
    if output_format.lower() != "wav":
        # TODO: Add format conversion using utils.encode_audio
        logger.warning(f"Format conversion to {output_format} not yet implemented, returning as WAV")
        output_format = "wav"
    
    media_type = f"audio/{output_format}"
    timestamp_str = time.strftime("%Y%m%d_%H%M%S")
    filename = utils.sanitize_filename(f"conversation_response_{timestamp_str}.{output_format}")
    headers = {"Content-Disposition": f'attachment; filename="{filename}"'}
    
    logger.info(f"Conversation complete: returning {len(audio_data.data)} bytes as {output_format}")
    
    return StreamingResponse(
        io.BytesIO(audio_data.data),
        media_type=media_type,
        headers=headers
    )


@router.get("/status", summary="Conversation system status")
async def get_conversation_status():
    """Get status of the conversation system."""
    try:
        conversation_system = await get_conversation_system()
        
        if LIBRARY_AVAILABLE and isinstance(conversation_system, ConversationEngine):
            # Library-based system
            status = {
                "system_type": "ConversationEngine",
                "library_available": True,
                "stt_available": conversation_system.stt_engine and await conversation_system.stt_engine.is_available(),
                "tts_available": conversation_system.tts_engine and conversation_system.tts_engine.model_loaded,
                "response_generator_available": conversation_system.response_generator is not None,
                "middleware_count": len(conversation_system.pipeline.middleware) if hasattr(conversation_system, 'pipeline') else 0
            }
        else:
            # Adapter-based system
            adapters = conversation_system
            status = {
                "system_type": "Legacy Adapters",
                "library_available": False,
                "stt_available": adapters["stt"] and await adapters["stt"].is_available(),
                "tts_available": adapters["tts"] and adapters["tts"].model_loaded,
                "config_available": adapters["config"] is not None,
                "response_generator_available": True  # Simple echo always available
            }
        
        return status
        
    except Exception as e:
        logger.error(f"Error getting conversation status: {e}")
        raise HTTPException(status_code=500, detail="Failed to get conversation status")


@router.post("/reload", summary="Reload conversation system")
async def reload_conversation_system():
    """Reload the conversation system components."""
    global _conversation_engine_instance, _adapters
    
    try:
        # Clear current instances
        _conversation_engine_instance = None
        _adapters = None
        
        # Recreate system
        conversation_system = await get_conversation_system()
        
        return {
            "status": "success",
            "message": "Conversation system reloaded",
            "system_type": "ConversationEngine" if LIBRARY_AVAILABLE else "Legacy Adapters"
        }
        
    except Exception as e:
        logger.error(f"Error reloading conversation system: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to reload: {str(e)}")


# Legacy compatibility endpoint
@router.post("/legacy", include_in_schema=False)
async def process_conversation_legacy(
    audio_file: UploadFile = File(...),
    voice_mode: str = Form("predefined"),
    predefined_voice_id: Optional[str] = Form(None),
    reference_audio_filename: Optional[str] = Form(None),
    output_format: str = Form("wav"),
    language: Optional[str] = Form(None),
    legacy_stt_engine=Depends(get_legacy_stt_engine)
):
    """
    Legacy compatibility endpoint using original implementation.
    
    This maintains 100% backward compatibility during transition.
    """
    # This replicates the original conversation.py logic
    if not legacy_stt_engine.model_loaded:
        raise HTTPException(status_code=503, detail="STT engine not available")
    
    if not engine.MODEL_LOADED:
        raise HTTPException(status_code=503, detail="TTS engine not available")
    
    if not audio_file.filename:
        raise HTTPException(status_code=400, detail="No audio file provided")
    
    allowed_extensions = {".wav", ".mp3", ".m4a", ".flac", ".ogg"}
    file_ext = Path(audio_file.filename).suffix.lower()
    if file_ext not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported audio format: {file_ext}. Supported: {', '.join(allowed_extensions)}"
        )
    
    temp_audio_path = get_output_path() / f"temp_conversation_legacy_{uuid.uuid4().hex[:8]}{file_ext}"
    
    try:
        # Save file
        with open(temp_audio_path, "wb") as buffer:
            import shutil
            shutil.copyfileobj(audio_file.file, buffer)
        
        # STT
        transcribed_text = legacy_stt_engine.transcribe_file(str(temp_audio_path), language)
        if transcribed_text is None:
            raise HTTPException(status_code=500, detail="Speech transcription failed")
        
        logger.info(f"Legacy transcription: '{transcribed_text[:100]}...'")
        
        # Simple echo response
        response_text = f"You said: {transcribed_text}"
        
        # Resolve voice path (original logic)
        audio_prompt_path_for_engine = None
        if voice_mode == "predefined":
            if not predefined_voice_id:
                predefined_voice_id = get_default_voice_id()
            voices_dir = get_predefined_voices_path(ensure_absolute=True)
            potential_path = voices_dir / predefined_voice_id
            if potential_path.is_file():
                audio_prompt_path_for_engine = potential_path
        elif voice_mode == "clone" and reference_audio_filename:
            ref_dir = get_reference_audio_path(ensure_absolute=True)
            potential_path = ref_dir / reference_audio_filename
            if potential_path.is_file():
                audio_prompt_path_for_engine = potential_path
        
        # TTS (original engine)
        from config import (
            get_gen_default_temperature, get_gen_default_exaggeration,
            get_gen_default_cfg_weight, get_gen_default_seed, get_gen_default_speed_factor
        )
        
        audio_tensor, sample_rate = engine.synthesize(
            text=response_text,
            audio_prompt_path=str(audio_prompt_path_for_engine) if audio_prompt_path_for_engine else None,
            temperature=get_gen_default_temperature(),
            exaggeration=get_gen_default_exaggeration(),
            cfg_weight=get_gen_default_cfg_weight(),
            seed=get_gen_default_seed(),
        )
        
        if audio_tensor is None or sample_rate is None:
            raise HTTPException(status_code=500, detail="TTS generation failed")
        
        # Process audio
        audio_np = audio_tensor.cpu().numpy().squeeze()
        
        # Encode
        encoded_audio_bytes = utils.encode_audio(
            audio_array=audio_np,
            sample_rate=sample_rate,
            output_format=output_format,
            target_sample_rate=get_audio_sample_rate(),
        )
        
        if encoded_audio_bytes is None:
            raise HTTPException(status_code=500, detail="Audio encoding failed")
        
        # Return response
        media_type = f"audio/{output_format}"
        timestamp_str = time.strftime("%Y%m%d_%H%M%S")
        filename = utils.sanitize_filename(f"conversation_legacy_{timestamp_str}.{output_format}")
        headers = {"Content-Disposition": f'attachment; filename="{filename}"'}
        
        return StreamingResponse(
            io.BytesIO(encoded_audio_bytes),
            media_type=media_type,
            headers=headers
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in legacy conversation: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Legacy conversation error: {str(e)}")
    finally:
        if temp_audio_path.exists():
            temp_audio_path.unlink(missing_ok=True)
        await audio_file.close()