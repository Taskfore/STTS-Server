# File: routers/websocket_conversation.py
# Real-time WebSocket audio conversation endpoints

import asyncio
import base64
import io
import logging
import time
from typing import Optional, Dict, Any
from concurrent.futures import ThreadPoolExecutor

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Query, Request
import numpy as np

# Import existing components
from stt_engine import STTEngine
from models import TranscriptionResult
import engine
import utils
from routers.websocket_stt import PCMAudioDecoder, OptimizedAudioBuffer

# Import new conversation components
from pause_detection import PauseDetector, EnergyFallbackDetector, WEBRTC_AVAILABLE
from conversation_engine import ConversationResponseGenerator

# Import config functions
from config import (
    get_default_voice_id,
    get_predefined_voices_path,
    get_reference_audio_path,
    get_gen_default_temperature,
    get_gen_default_exaggeration,
    get_gen_default_cfg_weight,
    get_gen_default_seed,
    get_gen_default_speed_factor,
    get_audio_sample_rate,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/ws", tags=["WebSocket Conversation"])

# Dedicated thread pool for conversation processing
CONVERSATION_THREAD_POOL = ThreadPoolExecutor(max_workers=3, thread_name_prefix="Conversation-Worker")


class ConversationProcessor:
    """
    Handles real-time audio conversation processing with pause detection and response generation.
    """
    
    def __init__(self, 
                 stt_engine: STTEngine,
                 websocket: WebSocket,
                 language: Optional[str] = None,
                 voice_mode: str = "predefined",
                 predefined_voice_id: Optional[str] = None,
                 reference_audio_filename: Optional[str] = None,
                 response_mode: str = "echo",
                 pause_aggressiveness: int = 2):
        """
        Initialize conversation processor.
        
        Args:
            stt_engine: Speech-to-text engine
            websocket: WebSocket connection for sending responses
            language: STT language (None for auto-detect)
            voice_mode: TTS voice mode ("predefined" or "clone")
            predefined_voice_id: Predefined voice ID for TTS
            reference_audio_filename: Reference audio for voice cloning
            response_mode: Response generation mode ("echo" or "template")
            pause_aggressiveness: WebRTC VAD aggressiveness (0-3)
        """
        self.stt_engine = stt_engine
        self.websocket = websocket
        self.language = language
        self.voice_mode = voice_mode
        self.predefined_voice_id = predefined_voice_id or get_default_voice_id()
        self.reference_audio_filename = reference_audio_filename
        
        # Initialize audio processing
        self.audio_buffer = OptimizedAudioBuffer(max_duration_seconds=10.0)
        self.decoder = PCMAudioDecoder()
        
        # Initialize pause detection
        try:
            if WEBRTC_AVAILABLE:
                self.pause_detector = PauseDetector(
                    aggressiveness=pause_aggressiveness,
                    min_speech_frames=8,   # ~240ms
                    min_pause_frames=20,   # ~600ms
                )
                logger.info("Using WebRTC VAD for pause detection")
            else:
                self.pause_detector = EnergyFallbackDetector()
                logger.info("Using energy-based fallback for pause detection")
        except Exception as e:
            logger.warning(f"Failed to initialize WebRTC VAD: {e}, using fallback")
            self.pause_detector = EnergyFallbackDetector()
        
        # Initialize response generation
        self.response_generator = ConversationResponseGenerator(response_mode)
        
        # State tracking
        self.processing_transcription = False
        self.last_transcription_time = 0
        self.conversation_state = "listening"  # "listening", "processing", "speaking"
        self.pending_audio_for_transcription = None
        
        # Resolve voice path for TTS
        self.voice_path = self._resolve_voice_path()
        
        logger.info(f"ConversationProcessor initialized: voice_mode={voice_mode}, "
                   f"response_mode={response_mode}, voice_path={self.voice_path}")
    
    def _resolve_voice_path(self) -> Optional[str]:
        """Resolve the voice path for TTS synthesis."""
        try:
            if self.voice_mode == "predefined":
                voices_dir = get_predefined_voices_path(ensure_absolute=True)
                voice_path = voices_dir / self.predefined_voice_id
                if voice_path.is_file():
                    return str(voice_path)
                else:
                    logger.warning(f"Predefined voice not found: {voice_path}")
                    return None
            
            elif self.voice_mode == "clone":
                if not self.reference_audio_filename:
                    logger.warning("No reference audio filename provided for voice cloning")
                    return None
                
                ref_dir = get_reference_audio_path(ensure_absolute=True)
                voice_path = ref_dir / self.reference_audio_filename
                if voice_path.is_file():
                    return str(voice_path)
                else:
                    logger.warning(f"Reference audio not found: {voice_path}")
                    return None
            
            return None
            
        except Exception as e:
            logger.error(f"Error resolving voice path: {e}")
            return None
    
    async def process_audio_chunk(self, audio_data: bytes) -> Optional[Dict[str, Any]]:
        """
        Process incoming PCM audio chunk with pause detection and conversation flow.
        
        Args:
            audio_data: Raw PCM audio bytes
            
        Returns:
            Dict with conversation events and state, or None if no events
        """
        if len(audio_data) == 0:
            return None
        
        # Convert PCM to numpy and add to buffer
        audio_np = await self.decoder.decode_pcm_to_numpy(audio_data)
        if audio_np is None:
            return None
        
        self.audio_buffer.add_audio(audio_np)
        
        # Process pause detection
        pause_result = self.pause_detector.process_pcm_chunk(audio_data)
        events = pause_result.get('events', [])
        
        conversation_events = []
        
        # Handle speech start
        if 'speech_start' in events:
            if self.conversation_state != "listening":
                logger.debug("Speech started while not listening - resetting state")
            self.conversation_state = "listening"
            conversation_events.append('speech_start')
        
        # Handle speech end - trigger transcription and response
        if 'speech_end' in events and not self.processing_transcription:
            self.conversation_state = "processing"
            conversation_events.append('speech_end')
            
            # Get recent audio for transcription
            recent_audio = self.audio_buffer.get_recent_audio(duration_seconds=5.0)
            if len(recent_audio) > 0:
                # Process transcription and response in background
                asyncio.create_task(self._process_speech_to_response(recent_audio))
        
        # Return current state
        response = {
            'type': 'conversation_state',
            'state': self.conversation_state,
            'is_speaking': pause_result.get('is_speaking', False),
            'silence_duration_ms': pause_result.get('silence_duration_ms', 0),
            'events': conversation_events,
            'pause_detection': pause_result
        }
        
        return response if conversation_events else None
    
    async def _process_speech_to_response(self, audio_data: np.ndarray):
        """
        Process speech to response in the background.
        
        Args:
            audio_data: Audio data to transcribe and respond to
        """
        if self.processing_transcription:
            logger.debug("Already processing transcription, skipping")
            return
        
        self.processing_transcription = True
        
        try:
            # Step 1: Transcribe speech
            logger.debug("Starting transcription...")
            transcription_result = await self._transcribe_audio(audio_data)
            
            if not transcription_result or not transcription_result.text.strip():
                logger.debug("No transcription result or empty text")
                return None
            
            transcribed_text = transcription_result.text.strip()
            logger.info(f"Transcribed: '{transcribed_text}'")
            
            # Step 2: Generate text response
            logger.debug("Generating text response...")
            response_text = self.response_generator.generate_response(transcribed_text)
            logger.info(f"Response: '{response_text}'")
            
            # Extract timing from segments
            start_time = transcription_result.segments[0].start if transcription_result.segments else 0.0
            end_time = transcription_result.segments[-1].end if transcription_result.segments else 0.0
            duration = end_time - start_time
            
            # Send transcription to client
            await self.websocket.send_json({
                "type": "transcription",
                "text": transcribed_text,
                "language": transcription_result.language if transcription_result.language else "unknown",
                "partial": False,
                "timing": {
                    "start": start_time,
                    "end": end_time,
                    "duration": duration
                },
                "segments": [
                    {
                        "text": seg.text,
                        "start": seg.start,
                        "end": seg.end
                    } for seg in transcription_result.segments
                ]
            })
            
            # Step 3: Convert response to speech
            logger.debug("Converting response to speech...")
            audio_response = await self._text_to_speech(response_text)
            
            if audio_response:
                # Send audio response to client
                await self.websocket.send_json({
                    "type": "response_audio",
                    "audio_data": audio_response['audio_data'],
                    "format": audio_response['format'],
                    "sample_rate": audio_response['sample_rate'],
                    "duration_ms": audio_response['duration_ms'],
                    "response_text": response_text
                })
                logger.info("Audio response sent to client")
            else:
                logger.warning("Failed to generate audio response")
                
        except Exception as e:
            logger.error(f"Error in speech-to-response processing: {e}", exc_info=True)
            return None
        
        finally:
            self.processing_transcription = False
            self.conversation_state = "listening"
    
    async def _transcribe_audio(self, audio_data: np.ndarray) -> Optional[TranscriptionResult]:
        """Transcribe audio data using STT engine."""
        try:
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                CONVERSATION_THREAD_POOL,
                self.stt_engine.transcribe_numpy_with_timing,
                audio_data,
                self.language
            )
            return result
        except Exception as e:
            logger.error(f"Transcription error: {e}")
            return None
    
    async def _text_to_speech(self, text: str) -> Optional[Dict[str, Any]]:
        """Convert text to speech using TTS engine."""
        try:
            # Check if TTS engine is loaded
            if not engine.MODEL_LOADED:
                logger.error("TTS engine not loaded")
                return None
            
            # Synthesize speech
            loop = asyncio.get_event_loop()
            audio_tensor, sample_rate = await loop.run_in_executor(
                CONVERSATION_THREAD_POOL,
                engine.synthesize,
                text,
                self.voice_path,
                get_gen_default_temperature(),
                get_gen_default_exaggeration(),
                get_gen_default_cfg_weight(),
                get_gen_default_seed()
            )
            
            if audio_tensor is None or sample_rate is None:
                logger.error("TTS synthesis failed")
                return None
            
            # Convert to numpy and apply speed factor
            audio_np = audio_tensor.cpu().numpy().squeeze()
            speed_factor = get_gen_default_speed_factor()
            if speed_factor != 1.0:
                audio_np, _ = utils.apply_speed_factor(audio_np, sample_rate, speed_factor)
            
            # Encode to WAV format for WebSocket transmission
            encoded_audio = utils.encode_audio(
                audio_array=audio_np,
                sample_rate=sample_rate,
                output_format="wav",
                target_sample_rate=get_audio_sample_rate()
            )
            
            if encoded_audio is None:
                logger.error("Audio encoding failed")
                return None
            
            # Encode to base64 for JSON transmission
            audio_base64 = base64.b64encode(encoded_audio).decode('utf-8')
            
            return {
                'audio_data': audio_base64,
                'format': 'wav',
                'sample_rate': get_audio_sample_rate(),
                'duration_ms': int(len(audio_np) / sample_rate * 1000)
            }
            
        except Exception as e:
            logger.error(f"TTS error: {e}", exc_info=True)
            return None
    
    def reset_conversation(self):
        """Reset conversation state."""
        self.audio_buffer.clear()
        self.pause_detector.reset()
        self.response_generator.clear_history()
        self.conversation_state = "listening"
        self.processing_transcription = False
        logger.info("Conversation state reset")


@router.websocket("/conversation")
async def websocket_conversation(
    websocket: WebSocket,
    language: Optional[str] = Query(None, description="STT language (auto-detect if None)"),
    voice_mode: str = Query("predefined", description="TTS voice mode"),
    predefined_voice_id: Optional[str] = Query(None, description="Predefined voice ID"),
    reference_audio_filename: Optional[str] = Query(None, description="Reference audio filename"),
    response_mode: str = Query("echo", description="Response generation mode"),
    pause_aggressiveness: int = Query(2, ge=0, le=3, description="Pause detection aggressiveness")
):
    """
    Real-time audio conversation via WebSocket.
    
    Flow:
    1. Receives PCM audio streams
    2. Detects speech pauses using WebRTC VAD
    3. Transcribes speech when pause detected
    4. Generates text response (echo/template modes)
    5. Converts response to speech and streams back
    
    Expected audio format: 16-bit PCM, 16kHz, mono
    """
    await websocket.accept()
    
    # Get engines from app state
    request = websocket.scope.get("app")
    if not hasattr(request.state, 'stt_engine'):
        await websocket.send_json({
            "type": "error",
            "message": "STT engine not initialized"
        })
        await websocket.close()
        return
    
    stt_engine = request.state.stt_engine
    
    # Validate engines
    if not stt_engine.model_loaded:
        await websocket.send_json({
            "type": "error",
            "message": "STT engine not available"
        })
        await websocket.close()
        return
    
    if not engine.MODEL_LOADED:
        await websocket.send_json({
            "type": "error",
            "message": "TTS engine not available"
        })
        await websocket.close()
        return
    
    logger.info(f"WebSocket conversation connection established: "
               f"voice_mode={voice_mode}, response_mode={response_mode}, "
               f"language={language}, pause_aggressiveness={pause_aggressiveness}")
    
    # Initialize conversation processor
    try:
        conversation = ConversationProcessor(
            stt_engine=stt_engine,
            websocket=websocket,
            language=language,
            voice_mode=voice_mode,
            predefined_voice_id=predefined_voice_id,
            reference_audio_filename=reference_audio_filename,
            response_mode=response_mode,
            pause_aggressiveness=pause_aggressiveness
        )
    except Exception as e:
        await websocket.send_json({
            "type": "error",
            "message": f"Failed to initialize conversation processor: {str(e)}"
        })
        await websocket.close()
        return
    
    
    try:
        await websocket.send_json({
            "type": "ready",
            "message": "Real-time conversation ready",
            "config": {
                "voice_mode": voice_mode,
                "response_mode": response_mode,
                "language": language,
                "pause_aggressiveness": pause_aggressiveness
            }
        })
        
        while True:
            # Receive data from client
            data = await websocket.receive()
            
            if data["type"] == "websocket.disconnect":
                break
            
            if data["type"] == "websocket.receive":
                if "bytes" in data:
                    # Process audio chunk
                    result = await conversation.process_audio_chunk(data["bytes"])
                    
                    if result:
                        await websocket.send_json(result)
                
                elif "text" in data:
                    # Handle text commands
                    import json
                    try:
                        command = json.loads(data["text"])
                        
                        if command.get("action") == "reset":
                            conversation.reset_conversation()
                            await websocket.send_json({
                                "type": "reset_complete",
                                "message": "Conversation state reset"
                            })
                        
                        elif command.get("action") == "change_response_mode":
                            new_mode = command.get("mode", "echo")
                            conversation.response_generator.set_response_mode(new_mode)
                            await websocket.send_json({
                                "type": "mode_changed",
                                "message": f"Response mode changed to {new_mode}"
                            })
                        
                        elif command.get("action") == "ping":
                            await websocket.send_json({
                                "type": "pong",
                                "message": "WebSocket conversation active"
                            })
                        
                        elif command.get("action") == "stats":
                            stats = conversation.response_generator.get_statistics()
                            await websocket.send_json({
                                "type": "statistics",
                                "data": stats
                            })
                    
                    except json.JSONDecodeError:
                        await websocket.send_json({
                            "type": "error",
                            "message": "Invalid command format"
                        })
    
    except WebSocketDisconnect:
        logger.info("WebSocket conversation connection closed")
    except Exception as e:
        logger.error(f"WebSocket conversation error: {e}", exc_info=True)
        try:
            await websocket.send_json({
                "type": "error",
                "message": f"Server error: {str(e)}"
            })
        except:
            pass
    finally:
        logger.info("WebSocket conversation session ended")


# Additional endpoint for conversation configuration
@router.websocket("/conversation/test")
async def websocket_conversation_test(websocket: WebSocket):
    """
    Simple test endpoint for conversation functionality.
    Returns echo responses to test the basic flow.
    """
    await websocket.accept()
    
    try:
        await websocket.send_json({
            "type": "ready",
            "message": "Conversation test endpoint ready"
        })
        
        while True:
            data = await websocket.receive()
            
            if data["type"] == "websocket.disconnect":
                break
            
            if data["type"] == "websocket.receive":
                if "text" in data:
                    # Simple text echo for testing
                    await websocket.send_json({
                        "type": "echo",
                        "message": f"Echo: {data['text']}"
                    })
                elif "bytes" in data:
                    # Report audio data received
                    await websocket.send_json({
                        "type": "audio_received",
                        "bytes_count": len(data["bytes"]),
                        "message": "Audio data received"
                    })
    
    except WebSocketDisconnect:
        logger.info("WebSocket conversation test connection closed")
    except Exception as e:
        logger.error(f"WebSocket conversation test error: {e}")
    finally:
        logger.info("WebSocket conversation test session ended")
