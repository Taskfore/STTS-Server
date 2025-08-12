# File: routers/websocket_stt.py
# Real-time WebSocket STT endpoints inspired by whisper_real_time

import asyncio
import logging
import json
import threading
from typing import Optional
from concurrent.futures import ThreadPoolExecutor
from collections import deque

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Depends
import numpy as np

from stt_engine import STTEngine
from models import TranscriptionResult

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/ws", tags=["WebSocket STT"])

# Dedicated thread pool for CPU-intensive transcription tasks
TRANSCRIPTION_THREAD_POOL = ThreadPoolExecutor(max_workers=2, thread_name_prefix="STT-Worker")


class PCMAudioDecoder:
    """Simple PCM to numpy array decoder."""
    
    @staticmethod
    async def decode_pcm_to_numpy(pcm_data: bytes) -> Optional[np.ndarray]:
        """
        Convert raw PCM data to numpy array.
        
        Args:
            pcm_data: Raw 16-bit little-endian PCM audio bytes
            
        Returns:
            Float32 numpy array, or None if conversion fails
        """
        try:
            if len(pcm_data) == 0:
                return None
            
            # Convert PCM bytes directly to numpy array
            # Frontend sends 16-bit signed integers in little-endian format
            audio_np = np.frombuffer(pcm_data, dtype=np.int16).astype(np.float32) / 32768.0
            return audio_np
            
        except Exception as e:
            logger.error(f"Error converting PCM to numpy: {e}")
            return None




class OptimizedAudioBuffer:
    """Optimized thread-safe audio buffer using deque for better performance."""
    
    def __init__(self, max_duration_seconds: float = 30.0, sample_rate: int = 16000):
        self.max_duration = max_duration_seconds
        self.sample_rate = sample_rate
        self.max_samples = int(max_duration_seconds * sample_rate)
        self.chunks = deque()  # Use deque for efficient append/pop operations
        self.total_samples = 0
        self.lock = threading.Lock()
    
    def add_audio(self, audio_data: np.ndarray):
        """Add audio data to the buffer efficiently."""
        with self.lock:
            self.chunks.append(audio_data)
            self.total_samples += len(audio_data)
            
            # Remove old chunks if we exceed max duration
            while self.total_samples > self.max_samples and self.chunks:
                removed_chunk = self.chunks.popleft()
                self.total_samples -= len(removed_chunk)
    
    def get_recent_audio(self, duration_seconds: float = 5.0) -> np.ndarray:
        """Get recent audio for transcription (more efficient than full buffer)."""
        target_samples = int(duration_seconds * self.sample_rate)
        
        with self.lock:
            if not self.chunks:
                return np.array([], dtype=np.float32)
            
            # Collect recent chunks up to target duration
            collected_chunks = []
            collected_samples = 0
            
            # Work backwards from most recent chunks
            for chunk in reversed(self.chunks):
                collected_chunks.insert(0, chunk)
                collected_samples += len(chunk)
                if collected_samples >= target_samples:
                    break
            
            if collected_chunks:
                return np.concatenate(collected_chunks)
            return np.array([], dtype=np.float32)
    
    def get_audio_since_sample(self, start_sample_count: int) -> np.ndarray:
        """Get all audio data since a specific sample count position."""
        with self.lock:
            if not self.chunks:
                return np.array([], dtype=np.float32)
            
            # Calculate how many samples to skip from the beginning
            samples_to_get = self.total_samples - start_sample_count
            
            if samples_to_get <= 0:
                return np.array([], dtype=np.float32)
            
            # Collect chunks from the end working backwards
            collected_chunks = []
            collected_samples = 0
            
            # Work backwards from most recent chunks
            for chunk in reversed(self.chunks):
                collected_chunks.insert(0, chunk)
                collected_samples += len(chunk)
                
                # Stop when we have enough samples
                if collected_samples >= samples_to_get:
                    break
            
            if not collected_chunks:
                return np.array([], dtype=np.float32)
            
            # Concatenate and trim to exact sample count if needed
            full_audio = np.concatenate(collected_chunks)
            
            # Trim from the beginning if we collected too much
            if len(full_audio) > samples_to_get:
                full_audio = full_audio[-samples_to_get:]
            
            return full_audio
    
    def clear(self):
        """Clear the audio buffer."""
        with self.lock:
            self.chunks.clear()
            self.total_samples = 0


class OptimizedRealtimeSTT:
    """Optimized real-time STT processor with PCM audio processing."""
    
    def __init__(self, stt_engine: STTEngine, language: Optional[str] = None):
        self.stt_engine = stt_engine
        self.language = language
        self.audio_buffer = OptimizedAudioBuffer()
        self.processing = False
        self.last_transcription = ""
        self.chunk_count = 0
        self.process_every_n_chunks = 4  # Process every 4 chunks (~256ms of audio)
        self.decoder = PCMAudioDecoder()
        
    async def process_audio_chunk(self, audio_data: bytes) -> Optional[str]:
        """Process incoming PCM audio chunk."""
        try:
            logger.debug(f"Received PCM chunk {self.chunk_count}, size: {len(audio_data)} bytes")
            
            # Convert PCM to numpy array directly
            audio_np = await self.decoder.decode_pcm_to_numpy(audio_data)
            
            if audio_np is None or len(audio_np) == 0:
                logger.debug("No audio data decoded from PCM chunk")
                return None
            
            # Add to optimized buffer
            self.audio_buffer.add_audio(audio_np)
            self.chunk_count += 1

            # Process every N chunks to accumulate enough audio
            if self.chunk_count % self.process_every_n_chunks != 0:
                return None
            
            # Prevent concurrent processing
            if self.processing:
                return None
                
            self.processing = True
            
            try:
                # Get recent audio for transcription (2 seconds worth)
                recent_audio = self.audio_buffer.get_recent_audio(duration_seconds=2.0)
                
                if len(recent_audio) == 0:
                    return None
                
                # Transcribe using dedicated thread pool
                loop = asyncio.get_event_loop()
                transcription_result = await loop.run_in_executor(
                    TRANSCRIPTION_THREAD_POOL,
                    self._transcribe_numpy_audio,
                    recent_audio
                )
                
                # Extract text and timing information
                if transcription_result and isinstance(transcription_result, TranscriptionResult):
                    transcription_text = transcription_result.text.strip()
                    
                    # Only return if transcription changed significantly
                    if transcription_text and transcription_text != self.last_transcription:
                        self.last_transcription = transcription_text
                        return transcription_result
                    
                return None
                
            finally:
                self.processing = False
                
        except Exception as e:
            logger.error(f"Error processing PCM chunk: {e}")
            self.processing = False
            return None
    
    def _transcribe_numpy_audio(self, audio_data: np.ndarray) -> Optional[TranscriptionResult]:
        """Transcribe numpy audio directly using STT engine with timing information."""
        try:
            # Use the new direct numpy transcription method with timing
            result = self.stt_engine.transcribe_numpy_with_timing(audio_data, self.language)
            return result
        except Exception as e:
            logger.error(f"Error in numpy transcription: {e}")
            return None


# Backward compatibility alias
RealtimeSTT = OptimizedRealtimeSTT


@router.websocket("/transcribe")
async def websocket_transcribe(
    websocket: WebSocket,
    language: Optional[str] = None
):
    """
    Real-time speech transcription via WebSocket.
    
    Expects raw PCM audio data as binary frames (16-bit little-endian, 16kHz mono).
    Returns JSON messages with transcription results.
    """
    await websocket.accept()

    stt_engine = websocket.app.state.stt_engine

    if not await check_all_good(websocket, stt_engine):
        return
    logger.info(f"WebSocket STT connection established, language: {language}")

    # Initialize optimized real-time STT processor
    realtime_stt = OptimizedRealtimeSTT(stt_engine, language)

    try:
        await websocket.send_json({
            "type": "ready",
            "message": "Real-time transcription ready"
        })

        while True:
            # Receive audio data
            data = await websocket.receive()

            if data["type"] == "websocket.disconnect":
                break

            if data["type"] == "websocket.receive":
                if "bytes" in data:
                    # Process audio chunk
                    transcription_result = await realtime_stt.process_audio_chunk(data["bytes"])

                    if transcription_result:
                        # Extract text and timing information from typed result
                        text = transcription_result.text.strip()
                        segments = transcription_result.segments
                        
                        # Calculate overall timing if segments are available
                        timing_info = None
                        if segments:
                            start_time = min(seg.start for seg in segments)
                            end_time = max(seg.end for seg in segments)
                            timing_info = {
                                "start": start_time,
                                "end": end_time,
                                "duration": end_time - start_time
                            }
                        
                        # Prepare enhanced response
                        response = {
                            "type": "transcription",
                            "text": text,
                            "language": language,
                            "partial": True
                        }
                        
                        # Add timing information if available
                        if timing_info:
                            response["timing"] = timing_info
                        
                        # Add segment details for more granular timing
                        if segments:
                            response["segments"] = [
                                {
                                    "text": seg.text.strip(),
                                    "start": seg.start,
                                    "end": seg.end
                                }
                                for seg in segments
                                if seg.text.strip()
                            ]
                        
                        await websocket.send_json(response)

                elif "text" in data:
                    # Handle text commands
                    try:
                        command = json.loads(data["text"])

                        if command.get("action") == "clear":
                            realtime_stt.audio_buffer.clear()
                            realtime_stt.chunk_count = 0
                            realtime_stt.last_transcription = ""
                            await websocket.send_json({
                                "type": "cleared",
                                "message": "Audio buffer cleared"
                            })

                        elif command.get("action") == "ping":
                            await websocket.send_json({
                                "type": "pong",
                                "message": "WebSocket connection active"
                            })

                    except json.JSONDecodeError:
                        await websocket.send_json({
                            "type": "error",
                            "message": "Invalid command format"
                        })

    except WebSocketDisconnect:
        logger.info("WebSocket STT connection closed")
    except Exception as e:
        logger.error(f"WebSocket STT error: {e}", exc_info=True)
        try:
            await websocket.send_json({
                "type": "error",
                "message": f"Server error: {str(e)}"
            })
        except:
            pass
    finally:
        logger.info("WebSocket STT session ended")


async def check_all_good(websocket, stt_engine) -> bool:
    # Get STT engine from app state
    if not hasattr(websocket.app.state, 'stt_engine'):
        await websocket.send_json({
            "type": "error",
            "message": "STT engine not initialized"
        })
        await websocket.close()
        return False

    if not stt_engine.model_loaded:
        await websocket.send_json({
            "type": "error",
            "message": "STT engine not available"
        })
        await websocket.close()
        return False
    return True

