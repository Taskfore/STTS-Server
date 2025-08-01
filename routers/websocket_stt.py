# File: routers/websocket_stt.py
# Real-time WebSocket STT endpoints inspired by whisper_real_time

import asyncio
import logging
import json
import threading
import queue
from typing import Optional
from concurrent.futures import ThreadPoolExecutor
from collections import deque

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Depends
import numpy as np

from stt_engine import STTEngine

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/ws", tags=["WebSocket STT"])

# Dedicated thread pool for CPU-intensive transcription tasks
TRANSCRIPTION_THREAD_POOL = ThreadPoolExecutor(max_workers=2, thread_name_prefix="STT-Worker")


class AsyncAudioDecoder:
    """Async WebM to numpy array decoder using ffmpeg."""
    
    @staticmethod
    async def decode_webm_to_numpy(webm_data: bytes) -> Optional[np.ndarray]:
        """
        Decode WebM audio data to numpy array using async subprocess.
        
        Args:
            webm_data: Raw WebM audio bytes
            
        Returns:
            Float32 numpy array at 16kHz mono, or None if decoding fails
        """
        try:
            # Use ffmpeg with pipes to avoid file I/O
            cmd = [
                'ffmpeg', '-f', 'webm', '-i', 'pipe:0',  # Read from stdin
                '-ar', '16000',         # Sample rate 16kHz
                '-ac', '1',            # Mono
                '-f', 's16le',         # 16-bit little-endian PCM
                '-loglevel', 'error',  # Suppress ffmpeg output
                '-avoid_negative_ts', 'make_zero',
                '-fflags', '+genpts',
                'pipe:1'               # Write to stdout
            ]
            
            # Create async subprocess with pipes
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            # Send WebM data and get PCM output
            stdout, stderr = await asyncio.wait_for(
                process.communicate(input=webm_data),
                timeout=5.0  # 5 second timeout
            )
            
            if process.returncode != 0:
                if stderr:
                    logger.warning(f"ffmpeg decode error: {stderr.decode('utf-8', errors='ignore')}")
                return None
            
            if len(stdout) == 0:
                return None
            
            # Convert PCM bytes to numpy array
            audio_np = np.frombuffer(stdout, dtype=np.int16).astype(np.float32) / 32768.0
            return audio_np
            
        except asyncio.TimeoutError:
            logger.warning("ffmpeg timeout during WebM decoding")
            if 'process' in locals():
                process.kill()
            return None
        except Exception as e:
            logger.error(f"Error in async WebM decoding: {e}")
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
    
    def clear(self):
        """Clear the audio buffer."""
        with self.lock:
            self.chunks.clear()
            self.total_samples = 0


class OptimizedRealtimeSTT:
    """Optimized real-time STT processor with async audio processing."""
    
    def __init__(self, stt_engine: STTEngine, language: Optional[str] = None):
        self.stt_engine = stt_engine
        self.language = language
        self.audio_buffer = OptimizedAudioBuffer()
        self.processing = False
        self.last_transcription = ""
        self.webm_accumulator = bytearray()  # More efficient than list of chunks
        self.chunk_count = 0
        self.process_every_n_chunks = 3  # Reduced for better responsiveness
        self.decoder = AsyncAudioDecoder()
        
    async def process_audio_chunk(self, audio_data: bytes) -> Optional[str]:
        """Process incoming audio chunk with optimized pipeline."""
        try:
            # Accumulate WebM data efficiently
            self.webm_accumulator.extend(audio_data)
            self.chunk_count += 1
            logger.debug(f"Received chunk {self.chunk_count}, total size: {len(self.webm_accumulator)} bytes")
            
            # Only process every N chunks to allow accumulation
            if self.chunk_count % self.process_every_n_chunks != 0:
                return None
            
            # Prevent concurrent processing
            if self.processing:
                return None
                
            self.processing = True
            
            try:
                # Convert WebM to numpy array using async decoder
                webm_bytes = bytes(self.webm_accumulator)
                audio_np = await self.decoder.decode_webm_to_numpy(webm_bytes)
                
                if audio_np is None or len(audio_np) == 0:
                    logger.debug("No audio data decoded from WebM chunk")
                    return None
                
                # Add to optimized buffer for potential future use
                self.audio_buffer.add_audio(audio_np)
                
                # Transcribe using dedicated thread pool
                loop = asyncio.get_event_loop()
                transcription = await loop.run_in_executor(
                    TRANSCRIPTION_THREAD_POOL,
                    self._transcribe_numpy_audio,
                    audio_np
                )
                
                # Only return if transcription changed significantly
                if transcription and transcription != self.last_transcription:
                    self.last_transcription = transcription
                    # Keep recent WebM data for context (last 2 processing cycles)
                    keep_size = len(self.webm_accumulator) // 2
                    self.webm_accumulator = self.webm_accumulator[-keep_size:]
                    return transcription
                    
                return None
                
            finally:
                self.processing = False
            
        except Exception as e:
            logger.error(f"Error processing audio chunk: {e}")
            self.processing = False
            return None
    
    def _transcribe_numpy_audio(self, audio_data: np.ndarray) -> Optional[str]:
        """Transcribe numpy audio directly using STT engine (no temp files)."""
        try:
            # Use the new direct numpy transcription method
            result = self.stt_engine.transcribe_numpy(audio_data, self.language)
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
    
    Expects audio data as binary frames (16-bit PCM, 16kHz recommended).
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
                    transcription = await realtime_stt.process_audio_chunk(data["bytes"])

                    if transcription:
                        await websocket.send_json({
                            "type": "transcription",
                            "text": transcription,
                            "language": language,
                            "partial": True  # Could implement partial results
                        })

                elif "text" in data:
                    # Handle text commands
                    try:
                        command = json.loads(data["text"])

                        if command.get("action") == "clear":
                            realtime_stt.audio_buffer.clear()
                            realtime_stt.webm_accumulator.clear()
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

