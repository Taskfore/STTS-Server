# File: routers/websocket_stt.py
# Real-time WebSocket STT endpoints inspired by whisper_real_time

import asyncio
import logging
import json
import threading
import queue
from typing import Optional

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Depends
import numpy as np

from stt_engine import STTEngine

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/ws", tags=["WebSocket STT"])




class AudioBuffer:
    """Thread-safe audio buffer for real-time processing."""
    
    def __init__(self, max_duration_seconds: float = 30.0, sample_rate: int = 16000):
        self.max_duration = max_duration_seconds
        self.sample_rate = sample_rate
        self.max_samples = int(max_duration_seconds * sample_rate)
        self.buffer = np.array([], dtype=np.float32)
        self.lock = threading.Lock()
    
    def add_audio(self, audio_data: np.ndarray):
        """Add audio data to the buffer."""
        with self.lock:
            self.buffer = np.concatenate([self.buffer, audio_data])
            # Keep only the last max_duration seconds
            if len(self.buffer) > self.max_samples:
                self.buffer = self.buffer[-self.max_samples:]
    
    def get_audio(self) -> np.ndarray:
        """Get current audio buffer."""
        with self.lock:
            return self.buffer.copy()
    
    def clear(self):
        """Clear the audio buffer."""
        with self.lock:
            self.buffer = np.array([], dtype=np.float32)


class RealtimeSTT:
    """Real-time STT processor using continuous audio buffering."""
    
    def __init__(self, stt_engine: STTEngine, language: Optional[str] = None):
        self.stt_engine = stt_engine
        self.language = language
        self.audio_buffer = AudioBuffer()
        self.processing = False
        self.last_transcription = ""
        self.webm_chunks = []
        self.chunk_count = 0
        self.process_every_n_chunks = 5  # Process every 5 chunks for better performance
        
    async def process_audio_chunk(self, audio_data: bytes) -> Optional[str]:
        """Process incoming audio chunk and return transcription if available."""
        try:
            # Accumulate WebM chunks
            self.webm_chunks.append(audio_data)
            self.chunk_count += 1
            logger.debug(f"Received chunk {self.chunk_count}, size: {len(audio_data)} bytes")
            
            # Only process every N chunks to allow accumulation
            if self.chunk_count % self.process_every_n_chunks != 0:
                return None
            
            # Prevent concurrent processing
            if self.processing:
                return None
                
            self.processing = True
            
            # Combine accumulated chunks
            combined_webm = b''.join(self.webm_chunks)
            logger.info(f"Processing {len(self.webm_chunks)} accumulated chunks, total size: {len(combined_webm)} bytes")
            
            # Convert WebM audio data to numpy array
            audio_np = await self._decode_combined_webm_audio(combined_webm)
            if audio_np is None or len(audio_np) == 0:
                # Try fallback approach with file-based processing
                transcription = await self._process_webm_as_file(combined_webm)
                self.processing = False
                if transcription and transcription != self.last_transcription:
                    self.last_transcription = transcription
                    self.webm_chunks = self.webm_chunks[-2:]
                    return transcription
                return None
            
            # Process in background thread to avoid blocking WebSocket
            loop = asyncio.get_event_loop()
            transcription = await loop.run_in_executor(
                None, 
                self._transcribe_audio, 
                audio_np
            )
            
            self.processing = False
            
            # Only return if transcription changed significantly
            if transcription and transcription != self.last_transcription:
                self.last_transcription = transcription
                # Keep only the last few chunks for context
                self.webm_chunks = self.webm_chunks[-2:]
                return transcription
                
            return None
            
        except Exception as e:
            logger.error(f"Error processing audio chunk: {e}")
            self.processing = False
            return None
    
    async def _decode_combined_webm_audio(self, webm_data: bytes) -> Optional[np.ndarray]:
        """Decode WebM audio data to numpy array."""
        webm_path = None
        pcm_path = None
        
        try:
            import tempfile
            import subprocess
            import os
            
            # Save WebM data to temporary file
            with tempfile.NamedTemporaryFile(suffix=".webm", delete=False) as webm_file:
                webm_file.write(webm_data)
                webm_path = webm_file.name
            
            # Convert to raw PCM using ffmpeg
            with tempfile.NamedTemporaryFile(suffix=".raw", delete=False) as pcm_file:
                pcm_path = pcm_file.name
            
            # Use ffmpeg to decode WebM to 16kHz mono 16-bit PCM
            cmd = [
                'ffmpeg', '-i', webm_path,
                '-ar', '16000',         # Sample rate 16kHz
                '-ac', '1',            # Mono
                '-f', 's16le',         # 16-bit little-endian PCM
                '-loglevel', 'error',  # Suppress ffmpeg output
                '-avoid_negative_ts', 'make_zero',  # Handle timing issues
                '-fflags', '+genpts',  # Generate presentation timestamps
                '-y', pcm_path         # Overwrite output
            ]
            
            # Run ffmpeg with suppressed output
            result = subprocess.run(
                cmd, 
                capture_output=True, 
                text=False,
                timeout=5  # 5 second timeout for real-time processing
            )
            
            if result.returncode != 0:
                stderr_output = result.stderr.decode('utf-8') if result.stderr else "No error output"
                logger.warning(f"ffmpeg failed to decode WebM audio chunk. Error: {stderr_output}")
                logger.warning("Make sure ffmpeg is installed and accessible in PATH")
                return None
            
            # Read the PCM data
            with open(pcm_path, 'rb') as f:
                pcm_data = f.read()
            
            if len(pcm_data) == 0:
                return None
            
            # Convert to numpy array
            audio_np = np.frombuffer(pcm_data, dtype=np.int16).astype(np.float32) / 32768.0
            return audio_np
            
        except subprocess.TimeoutExpired:
            logger.warning("ffmpeg timeout while decoding WebM audio chunk")
            return None
        except Exception as e:
            logger.error(f"Error decoding WebM audio: {e}")
            return None
        finally:
            # Clean up temporary files
            try:
                if webm_path and os.path.exists(webm_path):
                    os.unlink(webm_path)
                if pcm_path and os.path.exists(pcm_path):
                    os.unlink(pcm_path)
            except Exception as cleanup_error:
                logger.warning(f"Error cleaning up temp files: {cleanup_error}")
    
    async def _process_webm_as_file(self, webm_data: bytes) -> Optional[str]:
        """Fallback: Process WebM data directly as a complete file."""
        webm_path = None
        try:
            import tempfile
            import os
            
            # Save combined WebM data to a temporary file
            with tempfile.NamedTemporaryFile(suffix=".webm", delete=False) as webm_file:
                webm_file.write(webm_data)
                webm_path = webm_file.name
            
            # Process in background thread
            loop = asyncio.get_event_loop()
            transcription = await loop.run_in_executor(
                None,
                self.stt_engine.transcribe_file,
                webm_path,
                self.language
            )
            
            return transcription
            
        except Exception as e:
            logger.error(f"Error processing WebM file directly: {e}")
            return None
        finally:
            # Clean up temporary file
            try:
                if webm_path and os.path.exists(webm_path):
                    os.unlink(webm_path)
            except Exception as cleanup_error:
                logger.warning(f"Error cleaning up WebM temp file: {cleanup_error}")
    
    def _transcribe_audio(self, audio_data: np.ndarray) -> Optional[str]:
        """Transcribe audio data using STT engine."""
        try:
            import tempfile
            import soundfile as sf
            
            # Save audio to temporary file for Whisper
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                sf.write(temp_file.name, audio_data, self.audio_buffer.sample_rate)
                
                # Transcribe using STT engine
                result = self.stt_engine.transcribe_file(temp_file.name, self.language)
                
                # Clean up temp file
                import os
                os.unlink(temp_file.name)
                
                return result
                
        except Exception as e:
            logger.error(f"Error in transcription: {e}")
            return None


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
    
    # Get STT engine from app state
    if not hasattr(websocket.app.state, 'stt_engine'):
        await websocket.send_json({
            "type": "error",
            "message": "STT engine not initialized"
        })
        await websocket.close()
        return
    
    stt_engine = websocket.app.state.stt_engine
    
    if not stt_engine.model_loaded:
        await websocket.send_json({
            "type": "error",
            "message": "STT engine not available"
        })
        await websocket.close()
        return
    
    logger.info(f"WebSocket STT connection established, language: {language}")
    
    # Initialize real-time STT processor
    realtime_stt = RealtimeSTT(stt_engine, language)
    
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
                            "partial": False  # Could implement partial results
                        })
                        
                elif "text" in data:
                    # Handle text commands
                    try:
                        command = json.loads(data["text"])
                        
                        if command.get("action") == "clear":
                            realtime_stt.audio_buffer.clear()
                            realtime_stt.webm_chunks.clear()
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