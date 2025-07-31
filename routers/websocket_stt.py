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
        
    async def process_audio_chunk(self, audio_data: bytes) -> Optional[str]:
        """Process incoming audio chunk and return transcription if available."""
        try:
            # Convert bytes to numpy array (assuming 16-bit PCM, 16kHz)
            audio_np = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
            
            # Add to buffer
            self.audio_buffer.add_audio(audio_np)
            
            # Get current buffer
            current_audio = self.audio_buffer.get_audio()
            
            # Only process if we have enough audio (minimum 1 second)
            if len(current_audio) < self.audio_buffer.sample_rate:
                return None
            
            # Prevent concurrent processing
            if self.processing:
                return None
                
            self.processing = True
            
            # Process in background thread to avoid blocking WebSocket
            loop = asyncio.get_event_loop()
            transcription = await loop.run_in_executor(
                None, 
                self._transcribe_audio, 
                current_audio
            )
            
            self.processing = False
            
            # Only return if transcription changed significantly
            if transcription and transcription != self.last_transcription:
                self.last_transcription = transcription
                return transcription
                
            return None
            
        except Exception as e:
            logger.error(f"Error processing audio chunk: {e}")
            self.processing = False
            return None
    
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