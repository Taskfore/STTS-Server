"""
Base WebSocket adapter implementation.

Provides common functionality for WebSocket adapters across different frameworks.
"""

import asyncio
import json
import logging
from typing import Optional, Dict, Any
from ...core.interfaces import WebSocketAdapter, AudioData
from ...audio.codecs import decode_pcm

logger = logging.getLogger(__name__)


class BaseWebSocketAdapter(WebSocketAdapter):
    """
    Base implementation of WebSocket adapter with common functionality.
    
    Subclasses should implement the framework-specific methods.
    """
    
    def __init__(self):
        self._is_connected = False
        self._connection = None
        self._command_queue = asyncio.Queue()
        
    async def accept_connection(self) -> None:
        """Accept the WebSocket connection."""
        await self._accept_connection_impl()
        self._is_connected = True
        logger.debug("WebSocket connection accepted")
    
    async def receive_audio(self) -> Optional[AudioData]:
        """Receive audio data from the client."""
        try:
            data = await self._receive_data()
            
            if data is None:
                return None
            
            # Handle binary audio data
            if isinstance(data, bytes):
                return AudioData(
                    data=data,
                    sample_rate=16000,  # Default, should be configurable
                    channels=1,
                    format="pcm"
                )
            
            # Handle structured data with audio
            if isinstance(data, dict) and "audio_data" in data:
                audio_bytes = data["audio_data"]
                if isinstance(audio_bytes, str):
                    # Base64 encoded
                    import base64
                    audio_bytes = base64.b64decode(audio_bytes)
                
                return AudioData(
                    data=audio_bytes,
                    sample_rate=data.get("sample_rate", 16000),
                    channels=data.get("channels", 1),
                    format=data.get("format", "pcm")
                )
            
            return None
            
        except Exception as e:
            logger.error(f"Error receiving audio data: {e}")
            return None
    
    async def receive_command(self) -> Optional[Dict[str, Any]]:
        """Receive command data from the client."""
        try:
            # Try to get from command queue first (non-blocking)
            try:
                return self._command_queue.get_nowait()
            except asyncio.QueueEmpty:
                pass
            
            # Check for new data
            data = await self._receive_data_non_blocking()
            
            if data is None:
                return None
            
            # Handle text commands
            if isinstance(data, str):
                try:
                    command = json.loads(data)
                    return command
                except json.JSONDecodeError:
                    logger.warning(f"Invalid JSON command: {data}")
                    return None
            
            if isinstance(data, dict) and "action" in data:
                return data
            
            return None
            
        except Exception as e:
            logger.error(f"Error receiving command: {e}")
            return None
    
    async def send_json(self, data: Dict[str, Any]) -> None:
        """Send JSON data to the client."""
        try:
            await self._send_json_impl(data)
        except Exception as e:
            logger.error(f"Error sending JSON data: {e}")
            self._is_connected = False
    
    async def send_audio(self, audio: AudioData) -> None:
        """Send audio data to the client."""
        try:
            # Encode audio as base64 for JSON transmission
            import base64
            
            # For now, convert to WAV format
            from ...audio.codecs import encode_wav
            wav_data = encode_wav(
                audio.data, 
                audio.sample_rate, 
                audio.channels
            )
            
            audio_b64 = base64.b64encode(wav_data).decode('utf-8')
            
            await self.send_json({
                "type": "audio_response",
                "audio_data": audio_b64,
                "format": "wav",
                "sample_rate": audio.sample_rate,
                "duration_ms": audio.duration_ms
            })
            
        except Exception as e:
            logger.error(f"Error sending audio data: {e}")
            self._is_connected = False
    
    async def close(self, code: int = 1000) -> None:
        """Close the WebSocket connection."""
        try:
            await self._close_connection_impl(code)
        except Exception as e:
            logger.error(f"Error closing connection: {e}")
        finally:
            self._is_connected = False
            logger.debug("WebSocket connection closed")
    
    @property
    def is_connected(self) -> bool:
        """Check if WebSocket is connected."""
        return self._is_connected
    
    # Abstract methods to be implemented by subclasses
    
    async def _accept_connection_impl(self) -> None:
        """Framework-specific connection acceptance."""
        raise NotImplementedError
    
    async def _receive_data(self) -> Optional[Any]:
        """Framework-specific data reception (blocking)."""
        raise NotImplementedError
    
    async def _receive_data_non_blocking(self) -> Optional[Any]:
        """Framework-specific data reception (non-blocking)."""
        return None  # Default implementation returns None
    
    async def _send_json_impl(self, data: Dict[str, Any]) -> None:
        """Framework-specific JSON sending."""
        raise NotImplementedError
    
    async def _close_connection_impl(self, code: int) -> None:
        """Framework-specific connection closing."""
        raise NotImplementedError