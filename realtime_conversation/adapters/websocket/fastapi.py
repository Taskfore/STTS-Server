"""
FastAPI WebSocket adapter implementation.

Provides WebSocket integration for FastAPI applications.
"""

import asyncio
import json
import logging
from typing import Optional, Any, Dict, Union
from fastapi import WebSocket, WebSocketDisconnect

from .base import BaseWebSocketAdapter
from ...core.interfaces import AudioData

logger = logging.getLogger(__name__)


class FastAPIWebSocketAdapter(BaseWebSocketAdapter):
    """FastAPI WebSocket adapter implementation."""
    
    def __init__(self, websocket: WebSocket):
        """
        Initialize FastAPI WebSocket adapter.
        
        Args:
            websocket: FastAPI WebSocket instance
        """
        super().__init__()
        self.websocket = websocket
        self._pending_data = None
        
    async def _accept_connection_impl(self) -> None:
        """Accept the FastAPI WebSocket connection."""
        await self.websocket.accept()
    
    async def _receive_data(self) -> Optional[Any]:
        """Receive data from FastAPI WebSocket (blocking)."""
        try:
            # If we have pending data, return it first
            if self._pending_data is not None:
                data = self._pending_data
                self._pending_data = None
                return data
            
            # Receive new data
            message = await self.websocket.receive()
            
            if message["type"] == "websocket.disconnect":
                self._is_connected = False
                return None
            
            if message["type"] == "websocket.receive":
                # Handle binary data (audio)
                if "bytes" in message:
                    return message["bytes"]
                
                # Handle text data (commands)
                if "text" in message:
                    text_data = message["text"]
                    
                    # Try to parse as JSON command
                    try:
                        command = json.loads(text_data)
                        # Put command in queue for command handling
                        await self._command_queue.put(command)
                        return None  # Commands are handled separately
                    except json.JSONDecodeError:
                        # Not JSON, return as plain text
                        return text_data
            
            return None
            
        except WebSocketDisconnect:
            self._is_connected = False
            return None
        except Exception as e:
            logger.error(f"Error receiving data from FastAPI WebSocket: {e}")
            self._is_connected = False
            return None
    
    async def _receive_data_non_blocking(self) -> Optional[Any]:
        """Receive data from FastAPI WebSocket (non-blocking)."""
        try:
            # Check if there's data available without blocking
            # FastAPI doesn't provide a direct non-blocking receive,
            # so we'll use a timeout approach
            
            try:
                message = await asyncio.wait_for(
                    self.websocket.receive(), 
                    timeout=0.001  # 1ms timeout
                )
                
                if message["type"] == "websocket.disconnect":
                    self._is_connected = False
                    return None
                
                if message["type"] == "websocket.receive":
                    # Handle text data (commands)
                    if "text" in message:
                        text_data = message["text"]
                        
                        try:
                            return json.loads(text_data)
                        except json.JSONDecodeError:
                            return text_data
                    
                    # Handle binary data (audio) - store for next receive_audio call
                    if "bytes" in message:
                        self._pending_data = message["bytes"]
                        return None
                
                return None
                
            except asyncio.TimeoutError:
                # No data available, return None
                return None
            
        except WebSocketDisconnect:
            self._is_connected = False
            return None
        except Exception as e:
            logger.error(f"Error in non-blocking receive: {e}")
            return None
    
    async def _send_json_impl(self, data: Dict[str, Any]) -> None:
        """Send JSON data via FastAPI WebSocket."""
        await self.websocket.send_json(data)
    
    async def _close_connection_impl(self, code: int) -> None:
        """Close the FastAPI WebSocket connection."""
        await self.websocket.close(code)
    
    @property
    def is_connected(self) -> bool:
        """Check if FastAPI WebSocket is connected."""
        # FastAPI doesn't provide a direct connected check,
        # so we use our internal state
        return self._is_connected and hasattr(self.websocket, 'client')
    
    # Additional FastAPI-specific methods
    
    async def send_text(self, text: str) -> None:
        """Send plain text data."""
        try:
            await self.websocket.send_text(text)
        except Exception as e:
            logger.error(f"Error sending text data: {e}")
            self._is_connected = False
    
    async def send_bytes(self, data: bytes) -> None:
        """Send binary data directly."""
        try:
            await self.websocket.send_bytes(data)
        except Exception as e:
            logger.error(f"Error sending binary data: {e}")
            self._is_connected = False
    
    def get_client_info(self) -> Dict[str, Any]:
        """Get client connection information."""
        try:
            return {
                "client": str(self.websocket.client) if hasattr(self.websocket, 'client') else None,
                "url": str(self.websocket.url) if hasattr(self.websocket, 'url') else None,
                "headers": dict(self.websocket.headers) if hasattr(self.websocket, 'headers') else {},
                "query_params": dict(self.websocket.query_params) if hasattr(self.websocket, 'query_params') else {}
            }
        except Exception as e:
            logger.error(f"Error getting client info: {e}")
            return {}


class FastAPIWebSocketAdapterFactory:
    """Factory for creating FastAPI WebSocket adapters."""
    
    @staticmethod
    def create(websocket: WebSocket) -> FastAPIWebSocketAdapter:
        """
        Create a FastAPI WebSocket adapter.
        
        Args:
            websocket: FastAPI WebSocket instance
            
        Returns:
            Configured FastAPI WebSocket adapter
        """
        adapter = FastAPIWebSocketAdapter(websocket)
        logger.debug(f"Created FastAPI WebSocket adapter for client: {websocket.client}")
        return adapter
    
    @staticmethod
    def create_with_config(
        websocket: WebSocket, 
        config: Dict[str, Any]
    ) -> FastAPIWebSocketAdapter:
        """
        Create a FastAPI WebSocket adapter with configuration.
        
        Args:
            websocket: FastAPI WebSocket instance
            config: Adapter configuration
            
        Returns:
            Configured FastAPI WebSocket adapter
        """
        adapter = FastAPIWebSocketAdapter(websocket)
        
        # Apply configuration if needed
        # This could include audio format settings, buffer sizes, etc.
        
        logger.debug(
            f"Created FastAPI WebSocket adapter with config for client: {websocket.client}"
        )
        return adapter