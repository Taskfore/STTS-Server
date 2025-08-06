"""
Base pause detection implementation.

Provides common functionality for pause detection plugins.
"""

import asyncio
import logging
from typing import Dict, Any
from ...core.interfaces import PauseDetector, AudioData

logger = logging.getLogger(__name__)


class BasePauseDetector(PauseDetector):
    """
    Base implementation of pause detector with common functionality.
    
    Subclasses should implement the actual detection logic.
    """
    
    def __init__(self):
        self._is_speaking = False
        self._last_speech_time = 0
        self._last_silence_time = 0
        self._speech_frames = 0
        self._silence_frames = 0
        self._chunk_count = 0
        
    async def process_chunk(self, audio: AudioData) -> Dict[str, Any]:
        """
        Process audio chunk and return VAD results.
        
        Args:
            audio: Audio chunk to process
            
        Returns:
            VAD result containing events, state, and confidence information
        """
        if not audio.data or len(audio.data) == 0:
            return self._get_current_state()
        
        self._chunk_count += 1
        
        try:
            # Run detection in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None,
                self._process_chunk_sync,
                audio
            )
            return result
            
        except Exception as e:
            logger.error(f"Error processing audio chunk: {e}")
            return self._get_current_state()
    
    def reset(self) -> None:
        """Reset detector state."""
        self._is_speaking = False
        self._last_speech_time = 0
        self._last_silence_time = 0
        self._speech_frames = 0
        self._silence_frames = 0
        self._chunk_count = 0
        logger.debug("Pause detector state reset")
    
    @property
    def is_speaking(self) -> bool:
        """Check if currently detecting speech."""
        return self._is_speaking
    
    def _process_chunk_sync(self, audio: AudioData) -> Dict[str, Any]:
        """
        Synchronous chunk processing implementation.
        
        This method runs in a thread pool and should be implemented by subclasses.
        """
        raise NotImplementedError
    
    def _get_current_state(self) -> Dict[str, Any]:
        """Get current detection state."""
        return {
            "is_speaking": self._is_speaking,
            "speech_frames": self._speech_frames,
            "silence_frames": self._silence_frames,
            "events": [],
            "silence_duration_ms": 0,
            "chunk_count": self._chunk_count
        }
    
    def _create_state_result(
        self,
        events: list = None,
        confidence: float = None,
        silence_duration_ms: int = 0
    ) -> Dict[str, Any]:
        """Create a state result dictionary."""
        return {
            "is_speaking": self._is_speaking,
            "speech_frames": self._speech_frames,
            "silence_frames": self._silence_frames,
            "events": events or [],
            "silence_duration_ms": silence_duration_ms,
            "speech_confidence": confidence,
            "chunk_count": self._chunk_count
        }