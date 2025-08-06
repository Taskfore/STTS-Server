"""
Base STT engine implementation.

Provides common functionality for STT engine adapters.
"""

import asyncio
import logging
from typing import Optional
from ...core.interfaces import STTEngine, AudioData, TranscriptionResult

logger = logging.getLogger(__name__)


class BaseSTTEngine(STTEngine):
    """
    Base implementation of STT engine with common functionality.
    
    Subclasses should implement the actual transcription logic.
    """
    
    def __init__(self):
        self._model_loaded = False
        self._device = "cpu"
        self._model = None
        
    async def transcribe(
        self, 
        audio: AudioData, 
        language: Optional[str] = None
    ) -> Optional[TranscriptionResult]:
        """
        Transcribe audio data to text with timing information.
        
        Args:
            audio: Audio data to transcribe
            language: Target language (None for auto-detection)
            
        Returns:
            Transcription result with timing information, or None on failure
        """
        if not self.model_loaded:
            logger.error("STT model not loaded")
            return None
        
        if not audio.data or len(audio.data) == 0:
            logger.warning("Empty audio data provided")
            return None
        
        try:
            # Run transcription in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None, 
                self._transcribe_sync, 
                audio, 
                language
            )
            return result
            
        except Exception as e:
            logger.error(f"Transcription error: {e}", exc_info=True)
            return None
    
    async def is_available(self) -> bool:
        """Check if the STT engine is ready for transcription."""
        return self.model_loaded
    
    @property
    def model_loaded(self) -> bool:
        """Check if the STT model is loaded."""
        return self._model_loaded
    
    @property
    def device(self) -> str:
        """Get the device being used for processing."""
        return self._device
    
    # Methods to be implemented by subclasses
    
    def _transcribe_sync(
        self, 
        audio: AudioData, 
        language: Optional[str] = None
    ) -> Optional[TranscriptionResult]:
        """
        Synchronous transcription implementation.
        
        This method runs in a thread pool and should be implemented by subclasses.
        """
        raise NotImplementedError
    
    def load_model(self, **kwargs) -> bool:
        """
        Load the STT model.
        
        Should be implemented by subclasses.
        """
        raise NotImplementedError
    
    def unload_model(self) -> None:
        """
        Unload the STT model to free memory.
        
        Should be implemented by subclasses if needed.
        """
        pass