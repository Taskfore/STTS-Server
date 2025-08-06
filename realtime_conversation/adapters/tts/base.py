"""
Base TTS engine implementation.

Provides common functionality for TTS engine adapters.
"""

import asyncio
import logging
from typing import Optional, List, Dict, Any
from ...core.interfaces import TTSEngine, SynthesisResult, AudioData

logger = logging.getLogger(__name__)


class BaseTTSEngine(TTSEngine):
    """
    Base implementation of TTS engine with common functionality.
    
    Subclasses should implement the actual synthesis logic.
    """
    
    def __init__(self):
        self._model_loaded = False
        self._device = "cpu"
        self._model = None
        self._sample_rate = 22050  # Default sample rate
        
    async def synthesize(
        self, 
        text: str, 
        voice_config: Dict[str, Any]
    ) -> Optional[SynthesisResult]:
        """
        Synthesize text to speech audio.
        
        Args:
            text: Text to synthesize
            voice_config: Voice configuration dictionary
            
        Returns:
            Synthesis result with audio data, or None on failure
        """
        if not self.model_loaded:
            logger.error("TTS model not loaded")
            return None
        
        if not text or not text.strip():
            logger.warning("Empty text provided for synthesis")
            return None
        
        try:
            # Run synthesis in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None,
                self._synthesize_sync,
                text,
                voice_config
            )
            return result
            
        except Exception as e:
            logger.error(f"TTS synthesis error: {e}", exc_info=True)
            return None
    
    async def list_voices(self) -> List[Dict[str, Any]]:
        """List available voices."""
        # Default implementation returns empty list
        # Subclasses should override this
        return []
    
    @property
    def model_loaded(self) -> bool:
        """Check if the TTS model is loaded."""
        return self._model_loaded
    
    @property
    def device(self) -> str:
        """Get the device being used for processing."""
        return self._device
    
    @property
    def sample_rate(self) -> int:
        """Get the model's sample rate."""
        return self._sample_rate
    
    # Methods to be implemented by subclasses
    
    def _synthesize_sync(
        self, 
        text: str, 
        voice_config: Dict[str, Any]
    ) -> Optional[SynthesisResult]:
        """
        Synchronous synthesis implementation.
        
        This method runs in a thread pool and should be implemented by subclasses.
        """
        raise NotImplementedError
    
    def load_model(self, **kwargs) -> bool:
        """
        Load the TTS model.
        
        Should be implemented by subclasses.
        """
        raise NotImplementedError
    
    def unload_model(self) -> None:
        """
        Unload the TTS model to free memory.
        
        Should be implemented by subclasses if needed.
        """
        pass
    
    # Utility methods
    
    def _create_audio_data(
        self, 
        audio_tensor, 
        voice_id: str = "unknown"
    ) -> AudioData:
        """
        Create AudioData from tensor.
        
        This is a helper method for subclasses to convert audio tensors
        to the standard AudioData format.
        """
        try:
            import numpy as np
            import torch
            
            # Convert tensor to numpy if needed
            if torch.is_tensor(audio_tensor):
                audio_np = audio_tensor.cpu().numpy()
            else:
                audio_np = np.array(audio_tensor)
            
            # Ensure it's the right shape (1D)
            if audio_np.ndim > 1:
                audio_np = audio_np.squeeze()
            
            # Convert to 16-bit PCM format
            if audio_np.dtype != np.int16:
                # Assuming float32 in range [-1.0, 1.0]
                audio_np = (audio_np * 32767).astype(np.int16)
            
            # Convert to bytes
            audio_bytes = audio_np.tobytes()
            
            return AudioData(
                data=audio_bytes,
                sample_rate=self._sample_rate,
                channels=1,
                format="pcm"
            )
            
        except Exception as e:
            logger.error(f"Error creating audio data: {e}")
            # Return empty audio data as fallback
            return AudioData(
                data=b"",
                sample_rate=self._sample_rate,
                channels=1,
                format="pcm"
            )
    
    def _validate_voice_config(self, voice_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate and set defaults for voice configuration.
        
        Args:
            voice_config: Input voice configuration
            
        Returns:
            Validated voice configuration with defaults
        """
        validated_config = voice_config.copy()
        
        # Set common defaults
        validated_config.setdefault("voice_id", None)
        validated_config.setdefault("temperature", 0.7)
        validated_config.setdefault("speed_factor", 1.0)
        validated_config.setdefault("voice_path", None)
        
        return validated_config