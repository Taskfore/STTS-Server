"""
Chatterbox TTS engine adapter.

Provides TTS functionality using the Chatterbox TTS model.
"""

import logging
import random
import numpy as np
import torch
from typing import Optional, List, Dict, Any, Tuple
from pathlib import Path

try:
    from chatterbox.tts import ChatterboxTTS
    from chatterbox.models.s3gen.const import S3GEN_SR
    CHATTERBOX_AVAILABLE = True
except ImportError:
    CHATTERBOX_AVAILABLE = False
    ChatterboxTTS = None
    S3GEN_SR = 22050

from .base import BaseTTSEngine
from ...core.interfaces import SynthesisResult, AudioData
from ...audio.processing import AudioProcessor

logger = logging.getLogger(__name__)


class ChatterboxTTSEngine(BaseTTSEngine):
    """Chatterbox TTS engine implementation."""
    
    def __init__(
        self,
        device: str = "auto",
        model_cache_path: Optional[str] = None
    ):
        """
        Initialize Chatterbox TTS engine.
        
        Args:
            device: Processing device ("auto", "cuda", "mps", "cpu")
            model_cache_path: Path to model cache directory
        """
        super().__init__()
        
        if not CHATTERBOX_AVAILABLE:
            raise ImportError("Chatterbox TTS library not available")
        
        self.device_setting = device
        self.model_cache_path = model_cache_path
        self._sample_rate = S3GEN_SR
        
        logger.info(f"ChatterboxTTSEngine initialized: device={device}")
    
    def _test_device_functionality(self, device: str) -> bool:
        """Test if the specified device is functional for torch operations."""
        try:
            test_tensor = torch.tensor([1.0])
            if device == "cuda":
                test_tensor = test_tensor.cuda()
            elif device == "mps":
                test_tensor = test_tensor.to("mps")
            test_tensor = test_tensor.cpu()
            return True
        except Exception as e:
            logger.debug(f"{device.upper()} functionality test failed: {e}")
            return False
    
    def _resolve_device(self) -> str:
        """Resolve the best available device for processing."""
        if self.device_setting == "auto":
            if self._test_device_functionality("cuda"):
                logger.info("CUDA functionality test passed. Using CUDA for TTS.")
                return "cuda"
            elif self._test_device_functionality("mps"):
                logger.info("MPS functionality test passed. Using MPS for TTS.")
                return "mps"
            else:
                logger.info("Using CPU for TTS.")
                return "cpu"
        else:
            if self._test_device_functionality(self.device_setting):
                return self.device_setting
            else:
                logger.warning(f"Requested device {self.device_setting} failed test. Falling back to CPU.")
                return "cpu"
    
    def _set_seed(self, seed_value: int) -> None:
        """Set random seed for reproducibility."""
        torch.manual_seed(seed_value)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed_value)
            torch.cuda.manual_seed_all(seed_value)
        if torch.backends.mps.is_available():
            torch.mps.manual_seed(seed_value)
        random.seed(seed_value)
        np.random.seed(seed_value)
        logger.debug(f"Global seed set to: {seed_value}")
    
    def load_model(self, **kwargs) -> bool:
        """Load the Chatterbox TTS model."""
        if self._model_loaded:
            logger.info("Chatterbox TTS model is already loaded.")
            return True
        
        try:
            # Resolve processing device
            self._device = self._resolve_device()
            
            logger.info(f"Loading Chatterbox TTS model on device '{self._device}'...")
            
            # Initialize ChatterboxTTS
            if self.model_cache_path:
                self._model = ChatterboxTTS.from_pretrained(
                    "careerlink/chatterbox",
                    device=self._device,
                    cache_dir=self.model_cache_path
                )
            else:
                self._model = ChatterboxTTS.from_pretrained(
                    "careerlink/chatterbox",
                    device=self._device
                )
            
            self._model_loaded = True
            self._sample_rate = self._model.sr
            
            logger.info(f"Chatterbox TTS model loaded successfully on {self._device}, sample_rate={self._sample_rate}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load Chatterbox TTS model: {e}", exc_info=True)
            self._model = None
            self._model_loaded = False
            return False
    
    def unload_model(self) -> None:
        """Unload the Chatterbox TTS model to free memory."""
        if self._model is not None:
            del self._model
            self._model = None
            self._model_loaded = False
            
            # Clear GPU cache if using CUDA
            if torch.cuda.is_available() and self._device.startswith("cuda"):
                torch.cuda.empty_cache()
            
            logger.info("Chatterbox TTS model unloaded")
    
    def _synthesize_sync(
        self, 
        text: str, 
        voice_config: Dict[str, Any]
    ) -> Optional[SynthesisResult]:
        """Synchronous synthesis implementation."""
        if not self._model_loaded or self._model is None:
            logger.error("Chatterbox model is not loaded. Cannot synthesize audio.")
            return None
        
        try:
            # Validate and extract voice configuration
            config = self._validate_voice_config(voice_config)
            
            voice_path = config.get("voice_path")
            temperature = config.get("temperature", 0.8)
            exaggeration = config.get("exaggeration", 0.5)
            cfg_weight = config.get("cfg_weight", 0.5)
            seed = config.get("seed", 0)
            speed_factor = config.get("speed_factor", 1.0)
            
            # Set seed if provided
            if seed != 0:
                logger.debug(f"Applying user-provided seed for generation: {seed}")
                self._set_seed(seed)
            
            logger.debug(
                f"Synthesizing with Chatterbox: voice='{voice_path}', temp={temperature}, "
                f"exag={exaggeration}, cfg_weight={cfg_weight}, seed={seed}, speed={speed_factor}"
            )
            
            # Perform synthesis
            start_time = torch.cuda.Event(enable_timing=True) if self._device.startswith("cuda") else None
            end_time = torch.cuda.Event(enable_timing=True) if self._device.startswith("cuda") else None
            
            if start_time:
                start_time.record()
            
            wav_tensor = self._model.generate(
                text=text,
                audio_prompt_path=voice_path,
                temperature=temperature,
                exaggeration=exaggeration,
                cfg_weight=cfg_weight
            )
            
            if end_time:
                end_time.record()
                torch.cuda.synchronize()
                synthesis_time = start_time.elapsed_time(end_time) / 1000.0  # Convert to seconds
            else:
                synthesis_time = None
            
            if wav_tensor is None:
                logger.error("Chatterbox synthesis returned None")
                return None
            
            # Apply speed factor if needed
            if speed_factor != 1.0:
                logger.debug(f"Applying speed factor: {speed_factor}")
                wav_np = wav_tensor.cpu().numpy().squeeze()
                wav_np, _ = AudioProcessor.apply_speed_factor(
                    wav_np, self._sample_rate, speed_factor
                )
                wav_tensor = torch.tensor(wav_np).unsqueeze(0)
            
            # Create AudioData
            audio_data = self._create_audio_data(wav_tensor, config.get("voice_id", "unknown"))
            
            # Create synthesis result
            result = SynthesisResult(
                audio_data=audio_data,
                text=text,
                voice_id=config.get("voice_id", voice_path or "default"),
                synthesis_time=synthesis_time
            )
            
            logger.debug(
                f"Chatterbox synthesis completed: {len(text)} chars -> "
                f"{audio_data.duration_ms}ms audio"
                + (f" in {synthesis_time:.3f}s" if synthesis_time else "")
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Error during Chatterbox synthesis: {e}", exc_info=True)
            return None
    
    async def list_voices(self) -> List[Dict[str, Any]]:
        """
        List available voices.
        
        For Chatterbox, this would typically list predefined voice files
        and reference audio files.
        """
        voices = []
        
        # This is a placeholder implementation
        # In a real implementation, you would scan voice directories
        # and return information about available voices
        
        try:
            # Example of how you might list voices
            voices.append({
                "id": "default",
                "name": "Default Voice",
                "type": "builtin",
                "language": "en",
                "description": "Default Chatterbox voice"
            })
            
        except Exception as e:
            logger.error(f"Error listing voices: {e}")
        
        return voices
    
    # Additional utility methods
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model."""
        if not self._model_loaded or self._model is None:
            return {}
        
        try:
            return {
                "model_type": "chatterbox",
                "device": self._device,
                "sample_rate": self._sample_rate,
                "model_loaded": self._model_loaded
            }
        except Exception as e:
            logger.error(f"Error getting model info: {e}")
            return {"error": str(e)}
    
    def synthesize_with_timing(
        self,
        text: str,
        voice_path: Optional[str] = None,
        temperature: float = 0.8,
        exaggeration: float = 0.5,
        cfg_weight: float = 0.5,
        seed: int = 0
    ) -> Tuple[Optional[torch.Tensor], Optional[int]]:
        """
        Direct synthesis method compatible with the original engine.py interface.
        
        This method provides backward compatibility with the existing code.
        """
        if not self._model_loaded or self._model is None:
            logger.error("Chatterbox model is not loaded")
            return None, None
        
        try:
            # Set seed if provided
            if seed != 0:
                self._set_seed(seed)
            
            # Perform synthesis
            wav_tensor = self._model.generate(
                text=text,
                audio_prompt_path=voice_path,
                temperature=temperature,
                exaggeration=exaggeration,
                cfg_weight=cfg_weight
            )
            
            return wav_tensor, self._sample_rate
            
        except Exception as e:
            logger.error(f"Error during direct synthesis: {e}", exc_info=True)
            return None, None


# Factory functions

def create_chatterbox_engine(
    device: str = "auto", 
    auto_load: bool = True
) -> ChatterboxTTSEngine:
    """
    Create and optionally load a Chatterbox TTS engine.
    
    Args:
        device: Processing device
        auto_load: Whether to automatically load the model
        
    Returns:
        Configured ChatterboxTTSEngine instance
    """
    engine = ChatterboxTTSEngine(device=device)
    
    if auto_load:
        if not engine.load_model():
            logger.error("Failed to load Chatterbox model during creation")
    
    return engine


def create_chatterbox_engine_from_config(config: Dict[str, Any]) -> ChatterboxTTSEngine:
    """
    Create Chatterbox TTS engine from configuration dictionary.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Configured ChatterboxTTSEngine instance
    """
    engine = ChatterboxTTSEngine(
        device=config.get("device", "auto"),
        model_cache_path=config.get("model_cache_path")
    )
    
    if config.get("auto_load", True):
        engine.load_model()
    
    return engine