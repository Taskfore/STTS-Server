"""
OpenAI Whisper STT engine adapter.

Provides STT functionality using the OpenAI Whisper model.
"""

import logging
import numpy as np
from typing import Optional, Dict, Any
from pathlib import Path

try:
    import whisper
    import torch
    WHISPER_AVAILABLE = True
except ImportError:
    WHISPER_AVAILABLE = False
    whisper = None
    torch = None

from .base import BaseSTTEngine
from ...core.interfaces import AudioData, TranscriptionResult, TranscriptionSegment
from ...audio.codecs import decode_pcm

logger = logging.getLogger(__name__)


class WhisperSTTEngine(BaseSTTEngine):
    """OpenAI Whisper STT engine implementation."""
    
    def __init__(
        self,
        model_size: str = "base",
        device: str = "auto",
        model_path: Optional[str] = None,
        download_root: Optional[str] = None
    ):
        """
        Initialize Whisper STT engine.
        
        Args:
            model_size: Whisper model size ("tiny", "base", "small", "medium", "large")
            device: Processing device ("auto", "cuda", "mps", "cpu")
            model_path: Path to custom model file
            download_root: Directory to store downloaded models
        """
        super().__init__()
        
        if not WHISPER_AVAILABLE:
            raise ImportError("Whisper library not available. Install with: pip install openai-whisper")
        
        self.model_size = model_size
        self.device_setting = device
        self.model_path = model_path
        self.download_root = download_root
        
        logger.info(f"WhisperSTTEngine initialized: model={model_size}, device={device}")
    
    def _test_device_functionality(self, device: str) -> bool:
        """Test if the specified device is functional for torch operations."""
        if not torch:
            return False
            
        try:
            test_tensor = torch.tensor([1.0])
            test_tensor = test_tensor.to(device)
            test_tensor = test_tensor.cpu()
            return True
        except Exception as e:
            logger.debug(f"{device.upper()} functionality test failed: {e}")
            return False
    
    def _resolve_device(self) -> str:
        """Resolve the best available device for processing."""
        if self.device_setting == "auto":
            # Test devices in order of preference
            if self._test_device_functionality("cuda"):
                logger.info("CUDA functionality test passed. Using CUDA for STT.")
                return "cuda"
            elif self._test_device_functionality("mps"):
                logger.info("MPS functionality test passed. Using MPS for STT.")
                return "mps"
            else:
                logger.info("Using CPU for STT.")
                return "cpu"
        else:
            # Use specified device, fallback to CPU if it fails
            if self._test_device_functionality(self.device_setting):
                return self.device_setting
            else:
                logger.warning(f"Requested device {self.device_setting} failed test. Falling back to CPU.")
                return "cpu"
    
    def load_model(self, **kwargs) -> bool:
        """Load the Whisper model."""
        if self._model_loaded:
            logger.info("Whisper STT model is already loaded.")
            return True
        
        try:
            # Resolve processing device
            self._device = self._resolve_device()
            
            logger.info(f"Loading Whisper model '{self.model_size}' on device '{self._device}'...")
            
            # Load model
            load_kwargs = {
                "device": self._device
            }
            
            if self.download_root:
                load_kwargs["download_root"] = self.download_root
            
            if self.model_path:
                # Load from custom path
                self._model = whisper.load_model(self.model_path, **load_kwargs)
            else:
                # Load standard model
                self._model = whisper.load_model(self.model_size, **load_kwargs)
            
            self._model_loaded = True
            logger.info(f"Whisper STT model '{self.model_size}' loaded successfully on {self._device}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load Whisper STT model: {e}", exc_info=True)
            self._model = None
            self._model_loaded = False
            return False
    
    def unload_model(self) -> None:
        """Unload the Whisper model to free memory."""
        if self._model is not None:
            del self._model
            self._model = None
            self._model_loaded = False
            
            # Clear GPU cache if using CUDA
            if torch and self._device.startswith("cuda"):
                torch.cuda.empty_cache()
            
            logger.info("Whisper STT model unloaded")
    
    def _transcribe_sync(
        self, 
        audio: AudioData, 
        language: Optional[str] = None
    ) -> Optional[TranscriptionResult]:
        """Synchronous transcription implementation."""
        if not self._model_loaded or self._model is None:
            logger.error("Whisper model is not loaded. Cannot transcribe audio.")
            return None
        
        try:
            # Convert audio data to numpy array
            if audio.format == "pcm":
                audio_np = decode_pcm(audio.data, audio.sample_rate, audio.channels)
                if audio_np is None:
                    logger.error("Failed to decode PCM audio data")
                    return None
            else:
                # For other formats, assume the data is already in the correct format
                # This would need to be extended for other audio formats
                logger.warning(f"Audio format '{audio.format}' not fully supported, assuming raw float data")
                audio_np = np.frombuffer(audio.data, dtype=np.float32)
            
            if len(audio_np) == 0:
                logger.warning("Empty audio array after decoding")
                return None
            
            # Prepare language parameter
            language_param = None if language == "auto" or language is None else language
            
            logger.debug(f"Transcribing audio: shape={audio_np.shape}, dtype={audio_np.dtype}, language={language_param}")
            
            # Transcribe with Whisper
            raw_result = self._model.transcribe(audio_np, language=language_param)
            
            # Convert raw result to our interface format
            transcription_result = self._convert_whisper_result(raw_result)
            
            logger.debug(f"Transcription completed: '{transcription_result.text[:50]}...', segments={len(transcription_result.segments)}")
            
            return transcription_result
            
        except Exception as e:
            logger.error(f"Error during Whisper transcription: {e}", exc_info=True)
            return None
    
    def _convert_whisper_result(self, raw_result: Dict[str, Any]) -> TranscriptionResult:
        """Convert raw Whisper result to our TranscriptionResult format."""
        try:
            # Extract segments
            segments = []
            raw_segments = raw_result.get("segments", [])
            
            for segment in raw_segments:
                transcription_segment = TranscriptionSegment(
                    text=segment.get("text", "").strip(),
                    start=float(segment.get("start", 0.0)),
                    end=float(segment.get("end", 0.0)),
                    confidence=float(segment.get("avg_logprob", 0.0)) if segment.get("avg_logprob") is not None else None
                )
                segments.append(transcription_segment)
            
            # Create TranscriptionResult
            result = TranscriptionResult(
                text=raw_result.get("text", "").strip(),
                language=raw_result.get("language", "unknown"),
                segments=segments,
                confidence=None,  # Whisper doesn't provide overall confidence
                partial=False
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Error converting Whisper result: {e}")
            # Return minimal result
            return TranscriptionResult(
                text=raw_result.get("text", "").strip(),
                language=raw_result.get("language", "unknown"),
                segments=[],
                confidence=None,
                partial=False
            )
    
    # Additional utility methods
    
    def get_available_models(self) -> list:
        """Get list of available Whisper models."""
        if not WHISPER_AVAILABLE:
            return []
        
        return list(whisper.available_models())
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model."""
        if not self._model_loaded or self._model is None:
            return {}
        
        try:
            return {
                "model_size": self.model_size,
                "device": self._device,
                "is_multilingual": self._model.is_multilingual,
                "n_mels": self._model.dims.n_mels,
                "n_vocab": self._model.dims.n_vocab,
                "n_text_ctx": self._model.dims.n_text_ctx,
                "n_audio_ctx": self._model.dims.n_audio_ctx
            }
        except Exception as e:
            logger.error(f"Error getting model info: {e}")
            return {"error": str(e)}
    
    def transcribe_file(self, file_path: str, language: Optional[str] = None) -> Optional[TranscriptionResult]:
        """
        Transcribe audio file directly (synchronous).
        
        Args:
            file_path: Path to audio file
            language: Target language
            
        Returns:
            Transcription result or None
        """
        if not self._model_loaded or self._model is None:
            logger.error("Whisper model is not loaded")
            return None
        
        try:
            audio_path = Path(file_path)
            if not audio_path.exists():
                logger.error(f"Audio file not found: {file_path}")
                return None
            
            language_param = None if language == "auto" or language is None else language
            
            logger.info(f"Transcribing audio file: {file_path}")
            raw_result = self._model.transcribe(str(audio_path), language=language_param)
            
            return self._convert_whisper_result(raw_result)
            
        except Exception as e:
            logger.error(f"Error transcribing file: {e}", exc_info=True)
            return None


# Factory functions

def create_whisper_engine(
    model_size: str = "base", 
    device: str = "auto",
    auto_load: bool = True
) -> WhisperSTTEngine:
    """
    Create and optionally load a Whisper STT engine.
    
    Args:
        model_size: Whisper model size
        device: Processing device
        auto_load: Whether to automatically load the model
        
    Returns:
        Configured WhisperSTTEngine instance
    """
    engine = WhisperSTTEngine(model_size=model_size, device=device)
    
    if auto_load:
        if not engine.load_model():
            logger.error("Failed to load Whisper model during creation")
            
    return engine


def create_whisper_engine_from_config(config: Dict[str, Any]) -> WhisperSTTEngine:
    """
    Create Whisper STT engine from configuration dictionary.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Configured WhisperSTTEngine instance
    """
    engine = WhisperSTTEngine(
        model_size=config.get("model_size", "base"),
        device=config.get("device", "auto"),
        model_path=config.get("model_path"),
        download_root=config.get("download_root")
    )
    
    if config.get("auto_load", True):
        engine.load_model()
    
    return engine