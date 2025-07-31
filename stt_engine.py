# File: stt_engine.py
# Core STT model loading and speech-to-text transcription logic.

import logging
import torch
from typing import Optional
from pathlib import Path

try:
    import whisper
    WHISPER_AVAILABLE = True
except ImportError:
    WHISPER_AVAILABLE = False
    whisper = None

from config import get_stt_device, get_stt_model_size, get_stt_language

logger = logging.getLogger(__name__)


class STTEngine:
    """Speech-to-Text engine using OpenAI Whisper."""
    
    def __init__(self):
        self.model: Optional[whisper.Whisper] = None
        self.model_loaded: bool = False
        self.device: Optional[str] = None


    def _test_device_functionality(self, device: str) -> bool:
        """
        Tests if the specified device is functional for torch operations.
        
        Args:
            device: Device string ('cuda', 'mps', or 'cpu')
            
        Returns:
            bool: True if device works, False otherwise.
        """
        try:
            test_tensor = torch.tensor([1.0])
            test_tensor = test_tensor.to(device)
            test_tensor = test_tensor.cpu()
            return True
        except Exception as e:
            logger.warning(f"{device.upper()} functionality test failed: {e}")
            return False

    def load_model(self) -> bool:
        """
        Loads the Whisper STT model.
        
        Returns:
            bool: True if the model was loaded successfully, False otherwise.
        """
        if not WHISPER_AVAILABLE:
            logger.error("Whisper library not available. Cannot load STT model.")
            return False
        
        if self.model_loaded:
            logger.info("STT model is already loaded.")
            return True
        
        try:
            # Determine processing device
            device_setting = get_stt_device()
            
            if device_setting == "auto":
                if self._test_device_functionality("cuda"):
                    resolved_device_str = "cuda"
                    logger.info("CUDA functionality test passed. Using CUDA for STT.")
                elif self._test_device_functionality("mps"):
                    resolved_device_str = "mps" 
                    logger.info("MPS functionality test passed. Using MPS for STT.")
                else:
                    resolved_device_str = "cpu"
                    logger.info("Using CPU for STT.")
            else:
                resolved_device_str = device_setting
                if not self._test_device_functionality(resolved_device_str):
                    logger.warning(f"Requested device {resolved_device_str} failed test. Falling back to CPU.")
                    resolved_device_str = "cpu"
            
            self.device = resolved_device_str
            model_size = get_stt_model_size()
            
            logger.info(f"Loading Whisper model '{model_size}' on device '{self.device}'...")
            self.model = whisper.load_model(model_size, device=self.device)
            
            self.model_loaded = True
            logger.info(f"STT model '{model_size}' loaded successfully on {self.device}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load STT model: {e}", exc_info=True)
            self.model = None
            self.model_loaded = False
            return False

    def transcribe_file(self, audio_file_path: str, language: Optional[str] = None) -> Optional[str]:
        """
        Transcribes audio from a file path.
        
        Args:
            audio_file_path: Path to the audio file
            language: Language code or None for auto-detection
            
        Returns:
            Transcribed text or None if transcription fails
        """
        if not self.model_loaded or self.model is None:
            logger.error("STT model is not loaded. Cannot transcribe audio.")
            return None
        
        try:
            audio_path = Path(audio_file_path)
            if not audio_path.exists():
                logger.error(f"Audio file not found: {audio_file_path}")
                return None
            
            # Use configured language or auto-detection
            detect_language = language or get_stt_language()
            language_param = None if detect_language == "auto" else detect_language
            
            logger.info(f"Transcribing audio file: {audio_file_path}")
            result = self.model.transcribe(str(audio_path), language=language_param)
            
            transcribed_text = result["text"].strip()
            logger.info(f"Transcription completed. Length: {len(transcribed_text)} characters")
            return transcribed_text
            
        except Exception as e:
            logger.error(f"Error during transcription: {e}", exc_info=True)
            return None

    def transcribe_numpy(self, audio_array: 'np.ndarray', language: Optional[str] = None) -> Optional[str]:
        """
        Transcribes audio from a numpy array directly.
        
        Args:
            audio_array: Float32 numpy array with audio data (mono, any sample rate)
            language: Language code or None for auto-detection
            
        Returns:
            Transcribed text or None if transcription fails
        """
        if not self.model_loaded or self.model is None:
            logger.error("STT model is not loaded. Cannot transcribe audio.")
            return None
        
        try:
            import numpy as np
            
            if not isinstance(audio_array, np.ndarray):
                logger.error("Audio input must be a numpy array")
                return None
            
            if len(audio_array) == 0:
                logger.warning("Empty audio array provided")
                return None
            
            # Use configured language or auto-detection
            detect_language = language or get_stt_language()
            language_param = None if detect_language == "auto" else detect_language
            
            logger.debug(f"Transcribing numpy array: shape={audio_array.shape}, dtype={audio_array.dtype}")
            result = self.model.transcribe(audio_array, language=language_param)
            
            transcribed_text = result["text"].strip()
            logger.debug(f"Numpy transcription completed. Length: {len(transcribed_text)} characters")
            return transcribed_text
            
        except Exception as e:
            logger.error(f"Error during numpy transcription: {e}", exc_info=True)
            return None