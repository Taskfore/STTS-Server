# File: adapters/legacy_engines.py
# Adapter wrappers that bridge existing engines with realtime_conversation library interfaces

import logging
import numpy as np
import torch
from typing import Optional, Dict, Any, List
from pathlib import Path

# Import library interfaces
from realtime_conversation.core.interfaces import (
    STTEngine as STTEngineProtocol,
    TTSEngine as TTSEngineProtocol, 
    AudioData,
    TranscriptionResult,
    TranscriptionSegment,
    SynthesisResult
)

# Import existing engines
import engine as legacy_tts_engine
from stt_engine import STTEngine as LegacySTTEngine

logger = logging.getLogger(__name__)


class LegacySTTEngineAdapter:
    """
    Adapter that wraps the existing STTEngine to implement the library's STTEngine protocol.
    This allows gradual migration from the old engine to the new library interface.
    """
    
    def __init__(self, legacy_engine: LegacySTTEngine):
        """
        Initialize with existing STT engine instance.
        
        Args:
            legacy_engine: The existing STTEngine instance
        """
        self.legacy_engine = legacy_engine
        logger.info("LegacySTTEngineAdapter initialized")
    
    async def transcribe(
        self, 
        audio: AudioData, 
        language: Optional[str] = None
    ) -> Optional[TranscriptionResult]:
        """
        Transcribe audio data using the legacy engine.
        
        Args:
            audio: Audio data to transcribe
            language: Target language (None for auto-detection)
            
        Returns:
            Transcription result with timing information, or None on failure
        """
        try:
            # Convert AudioData to numpy array for legacy engine
            if audio.format == "pcm":
                # Convert PCM bytes to numpy array
                audio_np = np.frombuffer(audio.data, dtype=np.int16).astype(np.float32) / 32768.0
            else:
                # Assume already in correct format
                audio_np = np.frombuffer(audio.data, dtype=np.float32)
            
            if len(audio_np) == 0:
                logger.warning("Empty audio array for transcription")
                return None
            
            # Use the legacy engine's transcribe_numpy_with_timing method
            legacy_result = self.legacy_engine.transcribe_numpy_with_timing(audio_np, language)
            
            if legacy_result is None:
                return None
            
            # Convert legacy result to library format
            return self._convert_legacy_transcription_result(legacy_result)
            
        except Exception as e:
            logger.error(f"Error in legacy STT adapter transcription: {e}", exc_info=True)
            return None
    
    async def is_available(self) -> bool:
        """Check if the STT engine is ready for transcription."""
        return self.legacy_engine.model_loaded
    
    @property
    def model_loaded(self) -> bool:
        """Check if the STT model is loaded."""
        return self.legacy_engine.model_loaded
    
    def _convert_legacy_transcription_result(self, legacy_result) -> TranscriptionResult:
        """Convert legacy transcription result to library format."""
        try:
            # Convert segments
            segments = []
            if hasattr(legacy_result, 'segments') and legacy_result.segments:
                for seg in legacy_result.segments:
                    segment = TranscriptionSegment(
                        text=seg.text.strip(),
                        start=float(seg.start),
                        end=float(seg.end),
                        confidence=getattr(seg, 'confidence', None)
                    )
                    segments.append(segment)
            
            # Create result
            result = TranscriptionResult(
                text=legacy_result.text.strip(),
                language=getattr(legacy_result, 'language', 'unknown'),
                segments=segments,
                confidence=getattr(legacy_result, 'confidence', None),
                partial=getattr(legacy_result, 'partial', False)
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Error converting legacy transcription result: {e}")
            # Return minimal result
            return TranscriptionResult(
                text=getattr(legacy_result, 'text', '').strip(),
                language='unknown',
                segments=[],
                confidence=None,
                partial=False
            )


class LegacyTTSEngineAdapter:
    """
    Adapter that wraps the existing engine.py module to implement the library's TTS protocol.
    """
    
    def __init__(self):
        """Initialize the TTS engine adapter."""
        logger.info("LegacyTTSEngineAdapter initialized")
    
    async def synthesize(
        self, 
        text: str, 
        voice_config: Dict[str, Any]
    ) -> Optional[SynthesisResult]:
        """
        Synthesize text to speech using the legacy engine.
        
        Args:
            text: Text to synthesize
            voice_config: Voice configuration (voice_path, temperature, speed, etc.)
            
        Returns:
            Synthesis result with audio data, or None on failure
        """
        try:
            # Extract configuration
            voice_path = voice_config.get('voice_path')
            temperature = voice_config.get('temperature', 0.8)
            exaggeration = voice_config.get('exaggeration', 0.5)
            cfg_weight = voice_config.get('cfg_weight', 0.5)
            seed = voice_config.get('seed', 0)
            speed_factor = voice_config.get('speed_factor', 1.0)
            
            logger.debug(f"Synthesizing with legacy engine: {len(text)} chars, voice={voice_path}")
            
            # Use legacy engine
            audio_tensor, sample_rate = legacy_tts_engine.synthesize(
                text=text,
                audio_prompt_path=voice_path,
                temperature=temperature,
                exaggeration=exaggeration,
                cfg_weight=cfg_weight,
                seed=seed
            )
            
            if audio_tensor is None or sample_rate is None:
                logger.error("Legacy TTS engine returned None")
                return None
            
            # Apply speed factor if needed
            if speed_factor != 1.0:
                import utils
                audio_tensor, _ = utils.apply_speed_factor(audio_tensor, sample_rate, speed_factor)
            
            # Convert to numpy
            audio_np = audio_tensor.cpu().numpy().squeeze()
            
            # Create AudioData
            audio_data = self._create_audio_data(audio_np, sample_rate)
            
            # Create synthesis result
            result = SynthesisResult(
                audio_data=audio_data,
                text=text,
                voice_id=voice_config.get('voice_id', voice_path or 'default'),
                synthesis_time=None  # Legacy engine doesn't provide timing
            )
            
            logger.debug(f"Legacy TTS synthesis completed: {audio_data.duration_ms}ms audio")
            return result
            
        except Exception as e:
            logger.error(f"Error in legacy TTS adapter synthesis: {e}", exc_info=True)
            return None
    
    async def list_voices(self) -> List[Dict[str, Any]]:
        """List available voices using existing utility functions."""
        try:
            import utils
            from config import get_predefined_voices_path, get_reference_audio_path
            
            voices = []
            
            # Add predefined voices
            predefined_voices = utils.get_predefined_voices()
            for voice in predefined_voices:
                voices.append({
                    "id": voice["filename"],
                    "name": voice["display_name"],
                    "type": "predefined",
                    "path": voice["filename"],
                    "language": "en"  # Default assumption
                })
            
            # Add reference audio files
            reference_files = utils.get_valid_reference_files()
            for ref_file in reference_files:
                voices.append({
                    "id": ref_file,
                    "name": ref_file.replace(".wav", "").replace(".mp3", ""),
                    "type": "reference",
                    "path": ref_file,
                    "language": "en"  # Default assumption
                })
            
            return voices
            
        except Exception as e:
            logger.error(f"Error listing voices in legacy adapter: {e}")
            return []
    
    @property
    def model_loaded(self) -> bool:
        """Check if the TTS model is loaded."""
        return getattr(legacy_tts_engine, 'MODEL_LOADED', False)
    
    def _create_audio_data(self, audio_np: np.ndarray, sample_rate: int) -> AudioData:
        """Create AudioData from numpy array."""
        # Convert to 16-bit PCM bytes
        audio_int16 = (audio_np * 32767).astype(np.int16)
        audio_bytes = audio_int16.tobytes()
        
        return AudioData(
            data=audio_bytes,
            sample_rate=sample_rate,
            channels=1,
            format="pcm"
        )


class ConfigurationAdapter:
    """
    Adapter that bridges the existing config system with the library's configuration provider.
    """
    
    def __init__(self):
        """Initialize with existing config system."""
        from config import config_manager
        self.config_manager = config_manager
        logger.info("ConfigurationAdapter initialized")
    
    def get_stt_config(self) -> Dict[str, Any]:
        """Get STT engine configuration."""
        return {
            "model_size": self.config_manager.get_string("stt_engine.model_size", "base"),
            "device": self.config_manager.get_string("stt_engine.device", "auto"),
            "language": self.config_manager.get_string("stt_engine.language", "auto")
        }
    
    def get_tts_config(self) -> Dict[str, Any]:
        """Get TTS engine configuration."""
        return {
            "device": self.config_manager.get_string("tts_engine.device", "auto"),
            "temperature": self.config_manager.get_float("gen.default_temperature", 0.8),
            "exaggeration": self.config_manager.get_float("gen.default_exaggeration", 0.5),
            "cfg_weight": self.config_manager.get_float("gen.default_cfg_weight", 0.5),
            "speed_factor": self.config_manager.get_float("gen.default_speed_factor", 1.0),
            "seed": self.config_manager.get_int("gen.default_seed", 0)
        }
    
    def get_conversation_config(self) -> Dict[str, Any]:
        """Get conversation engine configuration."""
        return {
            "response_mode": "echo",  # Default for compatibility
            "max_history_length": 50,
            "enable_analytics": False
        }
    
    def get_pause_detection_config(self) -> Dict[str, Any]:
        """Get pause detection configuration."""
        return {
            "aggressiveness": 2,
            "min_speech_frames": 3,
            "min_pause_frames": 10,
            "sample_rate": self.config_manager.get_int("audio.sample_rate", 16000)
        }
    
    def get_audio_config(self) -> Dict[str, Any]:
        """Get audio processing configuration."""
        return {
            "sample_rate": self.config_manager.get_int("audio.sample_rate", 16000),
            "channels": 1,
            "format": "pcm",
            "output_format": self.config_manager.get_string("audio_output.format", "wav")
        }


# Factory functions for creating adapters

def create_legacy_stt_adapter() -> LegacySTTEngineAdapter:
    """Create a legacy STT engine adapter using the existing engine."""
    from stt_engine import STTEngine
    legacy_engine = STTEngine()
    return LegacySTTEngineAdapter(legacy_engine)


def create_legacy_tts_adapter() -> LegacyTTSEngineAdapter:
    """Create a legacy TTS engine adapter."""
    return LegacyTTSEngineAdapter()


def create_config_adapter() -> ConfigurationAdapter:
    """Create a configuration adapter."""
    return ConfigurationAdapter()


def get_or_create_adapters() -> Dict[str, Any]:
    """Get or create all adapters as a convenience function."""
    return {
        "stt_adapter": create_legacy_stt_adapter(),
        "tts_adapter": create_legacy_tts_adapter(), 
        "config_adapter": create_config_adapter()
    }