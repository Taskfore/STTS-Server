"""
Environment variable-based configuration provider.

Provides configuration from environment variables with optional prefixes
and type conversion.
"""

import logging
import os
from typing import Dict, Any, Optional, Union, Type
from ...core.interfaces import ConfigurationProvider

logger = logging.getLogger(__name__)


class EnvConfigurationProvider(ConfigurationProvider):
    """Environment variable-based configuration provider."""
    
    def __init__(
        self, 
        prefix: str = "CONV_", 
        separator: str = "_",
        case_sensitive: bool = False
    ):
        """
        Initialize environment configuration provider.
        
        Args:
            prefix: Prefix for environment variables
            separator: Separator for nested keys
            case_sensitive: Whether to treat environment variables as case-sensitive
        """
        self.prefix = prefix
        self.separator = separator
        self.case_sensitive = case_sensitive
        
        logger.debug(
            f"Environment configuration provider initialized: "
            f"prefix='{prefix}', separator='{separator}'"
        )
    
    def _get_env_key(self, config_path: str) -> str:
        """Convert configuration path to environment variable name."""
        # Convert dot notation to environment variable format
        env_path = config_path.replace(".", self.separator)
        env_key = f"{self.prefix}{env_path}"
        
        if not self.case_sensitive:
            env_key = env_key.upper()
        
        return env_key
    
    def _get_env_value(
        self, 
        key: str, 
        default: Any = None, 
        value_type: Type = str
    ) -> Any:
        """Get environment variable value with type conversion."""
        env_key = self._get_env_key(key)
        raw_value = os.environ.get(env_key)
        
        if raw_value is None:
            return default
        
        try:
            # Type conversion
            if value_type == bool:
                return raw_value.lower() in ('true', '1', 'yes', 'on')
            elif value_type == int:
                return int(raw_value)
            elif value_type == float:
                return float(raw_value)
            elif value_type == str:
                return raw_value
            else:
                return raw_value
        except (ValueError, TypeError) as e:
            logger.warning(
                f"Failed to convert environment variable {env_key}='{raw_value}' "
                f"to {value_type.__name__}: {e}, using default: {default}"
            )
            return default
    
    def _build_config_section(self, section_keys: Dict[str, tuple]) -> Dict[str, Any]:
        """
        Build configuration section from environment variables.
        
        Args:
            section_keys: Dict mapping config keys to (default_value, type) tuples
            
        Returns:
            Configuration section dictionary
        """
        config = {}
        
        for key, (default_value, value_type) in section_keys.items():
            config[key] = self._get_env_value(key, default_value, value_type)
        
        return config
    
    def get_stt_config(self) -> Dict[str, Any]:
        """Get STT engine configuration from environment variables."""
        stt_keys = {
            "stt_engine.model_size": ("base", str),
            "stt_engine.device": ("auto", str),
            "stt_engine.language": (None, str),
            "stt_engine.model_path": (None, str),
            "stt_engine.cache_dir": (None, str),
        }
        
        return self._build_config_section(stt_keys)
    
    def get_tts_config(self) -> Dict[str, Any]:
        """Get TTS engine configuration from environment variables."""
        tts_keys = {
            "tts_engine.device": ("auto", str),
            "tts_engine.model_path": (None, str),
            "tts_engine.temperature": (0.7, float),
            "tts_engine.speed_factor": (1.0, float),
            "tts_engine.voice_id": (None, str),
            "tts_engine.cache_dir": (None, str),
        }
        
        return self._build_config_section(tts_keys)
    
    def get_conversation_config(self) -> Dict[str, Any]:
        """Get conversation engine configuration from environment variables."""
        conversation_keys = {
            "conversation.response_mode": ("echo", str),
            "conversation.max_history_length": (50, int),
            "conversation.enable_middleware": (True, bool),
            "conversation.timeout_seconds": (30.0, float),
        }
        
        return self._build_config_section(conversation_keys)
    
    def get_pause_detection_config(self) -> Dict[str, Any]:
        """Get pause detection configuration from environment variables."""
        pause_keys = {
            "pause_detection.aggressiveness": (2, int),
            "pause_detection.min_speech_frames": (3, int),
            "pause_detection.min_pause_frames": (10, int),
            "pause_detection.frame_duration_ms": (30, int),
            "pause_detection.enable_webrtc": (True, bool),
        }
        
        return self._build_config_section(pause_keys)
    
    def get_audio_config(self) -> Dict[str, Any]:
        """Get audio processing configuration from environment variables."""
        audio_keys = {
            "audio.sample_rate": (16000, int),
            "audio.channels": (1, int),
            "audio.format": ("pcm", str),
            "audio.buffer_duration": (10.0, float),
            "audio.enable_normalization": (True, bool),
            "audio.target_level": (0.95, float),
        }
        
        return self._build_config_section(audio_keys)
    
    # Utility methods
    
    def get_value(self, key_path: str, default: Any = None, value_type: Type = str) -> Any:
        """
        Get arbitrary configuration value from environment variables.
        
        Args:
            key_path: Dot-separated configuration key path
            default: Default value if not found
            value_type: Type for value conversion
            
        Returns:
            Configuration value or default
        """
        return self._get_env_value(key_path, default, value_type)
    
    def list_env_vars(self) -> Dict[str, str]:
        """List all environment variables matching the prefix."""
        matching_vars = {}
        
        for key, value in os.environ.items():
            if key.startswith(self.prefix):
                matching_vars[key] = value
        
        return matching_vars
    
    def get_all_config(self) -> Dict[str, Any]:
        """Get complete configuration from all sections."""
        return {
            "stt_engine": self.get_stt_config(),
            "tts_engine": self.get_tts_config(),
            "conversation": self.get_conversation_config(),
            "pause_detection": self.get_pause_detection_config(),
            "audio": self.get_audio_config()
        }


# Factory functions

def create_env_provider(prefix: str = "CONV_") -> EnvConfigurationProvider:
    """Create environment configuration provider with custom prefix."""
    return EnvConfigurationProvider(prefix=prefix)


def create_docker_env_provider() -> EnvConfigurationProvider:
    """Create environment configuration provider optimized for Docker."""
    return EnvConfigurationProvider(
        prefix="REALTIME_CONV_", 
        separator="_",
        case_sensitive=False
    )


# Utility functions

def set_env_defaults() -> None:
    """Set default environment variables if not already set."""
    defaults = {
        "CONV_STT_ENGINE_MODEL_SIZE": "base",
        "CONV_STT_ENGINE_DEVICE": "auto",
        "CONV_TTS_ENGINE_DEVICE": "auto",
        "CONV_TTS_ENGINE_TEMPERATURE": "0.7",
        "CONV_CONVERSATION_RESPONSE_MODE": "echo",
        "CONV_PAUSE_DETECTION_AGGRESSIVENESS": "2",
        "CONV_AUDIO_SAMPLE_RATE": "16000",
    }
    
    for key, value in defaults.items():
        if key not in os.environ:
            os.environ[key] = value
    
    logger.debug(f"Set {len(defaults)} default environment variables")


def load_env_file(env_file_path: str) -> None:
    """
    Load environment variables from a .env file.
    
    Args:
        env_file_path: Path to .env file
    """
    try:
        with open(env_file_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    key = key.strip()
                    value = value.strip().strip('"').strip("'")
                    
                    # Only set if not already in environment
                    if key not in os.environ:
                        os.environ[key] = value
        
        logger.info(f"Environment variables loaded from: {env_file_path}")
        
    except Exception as e:
        logger.error(f"Error loading .env file {env_file_path}: {e}")