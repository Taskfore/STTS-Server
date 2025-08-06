"""
Dictionary-based configuration provider.

Provides configuration from Python dictionaries, useful for testing
and programmatic configuration.
"""

import logging
from typing import Dict, Any
from ...core.interfaces import ConfigurationProvider

logger = logging.getLogger(__name__)


class DictConfigurationProvider(ConfigurationProvider):
    """Dictionary-based configuration provider."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize dictionary configuration provider.
        
        Args:
            config: Configuration dictionary
        """
        self._config = config.copy() if config else {}
        logger.debug(f"Dictionary configuration initialized with {len(self._config)} keys")
    
    def _get_nested_value(
        self, 
        key_path: str, 
        default: Any = None
    ) -> Any:
        """Get value from nested configuration using dot notation."""
        keys = key_path.split('.')
        current = self._config
        
        try:
            for key in keys:
                if isinstance(current, dict) and key in current:
                    current = current[key]
                else:
                    return default
            return current
        except (KeyError, TypeError):
            return default
    
    def get_stt_config(self) -> Dict[str, Any]:
        """Get STT engine configuration."""
        return self._get_nested_value("stt_engine", {})
    
    def get_tts_config(self) -> Dict[str, Any]:
        """Get TTS engine configuration."""
        return self._get_nested_value("tts_engine", {})
    
    def get_conversation_config(self) -> Dict[str, Any]:
        """Get conversation engine configuration."""
        return self._get_nested_value("conversation", {})
    
    def get_pause_detection_config(self) -> Dict[str, Any]:
        """Get pause detection configuration."""
        return self._get_nested_value("pause_detection", {})
    
    def get_audio_config(self) -> Dict[str, Any]:
        """Get audio processing configuration."""
        return self._get_nested_value("audio", {})
    
    # Additional methods for programmatic configuration
    
    def update_config(self, config: Dict[str, Any]) -> None:
        """Update configuration with new values."""
        self._config.update(config)
        logger.debug("Configuration updated")
    
    def set_value(self, key_path: str, value: Any) -> None:
        """
        Set configuration value using dot notation.
        
        Args:
            key_path: Dot-separated key path
            value: Value to set
        """
        keys = key_path.split('.')
        current = self._config
        
        # Navigate to parent of target key
        for key in keys[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]
        
        # Set the final value
        current[keys[-1]] = value
        logger.debug(f"Set config value: {key_path} = {value}")
    
    def get_value(self, key_path: str, default: Any = None) -> Any:
        """
        Get configuration value using dot notation.
        
        Args:
            key_path: Dot-separated key path
            default: Default value if key not found
            
        Returns:
            Configuration value or default
        """
        return self._get_nested_value(key_path, default)
    
    def get_all_config(self) -> Dict[str, Any]:
        """Get complete configuration dictionary."""
        return self._config.copy()
    
    def clear_config(self) -> None:
        """Clear all configuration."""
        self._config.clear()
        logger.debug("Configuration cleared")


# Factory functions for common configurations

def create_default_config() -> "DictConfigurationProvider":
    """Create configuration provider with sensible defaults."""
    default_config = {
        "stt_engine": {
            "model_size": "base",
            "device": "auto",
            "language": None
        },
        "tts_engine": {
            "device": "auto",
            "temperature": 0.7,
            "speed_factor": 1.0
        },
        "conversation": {
            "response_mode": "echo",
            "max_history_length": 50
        },
        "pause_detection": {
            "aggressiveness": 2,
            "min_speech_frames": 3,
            "min_pause_frames": 10
        },
        "audio": {
            "sample_rate": 16000,
            "channels": 1,
            "format": "pcm",
            "buffer_duration": 10.0
        },
        "websocket": {
            "max_message_size": 1024 * 1024,  # 1MB
            "ping_interval": 20,
            "ping_timeout": 10
        }
    }
    
    return DictConfigurationProvider(default_config)


def create_test_config() -> DictConfigurationProvider:
    """Create configuration provider optimized for testing."""
    test_config = {
        "stt_engine": {
            "model_size": "tiny",
            "device": "cpu",
            "language": "en"
        },
        "tts_engine": {
            "device": "cpu",
            "temperature": 0.1,
            "speed_factor": 2.0  # Faster for testing
        },
        "conversation": {
            "response_mode": "echo",
            "max_history_length": 10
        },
        "pause_detection": {
            "aggressiveness": 1,
            "min_speech_frames": 1,
            "min_pause_frames": 5
        },
        "audio": {
            "sample_rate": 8000,  # Lower quality for testing
            "channels": 1,
            "format": "pcm",
            "buffer_duration": 5.0
        }
    }
    
    return DictConfigurationProvider(test_config)


def create_performance_config() -> DictConfigurationProvider:
    """Create configuration provider optimized for performance."""
    performance_config = {
        "stt_engine": {
            "model_size": "small",
            "device": "cuda",
            "language": None
        },
        "tts_engine": {
            "device": "cuda",
            "temperature": 0.7,
            "speed_factor": 1.0
        },
        "conversation": {
            "response_mode": "template",
            "max_history_length": 25
        },
        "pause_detection": {
            "aggressiveness": 3,  # More aggressive for faster detection
            "min_speech_frames": 2,
            "min_pause_frames": 8
        },
        "audio": {
            "sample_rate": 16000,
            "channels": 1,
            "format": "pcm",
            "buffer_duration": 8.0
        }
    }
    
    return DictConfigurationProvider(performance_config)