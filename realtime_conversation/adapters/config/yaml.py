"""
YAML-based configuration provider.

Provides configuration from YAML files with optional environment variable substitution.
"""

import logging
from typing import Dict, Any, Optional
from pathlib import Path
import yaml
import os
from ...core.interfaces import ConfigurationProvider

logger = logging.getLogger(__name__)


class YAMLConfigurationProvider(ConfigurationProvider):
    """YAML-based configuration provider."""
    
    def __init__(
        self, 
        config_path: Path, 
        env_substitution: bool = True,
        auto_reload: bool = False
    ):
        """
        Initialize YAML configuration provider.
        
        Args:
            config_path: Path to YAML configuration file
            env_substitution: Enable environment variable substitution
            auto_reload: Automatically reload config when file changes (not implemented)
        """
        self.config_path = Path(config_path)
        self.env_substitution = env_substitution
        self.auto_reload = auto_reload
        self._config: Dict[str, Any] = {}
        self._file_mtime: Optional[float] = None
        
        # Load initial configuration
        self._load_config()
        
        logger.info(f"YAML configuration loaded from: {self.config_path}")
    
    def _load_config(self) -> None:
        """Load configuration from YAML file."""
        try:
            if not self.config_path.exists():
                logger.warning(f"Configuration file not found: {self.config_path}")
                self._config = {}
                return
            
            # Check if file has changed
            current_mtime = self.config_path.stat().st_mtime
            if self._file_mtime == current_mtime and self._config:
                return  # No changes, skip reload
            
            self._file_mtime = current_mtime
            
            with open(self.config_path, 'r', encoding='utf-8') as f:
                raw_config = yaml.safe_load(f) or {}
            
            # Apply environment variable substitution
            if self.env_substitution:
                self._config = self._substitute_env_vars(raw_config)
            else:
                self._config = raw_config
            
            logger.debug(f"Configuration loaded: {len(self._config)} top-level keys")
            
        except Exception as e:
            logger.error(f"Error loading configuration from {self.config_path}: {e}")
            self._config = {}
    
    def _substitute_env_vars(self, config: Any) -> Any:
        """Recursively substitute environment variables in configuration."""
        if isinstance(config, dict):
            return {key: self._substitute_env_vars(value) for key, value in config.items()}
        elif isinstance(config, list):
            return [self._substitute_env_vars(item) for item in config]
        elif isinstance(config, str):
            # Simple environment variable substitution: ${VAR_NAME} or ${VAR_NAME:default}
            import re
            
            def replace_env_var(match):
                var_expr = match.group(1)
                if ':' in var_expr:
                    var_name, default_value = var_expr.split(':', 1)
                    return os.environ.get(var_name, default_value)
                else:
                    return os.environ.get(var_expr, match.group(0))  # Return original if not found
            
            return re.sub(r'\$\{([^}]+)\}', replace_env_var, config)
        else:
            return config
    
    def _get_nested_value(
        self, 
        config: Dict[str, Any], 
        key_path: str, 
        default: Any = None
    ) -> Any:
        """Get value from nested configuration using dot notation."""
        if self.auto_reload:
            self._load_config()
        
        keys = key_path.split('.')
        current = config
        
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
        return self._get_nested_value(self._config, "stt_engine", {})
    
    def get_tts_config(self) -> Dict[str, Any]:
        """Get TTS engine configuration."""
        return self._get_nested_value(self._config, "tts_engine", {})
    
    def get_conversation_config(self) -> Dict[str, Any]:
        """Get conversation engine configuration."""
        return self._get_nested_value(self._config, "conversation", {})
    
    def get_pause_detection_config(self) -> Dict[str, Any]:
        """Get pause detection configuration."""
        return self._get_nested_value(self._config, "pause_detection", {})
    
    def get_audio_config(self) -> Dict[str, Any]:
        """Get audio processing configuration."""
        return self._get_nested_value(self._config, "audio", {})
    
    # Additional configuration getters
    
    def get_websocket_config(self) -> Dict[str, Any]:
        """Get WebSocket configuration."""
        return self._get_nested_value(self._config, "websocket", {})
    
    def get_middleware_config(self) -> Dict[str, Any]:
        """Get middleware configuration."""
        return self._get_nested_value(self._config, "middleware", {})
    
    def get_logging_config(self) -> Dict[str, Any]:
        """Get logging configuration."""
        return self._get_nested_value(self._config, "logging", {})
    
    def get_value(self, key_path: str, default: Any = None) -> Any:
        """
        Get arbitrary configuration value using dot notation.
        
        Args:
            key_path: Dot-separated key path (e.g., "tts_engine.model_path")
            default: Default value if key not found
            
        Returns:
            Configuration value or default
        """
        return self._get_nested_value(self._config, key_path, default)
    
    def get_all_config(self) -> Dict[str, Any]:
        """Get complete configuration dictionary."""
        if self.auto_reload:
            self._load_config()
        return self._config.copy()
    
    def reload_config(self) -> None:
        """Manually reload configuration from file."""
        self._file_mtime = None  # Force reload
        self._load_config()
        logger.info("Configuration manually reloaded")
    
    @property
    def config_exists(self) -> bool:
        """Check if configuration file exists."""
        return self.config_path.exists()
    
    @property
    def last_modified(self) -> Optional[float]:
        """Get last modification time of configuration file."""
        return self._file_mtime


# Utility functions for working with YAML configuration

def load_yaml_config(config_path: Path) -> Dict[str, Any]:
    """
    Simple utility to load YAML configuration without provider overhead.
    
    Args:
        config_path: Path to YAML file
        
    Returns:
        Configuration dictionary
    """
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f) or {}
    except Exception as e:
        logger.error(f"Error loading YAML config from {config_path}: {e}")
        return {}


def save_yaml_config(config: Dict[str, Any], config_path: Path) -> bool:
    """
    Save configuration dictionary to YAML file.
    
    Args:
        config: Configuration dictionary to save
        config_path: Output YAML file path
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # Ensure parent directory exists
        config_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.safe_dump(
                config,
                f,
                default_flow_style=False,
                sort_keys=False,
                indent=2
            )
        
        logger.info(f"Configuration saved to: {config_path}")
        return True
        
    except Exception as e:
        logger.error(f"Error saving YAML config to {config_path}: {e}")
        return False


def merge_yaml_configs(base_config: Dict[str, Any], override_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Merge two configuration dictionaries with deep merging.
    
    Args:
        base_config: Base configuration
        override_config: Configuration to merge on top
        
    Returns:
        Merged configuration dictionary
    """
    def deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        result = base.copy()
        
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = deep_merge(result[key], value)
            else:
                result[key] = value
        
        return result
    
    return deep_merge(base_config, override_config)