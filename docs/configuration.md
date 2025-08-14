# Configuration Management Documentation

## Overview

The STTS Server features a comprehensive configuration management system that bridges legacy YAML-based configuration with modern schema validation, environment variable support, and library integration patterns.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                Configuration Architecture                    │
├─────────────────────┬─────────────────────┬─────────────────┤
│   config.yaml      │  Environment Vars   │  Default Config │
│   (User Settings)  │  (Runtime Override) │  (Fallback)     │
└─────────────────────┼─────────────────────┼─────────────────┘
                      │                     │
                      ▼                     ▼
┌─────────────────────────────────────────────────────────────┐
│                YamlConfigManager                            │
│              (Thread-safe Access)                          │
├─────────────────────┬─────────────────────┬─────────────────┤
│     Validation     │    Schema Export    │   Hot Reload    │
│     & Locking      │    & Documentation  │   & Updates     │
└─────────────────────┼─────────────────────┴─────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│                Configuration Adapter                       │
│              (Library Bridge Pattern)                      │
├─────────────────────┬─────────────────────┬─────────────────┤
│    STT Config      │    TTS Config       │ Middleware      │
│    Mapping         │    Mapping          │ Config          │
└─────────────────────┴─────────────────────┴─────────────────┘
```

## Configuration Structure

### Complete Configuration Schema

```yaml
# config.yaml - Complete configuration example
tts_engine:
  device: "auto"                    # "auto", "cuda", "cpu", "mps"
  
stt_engine:
  model_size: "base"                # "tiny", "base", "small", "medium", "large"
  device: "auto"                    # "auto", "cuda", "cpu", "mps"
  language: "auto"                  # "auto", "en", "es", "fr", etc.

gen:
  default_temperature: 0.8          # TTS temperature (0.0-2.0)
  default_speed_factor: 1.0         # TTS speed multiplier (0.5-2.0)
  default_exaggeration: 0.5         # TTS exaggeration (0.0-1.0)
  default_cfg_weight: 0.5           # TTS CFG weight (0.0-1.0)
  default_seed: 0                   # TTS random seed
  remove_silence: true              # Remove silence from TTS output
  chunk_size: 200                   # Text chunking size

paths:
  reference_audio: "./reference_audio"  # Reference audio directory
  outputs: "./outputs"                  # Output audio directory
  voices: "./voices"                    # Predefined voices directory

middleware:
  timing:
    enabled: true                    # Enable timing middleware
    slow_request_threshold: 5.0      # Slow request threshold (seconds)
    enable_historical_tracking: true # Track historical performance
    max_history_entries: 1000       # Max history entries to keep
  
  logging:
    enabled: true                    # Enable logging middleware
    log_level: "INFO"               # Log level
    include_request_data: true      # Log request data
    include_response_data: false    # Log response data
    log_slow_requests: true         # Log slow requests
    slow_request_threshold: 3.0     # Slow request threshold for logging
  
  analytics:
    enabled: true                    # Enable analytics middleware
    enable_usage_tracking: true     # Track usage statistics
    enable_voice_analytics: true    # Track voice usage patterns
    enable_performance_tracking: true # Track performance metrics
    retention_days: 30              # Data retention period
    aggregation_interval: 3600      # Aggregation interval (seconds)

routers:
  core:
    tts:
      enable_middleware: true       # Enable TTS middleware
      enable_statistics: true      # Enable TTS statistics
      max_text_length: 5000        # Maximum text length
    stt:
      enable_adapters: true        # Enable STT adapters
      max_file_size_mb: 50         # Maximum file size
      supported_formats: ["wav", "mp3", "m4a", "flac"]
    conversation:
      enable_library_integration: true  # Enable library integration
      fallback_to_adapters: true       # Fallback to adapters
  
  management:
    config:
      enable_validation: true       # Enable config validation
      enable_schema_export: true   # Enable schema export
    files:
      enable_analytics: true       # Enable file analytics
      enable_cleanup: true         # Enable automatic cleanup
      cleanup_interval_hours: 24   # Cleanup interval
  
  websocket:
    enable_timing_info: true       # Include timing information
    max_connections: 100           # Maximum WebSocket connections
    heartbeat_interval: 30         # Heartbeat interval (seconds)

features:
  use_adapters: true               # Enable adapter pattern
  use_new_routers: true           # Use new router architecture
  library_integration: true       # Enable library integration
  enhanced_monitoring: true       # Enable enhanced monitoring

logging:
  level: "INFO"                    # Application log level
  format: "detailed"               # Log format
  file: null                       # Log file (null = console only)
  max_size_mb: 50                 # Max log file size
  backup_count: 5                 # Number of backup files

server:
  host: "0.0.0.0"                 # Server host
  port: 8004                      # Server port
  debug: false                    # Debug mode
  reload: false                   # Auto-reload on changes
  workers: 1                      # Number of workers
```

## Configuration Management

### YamlConfigManager

The core configuration manager provides thread-safe access to configuration data:

```python
from config import YamlConfigManager

# Get instance
config_manager = YamlConfigManager()

# Basic access
device = config_manager.get_string("tts_engine.device", "auto")
temperature = config_manager.get_float("gen.default_temperature", 0.8)
enabled = config_manager.get_bool("middleware.timing.enabled", True)

# Nested object access
middleware_config = config_manager.get_dict("middleware.timing", {})
supported_formats = config_manager.get_list("routers.core.stt.supported_formats", ["wav"])

# Update configuration
config_manager.update_config({
    "gen": {"default_temperature": 0.9},
    "middleware": {"analytics": {"enabled": False}}
})

# Validation
validation_result = config_manager.validate_config()
if not validation_result.is_valid:
    print(f"Configuration errors: {validation_result.errors}")
```

### Default Configuration

All configuration has sensible defaults defined in `config.py`:

```python
DEFAULT_CONFIG = {
    "tts_engine": {
        "device": "auto",  # Automatic device detection
    },
    "stt_engine": {
        "model_size": "base",  # Good balance of speed/accuracy
        "device": "auto",
        "language": "auto"
    },
    "gen": {
        "default_temperature": 0.8,  # Optimal for most voices
        "default_speed_factor": 1.0,
        "default_exaggeration": 0.5,
        "default_cfg_weight": 0.5,
        "default_seed": 0,
        "remove_silence": True,
        "chunk_size": 200
    },
    "middleware": {
        "timing": {"enabled": True},
        "logging": {"enabled": True},
        "analytics": {"enabled": True}
    },
    "features": {
        "use_adapters": True,
        "library_integration": True,
        "enhanced_monitoring": True
    }
}
```

### Environment Variable Support

Configuration can be overridden via environment variables:

```bash
# Override TTS device
export STTS_TTS_ENGINE_DEVICE="cuda"

# Override log level
export STTS_LOGGING_LEVEL="DEBUG"

# Disable middleware
export STTS_MIDDLEWARE_ANALYTICS_ENABLED="false"

# Override paths
export STTS_PATHS_REFERENCE_AUDIO="/custom/audio/path"
```

**Environment Variable Naming:**
- Prefix: `STTS_`
- Nested keys: Separated by underscores
- Case: UPPERCASE
- Examples:
  - `tts_engine.device` → `STTS_TTS_ENGINE_DEVICE`
  - `middleware.timing.enabled` → `STTS_MIDDLEWARE_TIMING_ENABLED`

## Configuration Validation

### Schema Validation

The configuration system includes comprehensive schema validation:

```python
# Validation result structure
@dataclass
class ConfigValidationResult:
    is_valid: bool
    errors: List[str]
    warnings: List[str]
    schema_version: str

# Perform validation
result = config_manager.validate_config()

if not result.is_valid:
    print("Configuration errors found:")
    for error in result.errors:
        print(f"  - {error}")

if result.warnings:
    print("Configuration warnings:")
    for warning in result.warnings:
        print(f"  - {warning}")
```

### Validation Rules

#### Device Validation
```python
# Valid device options
valid_devices = ["auto", "cuda", "cpu", "mps"]

# Device compatibility checks
if device == "cuda" and not torch.cuda.is_available():
    errors.append("CUDA device specified but not available")

if device == "mps" and not torch.backends.mps.is_available():
    errors.append("MPS device specified but not available")
```

#### Parameter Range Validation
```python
# Temperature validation
if not 0.0 <= temperature <= 2.0:
    errors.append(f"Temperature {temperature} outside valid range [0.0, 2.0]")

# Speed factor validation
if not 0.5 <= speed_factor <= 2.0:
    errors.append(f"Speed factor {speed_factor} outside valid range [0.5, 2.0]")

# File size validation
if max_file_size_mb > 500:
    warnings.append(f"Large max file size ({max_file_size_mb}MB) may cause memory issues")
```

#### Path Validation
```python
# Directory existence checks
for path_key, path_value in paths.items():
    if not os.path.exists(path_value):
        warnings.append(f"Path {path_key} does not exist: {path_value}")
    elif not os.path.isdir(path_value):
        errors.append(f"Path {path_key} is not a directory: {path_value}")
```

### Schema Export

The configuration schema can be exported for documentation and validation:

```python
# Get schema
schema = config_manager.get_schema()

# Schema structure
{
    "type": "object",
    "properties": {
        "tts_engine": {
            "type": "object",
            "properties": {
                "device": {
                    "type": "string",
                    "enum": ["auto", "cuda", "cpu", "mps"],
                    "default": "auto",
                    "description": "Device for TTS processing"
                },
                "temperature": {
                    "type": "number",
                    "minimum": 0.0,
                    "maximum": 2.0,
                    "default": 0.8,
                    "description": "TTS synthesis temperature"
                }
            }
        }
    }
}
```

## Configuration Adapters

### Library Integration Bridge

The configuration adapter bridges legacy config with library interfaces:

```python
class ConfigurationAdapter:
    """Bridge legacy configuration to library interfaces."""
    
    def __init__(self):
        self.config_manager = YamlConfigManager()
    
    def get_stt_config(self) -> Dict[str, Any]:
        """Get STT configuration for library integration."""
        return {
            "model_size": self.config_manager.get_string("stt_engine.model_size", "base"),
            "device": self.config_manager.get_string("stt_engine.device", "auto"),
            "language": self.config_manager.get_string("stt_engine.language", "auto"),
            "enable_timing": self.config_manager.get_bool("features.enhanced_monitoring", True)
        }
    
    def get_tts_config(self) -> Dict[str, Any]:
        """Get TTS configuration for library integration."""
        return {
            "device": self.config_manager.get_string("tts_engine.device", "auto"),
            "temperature": self.config_manager.get_float("gen.default_temperature", 0.8),
            "speed_factor": self.config_manager.get_float("gen.default_speed_factor", 1.0),
            "exaggeration": self.config_manager.get_float("gen.default_exaggeration", 0.5),
            "cfg_weight": self.config_manager.get_float("gen.default_cfg_weight", 0.5),
            "seed": self.config_manager.get_int("gen.default_seed", 0),
            "remove_silence": self.config_manager.get_bool("gen.remove_silence", True),
            "chunk_size": self.config_manager.get_int("gen.chunk_size", 200)
        }
    
    def get_middleware_config(self) -> Dict[str, Any]:
        """Get middleware configuration."""
        return {
            "timing": self.config_manager.get_dict("middleware.timing", {}),
            "logging": self.config_manager.get_dict("middleware.logging", {}),
            "analytics": self.config_manager.get_dict("middleware.analytics", {})
        }
```

### Adapter Usage in Components

```python
# In routers and adapters
config_adapter = ConfigurationAdapter()

# STT adapter configuration
stt_config = config_adapter.get_stt_config()
stt_adapter = LegacySTTEngineAdapter(
    model_size=stt_config["model_size"],
    device=stt_config["device"]
)

# TTS adapter configuration
tts_config = config_adapter.get_tts_config()
tts_adapter = LegacyTTSEngineAdapter(
    device=tts_config["device"],
    default_temperature=tts_config["temperature"]
)

# Middleware configuration
middleware_config = config_adapter.get_middleware_config()
pipeline = create_middleware_pipeline(middleware_config)
```

## API Integration

### Configuration Endpoints

#### Get Configuration
```http
GET /config
```

Returns current active configuration:
```json
{
  "status": "success",
  "data": {
    "tts_engine": {"device": "auto"},
    "stt_engine": {"model_size": "base"},
    "middleware": {...},
    "source": "config.yaml",
    "last_modified": "2024-01-01T12:00:00Z"
  }
}
```

#### Update Configuration
```http
POST /config/update
```

Update configuration with validation:
```json
{
  "tts_engine": {"device": "cuda"},
  "middleware": {"analytics": {"enabled": false}}
}
```

Response:
```json
{
  "status": "success",
  "data": {
    "message": "Configuration updated successfully",
    "updated_keys": ["tts_engine.device", "middleware.analytics.enabled"],
    "restart_required": false,
    "validation_result": {
      "valid": true,
      "warnings": ["CUDA device may not be available on all systems"]
    }
  }
}
```

#### Validate Configuration
```http
GET /config/validation
```

Returns validation status:
```json
{
  "status": "success",
  "data": {
    "valid": true,
    "errors": [],
    "warnings": [
      "tts_engine.temperature is higher than recommended (0.9 > 0.8)"
    ],
    "schema_version": "1.0.0",
    "validated_at": "2024-01-01T12:00:00Z"
  }
}
```

#### Configuration Schema
```http
GET /config/schema
```

Returns complete configuration schema:
```json
{
  "status": "success",
  "data": {
    "schema": {...},
    "version": "1.0.0",
    "documentation": {...},
    "examples": {...}
  }
}
```

## Dynamic Configuration

### Hot Reload

Some configuration changes apply immediately without restart:

```python
# Hot-reloadable settings
HOT_RELOAD_KEYS = {
    "gen.default_temperature",
    "gen.default_speed_factor", 
    "gen.remove_silence",
    "middleware.timing.enabled",
    "middleware.logging.log_level",
    "middleware.analytics.enabled"
}

# Check if restart required
def requires_restart(updated_keys: List[str]) -> bool:
    restart_keys = {
        "tts_engine.device",
        "stt_engine.device",
        "stt_engine.model_size",
        "server.host",
        "server.port"
    }
    return any(key in restart_keys for key in updated_keys)
```

### Runtime Configuration Updates

```python
# Update configuration at runtime
async def update_runtime_config(updates: Dict[str, Any]):
    """Update configuration with immediate effect where possible."""
    
    # Validate updates
    validation_result = config_manager.validate_partial_config(updates)
    if not validation_result.is_valid:
        raise ValueError(f"Invalid configuration: {validation_result.errors}")
    
    # Apply updates
    config_manager.update_config(updates)
    
    # Apply runtime changes
    updated_keys = []
    for key, value in flatten_dict(updates):
        if key in HOT_RELOAD_KEYS:
            apply_runtime_update(key, value)
            updated_keys.append(key)
    
    return {
        "applied_immediately": updated_keys,
        "requires_restart": [k for k in flatten_dict(updates) if k not in HOT_RELOAD_KEYS]
    }
```

## Environment-Specific Configuration

### Configuration Profiles

Support for different environments:

```yaml
# config-development.yaml
server:
  debug: true
  reload: true
logging:
  level: "DEBUG"
middleware:
  timing:
    enable_historical_tracking: false

# config-production.yaml  
server:
  debug: false
  workers: 4
logging:
  level: "INFO"
  file: "/var/log/stts/server.log"
middleware:
  analytics:
    retention_days: 90
```

### Profile Loading

```python
# Load environment-specific configuration
def load_config_for_environment(env: str = None):
    """Load configuration for specific environment."""
    env = env or os.getenv("STTS_ENVIRONMENT", "development")
    
    # Load base configuration
    config = load_default_config()
    
    # Load environment-specific overrides
    env_config_path = f"config-{env}.yaml"
    if os.path.exists(env_config_path):
        env_config = load_yaml_config(env_config_path)
        config = deep_merge(config, env_config)
    
    # Apply environment variable overrides
    config = apply_env_overrides(config)
    
    return config
```

## Configuration Best Practices

### 1. **Environment Separation**

```yaml
# Use environment-specific configurations
development:
  debug: true
  log_level: "DEBUG"
  
production:
  debug: false
  log_level: "INFO"
  workers: 4
```

### 2. **Sensitive Data**

```yaml
# Don't store secrets in configuration files
# Use environment variables instead
database:
  host: "localhost"
  port: 5432
  # Don't do this:
  # password: "secret123"
  # Do this instead:
  password: "${DATABASE_PASSWORD}"  # Reads from env var
```

### 3. **Validation**

```python
# Always validate configuration on startup
config_manager = YamlConfigManager()
validation_result = config_manager.validate_config()

if not validation_result.is_valid:
    logger.error("Configuration validation failed:")
    for error in validation_result.errors:
        logger.error(f"  - {error}")
    sys.exit(1)

if validation_result.warnings:
    logger.warning("Configuration warnings:")
    for warning in validation_result.warnings:
        logger.warning(f"  - {warning}")
```

### 4. **Documentation**

```python
# Document configuration options
CONFIG_SCHEMA = {
    "tts_engine": {
        "device": {
            "type": "string",
            "description": "Device for TTS processing",
            "enum": ["auto", "cuda", "cpu", "mps"],
            "default": "auto",
            "examples": ["cuda", "cpu"]
        }
    }
}
```

## Troubleshooting

### Common Configuration Issues

#### 1. **Device Conflicts**
```yaml
# Problem: Specifying unavailable device
tts_engine:
  device: "cuda"  # But CUDA not available

# Solution: Use auto-detection
tts_engine:
  device: "auto"  # Automatically selects best available
```

#### 2. **Path Issues**
```yaml
# Problem: Invalid paths
paths:
  reference_audio: "/nonexistent/path"

# Solution: Use relative paths or validate existence
paths:
  reference_audio: "./reference_audio"  # Relative to project root
```

#### 3. **Memory Issues**
```yaml
# Problem: Settings too high for system
routers:
  core:
    stt:
      max_file_size_mb: 1000  # Too large for 8GB system

# Solution: Adjust based on available memory
routers:
  core:
    stt:
      max_file_size_mb: 50  # Reasonable for most systems
```

### Configuration Debugging

```python
# Debug configuration loading
def debug_config_loading():
    """Print detailed configuration loading information."""
    print("Configuration loading debug:")
    
    # Show default config
    print(f"Default config keys: {list(DEFAULT_CONFIG.keys())}")
    
    # Show loaded config file
    config_path = "config.yaml"
    if os.path.exists(config_path):
        print(f"Loaded config file: {config_path}")
        with open(config_path) as f:
            config_data = yaml.safe_load(f)
        print(f"Config file keys: {list(config_data.keys())}")
    else:
        print("No config.yaml found, using defaults")
    
    # Show environment overrides
    env_vars = {k: v for k, v in os.environ.items() if k.startswith("STTS_")}
    if env_vars:
        print(f"Environment overrides: {list(env_vars.keys())}")
    
    # Show final configuration
    config_manager = YamlConfigManager()
    final_config = config_manager.get_dict("", {})
    print(f"Final config keys: {list(final_config.keys())}")
```

### Migration Guide

#### From Legacy Configuration

```python
# Old style (direct access)
import config
temperature = config.DEFAULT_CONFIG["gen"]["default_temperature"]

# New style (managed access)
from config import YamlConfigManager
config_manager = YamlConfigManager()
temperature = config_manager.get_float("gen.default_temperature", 0.8)
```

#### Adding New Configuration Options

```python
# 1. Add to DEFAULT_CONFIG
DEFAULT_CONFIG["new_feature"] = {
    "enabled": True,
    "option1": "default_value"
}

# 2. Add validation rules
def validate_new_feature(config_data):
    new_feature = config_data.get("new_feature", {})
    if not isinstance(new_feature.get("enabled"), bool):
        return ["new_feature.enabled must be boolean"]
    return []

# 3. Add accessor functions
def get_new_feature_enabled() -> bool:
    return config_manager.get_bool("new_feature.enabled", True)

# 4. Update schema
SCHEMA["properties"]["new_feature"] = {
    "type": "object",
    "properties": {
        "enabled": {"type": "boolean", "default": True}
    }
}
```