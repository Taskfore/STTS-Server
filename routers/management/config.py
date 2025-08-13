# File: routers/management/config.py
# Configuration management endpoints

import logging
from typing import Dict, Any

from fastapi import APIRouter, HTTPException, Request

from config import config_manager
from models import UpdateStatusResponse

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/config", tags=["Configuration Management"])


@router.post("/save", response_model=UpdateStatusResponse)
async def save_settings(request: Request):
    """
    Saves partial configuration updates to the config.yaml file.
    Merges the update with the current configuration.
    """
    logger.info("Request received for /config/save.")
    try:
        partial_update = await request.json()
        if not isinstance(partial_update, dict):
            raise ValueError("Request body must be a JSON object for config save.")
        logger.debug(f"Received partial config data to save: {partial_update}")

        if config_manager.update_and_save(partial_update):
            restart_needed = any(
                key in partial_update
                for key in ["server", "tts_engine", "paths", "model"]
            )
            message = "Settings saved successfully."
            if restart_needed:
                message += " A server restart may be required for some changes to take full effect."
            return UpdateStatusResponse(message=message, restart_needed=restart_needed)
        else:
            logger.error("Failed to save configuration via config_manager.update_and_save.")
            raise HTTPException(
                status_code=500,
                detail="Failed to save configuration file due to an internal error.",
            )
    except ValueError as ve:
        logger.error(f"Invalid data format for config save: {ve}")
        raise HTTPException(status_code=400, detail=f"Invalid request data: {str(ve)}")
    except Exception as e:
        logger.error(f"Error processing config save request: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error during settings save: {str(e)}",
        )


@router.post("/reset", response_model=UpdateStatusResponse)
async def reset_settings():
    """Resets the configuration in config.yaml back to hardcoded defaults."""
    logger.warning("Request received to reset all configurations to default values.")
    try:
        if config_manager.reset_and_save():
            logger.info("Configuration successfully reset to defaults and saved.")
            return UpdateStatusResponse(
                message="Configuration reset to defaults. Please reload the page. A server restart may be beneficial.",
                restart_needed=True,
            )
        else:
            logger.error("Failed to reset and save configuration via config_manager.")
            raise HTTPException(
                status_code=500, detail="Failed to reset and save configuration file."
            )
    except Exception as e:
        logger.error(f"Error processing config reset request: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error during settings reset: {str(e)}",
        )


@router.get("/current")
async def get_current_config():
    """Get the current configuration."""
    try:
        from config import get_full_config_for_template
        return get_full_config_for_template()
    except Exception as e:
        logger.error(f"Error getting current config: {e}", exc_info=True)
        raise HTTPException(
            status_code=500, 
            detail="Failed to retrieve current configuration"
        )


@router.get("/defaults")
async def get_default_config():
    """Get the default configuration values."""
    try:
        from config import DEFAULT_CONFIG
        return DEFAULT_CONFIG
    except Exception as e:
        logger.error(f"Error getting default config: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="Failed to retrieve default configuration"
        )


@router.post("/restart", response_model=UpdateStatusResponse)
async def restart_server():
    """Attempts to trigger a server restart."""
    logger.info("Request received for server restart.")
    message = (
        "Server restart initiated. If running locally without a process manager, "
        "you may need to restart manually. For managed environments (Docker, systemd), "
        "the manager should handle the restart."
    )
    logger.warning(message)
    return UpdateStatusResponse(message=message, restart_needed=True)


@router.get("/validation")
async def validate_config():
    """Validate the current configuration and report any issues."""
    try:
        validation_results = {
            "valid": True,
            "errors": [],
            "warnings": [],
            "info": []
        }
        
        # Check TTS engine configuration
        try:
            tts_device = config_manager.get_string("tts_engine.device", "auto")
            if tts_device not in ["auto", "cuda", "mps", "cpu"]:
                validation_results["errors"].append(f"Invalid TTS device: {tts_device}")
                validation_results["valid"] = False
        except Exception as e:
            validation_results["errors"].append(f"TTS configuration error: {str(e)}")
            validation_results["valid"] = False
        
        # Check STT engine configuration  
        try:
            stt_device = config_manager.get_string("stt_engine.device", "auto")
            stt_model = config_manager.get_string("stt_engine.model_size", "base")
            if stt_device not in ["auto", "cuda", "mps", "cpu"]:
                validation_results["errors"].append(f"Invalid STT device: {stt_device}")
                validation_results["valid"] = False
            if stt_model not in ["tiny", "base", "small", "medium", "large"]:
                validation_results["warnings"].append(f"Unusual STT model size: {stt_model}")
        except Exception as e:
            validation_results["errors"].append(f"STT configuration error: {str(e)}")
            validation_results["valid"] = False
        
        # Check path configurations
        try:
            from config import get_output_path, get_reference_audio_path, get_predefined_voices_path
            
            paths_to_check = [
                ("output", get_output_path()),
                ("reference_audio", get_reference_audio_path()),
                ("predefined_voices", get_predefined_voices_path())
            ]
            
            for path_name, path_obj in paths_to_check:
                if not path_obj.exists():
                    validation_results["warnings"].append(f"{path_name} directory does not exist: {path_obj}")
                elif not path_obj.is_dir():
                    validation_results["errors"].append(f"{path_name} path is not a directory: {path_obj}")
                    validation_results["valid"] = False
                    
        except Exception as e:
            validation_results["errors"].append(f"Path configuration error: {str(e)}")
            validation_results["valid"] = False
        
        # Check audio settings
        try:
            sample_rate = config_manager.get_int("audio.sample_rate", 16000)
            if sample_rate not in [8000, 16000, 22050, 44100, 48000]:
                validation_results["warnings"].append(f"Unusual sample rate: {sample_rate}")
                
            output_format = config_manager.get_string("audio_output.format", "wav")
            if output_format not in ["wav", "opus", "mp3"]:
                validation_results["errors"].append(f"Invalid output format: {output_format}")
                validation_results["valid"] = False
        except Exception as e:
            validation_results["errors"].append(f"Audio configuration error: {str(e)}")
            validation_results["valid"] = False
        
        # Add summary info
        validation_results["info"].append(f"Configuration file location: {config_manager.config_file}")
        validation_results["info"].append(f"Total errors: {len(validation_results['errors'])}")
        validation_results["info"].append(f"Total warnings: {len(validation_results['warnings'])}")
        
        return validation_results
        
    except Exception as e:
        logger.error(f"Error during config validation: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="Failed to validate configuration"
        )


@router.get("/schema")
async def get_config_schema():
    """Get the configuration schema/structure."""
    try:
        schema = {
            "server": {
                "description": "Server configuration",
                "properties": {
                    "host": {"type": "string", "default": "127.0.0.1"},
                    "port": {"type": "integer", "default": 8004},
                    "log_level": {"type": "string", "default": "INFO"},
                    "enable_performance_monitor": {"type": "boolean", "default": False}
                }
            },
            "tts_engine": {
                "description": "Text-to-Speech engine configuration",
                "properties": {
                    "device": {"type": "string", "enum": ["auto", "cuda", "mps", "cpu"], "default": "auto"},
                    "model_cache_path": {"type": "string", "default": "./model_cache"}
                }
            },
            "stt_engine": {
                "description": "Speech-to-Text engine configuration", 
                "properties": {
                    "device": {"type": "string", "enum": ["auto", "cuda", "mps", "cpu"], "default": "auto"},
                    "model_size": {"type": "string", "enum": ["tiny", "base", "small", "medium", "large"], "default": "base"},
                    "language": {"type": "string", "default": "auto"}
                }
            },
            "gen": {
                "description": "Generation parameters",
                "properties": {
                    "default_temperature": {"type": "number", "default": 0.8, "min": 0.0, "max": 2.0},
                    "default_exaggeration": {"type": "number", "default": 0.5, "min": 0.0, "max": 2.0},
                    "default_cfg_weight": {"type": "number", "default": 0.5, "min": 0.0, "max": 1.0},
                    "default_speed_factor": {"type": "number", "default": 1.0, "min": 0.1, "max": 3.0},
                    "default_seed": {"type": "integer", "default": 0}
                }
            },
            "audio": {
                "description": "Audio processing configuration",
                "properties": {
                    "sample_rate": {"type": "integer", "enum": [8000, 16000, 22050, 44100, 48000], "default": 16000}
                }
            },
            "audio_output": {
                "description": "Audio output configuration",
                "properties": {
                    "format": {"type": "string", "enum": ["wav", "opus", "mp3"], "default": "wav"},
                    "max_reference_duration_sec": {"type": "integer", "default": 30}
                }
            },
            "audio_processing": {
                "description": "Audio processing options",
                "properties": {
                    "enable_silence_trimming": {"type": "boolean", "default": False},
                    "enable_internal_silence_fix": {"type": "boolean", "default": False},
                    "enable_unvoiced_removal": {"type": "boolean", "default": False}
                }
            },
            "paths": {
                "description": "File system paths",
                "properties": {
                    "output": {"type": "string", "default": "./outputs"},
                    "reference_audio": {"type": "string", "default": "./reference_audio"},
                    "predefined_voices": {"type": "string", "default": "./voices"},
                    "model_cache": {"type": "string", "default": "./model_cache"}
                }
            }
        }
        
        return schema
        
    except Exception as e:
        logger.error(f"Error getting config schema: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="Failed to retrieve configuration schema"
        )