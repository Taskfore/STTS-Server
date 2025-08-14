"""
WebSocket Conversation Router v2 - Using the new conversation library.

This is the modernized version of the websocket conversation router that uses
the new framework-agnostic conversation library instead of the tightly-coupled
original implementation.
"""

import asyncio
import logging
from typing import Optional
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Query, Request, HTTPException

# Import the new conversation library
from realtime_conversation import ConversationEngine
from realtime_conversation.adapters.websocket import FastAPIWebSocketAdapter
from realtime_conversation.adapters.stt import WhisperSTTEngine
from realtime_conversation.adapters.tts import ChatterboxTTSEngine
from realtime_conversation.adapters.config import YAMLConfigurationProvider
from realtime_conversation.plugins.pause_detection import WebRTCPauseDetector, EnergyPauseDetector
from realtime_conversation.plugins.response_generation import EchoResponseGenerator, TemplateResponseGenerator
from realtime_conversation.plugins.middleware import LoggingMiddleware, TimingMiddleware

# Import existing config system for compatibility
from config import (
    get_default_voice_id,
    get_predefined_voices_path,
    get_reference_audio_path,
    config_manager
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/ws", tags=["WebSocket Conversation v2"])

# Global conversation engine (initialized on first use)
_conversation_engine: Optional[ConversationEngine] = None
_engine_lock = asyncio.Lock()


async def get_conversation_engine() -> ConversationEngine:
    """
    Get or create the conversation engine singleton.
    
    This approach maintains compatibility with the existing app structure
    while using the new library.
    """
    global _conversation_engine
    
    if _conversation_engine is None:
        async with _engine_lock:
            if _conversation_engine is None:  # Double-check locking
                _conversation_engine = await _create_conversation_engine()
    
    return _conversation_engine


async def _create_conversation_engine() -> ConversationEngine:
    """Create and configure the conversation engine using existing config."""
    try:
        # Create configuration provider that bridges to existing config
        config_data = {
            "stt_engine": {
                "model_size": config_manager.get_string("stt_engine.model_size", "base"),
                "device": config_manager.get_string("stt_engine.device", "auto"),
                "language": config_manager.get_string("stt_engine.language", "auto")
            },
            "tts_engine": {
                "device": config_manager.get_string("tts_engine.device", "auto"),
                "temperature": config_manager.get_float("gen.default_temperature", 0.8),
                "speed_factor": config_manager.get_float("gen.default_speed_factor", 1.0)
            },
            "conversation": {
                "response_mode": "echo",  # Default for backward compatibility
                "max_history_length": 50
            },
            "pause_detection": {
                "aggressiveness": 2,
                "min_speech_frames": 3,
                "min_pause_frames": 10,
                "sample_rate": 16000
            },
            "audio": {
                "sample_rate": config_manager.get_int("audio.sample_rate", 16000),
                "channels": 1,
                "format": "pcm"
            }
        }
        
        from realtime_conversation.adapters.config import DictConfigurationProvider
        config_provider = DictConfigurationProvider(config_data)
        
        # Create conversation engine
        engine = ConversationEngine(config_provider)
        
        # Configure STT engine
        stt_config = config_provider.get_stt_config()
        stt_engine = WhisperSTTEngine(
            model_size=stt_config.get("model_size", "base"),
            device=stt_config.get("device", "auto")
        )
        
        if not stt_engine.load_model():
            raise RuntimeError("Failed to load STT model")
        
        engine.configure_stt(stt_engine)
        logger.info("STT engine configured successfully")
        
        # Configure TTS engine
        tts_config = config_provider.get_tts_config()
        tts_engine = ChatterboxTTSEngine(
            device=tts_config.get("device", "auto")
        )
        
        if not tts_engine.load_model():
            raise RuntimeError("Failed to load TTS model")
        
        engine.configure_tts(tts_engine)
        logger.info("TTS engine configured successfully")
        
        # Configure pause detection
        pause_config = config_provider.get_pause_detection_config()
        try:
            pause_detector = WebRTCPauseDetector(
                aggressiveness=pause_config.get("aggressiveness", 2),
                min_speech_frames=pause_config.get("min_speech_frames", 3),
                min_pause_frames=pause_config.get("min_pause_frames", 10),
                sample_rate=pause_config.get("sample_rate", 16000)
            )
            logger.info("WebRTC VAD configured successfully")
        except ImportError:
            logger.warning("WebRTC VAD not available, using energy-based fallback")
            pause_detector = EnergyPauseDetector(
                sample_rate=pause_config.get("sample_rate", 16000)
            )
        
        engine.configure_pause_detection(pause_detector)
        
        # Configure response generation (default to echo for compatibility)
        response_generator = EchoResponseGenerator()
        engine.configure_response_generation(response_generator)
        logger.info("Response generation configured successfully")
        
        # Add middleware
        engine.add_middleware(TimingMiddleware())
        engine.add_middleware(LoggingMiddleware(log_level=logging.INFO))
        
        logger.info("Conversation engine v2 initialized successfully")
        return engine
        
    except Exception as e:
        logger.error(f"Failed to create conversation engine v2: {e}")
        raise


class ConversationAdapterV2:
    """
    Adapter class to bridge the new conversation engine with the legacy interface.
    
    This provides backward compatibility for any code that expects the old
    ConversationProcessor interface.
    """
    
    def __init__(self, engine: ConversationEngine, websocket: WebSocket):
        self.engine = engine
        self.websocket = websocket
        self.adapter = FastAPIWebSocketAdapter(websocket)
        
        # Legacy compatibility properties
        self.conversation_state = "listening"
        
    async def handle_conversation(self) -> None:
        """Handle the complete conversation using the new engine."""
        try:
            await self.engine.handle_conversation(self.adapter)
        except Exception as e:
            logger.error(f"Conversation handling error: {e}")
            raise
    
    def reset_conversation(self) -> None:
        """Reset conversation state (legacy compatibility)."""
        # The new engine handles state internally
        logger.info("Conversation reset requested")


@router.websocket("/conversation/v2")
async def websocket_conversation_v2(
    websocket: WebSocket,
    language: Optional[str] = Query(
        None, description="STT language (auto-detect if None)"
    ),
    voice_mode: str = Query("predefined", description="TTS voice mode"),
    predefined_voice_id: Optional[str] = Query(None, description="Predefined voice ID"),
    reference_audio_filename: Optional[str] = Query(
        None, description="Reference audio filename"
    ),
    response_mode: str = Query("echo", description="Response generation mode"),
    pause_aggressiveness: int = Query(
        2, ge=0, le=3, description="Pause detection aggressiveness"
    ),
):
    """
    Real-time audio conversation via WebSocket - Version 2 using the new library.
    
    This endpoint provides the same functionality as the original conversation
    endpoint but uses the new modular conversation library architecture.
    
    Features:
    - Framework-agnostic conversation engine
    - Pluggable middleware system
    - Better error handling and logging
    - Configurable pause detection
    - Extensible response generation
    
    Expected audio format: 16-bit PCM, 16kHz, mono
    """
    logger.info(f"WebSocket conversation v2 connection attempt from {websocket.client}")
    
    try:
        # Get the conversation engine
        engine = await get_conversation_engine()
        
        # Create WebSocket adapter
        adapter = FastAPIWebSocketAdapter(websocket)
        
        # Handle conversation
        await engine.handle_conversation(adapter)
        
    except Exception as e:
        logger.error(f"WebSocket conversation v2 error: {e}")
        
        # Try to send error message if connection is still open
        try:
            if not websocket.client_state.disconnected:
                await websocket.send_json({
                    "type": "error",
                    "message": f"Conversation error: {str(e)}"
                })
                await websocket.close()
        except:
            pass  # Connection might already be closed
    
    finally:
        logger.info("WebSocket conversation v2 session ended")


@router.websocket("/conversation/custom")
async def websocket_conversation_custom(
    websocket: WebSocket,
    response_mode: str = Query("template", description="Response mode: echo, template"),
    enable_analytics: bool = Query(False, description="Enable analytics middleware"),
    log_level: str = Query("INFO", description="Logging level: DEBUG, INFO, WARNING, ERROR")
):
    """
    Custom conversation endpoint that demonstrates dynamic configuration.
    
    This endpoint shows how to create a conversation engine with custom
    configuration on a per-connection basis.
    """
    logger.info(f"Custom WebSocket conversation connection from {websocket.client}")
    
    try:
        # Create custom configuration
        config_data = {
            "stt_engine": {
                "model_size": "small",  # Better quality
                "device": "auto"
            },
            "tts_engine": {
                "device": "auto",
                "temperature": 0.7
            },
            "conversation": {
                "response_mode": response_mode,
                "max_history_length": 25
            },
            "pause_detection": {
                "aggressiveness": 2,
                "sample_rate": 16000
            }
        }
        
        from realtime_conversation.adapters.config import DictConfigurationProvider
        config_provider = DictConfigurationProvider(config_data)
        
        # Create custom engine instance
        custom_engine = ConversationEngine(config_provider)
        
        # Use the same components as the main engine but with different config
        main_engine = await get_conversation_engine()
        custom_engine.configure_stt(main_engine.stt_engine)
        custom_engine.configure_tts(main_engine.tts_engine)
        custom_engine.configure_pause_detection(main_engine.pause_detector)
        
        # Configure response generation based on mode
        if response_mode == "template":
            response_generator = TemplateResponseGenerator()
        else:
            response_generator = EchoResponseGenerator()
        
        custom_engine.configure_response_generation(response_generator)
        
        # Add middleware based on parameters
        custom_engine.add_middleware(TimingMiddleware())
        
        # Set logging level
        import logging
        log_level_int = getattr(logging, log_level.upper(), logging.INFO)
        custom_engine.add_middleware(LoggingMiddleware(log_level=log_level_int))
        
        # Add analytics if requested
        if enable_analytics:
            from realtime_conversation.plugins.middleware import AnalyticsMiddleware
            custom_engine.add_middleware(AnalyticsMiddleware())
        
        # Create adapter and handle conversation
        adapter = FastAPIWebSocketAdapter(websocket)
        await custom_engine.handle_conversation(adapter)
        
    except Exception as e:
        logger.error(f"Custom WebSocket conversation error: {e}")
        
        try:
            if not websocket.client_state.disconnected:
                await websocket.send_json({
                    "type": "error", 
                    "message": f"Custom conversation error: {str(e)}"
                })
                await websocket.close()
        except:
            pass
    
    finally:
        logger.info("Custom WebSocket conversation session ended")


# Compatibility endpoint that wraps the new implementation
@router.websocket("/conversation")
async def websocket_conversation_legacy(
    websocket: WebSocket,
    language: Optional[str] = Query(None, description="STT language"),
    voice_mode: str = Query("predefined", description="TTS voice mode"),
    predefined_voice_id: Optional[str] = Query(None, description="Predefined voice ID"),
    reference_audio_filename: Optional[str] = Query(None, description="Reference audio filename"),
    response_mode: str = Query("echo", description="Response generation mode"),
    pause_aggressiveness: int = Query(2, ge=0, le=3, description="Pause detection aggressiveness"),
):
    """
    Legacy compatibility endpoint.
    
    This endpoint maintains the same interface as the original conversation
    endpoint but uses the new library internally. This allows for gradual
    migration without breaking existing clients.
    """
    # Just redirect to the v2 endpoint with the same parameters
    await websocket_conversation_v2(
        websocket=websocket,
        language=language,
        voice_mode=voice_mode,
        predefined_voice_id=predefined_voice_id,
        reference_audio_filename=reference_audio_filename,
        response_mode=response_mode,
        pause_aggressiveness=pause_aggressiveness
    )


# Additional endpoints for managing the conversation engine

@router.get("/conversation/status")
async def get_conversation_status():
    """Get status of the conversation engine."""
    try:
        engine = await get_conversation_engine()
        
        return {
            "status": "ready",
            "stt_available": engine.stt_engine.model_loaded if engine.stt_engine else False,
            "tts_available": engine.tts_engine.model_loaded if engine.tts_engine else False,
            "pause_detection_available": engine.pause_detector is not None,
            "response_generation_available": engine.response_generator is not None,
            "middleware_count": len(engine.pipeline.middleware)
        }
        
    except Exception as e:
        logger.error(f"Error getting conversation status: {e}")
        return {
            "status": "error",
            "error": str(e)
        }


@router.get("/conversation/config")
async def get_conversation_config():
    """Get current conversation configuration."""
    try:
        engine = await get_conversation_engine()
        
        config_info = {}
        
        # Get component configurations
        if engine.stt_engine and hasattr(engine.stt_engine, 'get_model_info'):
            config_info["stt"] = engine.stt_engine.get_model_info()
        
        if engine.tts_engine and hasattr(engine.tts_engine, 'get_model_info'):
            config_info["tts"] = engine.tts_engine.get_model_info()
        
        if engine.pause_detector and hasattr(engine.pause_detector, 'get_config'):
            config_info["pause_detection"] = engine.pause_detector.get_config()
        
        if engine.response_generator and hasattr(engine.response_generator, 'get_config'):
            config_info["response_generation"] = engine.response_generator.get_config()
        
        # Get middleware configurations
        middleware_configs = []
        for middleware in engine.pipeline.middleware:
            if hasattr(middleware, 'get_config'):
                middleware_configs.append(middleware.get_config())
        
        config_info["middleware"] = middleware_configs
        
        return config_info
        
    except Exception as e:
        logger.error(f"Error getting conversation config: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/conversation/reload")
async def reload_conversation_engine():
    """
    Reload the conversation engine.
    
    This can be useful for applying configuration changes without
    restarting the entire server.
    """
    global _conversation_engine
    
    try:
        async with _engine_lock:
            if _conversation_engine:
                # Unload models to free memory
                if _conversation_engine.stt_engine and hasattr(_conversation_engine.stt_engine, 'unload_model'):
                    _conversation_engine.stt_engine.unload_model()
                
                if _conversation_engine.tts_engine and hasattr(_conversation_engine.tts_engine, 'unload_model'):
                    _conversation_engine.tts_engine.unload_model()
            
            # Create new engine
            _conversation_engine = await _create_conversation_engine()
        
        return {
            "status": "success",
            "message": "Conversation engine reloaded successfully"
        }
        
    except Exception as e:
        logger.error(f"Error reloading conversation engine: {e}")
        return {
            "status": "error",
            "message": f"Failed to reload conversation engine: {str(e)}"
        }


# Migration notes and documentation endpoint
@router.get("/conversation/migration-info")
async def get_migration_info():
    """
    Get information about the migration from v1 to v2 conversation system.
    
    This endpoint provides documentation for developers about the differences
    between the old and new systems.
    """
    return {
        "migration_info": {
            "version": "2.0",
            "changes": [
                "Modular architecture with pluggable components",
                "Framework-agnostic conversation engine",
                "Improved error handling and logging",
                "Middleware pipeline for extensibility",
                "Better configuration management",
                "Enhanced pause detection options",
                "Multiple response generation strategies"
            ],
            "backwards_compatibility": {
                "endpoint": "/ws/conversation",
                "description": "Legacy endpoint maintains same interface",
                "migration_path": "Gradually migrate to /ws/conversation/v2"
            },
            "new_features": {
                "custom_endpoint": "/ws/conversation/custom",
                "analytics_support": "Built-in analytics middleware",
                "authentication": "Token-based authentication middleware",
                "performance_monitoring": "Timing and performance tracking"
            },
            "breaking_changes": [
                "Internal architecture completely rewritten",
                "Global state replaced with dependency injection",
                "Configuration system abstracted",
                "Middleware replaces direct hooks"
            ]
        },
        "library_structure": {
            "core": "Core conversation engine and interfaces",
            "adapters": "Framework and service adapters (WebSocket, STT, TTS, Config)",
            "plugins": "Pause detection, response generation, middleware",
            "audio": "Audio processing utilities",
            "examples": "Usage examples and integration patterns"
        }
    }