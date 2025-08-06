"""
Custom pipeline example.

Demonstrates how to create a conversation system with custom middleware,
authentication, analytics, and specialized response generation.
"""

import logging
from pathlib import Path
from typing import Optional, Set
from fastapi import FastAPI, WebSocket, HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

# Import the conversation library components
from realtime_conversation import ConversationEngine
from realtime_conversation.adapters.websocket import FastAPIWebSocketAdapter
from realtime_conversation.adapters.stt import WhisperSTTEngine
from realtime_conversation.adapters.tts import ChatterboxTTSEngine
from realtime_conversation.adapters.config import DictConfigurationProvider
from realtime_conversation.plugins.pause_detection import WebRTCPauseDetector, EnergyPauseDetector
from realtime_conversation.plugins.response_generation import TemplateResponseGenerator
from realtime_conversation.plugins.middleware import (
    LoggingMiddleware, TimingMiddleware, 
    AuthenticationMiddleware, AnalyticsMiddleware
)

logger = logging.getLogger(__name__)

# Security
security = HTTPBearer()


def create_custom_pipeline_app(
    valid_tokens: Optional[Set[str]] = None,
    enable_analytics: bool = True
) -> FastAPI:
    """
    Create a FastAPI application with custom conversation pipeline.
    
    Features:
    - Token-based authentication
    - Advanced analytics
    - Custom response templates
    - Performance monitoring
    - Comprehensive logging
    
    Args:
        valid_tokens: Set of valid authentication tokens
        enable_analytics: Whether to enable analytics middleware
        
    Returns:
        Configured FastAPI application
    """
    app = FastAPI(
        title="Advanced Conversation API",
        description="WebSocket conversation with authentication, analytics, and custom middleware",
        version="1.0.0"
    )
    
    # Default configuration for custom pipeline
    config_data = {
        "stt_engine": {
            "model_size": "small",  # Better quality than base
            "device": "auto",
            "language": None
        },
        "tts_engine": {
            "device": "auto",
            "temperature": 0.7,
            "speed_factor": 1.0
        },
        "conversation": {
            "response_mode": "template",
            "max_history_length": 25
        },
        "pause_detection": {
            "aggressiveness": 3,  # More aggressive detection
            "min_speech_frames": 2,
            "min_pause_frames": 8,
            "sample_rate": 16000
        },
        "audio": {
            "sample_rate": 16000,
            "channels": 1,
            "format": "pcm"
        },
        "middleware": {
            "enable_auth": valid_tokens is not None,
            "enable_analytics": enable_analytics,
            "enable_timing": True,
            "enable_logging": True
        }
    }
    
    config_provider = DictConfigurationProvider(config_data)
    
    # Create conversation engine with custom configuration
    conversation_engine = ConversationEngine(config_provider)
    
    # Initialize components
    try:
        # STT Engine with better model
        stt_config = config_provider.get_stt_config()
        stt_engine = WhisperSTTEngine(
            model_size=stt_config.get("model_size", "small"),
            device=stt_config.get("device", "auto")
        )
        if not stt_engine.load_model():
            raise RuntimeError("Failed to load STT model")
        conversation_engine.configure_stt(stt_engine)
        
        # TTS Engine
        tts_config = config_provider.get_tts_config()
        tts_engine = ChatterboxTTSEngine(
            device=tts_config.get("device", "auto")
        )
        if not tts_engine.load_model():
            raise RuntimeError("Failed to load TTS model")
        conversation_engine.configure_tts(tts_engine)
        
        # Advanced pause detection
        pause_config = config_provider.get_pause_detection_config()
        try:
            pause_detector = WebRTCPauseDetector(
                aggressiveness=pause_config.get("aggressiveness", 3),
                min_speech_frames=pause_config.get("min_speech_frames", 2),
                min_pause_frames=pause_config.get("min_pause_frames", 8),
                sample_rate=pause_config.get("sample_rate", 16000)
            )
            logger.info("Using advanced WebRTC VAD configuration")
        except ImportError:
            logger.warning("WebRTC VAD not available, using advanced energy detection")
            pause_detector = EnergyPauseDetector(
                energy_threshold=0.005,  # More sensitive
                adaptive_threshold=True,
                sample_rate=pause_config.get("sample_rate", 16000)
            )
        
        conversation_engine.configure_pause_detection(pause_detector)
        
        # Template-based response generation
        response_generator = TemplateResponseGenerator()
        conversation_engine.configure_response_generation(response_generator)
        
        # Add middleware in order (executed in reverse order)
        
        # 1. Analytics (outermost - tracks everything)
        if enable_analytics:
            analytics_middleware = AnalyticsMiddleware(
                track_usage=True,
                track_performance=True,
                track_errors=True,
                retention_days=7  # Keep data for a week
            )
            conversation_engine.add_middleware(analytics_middleware)
        
        # 2. Authentication (second - validates before processing)
        if valid_tokens:
            from realtime_conversation.plugins.middleware.auth import TokenAuthenticationMiddleware
            auth_middleware = TokenAuthenticationMiddleware(
                valid_tokens=valid_tokens,
                allow_anonymous=False
            )
            conversation_engine.add_middleware(auth_middleware)
        
        # 3. Timing (third - measures processing time)
        timing_middleware = TimingMiddleware(
            track_total_time=True,
            track_stage_times=True,
            log_timing=True,
            log_threshold_ms=500.0  # Log requests taking more than 500ms
        )
        conversation_engine.add_middleware(timing_middleware)
        
        # 4. Logging (innermost - logs detailed information)
        logging_middleware = LoggingMiddleware(
            log_level=logging.INFO,
            log_audio_info=True,
            log_transcriptions=True,
            log_responses=True,
            log_timing=True,
            max_text_length=200
        )
        conversation_engine.add_middleware(logging_middleware)
        
        logger.info("Advanced conversation engine initialized successfully")
        
    except Exception as e:
        logger.error(f"Failed to initialize advanced conversation engine: {e}")
        raise RuntimeError(f"Advanced conversation engine initialization failed: {e}")
    
    # Store components in app state
    app.state.conversation_engine = conversation_engine
    app.state.config_provider = config_provider
    app.state.valid_tokens = valid_tokens or set()
    
    # Authentication dependency
    async def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
        if valid_tokens and credentials.credentials not in valid_tokens:
            raise HTTPException(status_code=401, detail="Invalid authentication token")
        return credentials.credentials
    
    # WebSocket endpoint with authentication support
    @app.websocket("/ws/conversation")
    async def websocket_conversation(websocket: WebSocket, token: Optional[str] = None):
        """
        Advanced real-time conversation via WebSocket.
        
        Supports authentication via token query parameter.
        """
        try:
            # Create WebSocket adapter
            adapter = FastAPIWebSocketAdapter(websocket)
            
            # Set up authentication data if token provided
            auth_data = {}
            if token:
                auth_data = {
                    "auth_token": token,
                    "permissions": ["conversation", "audio_processing"]  # Example permissions
                }
            
            # Create conversation context with auth data
            from realtime_conversation.core.interfaces import ConversationContext
            
            # Handle conversation (the middleware will handle authentication)
            # Note: In a real implementation, you'd pass auth_data through the context
            await conversation_engine.handle_conversation(adapter)
            
        except Exception as e:
            logger.error(f"Advanced WebSocket conversation error: {e}")
            if not websocket.client_state.disconnected:
                try:
                    await websocket.send_json({"type": "error", "message": str(e)})
                    await websocket.close()
                except:
                    pass
    
    # Advanced endpoints with authentication
    
    @app.get("/health")
    async def health_check():
        """Enhanced health check with component details."""
        engine = app.state.conversation_engine
        
        health_status = {
            "status": "healthy",
            "timestamp": "2024-01-01T00:00:00Z",  # Would use datetime.now()
            "components": {
                "stt": {
                    "available": engine.stt_engine.model_loaded if engine.stt_engine else False,
                    "model_info": engine.stt_engine.get_model_info() if engine.stt_engine and hasattr(engine.stt_engine, 'get_model_info') else {}
                },
                "tts": {
                    "available": engine.tts_engine.model_loaded if engine.tts_engine else False,
                    "model_info": engine.tts_engine.get_model_info() if engine.tts_engine and hasattr(engine.tts_engine, 'get_model_info') else {}
                },
                "pause_detection": {
                    "available": engine.pause_detector is not None,
                    "config": engine.pause_detector.get_config() if engine.pause_detector and hasattr(engine.pause_detector, 'get_config') else {}
                },
                "response_generation": {
                    "available": engine.response_generator is not None,
                    "config": engine.response_generator.get_config() if engine.response_generator and hasattr(engine.response_generator, 'get_config') else {}
                }
            },
            "middleware": {
                "count": len(engine.pipeline.middleware),
                "names": [mw.get_name() for mw in engine.pipeline.middleware if hasattr(mw, 'get_name')]
            }
        }
        
        # Determine overall status
        critical_available = all([
            health_status["components"]["stt"]["available"],
            health_status["components"]["tts"]["available"],
            health_status["components"]["response_generation"]["available"]
        ])
        
        if not critical_available:
            health_status["status"] = "degraded"
        
        return health_status
    
    @app.get("/analytics")
    async def get_analytics(token: str = Depends(verify_token) if valid_tokens else None):
        """Get comprehensive analytics data."""
        try:
            analytics_data = {}
            
            # Get analytics from middleware
            for middleware in conversation_engine.pipeline.middleware:
                if hasattr(middleware, 'get_usage_statistics'):
                    analytics_data["usage"] = middleware.get_usage_statistics()
                elif hasattr(middleware, 'get_performance_statistics'):
                    analytics_data["performance"] = middleware.get_performance_statistics()
                elif hasattr(middleware, 'get_error_statistics'):
                    analytics_data["errors"] = middleware.get_error_statistics()
                elif hasattr(middleware, 'get_statistics'):
                    stats = middleware.get_statistics()
                    analytics_data[middleware.get_name()] = stats
            
            return analytics_data
            
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to get analytics: {e}")
    
    @app.get("/analytics/detailed")
    async def get_detailed_analytics(token: str = Depends(verify_token) if valid_tokens else None):
        """Get detailed analytics including time-based usage and percentiles."""
        try:
            detailed_data = {}
            
            for middleware in conversation_engine.pipeline.middleware:
                if hasattr(middleware, 'get_time_based_usage'):
                    detailed_data["hourly_usage"] = middleware.get_time_based_usage("hourly")
                    detailed_data["daily_usage"] = middleware.get_time_based_usage("daily")
                
                if hasattr(middleware, 'get_percentile_stats'):
                    detailed_data["performance_percentiles"] = middleware.get_percentile_stats()
                
                if hasattr(middleware, 'get_recent_events'):
                    detailed_data["recent_events"] = middleware.get_recent_events(50)
            
            return detailed_data
            
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to get detailed analytics: {e}")
    
    @app.post("/analytics/reset")
    async def reset_analytics(token: str = Depends(verify_token) if valid_tokens else None):
        """Reset analytics data."""
        try:
            reset_count = 0
            
            for middleware in conversation_engine.pipeline.middleware:
                if hasattr(middleware, 'reset_analytics'):
                    middleware.reset_analytics()
                    reset_count += 1
                elif hasattr(middleware, 'reset_statistics'):
                    middleware.reset_statistics()
                    reset_count += 1
            
            return {
                "message": f"Analytics reset for {reset_count} middleware components",
                "reset_count": reset_count
            }
            
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to reset analytics: {e}")
    
    @app.get("/configuration")
    async def get_full_configuration(token: str = Depends(verify_token) if valid_tokens else None):
        """Get complete system configuration."""
        try:
            config_data = config_provider.get_all_config()
            
            # Add middleware configurations
            middleware_configs = []
            for middleware in conversation_engine.pipeline.middleware:
                if hasattr(middleware, 'get_config'):
                    middleware_configs.append(middleware.get_config())
            
            return {
                "engine_config": config_data,
                "middleware_config": middleware_configs,
                "system_info": {
                    "middleware_count": len(conversation_engine.pipeline.middleware),
                    "authentication_enabled": valid_tokens is not None,
                    "analytics_enabled": enable_analytics
                }
            }
            
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to get configuration: {e}")
    
    @app.post("/test/conversation")
    async def test_conversation_pipeline(
        test_data: dict,
        token: str = Depends(verify_token) if valid_tokens else None
    ):
        """Test the conversation pipeline with sample data."""
        try:
            # This is a simplified test - in reality you'd create proper AudioData
            from realtime_conversation.core.interfaces import ConversationContext, AudioData, TranscriptionResult, TranscriptionSegment
            
            # Create test context
            context = ConversationContext()
            
            # Create mock audio data
            if "audio_text" in test_data:
                # Simulate transcription result
                segments = [TranscriptionSegment(
                    text=test_data["audio_text"],
                    start=0.0,
                    end=2.0
                )]
                
                context.transcription = TranscriptionResult(
                    text=test_data["audio_text"],
                    language=test_data.get("language", "en"),
                    segments=segments,
                    confidence=0.95
                )
            
            # Process through response generation only (skip STT and TTS for testing)
            if conversation_engine.response_generator:
                response = await conversation_engine.response_generator.generate_response(
                    context.transcription,
                    context
                )
                context.response_text = response
            
            return {
                "input": context.transcription.text if context.transcription else None,
                "response": context.response_text,
                "language": context.transcription.language if context.transcription else None,
                "success": True
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    logger.info("Advanced conversation app with custom pipeline created successfully")
    return app


# Example usage
if __name__ == "__main__":
    import uvicorn
    
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create valid tokens for testing
    valid_tokens = {
        "demo_token_123",
        "test_token_456", 
        "admin_token_789"
    }
    
    # Create the advanced app
    app = create_custom_pipeline_app(
        valid_tokens=valid_tokens,
        enable_analytics=True
    )
    
    print("Starting advanced conversation server...")
    print("Valid tokens for testing:")
    for token in valid_tokens:
        print(f"  - {token}")
    print("\nWebSocket URL: ws://localhost:8001/ws/conversation?token=demo_token_123")
    print("Analytics URL: http://localhost:8001/analytics")
    
    # Run the server
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8001,
        log_level="info"
    )