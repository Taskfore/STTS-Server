"""
Basic FastAPI integration example.

Demonstrates how to integrate the conversation library with a FastAPI application
for real-time audio conversation.
"""

import logging
from pathlib import Path
from typing import Optional
from fastapi import FastAPI, WebSocket, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse

# Import the conversation library components
from realtime_conversation import ConversationEngine
from realtime_conversation.adapters.websocket import FastAPIWebSocketAdapter
from realtime_conversation.adapters.stt import WhisperSTTEngine
from realtime_conversation.adapters.tts import ChatterboxTTSEngine
from realtime_conversation.adapters.config import YAMLConfigurationProvider
from realtime_conversation.plugins.pause_detection import WebRTCPauseDetector, EnergyPauseDetector
from realtime_conversation.plugins.response_generation import EchoResponseGenerator, TemplateResponseGenerator
from realtime_conversation.plugins.middleware import LoggingMiddleware, TimingMiddleware

logger = logging.getLogger(__name__)


def create_basic_fastapi_app(
    config_path: Optional[str] = None,
    enable_middleware: bool = True,
    static_files_path: Optional[str] = None
) -> FastAPI:
    """
    Create a basic FastAPI application with conversation capabilities.
    
    Args:
        config_path: Path to configuration file
        enable_middleware: Whether to enable middleware
        static_files_path: Path to static files directory
        
    Returns:
        Configured FastAPI application
    """
    app = FastAPI(
        title="Real-time Conversation API",
        description="WebSocket-based real-time audio conversation using the conversation library",
        version="1.0.0"
    )
    
    # Initialize configuration
    if config_path:
        config_provider = YAMLConfigurationProvider(Path(config_path))
    else:
        # Use default configuration
        from realtime_conversation.adapters.config import create_default_config
        config_provider = create_default_config()
    
    # Create conversation engine
    conversation_engine = ConversationEngine(config_provider)
    
    # Initialize and configure components
    try:
        # STT Engine
        stt_config = config_provider.get_stt_config()
        stt_engine = WhisperSTTEngine(
            model_size=stt_config.get("model_size", "base"),
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
        
        # Pause Detection
        pause_config = config_provider.get_pause_detection_config()
        try:
            pause_detector = WebRTCPauseDetector(
                aggressiveness=pause_config.get("aggressiveness", 2),
                sample_rate=pause_config.get("sample_rate", 16000)
            )
            logger.info("Using WebRTC VAD for pause detection")
        except ImportError:
            logger.warning("WebRTC VAD not available, using energy-based detection")
            pause_detector = EnergyPauseDetector(
                sample_rate=pause_config.get("sample_rate", 16000)
            )
        
        conversation_engine.configure_pause_detection(pause_detector)
        
        # Response Generation
        conv_config = config_provider.get_conversation_config()
        response_mode = conv_config.get("response_mode", "echo")
        
        if response_mode == "template":
            response_generator = TemplateResponseGenerator()
        else:
            response_generator = EchoResponseGenerator()
        
        conversation_engine.configure_response_generation(response_generator)
        
        # Add middleware if enabled
        if enable_middleware:
            conversation_engine.add_middleware(TimingMiddleware())
            conversation_engine.add_middleware(LoggingMiddleware(log_level=logging.INFO))
        
        logger.info("Conversation engine initialized successfully")
        
    except Exception as e:
        logger.error(f"Failed to initialize conversation engine: {e}")
        raise RuntimeError(f"Conversation engine initialization failed: {e}")
    
    # Store engine in app state
    app.state.conversation_engine = conversation_engine
    app.state.config_provider = config_provider
    
    # WebSocket endpoint for conversation
    @app.websocket("/ws/conversation")
    async def websocket_conversation(websocket: WebSocket):
        """Real-time conversation via WebSocket."""
        try:
            # Create WebSocket adapter
            adapter = FastAPIWebSocketAdapter(websocket)
            
            # Handle conversation
            await conversation_engine.handle_conversation(adapter)
            
        except Exception as e:
            logger.error(f"WebSocket conversation error: {e}")
            if not websocket.client_state.disconnected:
                try:
                    await websocket.close()
                except:
                    pass
    
    # Health check endpoint
    @app.get("/health")
    async def health_check():
        """Health check endpoint."""
        engine = app.state.conversation_engine
        
        health_status = {
            "status": "healthy",
            "components": {
                "stt_available": engine.stt_engine.model_loaded if engine.stt_engine else False,
                "tts_available": engine.tts_engine.model_loaded if engine.tts_engine else False,
                "pause_detection_available": engine.pause_detector is not None,
                "response_generation_available": engine.response_generator is not None
            }
        }
        
        # Check if all critical components are available
        critical_components = ["stt_available", "tts_available", "response_generation_available"]
        all_critical_healthy = all(health_status["components"][comp] for comp in critical_components)
        
        if not all_critical_healthy:
            health_status["status"] = "degraded"
        
        return health_status
    
    # Configuration endpoint
    @app.get("/config")
    async def get_configuration():
        """Get current configuration."""
        try:
            return {
                "stt_config": config_provider.get_stt_config(),
                "tts_config": config_provider.get_tts_config(),
                "conversation_config": config_provider.get_conversation_config(),
                "audio_config": config_provider.get_audio_config()
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to get configuration: {e}")
    
    # Statistics endpoint
    @app.get("/stats")
    async def get_statistics():
        """Get conversation statistics."""
        try:
            stats = {}
            
            # Get middleware statistics if available
            for middleware in conversation_engine.pipeline.middleware:
                if hasattr(middleware, 'get_statistics'):
                    stats[middleware.get_name()] = middleware.get_statistics()
                elif hasattr(middleware, 'get_auth_statistics'):
                    stats[middleware.get_name()] = middleware.get_auth_statistics()
            
            return stats
            
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to get statistics: {e}")
    
    # Model info endpoint
    @app.get("/models/info")
    async def get_model_info():
        """Get information about loaded models."""
        try:
            info = {}
            
            if conversation_engine.stt_engine and hasattr(conversation_engine.stt_engine, 'get_model_info'):
                info["stt"] = conversation_engine.stt_engine.get_model_info()
            
            if conversation_engine.tts_engine and hasattr(conversation_engine.tts_engine, 'get_model_info'):
                info["tts"] = conversation_engine.tts_engine.get_model_info()
            
            return info
            
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to get model info: {e}")
    
    # Static files (if path provided)
    if static_files_path:
        static_path = Path(static_files_path)
        if static_path.exists():
            app.mount("/static", StaticFiles(directory=str(static_path)), name="static")
            
            # Serve basic HTML interface
            @app.get("/", response_class=HTMLResponse)
            async def get_index():
                """Serve basic conversation interface."""
                html_content = get_basic_html_interface()
                return HTMLResponse(content=html_content)
    
    logger.info("FastAPI conversation app created successfully")
    return app


def get_basic_html_interface() -> str:
    """Get basic HTML interface for testing the conversation system."""
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Real-time Conversation</title>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <style>
            body { font-family: Arial, sans-serif; margin: 40px; background: #f5f5f5; }
            .container { max-width: 800px; margin: 0 auto; background: white; padding: 30px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
            h1 { color: #333; text-align: center; }
            .status { padding: 10px; border-radius: 5px; margin: 10px 0; }
            .status.connected { background: #d4edda; color: #155724; border: 1px solid #c3e6cb; }
            .status.disconnected { background: #f8d7da; color: #721c24; border: 1px solid #f5c6cb; }
            .controls { text-align: center; margin: 20px 0; }
            button { background: #007bff; color: white; border: none; padding: 12px 24px; border-radius: 5px; cursor: pointer; margin: 5px; font-size: 16px; }
            button:hover { background: #0056b3; }
            button:disabled { background: #ccc; cursor: not-allowed; }
            .messages { height: 300px; border: 1px solid #ddd; padding: 15px; overflow-y: auto; background: #fafafa; border-radius: 5px; margin: 20px 0; }
            .message { margin: 10px 0; padding: 8px; border-radius: 5px; }
            .message.transcription { background: #e3f2fd; border-left: 4px solid #2196f3; }
            .message.response { background: #f3e5f5; border-left: 4px solid #9c27b0; }
            .message.system { background: #fff3e0; border-left: 4px solid #ff9800; }
            .timestamp { font-size: 12px; color: #666; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🎤 Real-time Conversation</h1>
            
            <div id="status" class="status disconnected">
                Disconnected
            </div>
            
            <div class="controls">
                <button id="connectBtn" onclick="connect()">Connect</button>
                <button id="disconnectBtn" onclick="disconnect()" disabled>Disconnect</button>
                <button onclick="clearMessages()">Clear Messages</button>
            </div>
            
            <div id="messages" class="messages">
                <div class="message system">
                    <div class="timestamp">${new Date().toLocaleTimeString()}</div>
                    <div>Welcome! Click Connect to start a conversation.</div>
                </div>
            </div>
            
            <div style="margin-top: 20px; padding: 15px; background: #f8f9fa; border-radius: 5px; font-size: 14px; color: #666;">
                <strong>Instructions:</strong><br>
                • Click "Connect" to establish WebSocket connection<br>
                • Allow microphone access when prompted<br>
                • Speak naturally - the system will detect speech and pauses<br>
                • Your speech will be transcribed and you'll hear a response<br>
                • Check browser console for technical details
            </div>
        </div>

        <script>
            let ws = null;
            let mediaRecorder = null;
            let audioStream = null;

            function addMessage(type, content, timestamp = new Date()) {
                const messages = document.getElementById('messages');
                const messageDiv = document.createElement('div');
                messageDiv.className = `message ${type}`;
                messageDiv.innerHTML = `
                    <div class="timestamp">${timestamp.toLocaleTimeString()}</div>
                    <div>${content}</div>
                `;
                messages.appendChild(messageDiv);
                messages.scrollTop = messages.scrollHeight;
            }

            function updateStatus(connected, message = null) {
                const status = document.getElementById('status');
                const connectBtn = document.getElementById('connectBtn');
                const disconnectBtn = document.getElementById('disconnectBtn');
                
                if (connected) {
                    status.textContent = message || 'Connected';
                    status.className = 'status connected';
                    connectBtn.disabled = true;
                    disconnectBtn.disabled = false;
                } else {
                    status.textContent = message || 'Disconnected';
                    status.className = 'status disconnected';
                    connectBtn.disabled = false;
                    disconnectBtn.disabled = true;
                }
            }

            async function connect() {
                try {
                    // Get WebSocket URL
                    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
                    const wsUrl = `${protocol}//${window.location.host}/ws/conversation`;
                    
                    // Connect WebSocket
                    ws = new WebSocket(wsUrl);
                    
                    ws.onopen = function() {
                        updateStatus(true, 'Connected');
                        addMessage('system', 'WebSocket connected successfully');
                        setupAudio();
                    };
                    
                    ws.onmessage = function(event) {
                        try {
                            const data = JSON.parse(event.data);
                            console.log('Received:', data);
                            
                            if (data.type === 'transcription') {
                                addMessage('transcription', `Transcription: "${data.text}" (${data.language})`);
                            } else if (data.type === 'response_text') {
                                addMessage('response', `Response: "${data.text}"`);
                            } else if (data.type === 'audio_response') {
                                addMessage('response', 'Audio response received');
                                playAudioResponse(data.audio_data);
                            } else if (data.type === 'ready') {
                                addMessage('system', data.message);
                            } else if (data.type === 'error') {
                                addMessage('system', `Error: ${data.message}`);
                            }
                        } catch (e) {
                            console.error('Error parsing message:', e);
                        }
                    };
                    
                    ws.onclose = function() {
                        updateStatus(false, 'Disconnected');
                        addMessage('system', 'WebSocket connection closed');
                        if (mediaRecorder && mediaRecorder.state !== 'inactive') {
                            mediaRecorder.stop();
                        }
                        if (audioStream) {
                            audioStream.getTracks().forEach(track => track.stop());
                        }
                    };
                    
                    ws.onerror = function(error) {
                        console.error('WebSocket error:', error);
                        addMessage('system', 'WebSocket error occurred');
                    };
                    
                } catch (error) {
                    console.error('Connection error:', error);
                    addMessage('system', `Connection error: ${error.message}`);
                }
            }

            async function setupAudio() {
                try {
                    audioStream = await navigator.mediaDevices.getUserMedia({ 
                        audio: {
                            sampleRate: 16000,
                            channelCount: 1,
                            echoCancellation: true,
                            noiseSuppression: true
                        } 
                    });
                    
                    mediaRecorder = new MediaRecorder(audioStream, { 
                        mimeType: 'audio/webm' 
                    });
                    
                    mediaRecorder.ondataavailable = function(event) {
                        if (event.data.size > 0 && ws && ws.readyState === WebSocket.OPEN) {
                            // Convert to PCM and send (simplified example)
                            event.data.arrayBuffer().then(buffer => {
                                ws.send(buffer);
                            });
                        }
                    };
                    
                    mediaRecorder.start(100); // Send data every 100ms
                    addMessage('system', 'Audio recording started');
                    
                } catch (error) {
                    console.error('Audio setup error:', error);
                    addMessage('system', `Audio setup error: ${error.message}`);
                }
            }

            function playAudioResponse(audioData) {
                try {
                    const audioBytes = atob(audioData);
                    const arrayBuffer = new ArrayBuffer(audioBytes.length);
                    const uint8Array = new Uint8Array(arrayBuffer);
                    
                    for (let i = 0; i < audioBytes.length; i++) {
                        uint8Array[i] = audioBytes.charCodeAt(i);
                    }
                    
                    const blob = new Blob([arrayBuffer], { type: 'audio/wav' });
                    const audioUrl = URL.createObjectURL(blob);
                    const audio = new Audio(audioUrl);
                    
                    audio.play().catch(error => {
                        console.error('Audio play error:', error);
                    });
                    
                } catch (error) {
                    console.error('Audio decode error:', error);
                }
            }

            function disconnect() {
                if (ws) {
                    ws.close();
                }
                if (mediaRecorder && mediaRecorder.state !== 'inactive') {
                    mediaRecorder.stop();
                }
                if (audioStream) {
                    audioStream.getTracks().forEach(track => track.stop());
                }
                updateStatus(false);
            }

            function clearMessages() {
                const messages = document.getElementById('messages');
                messages.innerHTML = '';
                addMessage('system', 'Messages cleared');
            }
        </script>
    </body>
    </html>
    """


# Example usage
if __name__ == "__main__":
    import uvicorn
    
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create the app
    app = create_basic_fastapi_app()
    
    # Run the server
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )