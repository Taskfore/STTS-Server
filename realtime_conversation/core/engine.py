"""
Core conversation engine implementation.

This module contains the main ConversationEngine class that orchestrates
STT, TTS, and response generation through a configurable middleware pipeline.
"""

import asyncio
import logging
from typing import Optional, List, Dict, Any, AsyncGenerator
from .interfaces import (
    STTEngine, TTSEngine, PauseDetector, ResponseGenerator,
    WebSocketAdapter, ConfigurationProvider, ConversationContext,
    AudioData, ConversationState, ConversationEvent
)
from .pipeline import ConversationPipeline

logger = logging.getLogger(__name__)


class ConversationEngine:
    """
    Core conversation engine that orchestrates STT, TTS, and response generation
    through a configurable middleware pipeline.
    """
    
    def __init__(self, config_provider: Optional[ConfigurationProvider] = None):
        """
        Initialize the conversation engine.
        
        Args:
            config_provider: Configuration provider for engine settings
        """
        self.config_provider = config_provider
        
        # Core components (initialized via configuration methods)
        self.stt_engine: Optional[STTEngine] = None
        self.tts_engine: Optional[TTSEngine] = None
        self.pause_detector: Optional[PauseDetector] = None
        self.response_generator: Optional[ResponseGenerator] = None
        
        # Pipeline and middleware
        self.pipeline = ConversationPipeline()
        
        # State tracking
        self._current_context: Optional[ConversationContext] = None
        self._is_running = False
        self._audio_buffer: List[AudioData] = []
        
        logger.info("ConversationEngine initialized")
    
    def configure_stt(self, stt_engine: STTEngine) -> None:
        """Configure the Speech-to-Text engine."""
        self.stt_engine = stt_engine
        logger.info(f"STT engine configured: {type(stt_engine).__name__}")
    
    def configure_tts(self, tts_engine: TTSEngine) -> None:
        """Configure the Text-to-Speech engine."""
        self.tts_engine = tts_engine
        logger.info(f"TTS engine configured: {type(tts_engine).__name__}")
    
    def configure_pause_detection(self, pause_detector: PauseDetector) -> None:
        """Configure the pause detection system."""
        self.pause_detector = pause_detector
        logger.info(f"Pause detector configured: {type(pause_detector).__name__}")
    
    def configure_response_generation(self, response_generator: ResponseGenerator) -> None:
        """Configure the response generation system."""
        self.response_generator = response_generator
        logger.info(f"Response generator configured: {type(response_generator).__name__}")
    
    def add_middleware(self, middleware) -> None:
        """Add middleware to the conversation pipeline."""
        self.pipeline.add_middleware(middleware)
        logger.info(f"Middleware added: {type(middleware).__name__}")
    
    def remove_middleware(self, middleware) -> None:
        """Remove middleware from the conversation pipeline."""
        self.pipeline.remove_middleware(middleware)
    
    def subscribe_to_events(self, event_type: str, handler) -> None:
        """Subscribe to conversation events."""
        self.pipeline.subscribe_event(event_type, handler)
    
    async def handle_conversation(self, websocket_adapter: WebSocketAdapter) -> None:
        """
        Main conversation handling loop.
        
        Args:
            websocket_adapter: Framework-specific WebSocket adapter
        """
        if not self._validate_configuration():
            raise ValueError("ConversationEngine not properly configured")
        
        self._is_running = True
        logger.info("Starting conversation session")
        
        try:
            await websocket_adapter.accept_connection()
            
            # Send ready message
            await websocket_adapter.send_json({
                "type": "ready",
                "message": "Conversation engine ready",
                "config": self._get_session_config()
            })
            
            # Main processing loop
            await self._conversation_loop(websocket_adapter)
            
        except Exception as e:
            logger.error(f"Conversation error: {e}", exc_info=True)
            if websocket_adapter.is_connected:
                try:
                    await websocket_adapter.send_json({
                        "type": "error",
                        "message": str(e)
                    })
                except:
                    pass  # Connection might be broken
        finally:
            self._is_running = False
            logger.info("Conversation session ended")
            if websocket_adapter.is_connected:
                await websocket_adapter.close()
    
    async def _conversation_loop(self, websocket_adapter: WebSocketAdapter) -> None:
        """Main conversation processing loop."""
        audio_chunk_count = 0
        
        while websocket_adapter.is_connected:
            try:
                # Handle audio input
                audio_data = await websocket_adapter.receive_audio()
                if audio_data:
                    audio_chunk_count += 1
                    if audio_chunk_count % 50 == 0:
                        logger.debug(f"Processing audio chunk #{audio_chunk_count}")
                    
                    await self._process_audio_chunk(audio_data, websocket_adapter)
                
                # Handle commands
                command = await websocket_adapter.receive_command()
                if command:
                    await self._handle_command(command, websocket_adapter)
                
                # Small delay to prevent busy waiting
                await asyncio.sleep(0.001)
                
            except asyncio.CancelledError:
                logger.info("Conversation loop cancelled")
                break
            except Exception as e:
                logger.error(f"Error in conversation loop: {e}")
                await asyncio.sleep(0.1)  # Brief pause before retrying
    
    async def _process_audio_chunk(
        self, 
        audio_data: AudioData, 
        websocket_adapter: WebSocketAdapter
    ) -> None:
        """Process incoming audio chunk through pause detection."""
        if not self.pause_detector:
            logger.warning("No pause detector configured, skipping audio processing")
            return
        
        # Add to buffer
        self._audio_buffer.append(audio_data)
        
        # Keep buffer manageable (last 10 seconds)
        max_duration = 10.0
        total_duration = sum(chunk.duration_seconds for chunk in self._audio_buffer)
        while total_duration > max_duration and len(self._audio_buffer) > 1:
            removed = self._audio_buffer.pop(0)
            total_duration -= removed.duration_seconds
        
        # Process through pause detector
        try:
            vad_result = await self.pause_detector.process_chunk(audio_data)
            events = vad_result.get("events", [])
            
            # Handle speech events
            if "speech_start" in events:
                await websocket_adapter.send_json({
                    "type": "speech_event",
                    "event": "speech_start",
                    "message": "Speech detected"
                })
            
            if "speech_end" in events:
                await websocket_adapter.send_json({
                    "type": "speech_event", 
                    "event": "speech_end",
                    "message": "Processing speech..."
                })
                
                # Process complete utterance
                await self._process_complete_utterance(websocket_adapter)
        
        except Exception as e:
            logger.error(f"Error processing audio chunk: {e}")
    
    async def _process_complete_utterance(self, websocket_adapter: WebSocketAdapter) -> None:
        """Process complete utterance through STT -> Response -> TTS pipeline."""
        if not self._audio_buffer:
            logger.warning("No audio data available for processing")
            return
        
        # Create conversation context
        context = ConversationContext()
        context.state = ConversationState.PROCESSING
        
        # Combine audio buffer into single audio data
        # For simplicity, we'll use the most recent audio chunk
        # In production, you might want to concatenate or use the full buffer
        context.audio_input = self._audio_buffer[-1] if self._audio_buffer else None
        
        try:
            # Process through pipeline
            result_context = await self.pipeline.process(context, self._core_processing)
            
            # Send results to client
            await self._send_results(result_context, websocket_adapter)
            
        except Exception as e:
            logger.error(f"Error processing utterance: {e}")
            await websocket_adapter.send_json({
                "type": "error",
                "message": f"Processing error: {str(e)}"
            })
    
    async def _core_processing(self, context: ConversationContext) -> ConversationContext:
        """Core processing pipeline: STT -> Response Generation -> TTS."""
        
        # Step 1: Speech-to-Text
        if self.stt_engine and context.audio_input:
            logger.debug("Starting STT processing")
            try:
                context.transcription = await self.stt_engine.transcribe(context.audio_input)
                logger.debug(f"STT result: {context.transcription.text if context.transcription else 'None'}")
            except Exception as e:
                logger.error(f"STT processing error: {e}")
                context.error = e
                return context
        
        # Step 2: Response Generation
        if self.response_generator and context.transcription:
            logger.debug("Starting response generation")
            try:
                context.response_text = await self.response_generator.generate_response(
                    context.transcription, context
                )
                logger.debug(f"Generated response: {context.response_text}")
            except Exception as e:
                logger.error(f"Response generation error: {e}")
                context.error = e
                return context
        
        # Step 3: Text-to-Speech
        if self.tts_engine and context.response_text:
            logger.debug("Starting TTS processing")
            try:
                tts_config = self.config_provider.get_tts_config() if self.config_provider else {}
                context.synthesis_result = await self.tts_engine.synthesize(
                    context.response_text, tts_config
                )
                logger.debug("TTS processing completed")
            except Exception as e:
                logger.error(f"TTS processing error: {e}")
                context.error = e
                return context
        
        context.state = ConversationState.LISTENING
        return context
    
    async def _send_results(
        self, 
        context: ConversationContext, 
        websocket_adapter: WebSocketAdapter
    ) -> None:
        """Send processing results to the client."""
        
        # Send transcription
        if context.transcription:
            await websocket_adapter.send_json({
                "type": "transcription",
                "text": context.transcription.text,
                "language": context.transcription.language,
                "partial": False,
                "timing": {
                    "duration": context.transcription.duration
                } if context.transcription.segments else {},
                "segments": [
                    {
                        "text": seg.text,
                        "start": seg.start,
                        "end": seg.end
                    } for seg in context.transcription.segments
                ]
            })
        
        # Send generated response text
        if context.response_text:
            await websocket_adapter.send_json({
                "type": "response_text", 
                "text": context.response_text
            })
        
        # Send synthesized audio
        if context.synthesis_result:
            await websocket_adapter.send_audio(context.synthesis_result.audio_data)
        
        # Send state update
        await websocket_adapter.send_json({
            "type": "state_update",
            "state": context.state.value,
            "message": "Processing complete"
        })
    
    async def _handle_command(
        self, 
        command: Dict[str, Any], 
        websocket_adapter: WebSocketAdapter
    ) -> None:
        """Handle client commands."""
        action = command.get("action")
        
        if action == "reset":
            await self._reset_conversation()
            await websocket_adapter.send_json({
                "type": "reset_complete",
                "message": "Conversation reset"
            })
        
        elif action == "change_mode":
            mode = command.get("mode")
            if self.response_generator and mode:
                self.response_generator.set_response_mode(mode)
                await websocket_adapter.send_json({
                    "type": "mode_changed",
                    "message": f"Response mode changed to {mode}"
                })
        
        elif action == "ping":
            await websocket_adapter.send_json({
                "type": "pong",
                "message": "Conversation engine active"
            })
        
        elif action == "stop":
            await websocket_adapter.send_json({
                "type": "stopping",
                "message": "Conversation stopping"
            })
            self._is_running = False
        
        else:
            logger.warning(f"Unknown command: {action}")
    
    async def _reset_conversation(self) -> None:
        """Reset conversation state."""
        self._audio_buffer.clear()
        
        if self.pause_detector:
            self.pause_detector.reset()
        
        if self.response_generator:
            self.response_generator.clear_history()
        
        self._current_context = None
        logger.info("Conversation state reset")
    
    def _validate_configuration(self) -> bool:
        """Validate that required components are configured."""
        required_components = [
            ("STT Engine", self.stt_engine),
            ("TTS Engine", self.tts_engine),
            ("Response Generator", self.response_generator)
        ]
        
        for name, component in required_components:
            if component is None:
                logger.error(f"{name} not configured")
                return False
        
        # Check if models are loaded
        if self.stt_engine and not self.stt_engine.model_loaded:
            logger.error("STT engine model not loaded")
            return False
        
        if self.tts_engine and not self.tts_engine.model_loaded:
            logger.error("TTS engine model not loaded")
            return False
        
        return True
    
    def _get_session_config(self) -> Dict[str, Any]:
        """Get current session configuration."""
        return {
            "stt_available": self.stt_engine is not None and self.stt_engine.model_loaded,
            "tts_available": self.tts_engine is not None and self.tts_engine.model_loaded,
            "pause_detection_available": self.pause_detector is not None,
            "response_generation_available": self.response_generator is not None,
            "middleware_count": self.pipeline.middleware_count
        }
    
    @property
    def is_running(self) -> bool:
        """Check if the conversation engine is currently running."""
        return self._is_running