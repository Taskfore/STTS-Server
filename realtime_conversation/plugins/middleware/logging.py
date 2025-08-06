"""
Logging middleware for conversation pipeline.

Logs conversation interactions, timing, and state changes.
"""

import logging
import json
from typing import Dict, Any, Optional
from datetime import datetime
from .base import BaseMiddleware
from ...core.interfaces import ConversationContext, ConversationState

logger = logging.getLogger(__name__)


class LoggingMiddleware(BaseMiddleware):
    """Middleware that logs conversation interactions and state changes."""
    
    def __init__(
        self, 
        log_level: int = logging.INFO,
        log_audio_info: bool = True,
        log_transcriptions: bool = True,
        log_responses: bool = True,
        log_timing: bool = True,
        max_text_length: int = 100
    ):
        """
        Initialize logging middleware.
        
        Args:
            log_level: Logging level for conversation logs
            log_audio_info: Whether to log audio information
            log_transcriptions: Whether to log transcription results
            log_responses: Whether to log generated responses
            log_timing: Whether to log timing information
            max_text_length: Maximum text length to log (truncated if longer)
        """
        super().__init__(name="logging")
        self.log_level = log_level
        self.log_audio_info = log_audio_info
        self.log_transcriptions = log_transcriptions
        self.log_responses = log_responses
        self.log_timing = log_timing
        self.max_text_length = max_text_length
        
        # Create conversation-specific logger
        self.conversation_logger = logging.getLogger(f"{__name__}.conversation")
        self.conversation_logger.setLevel(log_level)
        
        logger.info(f"Logging middleware initialized: level={logging.getLevelName(log_level)}")
    
    async def _pre_process(self, context: ConversationContext) -> None:
        """Log context state before processing."""
        if not self.conversation_logger.isEnabledFor(self.log_level):
            return
        
        log_data = {
            "event": "conversation_start",
            "timestamp": datetime.now().isoformat(),
            "state": context.state.value if context.state else "unknown",
            "has_audio": context.audio_input is not None,
            "has_transcription": context.transcription is not None,
            "has_response": context.response_text is not None
        }
        
        # Log audio information
        if self.log_audio_info and context.audio_input:
            log_data["audio_info"] = {
                "format": context.audio_input.format,
                "sample_rate": context.audio_input.sample_rate,
                "channels": context.audio_input.channels,
                "duration_ms": context.audio_input.duration_ms,
                "size_bytes": len(context.audio_input.data)
            }
        
        # Log existing transcription
        if self.log_transcriptions and context.transcription:
            log_data["input_transcription"] = {
                "text": self._truncate_text(context.transcription.text),
                "language": context.transcription.language,
                "segments": len(context.transcription.segments),
                "confidence": context.transcription.confidence,
                "duration": context.transcription.duration
            }
        
        # Log metadata
        if context.metadata:
            log_data["metadata"] = self._sanitize_metadata(context.metadata)
        
        self.conversation_logger.log(
            self.log_level,
            f"Conversation processing started: {json.dumps(log_data, default=str)}"
        )
    
    async def _post_process(self, context: ConversationContext) -> None:
        """Log context state after processing."""
        if not self.conversation_logger.isEnabledFor(self.log_level):
            return
        
        log_data = {
            "event": "conversation_complete",
            "timestamp": datetime.now().isoformat(),
            "state": context.state.value if context.state else "unknown",
            "success": context.error is None,
        }
        
        # Log transcription results
        if self.log_transcriptions and context.transcription:
            log_data["transcription"] = {
                "text": self._truncate_text(context.transcription.text),
                "language": context.transcription.language,
                "segments": len(context.transcription.segments),
                "confidence": context.transcription.confidence,
                "duration": context.transcription.duration
            }
        
        # Log response text
        if self.log_responses and context.response_text:
            log_data["response"] = {
                "text": self._truncate_text(context.response_text),
                "length": len(context.response_text)
            }
        
        # Log synthesis results
        if context.synthesis_result:
            log_data["synthesis"] = {
                "voice_id": context.synthesis_result.voice_id,
                "duration_ms": context.synthesis_result.audio_data.duration_ms,
                "synthesis_time": context.synthesis_result.synthesis_time
            }
        
        # Log errors
        if context.error:
            log_data["error"] = {
                "type": type(context.error).__name__,
                "message": str(context.error)
            }
        
        # Log timing if available
        if self.log_timing and context.metadata:
            timing_info = {}
            for key, value in context.metadata.items():
                if "time" in key.lower() or "duration" in key.lower():
                    timing_info[key] = value
            
            if timing_info:
                log_data["timing"] = timing_info
        
        self.conversation_logger.log(
            self.log_level,
            f"Conversation processing complete: {json.dumps(log_data, default=str)}"
        )
    
    async def _handle_error(self, context: ConversationContext, error: Exception) -> None:
        """Log error information."""
        error_data = {
            "event": "conversation_error",
            "timestamp": datetime.now().isoformat(),
            "error_type": type(error).__name__,
            "error_message": str(error),
            "state": context.state.value if context.state else "unknown",
            "has_audio": context.audio_input is not None,
            "has_transcription": context.transcription is not None,
            "has_response": context.response_text is not None
        }
        
        self.conversation_logger.error(
            f"Conversation error: {json.dumps(error_data, default=str)}",
            exc_info=True
        )
    
    def _truncate_text(self, text: str) -> str:
        """Truncate text for logging if it's too long."""
        if not text:
            return ""
        
        if len(text) <= self.max_text_length:
            return text
        
        return text[:self.max_text_length - 3] + "..."
    
    def _sanitize_metadata(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Sanitize metadata for logging (remove sensitive information)."""
        sanitized = {}
        sensitive_keys = {"password", "token", "key", "secret", "auth", "credential"}
        
        for key, value in metadata.items():
            key_lower = key.lower()
            
            # Check if key contains sensitive information
            if any(sensitive in key_lower for sensitive in sensitive_keys):
                sanitized[key] = "[REDACTED]"
            else:
                # Truncate long values
                if isinstance(value, str) and len(value) > 200:
                    sanitized[key] = value[:197] + "..."
                else:
                    sanitized[key] = value
        
        return sanitized
    
    # Configuration methods
    
    def set_log_level(self, level: int) -> None:
        """Set logging level."""
        self.log_level = level
        self.conversation_logger.setLevel(level)
        logger.info(f"Logging middleware level set to: {logging.getLevelName(level)}")
    
    def set_text_logging(
        self, 
        transcriptions: bool = True, 
        responses: bool = True,
        max_length: int = 100
    ) -> None:
        """Configure text logging settings."""
        self.log_transcriptions = transcriptions
        self.log_responses = responses
        self.max_text_length = max_length
        
        logger.info(
            f"Text logging configured: transcriptions={transcriptions}, "
            f"responses={responses}, max_length={max_length}"
        )
    
    def get_config(self) -> Dict[str, Any]:
        """Get current logging configuration."""
        return {
            "name": self.name,
            "enabled": self.enabled,
            "log_level": logging.getLevelName(self.log_level),
            "log_audio_info": self.log_audio_info,
            "log_transcriptions": self.log_transcriptions,
            "log_responses": self.log_responses,
            "log_timing": self.log_timing,
            "max_text_length": self.max_text_length
        }


# Factory functions

def create_logging_middleware(
    log_level: int = logging.INFO,
    detailed: bool = True
) -> LoggingMiddleware:
    """
    Create a logging middleware with standard settings.
    
    Args:
        log_level: Logging level
        detailed: Whether to enable detailed logging
        
    Returns:
        Configured logging middleware
    """
    return LoggingMiddleware(
        log_level=log_level,
        log_audio_info=detailed,
        log_transcriptions=detailed,
        log_responses=detailed,
        log_timing=detailed
    )


def create_logging_middleware_from_config(config: Dict[str, Any]) -> LoggingMiddleware:
    """
    Create logging middleware from configuration.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Configured logging middleware
    """
    log_level_name = config.get("log_level", "INFO")
    log_level = getattr(logging, log_level_name.upper(), logging.INFO)
    
    return LoggingMiddleware(
        log_level=log_level,
        log_audio_info=config.get("log_audio_info", True),
        log_transcriptions=config.get("log_transcriptions", True),
        log_responses=config.get("log_responses", True),
        log_timing=config.get("log_timing", True),
        max_text_length=config.get("max_text_length", 100)
    )