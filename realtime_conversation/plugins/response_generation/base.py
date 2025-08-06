"""
Base response generation implementation.

Provides common functionality for response generation plugins.
"""

import asyncio
import logging
from typing import List, Dict, Any
from datetime import datetime
from ...core.interfaces import ResponseGenerator, TranscriptionResult, ConversationContext

logger = logging.getLogger(__name__)


class BaseResponseGenerator(ResponseGenerator):
    """
    Base implementation of response generator with common functionality.
    
    Subclasses should implement the actual response generation logic.
    """
    
    def __init__(self, response_mode: str = "echo"):
        self.response_mode = response_mode
        self.conversation_history: List[Dict[str, Any]] = []
        self.max_history_length = 50
        
    async def generate_response(
        self, 
        transcription: TranscriptionResult, 
        context: ConversationContext
    ) -> str:
        """
        Generate a text response based on input transcription.
        
        Args:
            transcription: Input transcription result
            context: Conversation context with history and metadata
            
        Returns:
            Generated response text
        """
        if not transcription or not transcription.text.strip():
            return self._get_fallback_response()
        
        try:
            # Clean input text
            input_text = self._clean_text(transcription.text)
            
            # Add to conversation history
            self._add_to_history(input_text, transcription, context)
            
            # Run generation in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                self._generate_response_sync,
                input_text,
                transcription,
                context
            )
            
            return response or self._get_fallback_response()
            
        except Exception as e:
            logger.error(f"Error generating response: {e}", exc_info=True)
            return self._get_fallback_response()
    
    def set_response_mode(self, mode: str) -> None:
        """Set the response generation mode."""
        self.response_mode = mode
        logger.info(f"Response mode set to: {mode}")
    
    def clear_history(self) -> None:
        """Clear conversation history."""
        self.conversation_history.clear()
        logger.debug("Conversation history cleared")
    
    # Methods to be implemented by subclasses
    
    def _generate_response_sync(
        self, 
        input_text: str, 
        transcription: TranscriptionResult,
        context: ConversationContext
    ) -> str:
        """
        Synchronous response generation implementation.
        
        This method runs in a thread pool and should be implemented by subclasses.
        """
        raise NotImplementedError
    
    # Utility methods
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize input text."""
        if not text:
            return ""
        
        # Basic text cleaning
        cleaned = text.strip()
        
        # Remove excessive whitespace
        import re
        cleaned = re.sub(r'\s+', ' ', cleaned)
        
        return cleaned
    
    def _add_to_history(
        self, 
        input_text: str, 
        transcription: TranscriptionResult,
        context: ConversationContext
    ) -> None:
        """Add interaction to conversation history."""
        history_entry = {
            "timestamp": datetime.now(),
            "input": input_text,
            "language": transcription.language,
            "confidence": transcription.confidence,
            "duration": transcription.duration,
            "context": context.metadata.copy() if context.metadata else {}
        }
        
        self.conversation_history.append(history_entry)
        
        # Maintain history size
        if len(self.conversation_history) > self.max_history_length:
            # Keep the most recent half
            keep_count = self.max_history_length // 2
            self.conversation_history = self.conversation_history[-keep_count:]
    
    def _get_fallback_response(self) -> str:
        """Get fallback response when generation fails."""
        fallback_responses = [
            "I didn't catch that.",
            "Could you repeat that?",
            "I'm not sure I understood.",
            "Can you try again?",
            "Sorry, I didn't understand."
        ]
        
        import random
        return random.choice(fallback_responses)
    
    def _get_recent_history(self, count: int = 5) -> List[Dict[str, Any]]:
        """Get recent conversation history."""
        return self.conversation_history[-count:] if self.conversation_history else []
    
    def _extract_keywords(self, text: str) -> List[str]:
        """Extract keywords from text for template matching."""
        if not text:
            return []
        
        # Simple keyword extraction
        import re
        
        # Convert to lowercase and split into words
        words = re.findall(r'\b\w+\b', text.lower())
        
        # Filter out common stop words
        stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 
            'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 
            'been', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 
            'would', 'could', 'should', 'may', 'might', 'can', 'i', 'you',
            'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her', 'us', 'them'
        }
        
        keywords = [word for word in words if word not in stop_words and len(word) > 2]
        
        return keywords
    
    # Configuration and statistics
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get response generator statistics."""
        return {
            "response_mode": self.response_mode,
            "history_length": len(self.conversation_history),
            "max_history_length": self.max_history_length,
            "total_interactions": len(self.conversation_history)
        }
    
    def set_max_history_length(self, length: int) -> None:
        """Set maximum conversation history length."""
        if length <= 0:
            raise ValueError("History length must be positive")
        
        self.max_history_length = length
        
        # Trim current history if needed
        if len(self.conversation_history) > length:
            keep_count = length // 2
            self.conversation_history = self.conversation_history[-keep_count:]
        
        logger.debug(f"Max history length set to: {length}")
    
    def get_config(self) -> Dict[str, Any]:
        """Get current generator configuration."""
        return {
            "response_mode": self.response_mode,
            "max_history_length": self.max_history_length,
            "history_count": len(self.conversation_history)
        }