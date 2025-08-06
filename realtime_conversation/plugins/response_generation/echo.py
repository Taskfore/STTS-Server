"""
Echo response generator.

Simple response generator that echoes back the input text,
useful for testing and basic conversation loops.
"""

import logging
from typing import Dict, Any
from .base import BaseResponseGenerator
from ...core.interfaces import TranscriptionResult, ConversationContext

logger = logging.getLogger(__name__)


class EchoResponseGenerator(BaseResponseGenerator):
    """Echo response generator that returns the input text."""
    
    def __init__(self, add_prefix: bool = False, prefix_text: str = "You said: "):
        """
        Initialize echo response generator.
        
        Args:
            add_prefix: Whether to add a prefix to the echoed text
            prefix_text: Prefix text to add if add_prefix is True
        """
        super().__init__(response_mode="echo")
        self.add_prefix = add_prefix
        self.prefix_text = prefix_text
        
        logger.info(f"Echo response generator initialized: prefix={add_prefix}")
    
    def _generate_response_sync(
        self, 
        input_text: str, 
        transcription: TranscriptionResult,
        context: ConversationContext
    ) -> str:
        """Generate echo response."""
        try:
            if not input_text.strip():
                return self._get_fallback_response()
            
            # Simple echo with optional prefix
            if self.add_prefix:
                response = f"{self.prefix_text}{input_text}"
            else:
                response = input_text
            
            logger.debug(f"Echo response generated: '{response[:50]}...'")
            return response
            
        except Exception as e:
            logger.error(f"Error in echo response generation: {e}")
            return input_text  # Fallback to plain echo
    
    # Configuration methods
    
    def set_prefix(self, enabled: bool, prefix_text: str = "You said: ") -> None:
        """Set prefix configuration."""
        self.add_prefix = enabled
        self.prefix_text = prefix_text
        logger.info(f"Echo prefix {'enabled' if enabled else 'disabled'}: '{prefix_text}'")
    
    def get_config(self) -> Dict[str, Any]:
        """Get echo generator configuration."""
        config = super().get_config()
        config.update({
            "add_prefix": self.add_prefix,
            "prefix_text": self.prefix_text
        })
        return config


# Factory functions

def create_echo_generator(add_prefix: bool = False) -> EchoResponseGenerator:
    """
    Create a simple echo response generator.
    
    Args:
        add_prefix: Whether to add "You said: " prefix
        
    Returns:
        Configured echo response generator
    """
    return EchoResponseGenerator(add_prefix=add_prefix)


def create_echo_generator_from_config(config: Dict[str, Any]) -> EchoResponseGenerator:
    """
    Create echo response generator from configuration.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Configured echo response generator
    """
    return EchoResponseGenerator(
        add_prefix=config.get("add_prefix", False),
        prefix_text=config.get("prefix_text", "You said: ")
    )