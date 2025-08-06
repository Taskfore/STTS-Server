"""
Base middleware implementation.

Provides common functionality for conversation middleware plugins.
"""

import asyncio
import logging
from typing import Any, Callable, Awaitable
from ...core.interfaces import ConversationMiddleware, ConversationContext

logger = logging.getLogger(__name__)


class BaseMiddleware(ConversationMiddleware):
    """
    Base implementation of conversation middleware with common functionality.
    
    Subclasses should implement the actual middleware logic.
    """
    
    def __init__(self, name: str = "base"):
        self.name = name
        self.enabled = True
        
    async def process(
        self, 
        context: ConversationContext, 
        next_middleware: Callable[[ConversationContext], Awaitable[ConversationContext]]
    ) -> ConversationContext:
        """
        Process conversation context and call next middleware.
        
        Args:
            context: Current conversation context
            next_middleware: Next middleware function in the chain
            
        Returns:
            Modified conversation context
        """
        if not self.enabled:
            # Skip processing if disabled
            return await next_middleware(context)
        
        try:
            # Pre-processing hook
            await self._pre_process(context)
            
            # Call next middleware
            result_context = await next_middleware(context)
            
            # Post-processing hook
            await self._post_process(result_context)
            
            return result_context
            
        except Exception as e:
            # Error handling hook
            await self._handle_error(context, e)
            raise
    
    # Hooks to be implemented by subclasses
    
    async def _pre_process(self, context: ConversationContext) -> None:
        """
        Pre-processing hook called before next middleware.
        
        Subclasses should override this method to implement pre-processing logic.
        """
        pass
    
    async def _post_process(self, context: ConversationContext) -> None:
        """
        Post-processing hook called after next middleware.
        
        Subclasses should override this method to implement post-processing logic.
        """
        pass
    
    async def _handle_error(self, context: ConversationContext, error: Exception) -> None:
        """
        Error handling hook called when an exception occurs.
        
        Subclasses can override this method to implement custom error handling.
        """
        logger.error(f"Error in middleware '{self.name}': {error}")
    
    # Configuration methods
    
    def enable(self) -> None:
        """Enable this middleware."""
        self.enabled = True
        logger.debug(f"Middleware '{self.name}' enabled")
    
    def disable(self) -> None:
        """Disable this middleware."""
        self.enabled = False
        logger.debug(f"Middleware '{self.name}' disabled")
    
    def is_enabled(self) -> bool:
        """Check if middleware is enabled."""
        return self.enabled
    
    def get_name(self) -> str:
        """Get middleware name."""
        return self.name