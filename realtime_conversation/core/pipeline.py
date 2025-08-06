"""
Conversation processing pipeline with middleware support.
"""

import asyncio
import logging
from typing import List, Callable, Optional, Any
from .interfaces import ConversationContext, ConversationMiddleware, ConversationEvent, EventHandler

logger = logging.getLogger(__name__)


class ConversationPipeline:
    """
    Manages the conversation processing pipeline with middleware support.
    """
    
    def __init__(self):
        self.middleware: List[ConversationMiddleware] = []
        self._event_handlers: dict[str, List[EventHandler]] = {}
        self._is_processing = False
    
    def add_middleware(self, middleware: ConversationMiddleware) -> None:
        """Add middleware to the pipeline."""
        self.middleware.append(middleware)
        logger.debug(f"Added middleware: {type(middleware).__name__}")
    
    def remove_middleware(self, middleware: ConversationMiddleware) -> None:
        """Remove middleware from the pipeline."""
        if middleware in self.middleware:
            self.middleware.remove(middleware)
            logger.debug(f"Removed middleware: {type(middleware).__name__}")
    
    def clear_middleware(self) -> None:
        """Clear all middleware from the pipeline."""
        self.middleware.clear()
        logger.debug("Cleared all middleware")
    
    def subscribe_event(self, event_type: str, handler: EventHandler) -> None:
        """Subscribe to pipeline events."""
        if event_type not in self._event_handlers:
            self._event_handlers[event_type] = []
        self._event_handlers[event_type].append(handler)
        logger.debug(f"Subscribed to event: {event_type}")
    
    def unsubscribe_event(self, event_type: str, handler: EventHandler) -> None:
        """Unsubscribe from pipeline events."""
        if event_type in self._event_handlers:
            if handler in self._event_handlers[event_type]:
                self._event_handlers[event_type].remove(handler)
                logger.debug(f"Unsubscribed from event: {event_type}")
    
    async def emit_event(self, event: ConversationEvent) -> None:
        """Emit an event to all subscribers."""
        if event.event_type in self._event_handlers:
            for handler in self._event_handlers[event.event_type]:
                try:
                    await handler.handle(event)
                except Exception as e:
                    logger.error(f"Error in event handler for {event.event_type}: {e}")
    
    async def process(
        self, 
        context: ConversationContext, 
        core_processor: Callable[[ConversationContext], ConversationContext]
    ) -> ConversationContext:
        """
        Process conversation context through middleware pipeline.
        
        Args:
            context: Conversation context to process
            core_processor: Core processing function (STT -> Response -> TTS)
            
        Returns:
            Processed conversation context
        """
        if self._is_processing:
            logger.warning("Pipeline already processing, skipping concurrent request")
            return context
        
        self._is_processing = True
        
        try:
            # Emit processing start event
            await self.emit_event(ConversationEvent("processing_start", {"context": context}))
            
            # Build middleware chain
            async def pipeline_chain(ctx: ConversationContext) -> ConversationContext:
                return await core_processor(ctx)
            
            # Wrap with middleware (in reverse order so first added executes first)
            for middleware in reversed(self.middleware):
                current_middleware = middleware
                current_func = pipeline_chain
                
                async def middleware_wrapper(
                    ctx: ConversationContext,
                    mw=current_middleware,
                    next_func=current_func
                ) -> ConversationContext:
                    return await mw.process(ctx, next_func)
                
                pipeline_chain = middleware_wrapper
            
            # Execute the complete pipeline
            result_context = await pipeline_chain(context)
            
            # Emit processing complete event
            await self.emit_event(ConversationEvent("processing_complete", {"context": result_context}))
            
            return result_context
            
        except Exception as e:
            logger.error(f"Pipeline processing error: {e}", exc_info=True)
            context.error = e
            
            # Emit error event
            await self.emit_event(ConversationEvent("processing_error", {"context": context, "error": e}))
            
            return context
        finally:
            self._is_processing = False
    
    @property
    def is_processing(self) -> bool:
        """Check if pipeline is currently processing."""
        return self._is_processing
    
    @property
    def middleware_count(self) -> int:
        """Get number of registered middleware."""
        return len(self.middleware)