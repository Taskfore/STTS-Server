# File: middleware/base.py
# Base middleware system for processing TTS/STT requests

import time
import logging
import asyncio
from typing import Dict, Any, Callable, List, Optional
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class RequestContext:
    """Context object that flows through middleware pipeline."""
    
    # Request information
    request_id: str
    request_type: str  # "tts", "stt", "conversation"
    start_time: float = field(default_factory=time.time)
    
    # Request data
    input_data: Dict[str, Any] = field(default_factory=dict)
    config: Dict[str, Any] = field(default_factory=dict)
    
    # Processing results
    output_data: Dict[str, Any] = field(default_factory=dict)
    
    # Metadata and metrics
    metadata: Dict[str, Any] = field(default_factory=dict)
    metrics: Dict[str, Any] = field(default_factory=dict)
    
    # State
    status: str = "processing"  # "processing", "completed", "error"
    error: Optional[Exception] = None
    
    @property
    def duration_ms(self) -> float:
        """Get request duration in milliseconds."""
        return (time.time() - self.start_time) * 1000
    
    def add_metric(self, key: str, value: Any):
        """Add a metric to the context."""
        self.metrics[key] = value
        
    def add_metadata(self, key: str, value: Any):
        """Add metadata to the context."""
        self.metadata[key] = value


class BaseMiddleware(ABC):
    """Base class for all middleware components."""
    
    def __init__(self, name: str = None):
        self.name = name or self.__class__.__name__
        self.enabled = True
        
    @abstractmethod
    async def process(self, context: RequestContext, next_middleware: Callable) -> RequestContext:
        """
        Process the request context and call the next middleware.
        
        Args:
            context: Current request context
            next_middleware: Next middleware function in the chain
            
        Returns:
            Modified request context
        """
        pass
    
    def enable(self):
        """Enable this middleware."""
        self.enabled = True
        
    def disable(self):
        """Disable this middleware."""
        self.enabled = False


class MiddlewarePipeline:
    """Pipeline for executing middleware in sequence."""
    
    def __init__(self):
        self.middleware: List[BaseMiddleware] = []
        
    def add_middleware(self, middleware: BaseMiddleware):
        """Add middleware to the pipeline."""
        self.middleware.append(middleware)
        logger.debug(f"Added middleware: {middleware.name}")
        
    def remove_middleware(self, middleware: BaseMiddleware):
        """Remove middleware from the pipeline."""
        if middleware in self.middleware:
            self.middleware.remove(middleware)
            logger.debug(f"Removed middleware: {middleware.name}")
            
    def clear(self):
        """Clear all middleware."""
        self.middleware.clear()
        logger.debug("Cleared all middleware")
        
    async def process(self, context: RequestContext, core_processor: Callable) -> RequestContext:
        """
        Process request through middleware pipeline.
        
        Args:
            context: Request context to process
            core_processor: Core processing function (TTS/STT/etc.)
            
        Returns:
            Processed request context
        """
        try:
            # Build the middleware chain
            async def pipeline_chain(ctx: RequestContext) -> RequestContext:
                return await core_processor(ctx)
            
            # Wrap with middleware (in reverse order so first added executes first)
            for middleware in reversed(self.middleware):
                if not middleware.enabled:
                    continue
                    
                current_middleware = middleware
                current_func = pipeline_chain
                
                async def middleware_wrapper(
                    ctx: RequestContext,
                    mw=current_middleware,
                    next_func=current_func
                ) -> RequestContext:
                    return await mw.process(ctx, next_func)
                
                pipeline_chain = middleware_wrapper
            
            # Execute the complete pipeline
            result_context = await pipeline_chain(context)
            result_context.status = "completed"
            
            return result_context
            
        except Exception as e:
            logger.error(f"Pipeline processing error: {e}", exc_info=True)
            context.error = e
            context.status = "error"
            return context
    
    @property
    def enabled_middleware_count(self) -> int:
        """Get count of enabled middleware."""
        return sum(1 for mw in self.middleware if mw.enabled)
    
    def get_middleware_names(self) -> List[str]:
        """Get names of all middleware."""
        return [mw.name for mw in self.middleware]


# Built-in middleware components

class TimingMiddleware(BaseMiddleware):
    """Middleware that tracks request timing and performance metrics."""
    
    def __init__(self, name: str = "TimingMiddleware"):
        super().__init__(name)
        self.request_times: List[float] = []
        self.max_history = 1000
        
    async def process(self, context: RequestContext, next_middleware: Callable) -> RequestContext:
        """Add timing information to the request context."""
        start_time = time.time()
        
        # Add timing metadata
        context.add_metadata("middleware_start_time", start_time)
        context.add_metadata("processing_started_at", datetime.now().isoformat())
        
        try:
            # Process through next middleware
            result_context = await next_middleware(context)
            
            # Calculate timing
            end_time = time.time()
            duration_ms = (end_time - start_time) * 1000
            
            # Add timing metrics
            result_context.add_metric("processing_duration_ms", duration_ms)
            result_context.add_metric("total_duration_ms", result_context.duration_ms)
            result_context.add_metadata("processing_completed_at", datetime.now().isoformat())
            
            # Track history
            self.request_times.append(duration_ms)
            if len(self.request_times) > self.max_history:
                self.request_times.pop(0)
            
            logger.debug(f"Request {context.request_id} processed in {duration_ms:.2f}ms")
            
            return result_context
            
        except Exception as e:
            end_time = time.time()
            duration_ms = (end_time - start_time) * 1000
            
            context.add_metric("error_duration_ms", duration_ms)
            context.add_metadata("error_occurred_at", datetime.now().isoformat())
            
            logger.error(f"Request {context.request_id} failed after {duration_ms:.2f}ms: {e}")
            raise
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get timing statistics."""
        if not self.request_times:
            return {"request_count": 0}
        
        return {
            "request_count": len(self.request_times),
            "average_duration_ms": sum(self.request_times) / len(self.request_times),
            "min_duration_ms": min(self.request_times),
            "max_duration_ms": max(self.request_times),
            "recent_duration_ms": self.request_times[-1] if self.request_times else 0
        }


class LoggingMiddleware(BaseMiddleware):
    """Middleware that provides detailed logging of requests."""
    
    def __init__(self, name: str = "LoggingMiddleware", log_level: int = logging.INFO):
        super().__init__(name)
        self.log_level = log_level
        self.request_count = 0
        
    async def process(self, context: RequestContext, next_middleware: Callable) -> RequestContext:
        """Log request details and processing."""
        self.request_count += 1
        
        # Log request start
        logger.log(
            self.log_level,
            f"[{context.request_id}] {context.request_type.upper()} request started "
            f"(#{self.request_count})"
        )
        
        # Log input data summary
        input_summary = self._summarize_input(context.input_data)
        if input_summary:
            logger.log(self.log_level, f"[{context.request_id}] Input: {input_summary}")
        
        try:
            # Process through next middleware
            result_context = await next_middleware(context)
            
            # Log completion
            output_summary = self._summarize_output(result_context.output_data)
            duration = result_context.duration_ms
            
            logger.log(
                self.log_level,
                f"[{context.request_id}] {context.request_type.upper()} completed "
                f"in {duration:.2f}ms - {output_summary}"
            )
            
            return result_context
            
        except Exception as e:
            # Log error
            duration = context.duration_ms
            logger.error(
                f"[{context.request_id}] {context.request_type.upper()} failed "
                f"after {duration:.2f}ms: {str(e)}"
            )
            raise
    
    def _summarize_input(self, input_data: Dict[str, Any]) -> str:
        """Create a summary of input data for logging."""
        summary_parts = []
        
        if "text" in input_data:
            text = input_data["text"]
            if len(text) > 50:
                summary_parts.append(f"text='{text[:50]}...' ({len(text)} chars)")
            else:
                summary_parts.append(f"text='{text}'")
                
        if "voice_mode" in input_data:
            summary_parts.append(f"voice_mode={input_data['voice_mode']}")
            
        if "voice_id" in input_data:
            summary_parts.append(f"voice_id={input_data['voice_id']}")
            
        if "output_format" in input_data:
            summary_parts.append(f"format={input_data['output_format']}")
            
        if "audio_size" in input_data:
            summary_parts.append(f"audio_size={input_data['audio_size']} bytes")
        
        return ", ".join(summary_parts)
    
    def _summarize_output(self, output_data: Dict[str, Any]) -> str:
        """Create a summary of output data for logging."""
        summary_parts = []
        
        if "audio_size" in output_data:
            summary_parts.append(f"audio_size={output_data['audio_size']} bytes")
            
        if "transcribed_text" in output_data:
            text = output_data["transcribed_text"]
            if len(text) > 50:
                summary_parts.append(f"transcribed='{text[:50]}...' ({len(text)} chars)")
            else:
                summary_parts.append(f"transcribed='{text}'")
                
        if "audio_duration_ms" in output_data:
            summary_parts.append(f"duration={output_data['audio_duration_ms']}ms")
        
        return ", ".join(summary_parts) if summary_parts else "success"
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get logging statistics."""
        return {
            "total_requests": self.request_count,
            "log_level": logging.getLevelName(self.log_level)
        }


class AnalyticsMiddleware(BaseMiddleware):
    """Middleware that collects analytics and usage statistics."""
    
    def __init__(self, name: str = "AnalyticsMiddleware"):
        super().__init__(name)
        self.request_counts: Dict[str, int] = {}
        self.error_counts: Dict[str, int] = {}
        self.voice_usage: Dict[str, int] = {}
        self.format_usage: Dict[str, int] = {}
        
    async def process(self, context: RequestContext, next_middleware: Callable) -> RequestContext:
        """Collect analytics data."""
        # Track request type
        self.request_counts[context.request_type] = self.request_counts.get(context.request_type, 0) + 1
        
        # Track voice usage
        if "voice_id" in context.input_data:
            voice_id = context.input_data["voice_id"]
            self.voice_usage[voice_id] = self.voice_usage.get(voice_id, 0) + 1
            
        # Track output format
        if "output_format" in context.input_data:
            output_format = context.input_data["output_format"]
            self.format_usage[output_format] = self.format_usage.get(output_format, 0) + 1
        
        try:
            # Process through next middleware
            result_context = await next_middleware(context)
            
            # Add analytics metadata
            result_context.add_metadata("analytics_tracked", True)
            
            return result_context
            
        except Exception as e:
            # Track errors
            error_type = type(e).__name__
            self.error_counts[error_type] = self.error_counts.get(error_type, 0) + 1
            raise
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get analytics statistics."""
        total_requests = sum(self.request_counts.values())
        total_errors = sum(self.error_counts.values())
        
        return {
            "total_requests": total_requests,
            "total_errors": total_errors,
            "error_rate": (total_errors / total_requests * 100) if total_requests > 0 else 0,
            "request_counts": dict(self.request_counts),
            "error_counts": dict(self.error_counts),
            "voice_usage": dict(self.voice_usage),
            "format_usage": dict(self.format_usage)
        }