"""
Timing middleware for conversation pipeline.

Measures and tracks processing times for different stages of the conversation.
"""

import time
import logging
from typing import Dict, Any, Optional
from datetime import datetime
from .base import BaseMiddleware
from ...core.interfaces import ConversationContext

logger = logging.getLogger(__name__)


class TimingMiddleware(BaseMiddleware):
    """Middleware that measures and tracks processing times."""
    
    def __init__(
        self,
        track_total_time: bool = True,
        track_stage_times: bool = True,
        log_timing: bool = True,
        log_threshold_ms: float = 1000.0
    ):
        """
        Initialize timing middleware.
        
        Args:
            track_total_time: Whether to track total processing time
            track_stage_times: Whether to track individual stage times
            log_timing: Whether to log timing information
            log_threshold_ms: Log threshold in milliseconds (log if processing takes longer)
        """
        super().__init__(name="timing")
        self.track_total_time = track_total_time
        self.track_stage_times = track_stage_times
        self.log_timing = log_timing
        self.log_threshold_ms = log_threshold_ms
        
        # Timing statistics
        self.total_requests = 0
        self.total_processing_time = 0.0
        self.min_processing_time: Optional[float] = None
        self.max_processing_time: Optional[float] = None
        
        # Stage timing statistics
        self.stage_times = {
            "stt": [],
            "response_generation": [],
            "tts": [],
            "total": []
        }
        
        logger.info(f"Timing middleware initialized: threshold={log_threshold_ms}ms")
    
    async def _pre_process(self, context: ConversationContext) -> None:
        """Record processing start time."""
        if not self.track_total_time and not self.track_stage_times:
            return
        
        # Store start time in context metadata
        if not context.metadata:
            context.metadata = {}
        
        context.metadata["timing_start"] = time.time()
        context.metadata["timing_stages"] = {}
        
        # Initialize stage timers
        if self.track_stage_times:
            context.metadata["timing_stage_start"] = time.time()
    
    async def _post_process(self, context: ConversationContext) -> None:
        """Record processing end time and calculate statistics."""
        if not context.metadata or "timing_start" not in context.metadata:
            return
        
        end_time = time.time()
        start_time = context.metadata["timing_start"]
        total_time = end_time - start_time
        
        # Update statistics
        self._update_statistics(total_time, context)
        
        # Store timing information in context
        context.metadata["timing_total"] = total_time
        context.metadata["timing_end"] = end_time
        
        # Log timing if enabled and above threshold
        if self.log_timing and (total_time * 1000) >= self.log_threshold_ms:
            self._log_timing(context, total_time)
    
    def _update_statistics(self, total_time: float, context: ConversationContext) -> None:
        """Update timing statistics."""
        self.total_requests += 1
        self.total_processing_time += total_time
        
        # Update min/max times
        if self.min_processing_time is None or total_time < self.min_processing_time:
            self.min_processing_time = total_time
        
        if self.max_processing_time is None or total_time > self.max_processing_time:
            self.max_processing_time = total_time
        
        # Store total time in stage statistics
        self.stage_times["total"].append(total_time)
        
        # Limit stored times to prevent memory growth
        max_stored_times = 1000
        for stage, times in self.stage_times.items():
            if len(times) > max_stored_times:
                self.stage_times[stage] = times[-max_stored_times//2:]
        
        # Extract stage timing information from metadata
        if context.metadata and "timing_stages" in context.metadata:
            stages = context.metadata["timing_stages"]
            
            for stage, stage_time in stages.items():
                if stage in self.stage_times:
                    self.stage_times[stage].append(stage_time)
    
    def _log_timing(self, context: ConversationContext, total_time: float) -> None:
        """Log timing information."""
        timing_info = {
            "total_time_ms": round(total_time * 1000, 2),
            "has_transcription": context.transcription is not None,
            "has_response": context.response_text is not None,
            "has_synthesis": context.synthesis_result is not None
        }
        
        # Add stage timing if available
        if context.metadata and "timing_stages" in context.metadata:
            stages = context.metadata["timing_stages"]
            stage_times_ms = {
                stage: round(time_val * 1000, 2) 
                for stage, time_val in stages.items()
            }
            timing_info["stage_times_ms"] = stage_times_ms
        
        # Add text lengths for context
        if context.transcription:
            timing_info["transcription_length"] = len(context.transcription.text)
        
        if context.response_text:
            timing_info["response_length"] = len(context.response_text)
        
        logger.info(f"Conversation timing: {timing_info}")
    
    # Utility methods for stage timing
    
    def start_stage_timer(self, context: ConversationContext, stage: str) -> None:
        """Start timing a specific stage."""
        if not self.track_stage_times or not context.metadata:
            return
        
        context.metadata[f"timing_{stage}_start"] = time.time()
    
    def end_stage_timer(self, context: ConversationContext, stage: str) -> None:
        """End timing a specific stage."""
        if not self.track_stage_times or not context.metadata:
            return
        
        start_key = f"timing_{stage}_start"
        if start_key in context.metadata:
            end_time = time.time()
            start_time = context.metadata[start_key]
            stage_time = end_time - start_time
            
            # Store stage time
            if "timing_stages" not in context.metadata:
                context.metadata["timing_stages"] = {}
            
            context.metadata["timing_stages"][stage] = stage_time
    
    # Statistics methods
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get timing statistics."""
        if self.total_requests == 0:
            return {"total_requests": 0}
        
        avg_time = self.total_processing_time / self.total_requests
        
        stats = {
            "total_requests": self.total_requests,
            "total_processing_time": round(self.total_processing_time, 3),
            "average_time_ms": round(avg_time * 1000, 2),
            "min_time_ms": round(self.min_processing_time * 1000, 2) if self.min_processing_time else None,
            "max_time_ms": round(self.max_processing_time * 1000, 2) if self.max_processing_time else None
        }
        
        # Add stage statistics
        if self.track_stage_times:
            stage_stats = {}
            
            for stage, times in self.stage_times.items():
                if times:
                    stage_stats[stage] = {
                        "count": len(times),
                        "avg_ms": round(sum(times) / len(times) * 1000, 2),
                        "min_ms": round(min(times) * 1000, 2),
                        "max_ms": round(max(times) * 1000, 2)
                    }
            
            if stage_stats:
                stats["stage_statistics"] = stage_stats
        
        return stats
    
    def get_percentile_stats(self, percentiles: list = [50, 90, 95, 99]) -> Dict[str, Any]:
        """Get percentile statistics for processing times."""
        import numpy as np
        
        if not self.stage_times["total"]:
            return {}
        
        times_ms = [t * 1000 for t in self.stage_times["total"]]
        
        percentile_stats = {}
        for p in percentiles:
            percentile_stats[f"p{p}"] = round(np.percentile(times_ms, p), 2)
        
        return {
            "total_requests": len(times_ms),
            "percentiles_ms": percentile_stats
        }
    
    def reset_statistics(self) -> None:
        """Reset timing statistics."""
        self.total_requests = 0
        self.total_processing_time = 0.0
        self.min_processing_time = None
        self.max_processing_time = None
        
        for stage in self.stage_times:
            self.stage_times[stage].clear()
        
        logger.info("Timing statistics reset")
    
    # Configuration methods
    
    def set_log_threshold(self, threshold_ms: float) -> None:
        """Set logging threshold in milliseconds."""
        self.log_threshold_ms = threshold_ms
        logger.info(f"Timing log threshold set to: {threshold_ms}ms")
    
    def enable_stage_timing(self, enabled: bool = True) -> None:
        """Enable or disable stage timing."""
        self.track_stage_times = enabled
        logger.info(f"Stage timing {'enabled' if enabled else 'disabled'}")
    
    def get_config(self) -> Dict[str, Any]:
        """Get current timing configuration."""
        return {
            "name": self.name,
            "enabled": self.enabled,
            "track_total_time": self.track_total_time,
            "track_stage_times": self.track_stage_times,
            "log_timing": self.log_timing,
            "log_threshold_ms": self.log_threshold_ms,
            "total_requests": self.total_requests
        }


# Factory functions

def create_timing_middleware(
    log_slow_requests: bool = True,
    threshold_ms: float = 1000.0
) -> TimingMiddleware:
    """
    Create a timing middleware with standard settings.
    
    Args:
        log_slow_requests: Whether to log slow requests
        threshold_ms: Threshold for logging slow requests
        
    Returns:
        Configured timing middleware
    """
    return TimingMiddleware(
        log_timing=log_slow_requests,
        log_threshold_ms=threshold_ms
    )


def create_timing_middleware_from_config(config: Dict[str, Any]) -> TimingMiddleware:
    """
    Create timing middleware from configuration.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Configured timing middleware
    """
    return TimingMiddleware(
        track_total_time=config.get("track_total_time", True),
        track_stage_times=config.get("track_stage_times", True),
        log_timing=config.get("log_timing", True),
        log_threshold_ms=config.get("log_threshold_ms", 1000.0)
    )