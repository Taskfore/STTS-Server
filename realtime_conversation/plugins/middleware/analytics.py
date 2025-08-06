"""
Analytics middleware for conversation pipeline.

Collects usage metrics, performance data, and conversation statistics.
"""

import time
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from collections import defaultdict, deque
from .base import BaseMiddleware
from ...core.interfaces import ConversationContext, ConversationState

logger = logging.getLogger(__name__)


class AnalyticsMiddleware(BaseMiddleware):
    """Middleware that collects analytics and usage metrics."""
    
    def __init__(
        self,
        track_usage: bool = True,
        track_performance: bool = True,
        track_errors: bool = True,
        retention_days: int = 30,
        max_events: int = 10000
    ):
        """
        Initialize analytics middleware.
        
        Args:
            track_usage: Whether to track usage metrics
            track_performance: Whether to track performance metrics
            track_errors: Whether to track error metrics
            retention_days: Number of days to retain analytics data
            max_events: Maximum number of events to store
        """
        super().__init__(name="analytics")
        self.track_usage = track_usage
        self.track_performance = track_performance
        self.track_errors = track_errors
        self.retention_days = retention_days
        self.max_events = max_events
        
        # Usage metrics
        self.total_conversations = 0
        self.successful_conversations = 0
        self.failed_conversations = 0
        self.total_audio_duration = 0.0
        self.total_response_length = 0
        
        # Performance metrics
        self.processing_times = deque(maxlen=max_events)
        self.stt_times = deque(maxlen=max_events)
        self.tts_times = deque(maxlen=max_events)
        
        # Language statistics
        self.language_counts = defaultdict(int)
        
        # Error tracking
        self.error_counts = defaultdict(int)
        self.error_details = deque(maxlen=max_events)
        
        # Time-based metrics
        self.hourly_usage = defaultdict(int)
        self.daily_usage = defaultdict(int)
        
        # Events log
        self.events = deque(maxlen=max_events)
        
        logger.info(f"Analytics middleware initialized: retention={retention_days} days")
    
    async def _pre_process(self, context: ConversationContext) -> None:
        """Record conversation start analytics."""
        if not self.track_usage:
            return
        
        event_data = {
            "timestamp": datetime.now(),
            "event_type": "conversation_start",
            "has_audio": context.audio_input is not None,
            "state": context.state.value if context.state else "unknown"
        }
        
        # Record audio information
        if context.audio_input:
            event_data["audio_info"] = {
                "duration_ms": context.audio_input.duration_ms,
                "sample_rate": context.audio_input.sample_rate,
                "format": context.audio_input.format
            }
            
            # Add to total audio duration
            self.total_audio_duration += context.audio_input.duration_seconds
        
        # Record user information if available
        if context.user_data:
            user_info = {}
            for key in ["user_id", "session_id", "client_type"]:
                if key in context.user_data:
                    user_info[key] = context.user_data[key]
            
            if user_info:
                event_data["user_info"] = user_info
        
        self._add_event(event_data)
        self.total_conversations += 1
        
        # Update time-based usage
        now = datetime.now()
        hour_key = now.strftime("%Y-%m-%d_%H")
        day_key = now.strftime("%Y-%m-%d")
        
        self.hourly_usage[hour_key] += 1
        self.daily_usage[day_key] += 1
    
    async def _post_process(self, context: ConversationContext) -> None:
        """Record conversation completion analytics."""
        if not self.track_usage:
            return
        
        event_data = {
            "timestamp": datetime.now(),
            "event_type": "conversation_complete",
            "success": context.error is None,
            "state": context.state.value if context.state else "unknown"
        }
        
        # Record transcription metrics
        if context.transcription:
            transcription_data = {
                "language": context.transcription.language,
                "text_length": len(context.transcription.text),
                "segments": len(context.transcription.segments),
                "duration": context.transcription.duration,
                "confidence": context.transcription.confidence
            }
            event_data["transcription"] = transcription_data
            
            # Update language statistics
            self.language_counts[context.transcription.language] += 1
        
        # Record response metrics
        if context.response_text:
            response_data = {
                "text_length": len(context.response_text)
            }
            event_data["response"] = response_data
            
            # Add to total response length
            self.total_response_length += len(context.response_text)
        
        # Record synthesis metrics
        if context.synthesis_result:
            synthesis_data = {
                "voice_id": context.synthesis_result.voice_id,
                "duration_ms": context.synthesis_result.audio_data.duration_ms,
                "synthesis_time": context.synthesis_result.synthesis_time
            }
            event_data["synthesis"] = synthesis_data
        
        # Record performance metrics
        if self.track_performance and context.metadata:
            performance_data = self._extract_performance_data(context.metadata)
            if performance_data:
                event_data["performance"] = performance_data
                
                # Store in performance collections
                if "total_time" in performance_data:
                    self.processing_times.append(performance_data["total_time"])
                
                if "stt_time" in performance_data:
                    self.stt_times.append(performance_data["stt_time"])
                
                if "tts_time" in performance_data:
                    self.tts_times.append(performance_data["tts_time"])
        
        # Update success/failure counts
        if context.error is None:
            self.successful_conversations += 1
        else:
            self.failed_conversations += 1
        
        self._add_event(event_data)
    
    async def _handle_error(self, context: ConversationContext, error: Exception) -> None:
        """Record error analytics."""
        if not self.track_errors:
            return
        
        error_type = type(error).__name__
        error_message = str(error)
        
        # Update error counts
        self.error_counts[error_type] += 1
        
        # Record error details
        error_detail = {
            "timestamp": datetime.now(),
            "error_type": error_type,
            "error_message": error_message,
            "state": context.state.value if context.state else "unknown",
            "has_audio": context.audio_input is not None,
            "has_transcription": context.transcription is not None
        }
        
        self.error_details.append(error_detail)
        
        # Record as event
        event_data = {
            "timestamp": datetime.now(),
            "event_type": "conversation_error",
            "error": error_detail
        }
        
        self._add_event(event_data)
    
    def _add_event(self, event_data: Dict[str, Any]) -> None:
        """Add event to events log."""
        self.events.append(event_data)
        
        # Clean up old events based on retention policy
        self._cleanup_old_events()
    
    def _cleanup_old_events(self) -> None:
        """Clean up old events based on retention policy."""
        if not self.events:
            return
        
        cutoff_date = datetime.now() - timedelta(days=self.retention_days)
        
        # Remove old events
        while self.events and self.events[0]["timestamp"] < cutoff_date:
            self.events.popleft()
    
    def _extract_performance_data(self, metadata: Dict[str, Any]) -> Dict[str, float]:
        """Extract performance timing data from context metadata."""
        performance_data = {}
        
        # Extract total time
        if "timing_total" in metadata:
            performance_data["total_time"] = metadata["timing_total"]
        
        # Extract stage times
        if "timing_stages" in metadata:
            stages = metadata["timing_stages"]
            for stage, time_val in stages.items():
                performance_data[f"{stage}_time"] = time_val
        
        return performance_data
    
    # Analytics retrieval methods
    
    def get_usage_statistics(self) -> Dict[str, Any]:
        """Get usage statistics."""
        success_rate = (
            self.successful_conversations / self.total_conversations * 100
            if self.total_conversations > 0 else 0
        )
        
        avg_audio_duration = (
            self.total_audio_duration / self.total_conversations
            if self.total_conversations > 0 else 0
        )
        
        avg_response_length = (
            self.total_response_length / self.successful_conversations
            if self.successful_conversations > 0 else 0
        )
        
        return {
            "total_conversations": self.total_conversations,
            "successful_conversations": self.successful_conversations,
            "failed_conversations": self.failed_conversations,
            "success_rate_percent": round(success_rate, 2),
            "total_audio_duration_seconds": round(self.total_audio_duration, 2),
            "average_audio_duration_seconds": round(avg_audio_duration, 2),
            "total_response_length": self.total_response_length,
            "average_response_length": round(avg_response_length, 2),
            "languages_detected": dict(self.language_counts)
        }
    
    def get_performance_statistics(self) -> Dict[str, Any]:
        """Get performance statistics."""
        if not self.processing_times:
            return {"message": "No performance data available"}
        
        def calc_stats(times):
            if not times:
                return {}
            
            times_ms = [t * 1000 for t in times]
            return {
                "count": len(times_ms),
                "avg_ms": round(sum(times_ms) / len(times_ms), 2),
                "min_ms": round(min(times_ms), 2),
                "max_ms": round(max(times_ms), 2)
            }
        
        return {
            "total_processing": calc_stats(self.processing_times),
            "stt_processing": calc_stats(self.stt_times),
            "tts_processing": calc_stats(self.tts_times)
        }
    
    def get_error_statistics(self) -> Dict[str, Any]:
        """Get error statistics."""
        total_errors = sum(self.error_counts.values())
        
        return {
            "total_errors": total_errors,
            "error_types": dict(self.error_counts),
            "recent_errors": [
                {
                    "timestamp": error["timestamp"].isoformat(),
                    "type": error["error_type"],
                    "message": error["error_message"][:100] + "..." if len(error["error_message"]) > 100 else error["error_message"]
                }
                for error in list(self.error_details)[-10:]  # Last 10 errors
            ]
        }
    
    def get_time_based_usage(self, period: str = "daily") -> Dict[str, int]:
        """
        Get time-based usage statistics.
        
        Args:
            period: "hourly" or "daily"
            
        Returns:
            Usage counts by time period
        """
        if period == "hourly":
            return dict(self.hourly_usage)
        elif period == "daily":
            return dict(self.daily_usage)
        else:
            raise ValueError("Period must be 'hourly' or 'daily'")
    
    def get_recent_events(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get recent events."""
        recent_events = list(self.events)[-limit:]
        
        # Convert timestamps to ISO format for JSON serialization
        for event in recent_events:
            if "timestamp" in event:
                event["timestamp"] = event["timestamp"].isoformat()
        
        return recent_events
    
    # Configuration methods
    
    def reset_analytics(self) -> None:
        """Reset all analytics data."""
        self.total_conversations = 0
        self.successful_conversations = 0
        self.failed_conversations = 0
        self.total_audio_duration = 0.0
        self.total_response_length = 0
        
        self.processing_times.clear()
        self.stt_times.clear()
        self.tts_times.clear()
        
        self.language_counts.clear()
        self.error_counts.clear()
        self.error_details.clear()
        
        self.hourly_usage.clear()
        self.daily_usage.clear()
        self.events.clear()
        
        logger.info("Analytics data reset")
    
    def set_retention_policy(self, days: int, max_events: int) -> None:
        """Set data retention policy."""
        self.retention_days = days
        self.max_events = max_events
        
        # Update deque max lengths
        self.processing_times = deque(self.processing_times, maxlen=max_events)
        self.stt_times = deque(self.stt_times, maxlen=max_events)
        self.tts_times = deque(self.tts_times, maxlen=max_events)
        self.error_details = deque(self.error_details, maxlen=max_events)
        self.events = deque(self.events, maxlen=max_events)
        
        logger.info(f"Retention policy updated: {days} days, {max_events} max events")
    
    def get_config(self) -> Dict[str, Any]:
        """Get current analytics configuration."""
        return {
            "name": self.name,
            "enabled": self.enabled,
            "track_usage": self.track_usage,
            "track_performance": self.track_performance,
            "track_errors": self.track_errors,
            "retention_days": self.retention_days,
            "max_events": self.max_events,
            "total_conversations": self.total_conversations,
            "events_stored": len(self.events)
        }


# Factory functions

def create_analytics_middleware(
    full_tracking: bool = True,
    retention_days: int = 30
) -> AnalyticsMiddleware:
    """
    Create analytics middleware with standard settings.
    
    Args:
        full_tracking: Whether to enable all tracking features
        retention_days: Number of days to retain data
        
    Returns:
        Configured analytics middleware
    """
    return AnalyticsMiddleware(
        track_usage=full_tracking,
        track_performance=full_tracking,
        track_errors=full_tracking,
        retention_days=retention_days
    )


def create_analytics_middleware_from_config(config: Dict[str, Any]) -> AnalyticsMiddleware:
    """
    Create analytics middleware from configuration.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Configured analytics middleware
    """
    return AnalyticsMiddleware(
        track_usage=config.get("track_usage", True),
        track_performance=config.get("track_performance", True),
        track_errors=config.get("track_errors", True),
        retention_days=config.get("retention_days", 30),
        max_events=config.get("max_events", 10000)
    )