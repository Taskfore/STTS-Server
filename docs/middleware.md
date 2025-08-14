# Middleware System Documentation

## Overview

The middleware system provides a comprehensive request processing pipeline for the STTS Server. It implements cross-cutting concerns like timing, logging, analytics, and error handling in a modular, extensible way.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                  Middleware Pipeline                        │
├─────────────────────┬─────────────────────┬─────────────────┤
│   Timing           │     Logging         │   Analytics     │
│   Middleware       │     Middleware      │   Middleware    │
│                    │                     │                 │
│ • Request timing   │ • Context logging   │ • Usage stats   │
│ • Performance      │ • Error tracking    │ • Voice usage   │
│ • Duration metrics │ • Request correlation│ • Format prefs  │
└─────────────────────┼─────────────────────┴─────────────────┘
                      │
                      ▼
            ┌─────────────────┐
            │  Core Processor │
            │  (TTS/STT/etc)  │
            └─────────────────┘
```

## Core Concepts

### Request Context

The `RequestContext` is the foundation of the middleware system:

```python
@dataclass
class RequestContext:
    request_id: str              # Unique correlation ID
    request_type: str           # "tts", "stt", "conversation"
    start_time: float           # Request start timestamp
    input_data: Dict[str, Any]  # Request parameters
    output_data: Dict[str, Any] # Response data
    metrics: Dict[str, Any]     # Performance metrics
    metadata: Dict[str, Any]    # Processing metadata
    status: str                 # "processing", "completed", "error"
```

### Middleware Interface

All middleware implements the `BaseMiddleware` protocol:

```python
class BaseMiddleware(Protocol):
    async def process(
        self, 
        context: RequestContext, 
        next_processor: Callable[[RequestContext], Awaitable[RequestContext]]
    ) -> RequestContext:
        """Process request context through middleware."""
        pass
```

## Built-in Middleware

### 1. Timing Middleware

Tracks request duration and performance metrics.

#### Features
- **Request Duration**: Total time from start to completion
- **Processing Time**: Core processing time excluding middleware overhead
- **Performance Tracking**: Historical performance data
- **Threshold Alerts**: Warnings for slow requests

#### Usage
```python
# Automatic timing for all requests
timing_middleware = TimingMiddleware()
pipeline.add_middleware(timing_middleware)

# Access timing data
context = await pipeline.process(context, core_processor)
duration = context.metrics.get('duration', 0)
processing_time = context.metrics.get('processing_time', 0)
```

#### Configuration
```python
# config.yaml
middleware:
  timing:
    slow_request_threshold: 5.0  # seconds
    enable_historical_tracking: true
    max_history_entries: 1000
```

#### Metrics Collected
- `duration`: Total request duration (seconds)
- `processing_time`: Core processing time (seconds)
- `middleware_overhead`: Middleware processing time (seconds)
- `timestamp`: Request completion timestamp
- `slow_request`: Boolean flag for slow requests

### 2. Logging Middleware

Provides context-aware logging with request correlation.

#### Features
- **Request Correlation**: Unique request IDs for tracing
- **Context Logging**: Structured logging with request context
- **Error Tracking**: Detailed error logging with stack traces
- **Performance Logging**: Request timing and resource usage

#### Usage
```python
# Automatic logging for all requests
logging_middleware = LoggingMiddleware()
pipeline.add_middleware(logging_middleware)

# Logs include request context
# INFO: [req_abc123] TTS request started: text_length=50
# INFO: [req_abc123] TTS completed in 2.34s
```

#### Configuration
```python
# config.yaml
middleware:
  logging:
    log_level: "INFO"
    include_request_data: true
    include_response_data: false
    log_slow_requests: true
    slow_request_threshold: 3.0
```

#### Log Format
```
[{timestamp}] {level}: [{request_id}] {message}
Context: {context_data}
```

### 3. Analytics Middleware

Collects usage statistics and system insights.

#### Features
- **Usage Tracking**: Request counts, types, patterns
- **Voice Analytics**: Voice usage patterns and preferences
- **Performance Analytics**: System performance trends
- **Error Analytics**: Error rates and failure patterns

#### Usage
```python
# Automatic analytics collection
analytics_middleware = AnalyticsMiddleware()
pipeline.add_middleware(analytics_middleware)

# Access analytics data via API
GET /tts/statistics
```

#### Configuration
```python
# config.yaml
middleware:
  analytics:
    enable_usage_tracking: true
    enable_voice_analytics: true
    enable_performance_tracking: true
    retention_days: 30
    aggregation_interval: 3600  # seconds
```

#### Data Collected
- **Request Statistics**: Count, types, success rates
- **Voice Usage**: Voice ID usage frequency, preferences
- **Performance Data**: Average response times, throughput
- **Error Statistics**: Error rates, common failure types
- **System Metrics**: Resource usage, capacity planning data

## Pipeline Management

### Creating a Pipeline

```python
from middleware.base import MiddlewarePipeline

# Create pipeline
pipeline = MiddlewarePipeline()

# Add middleware in order
pipeline.add_middleware(TimingMiddleware())
pipeline.add_middleware(LoggingMiddleware())
pipeline.add_middleware(AnalyticsMiddleware())
```

### Processing Requests

```python
async def process_tts_request(request_data):
    # Create request context
    context = RequestContext(
        request_id=str(uuid.uuid4())[:8],
        request_type="tts",
        input_data=request_data
    )
    
    # Process through pipeline
    result_context = await pipeline.process(context, core_tts_processor)
    
    # Return response
    return result_context.output_data
```

### Custom Core Processor

```python
async def core_tts_processor(context: RequestContext) -> RequestContext:
    """Core TTS processing logic."""
    try:
        # Extract input data
        text = context.input_data.get('text', '')
        voice_config = context.input_data.get('voice_config', {})
        
        # Process with TTS adapter
        result = await tts_adapter.synthesize(text, voice_config)
        
        # Store output
        context.output_data = {
            'audio_data': result.audio_data,
            'voice_id': result.voice_id,
            'duration': result.duration
        }
        
        context.status = "completed"
        
    except Exception as e:
        context.status = "error"
        context.metadata['error'] = str(e)
        
    return context
```

## Dependency Injection

### Pipeline as Dependency

```python
from fastapi import Depends

def get_middleware_pipeline() -> MiddlewarePipeline:
    """Get the middleware pipeline instance."""
    global _pipeline_instance
    
    if _pipeline_instance is None:
        _pipeline_instance = create_middleware_pipeline()
    
    return _pipeline_instance

@router.post("/tts")
async def synthesize_speech(
    request: TTSRequest,
    pipeline: MiddlewarePipeline = Depends(get_middleware_pipeline)
):
    context = RequestContext(...)
    result = await pipeline.process(context, core_processor)
    return result.output_data
```

### Configuration Integration

```python
def create_middleware_pipeline() -> MiddlewarePipeline:
    """Create configured middleware pipeline."""
    pipeline = MiddlewarePipeline()
    
    # Add middleware based on configuration
    if config_manager.get_bool("middleware.timing.enabled", True):
        pipeline.add_middleware(TimingMiddleware())
    
    if config_manager.get_bool("middleware.logging.enabled", True):
        pipeline.add_middleware(LoggingMiddleware())
    
    if config_manager.get_bool("middleware.analytics.enabled", True):
        pipeline.add_middleware(AnalyticsMiddleware())
    
    return pipeline
```

## Custom Middleware

### Creating Custom Middleware

```python
class CustomMiddleware:
    """Example custom middleware."""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
    
    async def process(
        self, 
        context: RequestContext, 
        next_processor: Callable[[RequestContext], Awaitable[RequestContext]]
    ) -> RequestContext:
        # Pre-processing
        context.metadata['custom_start'] = time.time()
        
        try:
            # Call next middleware/processor
            context = await next_processor(context)
            
            # Post-processing
            context.metadata['custom_duration'] = time.time() - context.metadata['custom_start']
            
        except Exception as e:
            # Error handling
            context.metadata['custom_error'] = str(e)
            raise
        
        return context
```

### Registering Custom Middleware

```python
# Add to pipeline
pipeline.add_middleware(CustomMiddleware(config={'option': 'value'}))

# Or use factory function
def create_custom_middleware() -> CustomMiddleware:
    config = config_manager.get_dict("middleware.custom", {})
    return CustomMiddleware(config)

pipeline.add_middleware(create_custom_middleware())
```

## API Endpoints

### Middleware Status

```http
GET /tts/middleware/status
```

Response:
```json
{
  "total_middleware": 3,
  "enabled_middleware": 3,
  "middleware_names": [
    "TimingMiddleware",
    "LoggingMiddleware", 
    "AnalyticsMiddleware"
  ],
  "pipeline_status": "active"
}
```

### Middleware Reload

```http
POST /tts/middleware/reload
```

Response:
```json
{
  "status": "success",
  "message": "Middleware pipeline reloaded",
  "middleware_count": 3
}
```

### Statistics

```http
GET /tts/statistics
```

Response:
```json
{
  "system": {
    "adapter_type": "LegacyTTSEngineAdapter",
    "library_available": true,
    "middleware_enabled": true
  },
  "middleware": {
    "enabled_count": 3,
    "total_requests": 150,
    "average_duration": 2.34,
    "error_rate": 0.02
  },
  "analytics": {
    "popular_voices": ["female_voice_01", "male_voice_02"],
    "request_patterns": {...},
    "performance_trends": {...}
  }
}
```

## Error Handling

### Middleware Error Isolation

```python
async def process(self, context: RequestContext, next_processor) -> RequestContext:
    try:
        # Middleware processing
        return await next_processor(context)
        
    except MiddlewareError as e:
        # Handle middleware-specific errors
        logger.error(f"Middleware error: {e}")
        context.metadata['middleware_error'] = str(e)
        return context
        
    except Exception as e:
        # Handle unexpected errors
        logger.error(f"Unexpected middleware error: {e}", exc_info=True)
        # Don't break the pipeline
        return await next_processor(context)
```

### Error Context Preservation

```python
class ErrorHandlingMiddleware:
    async def process(self, context: RequestContext, next_processor) -> RequestContext:
        try:
            return await next_processor(context)
        except Exception as e:
            # Preserve error context
            context.status = "error"
            context.metadata.update({
                'error_type': type(e).__name__,
                'error_message': str(e),
                'error_timestamp': time.time()
            })
            
            # Re-raise for proper error handling
            raise
```

## Performance Optimization

### Middleware Overhead

- **Target**: < 5ms overhead per request
- **Measurement**: Built-in timing middleware tracks overhead
- **Optimization**: Efficient context handling and minimal processing

### Memory Management

```python
class OptimizedMiddleware:
    def __init__(self):
        self._stats_cache = {}
        self._last_cleanup = time.time()
    
    async def process(self, context: RequestContext, next_processor) -> RequestContext:
        # Periodic cleanup
        if time.time() - self._last_cleanup > 3600:  # 1 hour
            self._cleanup_cache()
        
        # Process with minimal memory footprint
        return await next_processor(context)
    
    def _cleanup_cache(self):
        # Clean up old entries
        self._stats_cache.clear()
        self._last_cleanup = time.time()
```

### Efficient Context Updates

```python
# Efficient context updates
context.metrics.update({
    'key1': value1,
    'key2': value2
})

# Avoid frequent individual updates
# context.metrics['key1'] = value1  # Less efficient
# context.metrics['key2'] = value2
```

## Testing Middleware

### Unit Testing

```python
import pytest
from middleware.base import TimingMiddleware, RequestContext

@pytest.fixture
def timing_middleware():
    return TimingMiddleware()

@pytest.fixture  
def sample_context():
    return RequestContext(
        request_id="test_123",
        request_type="tts"
    )

async def test_timing_middleware(timing_middleware, sample_context):
    """Test timing middleware functionality."""
    
    async def mock_processor(context):
        await asyncio.sleep(0.1)  # Simulate processing
        return context
    
    result = await timing_middleware.process(sample_context, mock_processor)
    
    assert 'duration' in result.metrics
    assert result.metrics['duration'] >= 0.1
    assert 'timestamp' in result.metrics
```

### Integration Testing

```python
async def test_full_pipeline():
    """Test complete middleware pipeline."""
    pipeline = MiddlewarePipeline()
    pipeline.add_middleware(TimingMiddleware())
    pipeline.add_middleware(LoggingMiddleware())
    
    context = RequestContext(request_id="integration_test", request_type="test")
    
    async def core_processor(ctx):
        ctx.output_data = {'result': 'success'}
        return ctx
    
    result = await pipeline.process(context, core_processor)
    
    assert result.output_data['result'] == 'success'
    assert 'duration' in result.metrics
    assert result.status == 'completed'
```

## Configuration Reference

### Complete Middleware Configuration

```yaml
middleware:
  timing:
    enabled: true
    slow_request_threshold: 5.0
    enable_historical_tracking: true
    max_history_entries: 1000
  
  logging:
    enabled: true
    log_level: "INFO"
    include_request_data: true
    include_response_data: false
    log_slow_requests: true
    slow_request_threshold: 3.0
  
  analytics:
    enabled: true
    enable_usage_tracking: true
    enable_voice_analytics: true
    enable_performance_tracking: true
    retention_days: 30
    aggregation_interval: 3600
```

## Best Practices

### 1. Middleware Design

- **Single Responsibility**: Each middleware handles one concern
- **Order Matters**: Place timing first, logging second, analytics last
- **Error Resilience**: Don't break the pipeline on middleware errors
- **Performance**: Minimize overhead and memory usage

### 2. Context Management

- **Immutable Data**: Don't modify input_data after initial creation
- **Efficient Updates**: Use bulk updates for metrics and metadata
- **Memory Cleanup**: Remove temporary data after processing
- **Error Context**: Preserve error information for debugging

### 3. Configuration

- **Environment Specific**: Different settings for dev/staging/prod
- **Feature Flags**: Allow enabling/disabling middleware components
- **Performance Tuning**: Adjust thresholds based on system capacity
- **Monitoring**: Track middleware performance and adjust as needed

## Troubleshooting

### Common Issues

1. **High Middleware Overhead**
   - Check middleware order (timing should be first)
   - Review custom middleware for inefficiencies
   - Monitor memory usage and cleanup

2. **Missing Context Data**
   - Verify middleware order in pipeline
   - Check error handling in custom middleware
   - Ensure proper context passing

3. **Pipeline Failures**
   - Review error handling in middleware
   - Check for circular dependencies
   - Verify proper async/await usage

### Debug Information

```python
# Get pipeline debug info
pipeline_info = pipeline.get_debug_info()
print(f"Middleware count: {pipeline_info['middleware_count']}")
print(f"Middleware names: {pipeline_info['middleware_names']}")

# Get middleware-specific debug info
for middleware in pipeline.middleware:
    if hasattr(middleware, 'get_debug_info'):
        debug_info = middleware.get_debug_info()
        print(f"Middleware {type(middleware).__name__}: {debug_info}")
```

## Future Enhancements

### Planned Features

1. **Async Middleware**: Full async/await support for all middleware
2. **Conditional Middleware**: Enable/disable based on request type
3. **Middleware Metrics**: Built-in performance monitoring
4. **Pipeline Branching**: Different pipelines for different request types
5. **Distributed Tracing**: OpenTelemetry integration for distributed systems