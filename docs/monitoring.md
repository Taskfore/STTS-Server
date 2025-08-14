# Monitoring & Analytics Documentation

## Overview

The STTS Server library integration includes a comprehensive monitoring and analytics system that provides deep insights into system performance, usage patterns, and operational health. This system is built on middleware patterns and provides both real-time monitoring and historical analytics.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                 Monitoring Architecture                     │
├─────────────────────┬─────────────────────┬─────────────────┤
│  Middleware Layer   │   Analytics Engine  │ Metrics Storage │
│                     │                     │                 │
│ • TimingMiddleware  │ • Usage Tracking    │ • In-Memory     │
│ • LoggingMiddleware │ • Performance       │ • File-based    │
│ • AnalyticsMiddleware│ • Error Analytics  │ • Time-series   │
└─────────────────────┼─────────────────────┼─────────────────┘
                      │                     │
                      ▼                     ▼
┌─────────────────────────────────────────────────────────────┐
│                Request Context System                       │
├─────────────────────┬─────────────────────┬─────────────────┤
│   Correlation IDs   │   Timing Data       │   Usage Data    │
│   Request Tracking  │   Performance       │   Analytics     │
└─────────────────────┴─────────────────────┴─────────────────┘
```

## Core Monitoring Components

### Request Context System

The foundation of monitoring is the `RequestContext` system:

```python
@dataclass
class RequestContext:
    request_id: str              # Unique correlation ID (8 chars)
    request_type: str           # "tts", "stt", "conversation"
    start_time: float           # Request start timestamp
    input_data: Dict[str, Any]  # Request parameters and data
    output_data: Dict[str, Any] # Response data and results
    metrics: Dict[str, Any]     # Performance metrics
    metadata: Dict[str, Any]    # Processing metadata
    status: str                 # "processing", "completed", "error"
```

#### Request Context Usage

```python
# Create context for new request
context = RequestContext(
    request_id=str(uuid.uuid4())[:8],
    request_type="tts",
    input_data={
        "text": "Hello world",
        "voice_id": "female_voice_01",
        "temperature": 0.8
    }
)

# Track processing through middleware
result_context = await middleware_pipeline.process(context, core_processor)

# Access metrics
duration = result_context.metrics.get('duration', 0)
processing_time = result_context.metrics.get('processing_time', 0)
voice_used = result_context.metadata.get('voice_id')
```

### Timing Middleware

Provides comprehensive timing and performance monitoring:

```python
class TimingMiddleware:
    """Comprehensive timing and performance monitoring."""
    
    def __init__(self):
        self.request_history: List[Dict[str, Any]] = []
        self.performance_stats = {
            'total_requests': 0,
            'total_duration': 0.0,
            'slow_requests': 0,
            'error_requests': 0
        }
    
    async def process(self, context: RequestContext, next_processor) -> RequestContext:
        # Record start time
        start_time = time.time()
        context.start_time = start_time
        
        try:
            # Process request
            result_context = await next_processor(context)
            
            # Calculate timing metrics
            end_time = time.time()
            duration = end_time - start_time
            processing_time = end_time - context.metadata.get('core_start_time', start_time)
            middleware_overhead = duration - processing_time
            
            # Update context metrics
            result_context.metrics.update({
                'duration': duration,
                'processing_time': processing_time,
                'middleware_overhead': middleware_overhead,
                'timestamp': end_time,
                'slow_request': duration > self.slow_threshold
            })
            
            # Update statistics
            self._update_stats(result_context)
            
            return result_context
            
        except Exception as e:
            # Record error timing
            duration = time.time() - start_time
            context.metrics.update({
                'duration': duration,
                'error': str(e),
                'timestamp': time.time()
            })
            context.status = "error"
            
            self._update_stats(context, error=True)
            raise
```

#### Timing Metrics Collected

- **Duration**: Total request processing time
- **Processing Time**: Core logic processing time (excluding middleware)
- **Middleware Overhead**: Time spent in middleware pipeline
- **Slow Request Flag**: Requests exceeding threshold
- **Error Timing**: Duration of failed requests
- **Throughput**: Requests per second
- **Percentiles**: 50th, 90th, 95th, 99th percentile response times

### Analytics Middleware

Collects detailed usage and system analytics:

```python
class AnalyticsMiddleware:
    """Comprehensive usage and system analytics."""
    
    def __init__(self):
        self.usage_stats = {
            'voice_usage': defaultdict(int),
            'format_preferences': defaultdict(int),
            'request_patterns': defaultdict(int),
            'error_types': defaultdict(int),
            'user_agents': defaultdict(int),
            'hourly_usage': defaultdict(int)
        }
        
        self.performance_trends = {
            'response_times': deque(maxlen=1000),
            'throughput': deque(maxlen=100),
            'error_rates': deque(maxlen=100)
        }
    
    async def process(self, context: RequestContext, next_processor) -> RequestContext:
        result_context = await next_processor(context)
        
        # Collect usage analytics
        await self._collect_usage_analytics(result_context)
        
        # Collect performance analytics
        await self._collect_performance_analytics(result_context)
        
        # Collect error analytics (if applicable)
        if result_context.status == "error":
            await self._collect_error_analytics(result_context)
        
        return result_context
    
    async def _collect_usage_analytics(self, context: RequestContext):
        """Collect usage pattern analytics."""
        request_type = context.request_type
        
        # Voice usage tracking
        if request_type in ["tts", "conversation"]:
            voice_id = context.input_data.get('voice_id') or \
                      context.input_data.get('voice_path', 'unknown')
            self.usage_stats['voice_usage'][voice_id] += 1
        
        # Format preferences
        if request_type == "tts":
            output_format = context.input_data.get('format', 'wav')
            self.usage_stats['format_preferences'][output_format] += 1
        
        # Request patterns
        hour = datetime.now().hour
        self.usage_stats['hourly_usage'][hour] += 1
        self.usage_stats['request_patterns'][request_type] += 1
    
    async def _collect_performance_analytics(self, context: RequestContext):
        """Collect performance trend analytics."""
        duration = context.metrics.get('duration', 0)
        
        # Response time trends
        self.performance_trends['response_times'].append({
            'timestamp': time.time(),
            'duration': duration,
            'request_type': context.request_type
        })
        
        # Calculate throughput (requests per minute)
        current_minute = int(time.time()) // 60
        if not hasattr(self, '_last_throughput_minute'):
            self._last_throughput_minute = current_minute
            self._current_minute_requests = 0
        
        if current_minute == self._last_throughput_minute:
            self._current_minute_requests += 1
        else:
            throughput = self._current_minute_requests
            self.performance_trends['throughput'].append({
                'timestamp': self._last_throughput_minute * 60,
                'throughput': throughput
            })
            self._last_throughput_minute = current_minute
            self._current_minute_requests = 1
```

### Logging Middleware

Provides context-aware logging with correlation:

```python
class LoggingMiddleware:
    """Context-aware logging with request correlation."""
    
    def __init__(self):
        self.logger = logging.getLogger("stts.middleware.logging")
        self.include_request_data = True
        self.include_response_data = False
        self.log_slow_requests = True
        self.slow_threshold = 3.0
    
    async def process(self, context: RequestContext, next_processor) -> RequestContext:
        request_id = context.request_id
        request_type = context.request_type
        
        # Log request start
        self.logger.info(
            f"[{request_id}] {request_type.upper()} request started",
            extra={'request_id': request_id, 'request_type': request_type}
        )
        
        if self.include_request_data:
            # Log key request parameters (sanitized)
            sanitized_input = self._sanitize_input_data(context.input_data)
            self.logger.debug(
                f"[{request_id}] Request data: {sanitized_input}",
                extra={'request_id': request_id}
            )
        
        try:
            result_context = await next_processor(context)
            
            # Log completion
            duration = result_context.metrics.get('duration', 0)
            self.logger.info(
                f"[{request_id}] {request_type.upper()} completed in {duration:.3f}s",
                extra={
                    'request_id': request_id,
                    'duration': duration,
                    'status': 'success'
                }
            )
            
            # Log slow requests
            if self.log_slow_requests and duration > self.slow_threshold:
                self.logger.warning(
                    f"[{request_id}] Slow request detected: {duration:.3f}s > {self.slow_threshold}s",
                    extra={'request_id': request_id, 'slow_request': True}
                )
            
            return result_context
            
        except Exception as e:
            # Log errors with context
            duration = time.time() - context.start_time
            self.logger.error(
                f"[{request_id}] {request_type.upper()} failed after {duration:.3f}s: {str(e)}",
                extra={
                    'request_id': request_id,
                    'duration': duration,
                    'status': 'error',
                    'error_type': type(e).__name__
                },
                exc_info=True
            )
            raise
    
    def _sanitize_input_data(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Sanitize input data for logging (remove sensitive info, truncate large data)."""
        sanitized = {}
        for key, value in input_data.items():
            if key in ['text']:
                # Truncate long text
                sanitized[key] = str(value)[:100] + "..." if len(str(value)) > 100 else str(value)
            elif key in ['voice_path', 'audio_file']:
                # Show only filename, not full path
                sanitized[key] = os.path.basename(str(value)) if value else None
            else:
                sanitized[key] = value
        return sanitized
```

## Performance Monitoring

### Real-time Performance Metrics

The system tracks comprehensive performance metrics in real-time:

```python
class PerformanceMonitor:
    """Real-time performance monitoring and alerting."""
    
    def __init__(self):
        self.metrics = {
            'requests_per_second': MovingAverage(window_size=60),
            'average_response_time': MovingAverage(window_size=100),
            'error_rate': MovingAverage(window_size=100),
            'memory_usage': MovingAverage(window_size=60),
            'cpu_usage': MovingAverage(window_size=60)
        }
        
        self.alerts = {
            'high_response_time': 5.0,
            'high_error_rate': 0.05,
            'high_memory_usage': 0.85,
            'high_cpu_usage': 0.80
        }
    
    def record_request(self, duration: float, success: bool):
        """Record a completed request."""
        current_time = time.time()
        
        # Update RPS
        self.metrics['requests_per_second'].add(1, current_time)
        
        # Update response time
        self.metrics['average_response_time'].add(duration)
        
        # Update error rate
        self.metrics['error_rate'].add(0 if success else 1)
        
        # Check for alerts
        self._check_alerts()
    
    def get_current_metrics(self) -> Dict[str, float]:
        """Get current performance metrics."""
        return {
            'requests_per_second': self.metrics['requests_per_second'].get_average(),
            'average_response_time': self.metrics['average_response_time'].get_average(),
            'error_rate': self.metrics['error_rate'].get_average(),
            'memory_usage': self._get_memory_usage(),
            'cpu_usage': self._get_cpu_usage()
        }
    
    def _check_alerts(self):
        """Check for performance alerts."""
        current_metrics = self.get_current_metrics()
        
        for metric, threshold in self.alerts.items():
            if current_metrics.get(metric, 0) > threshold:
                self._trigger_alert(metric, current_metrics[metric], threshold)
    
    def _trigger_alert(self, metric: str, value: float, threshold: float):
        """Trigger performance alert."""
        logger.warning(
            f"Performance alert: {metric} = {value:.3f} > {threshold:.3f}",
            extra={'alert_type': 'performance', 'metric': metric, 'value': value}
        )
```

### Historical Performance Analysis

```python
class PerformanceAnalyzer:
    """Historical performance analysis and reporting."""
    
    def __init__(self):
        self.historical_data = defaultdict(list)
    
    def analyze_performance_trends(self, days: int = 7) -> Dict[str, Any]:
        """Analyze performance trends over specified period."""
        cutoff_time = time.time() - (days * 24 * 3600)
        
        # Filter recent data
        recent_data = {
            metric: [d for d in data if d['timestamp'] > cutoff_time]
            for metric, data in self.historical_data.items()
        }
        
        analysis = {}
        
        # Response time analysis
        response_times = [d['value'] for d in recent_data.get('response_time', [])]
        if response_times:
            analysis['response_time'] = {
                'average': statistics.mean(response_times),
                'median': statistics.median(response_times),
                'p95': self._percentile(response_times, 95),
                'p99': self._percentile(response_times, 99),
                'trend': self._calculate_trend(recent_data['response_time'])
            }
        
        # Throughput analysis
        throughput_data = recent_data.get('throughput', [])
        if throughput_data:
            analysis['throughput'] = {
                'average_rps': statistics.mean([d['value'] for d in throughput_data]),
                'peak_rps': max([d['value'] for d in throughput_data]),
                'trend': self._calculate_trend(throughput_data)
            }
        
        # Error rate analysis
        error_data = recent_data.get('error_rate', [])
        if error_data:
            analysis['error_rate'] = {
                'average': statistics.mean([d['value'] for d in error_data]),
                'peak': max([d['value'] for d in error_data]),
                'trend': self._calculate_trend(error_data)
            }
        
        return analysis
    
    def _calculate_trend(self, data: List[Dict[str, Any]]) -> str:
        """Calculate trend direction (increasing, decreasing, stable)."""
        if len(data) < 2:
            return "insufficient_data"
        
        # Simple linear regression to determine trend
        x = list(range(len(data)))
        y = [d['value'] for d in data]
        
        # Calculate slope
        n = len(data)
        slope = (n * sum(x[i] * y[i] for i in range(n)) - sum(x) * sum(y)) / \
                (n * sum(x[i]**2 for i in range(n)) - sum(x)**2)
        
        if slope > 0.01:
            return "increasing"
        elif slope < -0.01:
            return "decreasing"
        else:
            return "stable"
```

## Usage Analytics

### Voice Usage Analytics

Track and analyze voice usage patterns:

```python
class VoiceAnalytics:
    """Voice usage pattern analysis."""
    
    def __init__(self):
        self.voice_usage = defaultdict(lambda: {
            'usage_count': 0,
            'total_duration': 0.0,
            'total_characters': 0,
            'last_used': None,
            'success_rate': 1.0,
            'error_count': 0
        })
    
    def record_voice_usage(self, voice_id: str, text_length: int, 
                          duration: float, success: bool):
        """Record voice usage for analytics."""
        stats = self.voice_usage[voice_id]
        
        stats['usage_count'] += 1
        stats['total_duration'] += duration
        stats['total_characters'] += text_length
        stats['last_used'] = time.time()
        
        if not success:
            stats['error_count'] += 1
        
        # Update success rate
        stats['success_rate'] = (stats['usage_count'] - stats['error_count']) / stats['usage_count']
    
    def get_voice_popularity_ranking(self) -> List[Dict[str, Any]]:
        """Get voices ranked by popularity."""
        ranked_voices = []
        
        for voice_id, stats in self.voice_usage.items():
            ranked_voices.append({
                'voice_id': voice_id,
                'usage_count': stats['usage_count'],
                'total_duration': stats['total_duration'],
                'average_duration': stats['total_duration'] / max(1, stats['usage_count']),
                'success_rate': stats['success_rate'],
                'last_used': stats['last_used']
            })
        
        # Sort by usage count
        ranked_voices.sort(key=lambda x: x['usage_count'], reverse=True)
        return ranked_voices
    
    def get_voice_usage_trends(self, days: int = 30) -> Dict[str, Any]:
        """Analyze voice usage trends over time."""
        cutoff_time = time.time() - (days * 24 * 3600)
        
        trends = {
            'total_voices_used': len(self.voice_usage),
            'active_voices': sum(1 for stats in self.voice_usage.values() 
                               if stats['last_used'] and stats['last_used'] > cutoff_time),
            'most_popular': None,
            'least_popular': None,
            'usage_distribution': {}
        }
        
        if self.voice_usage:
            popularity_ranking = self.get_voice_popularity_ranking()
            trends['most_popular'] = popularity_ranking[0] if popularity_ranking else None
            trends['least_popular'] = popularity_ranking[-1] if popularity_ranking else None
            
            # Usage distribution
            total_usage = sum(stats['usage_count'] for stats in self.voice_usage.values())
            for voice_id, stats in self.voice_usage.items():
                trends['usage_distribution'][voice_id] = stats['usage_count'] / total_usage
        
        return trends
```

### System Usage Analytics

```python
class SystemAnalytics:
    """System-wide usage analytics and insights."""
    
    def __init__(self):
        self.request_stats = defaultdict(int)
        self.hourly_patterns = defaultdict(lambda: defaultdict(int))
        self.format_preferences = defaultdict(int)
        self.performance_stats = defaultdict(list)
    
    def record_request(self, request_type: str, hour: int, format_type: str = None,
                      duration: float = None, success: bool = True):
        """Record system request for analytics."""
        self.request_stats[request_type] += 1
        self.hourly_patterns[hour][request_type] += 1
        
        if format_type:
            self.format_preferences[format_type] += 1
        
        if duration is not None:
            self.performance_stats[request_type].append({
                'duration': duration,
                'timestamp': time.time(),
                'success': success
            })
    
    def get_usage_summary(self) -> Dict[str, Any]:
        """Get comprehensive usage summary."""
        total_requests = sum(self.request_stats.values())
        
        summary = {
            'total_requests': total_requests,
            'request_breakdown': dict(self.request_stats),
            'peak_hour': self._get_peak_hour(),
            'format_preferences': dict(self.format_preferences),
            'average_performance': self._get_average_performance(),
            'success_rates': self._get_success_rates()
        }
        
        return summary
    
    def _get_peak_hour(self) -> Dict[str, Any]:
        """Get peak usage hour."""
        hour_totals = defaultdict(int)
        for hour, types in self.hourly_patterns.items():
            hour_totals[hour] = sum(types.values())
        
        if not hour_totals:
            return {'hour': None, 'requests': 0}
        
        peak_hour = max(hour_totals.keys(), key=lambda h: hour_totals[h])
        return {
            'hour': peak_hour,
            'requests': hour_totals[peak_hour],
            'breakdown': dict(self.hourly_patterns[peak_hour])
        }
    
    def _get_average_performance(self) -> Dict[str, float]:
        """Get average performance by request type."""
        performance = {}
        for request_type, data in self.performance_stats.items():
            if data:
                performance[request_type] = statistics.mean(d['duration'] for d in data)
        return performance
    
    def _get_success_rates(self) -> Dict[str, float]:
        """Get success rates by request type."""
        success_rates = {}
        for request_type, data in self.performance_stats.items():
            if data:
                successful = sum(1 for d in data if d['success'])
                success_rates[request_type] = successful / len(data)
        return success_rates
```

## API Endpoints for Monitoring

### Statistics Endpoints

#### TTS Statistics
```http
GET /tts/statistics
```

Returns comprehensive TTS statistics:
```json
{
  "status": "success",
  "data": {
    "system": {
      "adapter_type": "LegacyTTSEngineAdapter",
      "library_available": true,
      "middleware_enabled": true,
      "model_loaded": true,
      "uptime": 3600
    },
    "middleware": {
      "enabled_count": 3,
      "total_requests": 150,
      "average_duration": 2.34,
      "error_rate": 0.02,
      "slow_requests": 5,
      "throughput": 1.2
    },
    "analytics": {
      "popular_voices": [
        {"voice_id": "female_voice_01", "usage_count": 45},
        {"voice_id": "male_voice_02", "usage_count": 23}
      ],
      "format_preferences": {"wav": 60, "opus": 30, "mp3": 10},
      "usage_trends": {
        "peak_hour": 14,
        "requests_per_hour": {...}
      },
      "performance_metrics": {
        "p50_response_time": 1.8,
        "p95_response_time": 4.2,
        "p99_response_time": 6.1
      }
    }
  }
}
```

#### Middleware Status
```http
GET /tts/middleware/status
```

Returns middleware pipeline status:
```json
{
  "status": "success",
  "data": {
    "total_middleware": 3,
    "enabled_middleware": 3,
    "middleware_names": [
      "TimingMiddleware",
      "LoggingMiddleware",
      "AnalyticsMiddleware"
    ],
    "pipeline_status": "active",
    "configuration": {
      "timing": {"enabled": true, "slow_threshold": 5.0},
      "logging": {"enabled": true, "log_level": "INFO"},
      "analytics": {"enabled": true, "retention_days": 30}
    },
    "performance": {
      "middleware_overhead": 0.003,
      "requests_processed": 150,
      "average_processing_time": 0.002
    }
  }
}
```

#### Performance Metrics
```http
GET /api/metrics/performance
```

Returns detailed performance metrics:
```json
{
  "status": "success",
  "data": {
    "current": {
      "requests_per_second": 1.2,
      "average_response_time": 2.34,
      "error_rate": 0.02,
      "memory_usage": 0.45,
      "cpu_usage": 0.23
    },
    "trends": {
      "response_time": {
        "trend": "stable",
        "change_percentage": -0.05
      },
      "throughput": {
        "trend": "increasing",
        "change_percentage": 0.12
      }
    },
    "percentiles": {
      "p50": 1.8,
      "p90": 3.9,
      "p95": 4.2,
      "p99": 6.1
    }
  }
}
```

### Analytics Endpoints

#### Usage Analytics
```http
GET /api/analytics/usage
```

#### Voice Analytics
```http
GET /api/analytics/voices
```

#### System Health
```http
GET /api/health/detailed
```

## Alerting and Notifications

### Performance Alerts

```python
class AlertManager:
    """Manage performance alerts and notifications."""
    
    def __init__(self):
        self.alert_rules = {
            'high_response_time': {'threshold': 5.0, 'severity': 'warning'},
            'very_high_response_time': {'threshold': 10.0, 'severity': 'critical'},
            'high_error_rate': {'threshold': 0.05, 'severity': 'warning'},
            'very_high_error_rate': {'threshold': 0.15, 'severity': 'critical'},
            'low_throughput': {'threshold': 0.1, 'severity': 'warning'},
            'high_memory_usage': {'threshold': 0.85, 'severity': 'warning'},
            'very_high_memory_usage': {'threshold': 0.95, 'severity': 'critical'}
        }
        
        self.active_alerts = {}
        self.alert_history = []
    
    def check_metrics(self, metrics: Dict[str, float]):
        """Check metrics against alert rules."""
        current_time = time.time()
        
        for rule_name, rule_config in self.alert_rules.items():
            metric_name = self._get_metric_name_for_rule(rule_name)
            metric_value = metrics.get(metric_name, 0)
            
            if self._should_trigger_alert(rule_name, metric_value, rule_config):
                self._trigger_alert(rule_name, metric_value, rule_config, current_time)
            elif rule_name in self.active_alerts:
                self._resolve_alert(rule_name, current_time)
    
    def _trigger_alert(self, rule_name: str, value: float, 
                      rule_config: Dict[str, Any], timestamp: float):
        """Trigger an alert."""
        alert = {
            'rule_name': rule_name,
            'severity': rule_config['severity'],
            'value': value,
            'threshold': rule_config['threshold'],
            'triggered_at': timestamp,
            'status': 'active'
        }
        
        self.active_alerts[rule_name] = alert
        self.alert_history.append(alert.copy())
        
        # Log alert
        logger.warning(
            f"Alert triggered: {rule_name} - {value:.3f} > {rule_config['threshold']:.3f}",
            extra={'alert': alert}
        )
        
        # Send notification (if configured)
        self._send_notification(alert)
    
    def _resolve_alert(self, rule_name: str, timestamp: float):
        """Resolve an active alert."""
        if rule_name in self.active_alerts:
            alert = self.active_alerts[rule_name]
            alert['status'] = 'resolved'
            alert['resolved_at'] = timestamp
            
            logger.info(
                f"Alert resolved: {rule_name}",
                extra={'alert': alert}
            )
            
            del self.active_alerts[rule_name]
```

## Configuration

### Monitoring Configuration

```yaml
monitoring:
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
  
  alerts:
    enabled: true
    rules:
      high_response_time:
        threshold: 5.0
        severity: "warning"
      high_error_rate:
        threshold: 0.05
        severity: "warning"
  
  storage:
    type: "memory"  # "memory", "file", "database"
    max_entries: 10000
    cleanup_interval: 3600
```

## Best Practices

### 1. **Correlation IDs**
Always use correlation IDs for request tracking:
```python
# Generate short, readable correlation IDs
request_id = str(uuid.uuid4())[:8]

# Include in all log messages
logger.info(f"[{request_id}] Processing started")
```

### 2. **Metric Aggregation**
Aggregate metrics efficiently:
```python
# Use moving averages for real-time metrics
moving_avg = MovingAverage(window_size=100)

# Aggregate by time periods for historical data
hourly_stats = aggregate_by_hour(metrics_data)
```

### 3. **Storage Management**
Manage metric storage efficiently:
```python
# Implement data retention
if len(historical_data) > max_entries:
    historical_data = historical_data[-max_entries:]

# Periodic cleanup
if time.time() - last_cleanup > cleanup_interval:
    cleanup_old_metrics()
```

### 4. **Performance Impact**
Minimize monitoring overhead:
```python
# Use async processing for analytics
asyncio.create_task(process_analytics(context))

# Batch metric updates
metrics_batch.append(metric)
if len(metrics_batch) >= batch_size:
    process_metrics_batch(metrics_batch)
```