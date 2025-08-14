# Migration Guide Documentation

## Overview

This guide provides step-by-step instructions for migrating from the legacy STTS Server architecture to the library-integrated system. The migration is designed to be non-disruptive with zero downtime and full backward compatibility.

## Migration Phases

The migration follows a three-phase approach:

1. **Phase 1**: Clean Router Organization (No Library Integration)
2. **Phase 2**: Library Integration with Adapters and Middleware  
3. **Phase 3**: Native Library Integration (Future)

## Pre-Migration Assessment

### Current State Analysis

Before starting migration, assess your current deployment:

```bash
# Check current server version
curl http://localhost:8004/health | jq '.data.version'

# Check current architecture
curl http://localhost:8004/health | jq '.data.architecture'

# Check available features
curl http://localhost:8004/health | jq '.data.features'
```

### Compatibility Check

```python
#!/usr/bin/env python3
"""Pre-migration compatibility check."""

import os
import sys
import yaml
import subprocess
from pathlib import Path

def check_python_version():
    """Check Python version compatibility."""
    if sys.version_info < (3.8, 0):
        print("❌ Python 3.8+ required")
        return False
    print(f"✅ Python {sys.version}")
    return True

def check_dependencies():
    """Check required dependencies."""
    required_packages = [
        'fastapi', 'uvicorn', 'torch', 'numpy', 
        'pydantic', 'pyyaml', 'python-multipart'
    ]
    
    missing = []
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
            print(f"✅ {package}")
        except ImportError:
            missing.append(package)
            print(f"❌ {package}")
    
    return len(missing) == 0

def check_configuration():
    """Check configuration compatibility."""
    if not os.path.exists('config.yaml'):
        print("⚠️  No config.yaml found - will use defaults")
        return True
    
    try:
        with open('config.yaml') as f:
            config = yaml.safe_load(f)
        print("✅ Configuration valid")
        return True
    except Exception as e:
        print(f"❌ Configuration error: {e}")
        return False

def check_file_structure():
    """Check required file structure."""
    required_dirs = ['reference_audio', 'voices', 'outputs']
    for dir_name in required_dirs:
        if not os.path.exists(dir_name):
            print(f"⚠️  Directory {dir_name} missing - will be created")
        else:
            print(f"✅ Directory {dir_name}")
    return True

if __name__ == "__main__":
    print("🔍 Pre-Migration Compatibility Check")
    print("=" * 50)
    
    checks = [
        check_python_version(),
        check_dependencies(),
        check_configuration(),
        check_file_structure()
    ]
    
    if all(checks):
        print("\n✅ System ready for migration!")
    else:
        print("\n❌ Please resolve issues before migration")
        sys.exit(1)
```

## Phase 1: Clean Router Organization

### Overview

Phase 1 reorganizes the monolithic server into clean, focused routers without library integration.

### Migration Steps

#### 1. Backup Current System

```bash
# Create backup
cp server.py server_legacy_backup.py
cp config.yaml config_backup.yaml

# Create backup directory
mkdir -p migration_backup/$(date +%Y%m%d_%H%M%S)
cp -r . migration_backup/$(date +%Y%m%d_%H%M%S)/
```

#### 2. Update Server Configuration

Add router configuration to `config.yaml`:

```yaml
# Add to existing config.yaml
routers:
  core:
    tts:
      max_text_length: 5000
    stt:
      max_file_size_mb: 50
      supported_formats: ["wav", "mp3", "m4a", "flac"]
    conversation:
      fallback_to_adapters: true
  
  management:
    config:
      enable_validation: true
    files:
      enable_analytics: false  # Disabled in Phase 1
  
features:
  use_new_routers: true
  library_integration: false  # Phase 1: disabled
  enhanced_monitoring: false  # Phase 1: disabled
```

#### 3. Deploy Phase 1 Server

```bash
# Test Phase 1 server
python server_clean.py &
SERVER_PID=$!

# Verify functionality
python test_new_routers.py

# If tests pass, switch to Phase 1
kill $SERVER_PID
```

#### 4. Production Deployment

```bash
# Zero-downtime deployment
# 1. Start new server on different port
python server_clean.py --port 8005 &
NEW_SERVER_PID=$!

# 2. Verify new server
curl http://localhost:8005/health

# 3. Update load balancer/reverse proxy to point to new port
# 4. Gracefully shutdown old server
kill $OLD_SERVER_PID

# 5. Switch new server to production port
kill $NEW_SERVER_PID
python server_clean.py --port 8004 &
```

### Phase 1 Validation

```bash
# Run Phase 1 tests
python test_new_routers.py

# Check endpoints
curl http://localhost:8004/health
curl http://localhost:8004/tts/voices
curl http://localhost:8004/config/validation
```

## Phase 2: Library Integration

### Overview

Phase 2 introduces library integration with adapters, middleware, and enhanced monitoring.

### Prerequisites

```bash
# Install additional dependencies (if needed)
pip install -r requirements.txt

# Verify realtime_conversation library
python -c "
try:
    from realtime_conversation import ConversationEngine
    print('✅ Library available')
except ImportError:
    print('⚠️  Library not available - will use adapters only')
"
```

### Migration Steps

#### 1. Update Configuration

```yaml
# Update config.yaml for Phase 2
middleware:
  timing:
    enabled: true
    slow_request_threshold: 5.0
  logging:
    enabled: true
    log_level: "INFO"
  analytics:
    enabled: true
    retention_days: 30

features:
  use_adapters: true
  library_integration: true
  enhanced_monitoring: true

routers:
  core:
    tts:
      enable_middleware: true
      enable_statistics: true
    stt:
      enable_adapters: true
    conversation:
      enable_library_integration: true
```

#### 2. Gradual Feature Rollout

Enable features incrementally:

```python
# Step 1: Enable adapters only
config_manager.update_config({
    "features": {"use_adapters": True, "library_integration": False}
})

# Test adapters
run_adapter_tests()

# Step 2: Enable middleware
config_manager.update_config({
    "middleware": {"timing": {"enabled": True}}
})

# Test middleware
run_middleware_tests()

# Step 3: Enable full library integration
config_manager.update_config({
    "features": {"library_integration": True}
})
```

#### 3. Deploy Phase 2 Server

```bash
# Test Phase 2 locally
python server_v2.py &
SERVER_PID=$!

# Run comprehensive tests
python test_library_integration.py

# Check new features
curl http://localhost:8004/tts/statistics
curl http://localhost:8004/tts/middleware/status
curl http://localhost:8004/stt/status

# If all tests pass, proceed with deployment
```

#### 4. Production Deployment

```bash
# Blue-green deployment
# 1. Deploy to staging environment
ssh staging "cd /app && git pull && python server_v2.py --port 8004"

# 2. Run integration tests on staging
ssh staging "cd /app && python test_library_integration.py"

# 3. Deploy to production (zero downtime)
ssh production "
  cd /app &&
  git pull &&
  python server_v2.py --port 8005 &
  NEW_PID=\$! &&
  sleep 5 &&
  curl http://localhost:8005/health &&
  # Update load balancer to new port
  sudo systemctl reload nginx &&
  # Kill old server
  kill \$OLD_PID &&
  # Move new server to production port
  kill \$NEW_PID &&
  python server_v2.py --port 8004 &
"
```

### Phase 2 Validation

```bash
# Comprehensive validation
python test_library_integration.py

# Feature validation
curl http://localhost:8004/health | jq '.data.features'

# Performance validation
curl http://localhost:8004/tts/statistics | jq '.data.middleware'
```

## Configuration Migration

### Legacy to Library Integration

#### TTS Configuration

```yaml
# Legacy configuration
gen:
  default_temperature: 0.8
  default_speed_factor: 1.0

# Library-integrated configuration (additive)
gen:
  default_temperature: 0.8
  default_speed_factor: 1.0
  # New library-specific options
  default_exaggeration: 0.5
  default_cfg_weight: 0.5
  chunk_size: 200

middleware:
  timing:
    enabled: true
  analytics:
    enabled: true
```

#### Router Configuration

```yaml
# New router-specific configuration
routers:
  core:
    tts:
      enable_middleware: true
      enable_statistics: true
      max_text_length: 5000
    stt:
      enable_adapters: true
      max_file_size_mb: 50
    conversation:
      enable_library_integration: true
      fallback_to_adapters: true
```

### Configuration Validation

```python
# Validate configuration during migration
def validate_migration_config():
    """Validate configuration for migration compatibility."""
    config_manager = YamlConfigManager()
    
    # Check for required sections
    required_sections = ["middleware", "routers", "features"]
    for section in required_sections:
        if not config_manager.get_dict(section):
            print(f"⚠️  Missing configuration section: {section}")
    
    # Check for deprecated settings
    deprecated_settings = [
        "old_setting_name"  # Add any deprecated settings
    ]
    
    for setting in deprecated_settings:
        if config_manager.get_string(setting):
            print(f"⚠️  Deprecated setting found: {setting}")
    
    # Validate new settings
    validation_result = config_manager.validate_config()
    if not validation_result.is_valid:
        print("❌ Configuration validation failed:")
        for error in validation_result.errors:
            print(f"  - {error}")
        return False
    
    print("✅ Configuration valid for migration")
    return True
```

## Data Migration

### Analytics Data

If migrating from custom analytics:

```python
def migrate_analytics_data():
    """Migrate existing analytics data to new format."""
    
    # Load legacy analytics
    legacy_data = load_legacy_analytics()
    
    # Convert to new format
    new_analytics = AnalyticsMiddleware()
    
    for record in legacy_data:
        # Convert legacy record to new format
        new_record = {
            'voice_id': record.get('voice'),
            'usage_count': record.get('count', 1),
            'timestamp': record.get('timestamp'),
            'success': record.get('success', True)
        }
        
        # Add to new analytics
        new_analytics.add_historical_record(new_record)
    
    print(f"✅ Migrated {len(legacy_data)} analytics records")
```

### Log Data

Migrate log format if needed:

```python
def migrate_log_format():
    """Update log format for new correlation system."""
    
    # Update log configuration
    logging_config = {
        'version': 1,
        'formatters': {
            'detailed': {
                'format': '[%(asctime)s] %(levelname)s [%(request_id)s] %(message)s'
            }
        },
        'handlers': {
            'console': {
                'class': 'logging.StreamHandler',
                'formatter': 'detailed'
            }
        },
        'root': {
            'level': 'INFO',
            'handlers': ['console']
        }
    }
    
    logging.config.dictConfig(logging_config)
```

## Testing Migration

### Comprehensive Test Suite

```python
#!/usr/bin/env python3
"""Comprehensive migration test suite."""

import asyncio
import requests
import time
from typing import List, Dict, Any

class MigrationTester:
    """Test suite for migration validation."""
    
    def __init__(self, base_url: str = "http://localhost:8004"):
        self.base_url = base_url
        self.test_results = []
    
    async def run_all_tests(self):
        """Run complete migration test suite."""
        print("🧪 Running Migration Test Suite")
        print("=" * 50)
        
        # Basic functionality tests
        await self.test_basic_endpoints()
        await self.test_tts_functionality()
        await self.test_stt_functionality()
        await self.test_conversation_pipeline()
        
        # Library integration tests
        await self.test_adapter_functionality()
        await self.test_middleware_integration()
        await self.test_analytics_collection()
        
        # Performance tests
        await self.test_performance_regression()
        
        # Compatibility tests
        await self.test_backward_compatibility()
        
        # Report results
        self.report_results()
    
    async def test_basic_endpoints(self):
        """Test basic endpoint availability."""
        endpoints = [
            "/health",
            "/config",
            "/tts/voices",
            "/get_reference_files"
        ]
        
        for endpoint in endpoints:
            try:
                response = requests.get(f"{self.base_url}{endpoint}")
                success = response.status_code == 200
                self.log_test(f"Basic endpoint {endpoint}", success)
            except Exception as e:
                self.log_test(f"Basic endpoint {endpoint}", False, str(e))
    
    async def test_library_integration(self):
        """Test library integration features."""
        try:
            # Test adapter status
            response = requests.get(f"{self.base_url}/stt/status")
            if response.status_code == 200:
                data = response.json()
                adapter_available = data.get('data', {}).get('adapter_type') is not None
                self.log_test("STT Adapter Integration", adapter_available)
            
            # Test middleware status
            response = requests.get(f"{self.base_url}/tts/middleware/status")
            if response.status_code == 200:
                data = response.json()
                middleware_active = data.get('data', {}).get('pipeline_status') == 'active'
                self.log_test("Middleware Integration", middleware_active)
            
        except Exception as e:
            self.log_test("Library Integration", False, str(e))
    
    def log_test(self, test_name: str, success: bool, details: str = ""):
        """Log test result."""
        status = "✅ PASS" if success else "❌ FAIL"
        self.test_results.append((test_name, success, details))
        print(f"{status} {test_name}")
        if details and not success:
            print(f"   Details: {details}")
    
    def report_results(self):
        """Report final test results."""
        total_tests = len(self.test_results)
        passed_tests = sum(1 for _, success, _ in self.test_results if success)
        
        print("\n" + "=" * 50)
        print("📊 Migration Test Results")
        print("=" * 50)
        print(f"Total Tests: {total_tests}")
        print(f"Passed: {passed_tests}")
        print(f"Failed: {total_tests - passed_tests}")
        print(f"Success Rate: {(passed_tests/total_tests)*100:.1f}%")
        
        if passed_tests == total_tests:
            print("\n✅ Migration SUCCESSFUL!")
        else:
            print("\n❌ Migration issues detected:")
            for test_name, success, details in self.test_results:
                if not success:
                    print(f"  - {test_name}: {details}")

# Run migration tests
if __name__ == "__main__":
    tester = MigrationTester()
    asyncio.run(tester.run_all_tests())
```

## Rollback Procedures

### Automatic Rollback

```bash
#!/bin/bash
# automated_rollback.sh

echo "🔄 Initiating automatic rollback..."

# Check if backup exists
if [ ! -f "server_legacy_backup.py" ]; then
    echo "❌ No backup found - cannot rollback"
    exit 1
fi

# Stop current server
echo "Stopping current server..."
pkill -f "python server_v2.py"
pkill -f "python server_clean.py"

# Restore backup
echo "Restoring legacy server..."
cp server_legacy_backup.py server.py
cp config_backup.yaml config.yaml

# Start legacy server
echo "Starting legacy server..."
python server.py &

# Verify rollback
sleep 5
if curl -f http://localhost:8004/health > /dev/null 2>&1; then
    echo "✅ Rollback successful"
else
    echo "❌ Rollback failed"
    exit 1
fi
```

### Manual Rollback

```bash
# Manual rollback steps
# 1. Stop current server
sudo systemctl stop stts-server

# 2. Restore from backup
cd /app
cp migration_backup/20240101_120000/* .

# 3. Restart with legacy configuration
python server.py &

# 4. Verify functionality
curl http://localhost:8004/health
```

### Rollback Testing

```python
def test_rollback_functionality():
    """Test that rollback maintains functionality."""
    
    # Test basic endpoints
    basic_tests = [
        ("GET", "/health"),
        ("GET", "/get_reference_files"),
        ("POST", "/save_settings", {"gen": {"default_temperature": 0.8}})
    ]
    
    for method, endpoint, data in basic_tests:
        if method == "GET":
            response = requests.get(f"http://localhost:8004{endpoint}")
        else:
            response = requests.post(f"http://localhost:8004{endpoint}", json=data)
        
        if response.status_code != 200:
            print(f"❌ Rollback test failed: {endpoint}")
            return False
    
    print("✅ Rollback functionality verified")
    return True
```

## Performance Monitoring During Migration

### Migration Metrics

```python
class MigrationMonitor:
    """Monitor performance during migration."""
    
    def __init__(self):
        self.metrics = {
            'response_times': [],
            'error_rates': [],
            'throughput': [],
            'memory_usage': []
        }
    
    def collect_baseline_metrics(self, duration: int = 60):
        """Collect baseline metrics before migration."""
        print(f"📊 Collecting baseline metrics for {duration}s...")
        
        start_time = time.time()
        while time.time() - start_time < duration:
            # Test TTS endpoint
            start = time.time()
            try:
                response = requests.post(
                    "http://localhost:8004/tts",
                    json={"text": "test", "voice_id": "default"}
                )
                duration = time.time() - start
                success = response.status_code == 200
                
                self.metrics['response_times'].append(duration)
                self.metrics['error_rates'].append(0 if success else 1)
                
            except Exception:
                self.metrics['error_rates'].append(1)
            
            time.sleep(1)
        
        baseline = {
            'avg_response_time': sum(self.metrics['response_times']) / len(self.metrics['response_times']),
            'error_rate': sum(self.metrics['error_rates']) / len(self.metrics['error_rates']),
            'throughput': len(self.metrics['response_times']) / duration
        }
        
        print(f"📈 Baseline metrics:")
        print(f"  - Avg Response Time: {baseline['avg_response_time']:.3f}s")
        print(f"  - Error Rate: {baseline['error_rate']:.3f}")
        print(f"  - Throughput: {baseline['throughput']:.2f} req/s")
        
        return baseline
    
    def compare_post_migration(self, baseline: Dict[str, float]) -> bool:
        """Compare post-migration metrics to baseline."""
        post_migration = self.collect_baseline_metrics(60)
        
        # Check for regression
        response_time_regression = (
            post_migration['avg_response_time'] / baseline['avg_response_time'] - 1
        ) > 0.2  # Allow 20% degradation
        
        error_rate_increase = (
            post_migration['error_rate'] - baseline['error_rate']
        ) > 0.05  # Allow 5% error rate increase
        
        throughput_degradation = (
            baseline['throughput'] / post_migration['throughput'] - 1
        ) > 0.2  # Allow 20% throughput decrease
        
        if response_time_regression:
            print(f"⚠️  Response time regression detected")
        
        if error_rate_increase:
            print(f"⚠️  Error rate increase detected")
        
        if throughput_degradation:
            print(f"⚠️  Throughput degradation detected")
        
        return not (response_time_regression or error_rate_increase or throughput_degradation)
```

## Troubleshooting Migration Issues

### Common Issues and Solutions

#### 1. Configuration Conflicts

**Issue**: Configuration validation fails
```yaml
# Problem: Invalid configuration
middleware:
  invalid_middleware: true
```

**Solution**: Update configuration schema
```yaml
# Fix: Use valid configuration
middleware:
  timing:
    enabled: true
  logging:
    enabled: true
```

#### 2. Dependency Issues

**Issue**: Missing dependencies
```
ImportError: No module named 'realtime_conversation'
```

**Solution**: Install or disable library integration
```bash
# Option 1: Install dependency
pip install realtime-conversation

# Option 2: Disable library integration
# In config.yaml:
features:
  library_integration: false
```

#### 3. Port Conflicts

**Issue**: Port already in use
```
OSError: [Errno 48] Address already in use
```

**Solution**: Use different port or kill existing process
```bash
# Find process using port
lsof -i :8004

# Kill process
kill -9 <PID>

# Or use different port
python server_v2.py --port 8005
```

#### 4. Performance Regression

**Issue**: Significant performance decrease
```
Response times increased by 50%
```

**Solution**: Disable resource-intensive features
```yaml
# Disable analytics temporarily
middleware:
  analytics:
    enabled: false

# Reduce middleware overhead
middleware:
  timing:
    enable_historical_tracking: false
```

### Debug Tools

#### Migration Debug Script

```python
#!/usr/bin/env python3
"""Debug migration issues."""

def debug_migration():
    """Comprehensive migration debugging."""
    
    print("🔍 Migration Debug Information")
    print("=" * 50)
    
    # Check server status
    try:
        response = requests.get("http://localhost:8004/health")
        if response.status_code == 200:
            health_data = response.json()
            print(f"✅ Server running: {health_data.get('data', {}).get('version')}")
            print(f"   Architecture: {health_data.get('data', {}).get('architecture')}")
        else:
            print(f"❌ Server not responding: HTTP {response.status_code}")
    except Exception as e:
        print(f"❌ Server connection failed: {e}")
    
    # Check configuration
    try:
        config_manager = YamlConfigManager()
        config_valid = config_manager.validate_config()
        if config_valid.is_valid:
            print("✅ Configuration valid")
        else:
            print("❌ Configuration errors:")
            for error in config_valid.errors:
                print(f"   - {error}")
    except Exception as e:
        print(f"❌ Configuration check failed: {e}")
    
    # Check file structure
    required_files = [
        'server_v2.py',
        'adapters/legacy_engines.py',
        'middleware/base.py',
        'routers/core/tts.py'
    ]
    
    for file_path in required_files:
        if os.path.exists(file_path):
            print(f"✅ File exists: {file_path}")
        else:
            print(f"❌ Missing file: {file_path}")

if __name__ == "__main__":
    debug_migration()
```

## Best Practices

### 1. **Staged Migration**
- Always migrate in phases
- Test each phase thoroughly before proceeding
- Keep rollback procedures ready

### 2. **Zero Downtime**
- Use blue-green deployment
- Test new version before switching traffic
- Monitor performance during transition

### 3. **Configuration Management**
- Version control all configuration changes
- Validate configuration before deployment
- Document configuration differences

### 4. **Testing**
- Comprehensive test suite for each phase
- Performance regression testing
- Backward compatibility validation

### 5. **Monitoring**
- Monitor performance during migration
- Set up alerts for critical metrics
- Keep detailed logs of migration process

### 6. **Documentation**
- Document all changes made
- Keep migration runbooks updated
- Record lessons learned for future migrations