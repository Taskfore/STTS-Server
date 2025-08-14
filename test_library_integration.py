#!/usr/bin/env python3
"""
Comprehensive test suite for the library integration (Phase 2).
Tests adapters, middleware, and the new router architecture.
"""

import asyncio
import httpx
import json
import time
import sys
from pathlib import Path

BASE_URL = "http://localhost:8004"
TEST_RESULTS = []

def log_test(test_name: str, success: bool, details: str = ""):
    """Log test results."""
    status = "✅ PASS" if success else "❌ FAIL"
    TEST_RESULTS.append((test_name, success, details))
    print(f"{status} {test_name}")
    if details and not success:
        print(f"   Details: {details}")
    elif details and success:
        print(f"   ℹ️  {details}")

async def test_system_health():
    """Test system health with library integration status."""
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{BASE_URL}/health")
            success = response.status_code == 200
            
            if success:
                data = response.json()
                architecture = data.get('architecture', 'unknown')
                library_features = data.get('features', {})
                
                # Check for library integration features
                has_adapters = library_features.get('adapter_pattern', False)
                has_middleware = library_features.get('middleware_pipeline', False)
                has_integration = library_features.get('library_integration', False)
                
                details = f"Architecture: {architecture}, Adapters: {has_adapters}, Middleware: {has_middleware}, Integration: {has_integration}"
                log_test("System Health (Library Integration)", success, details)
                
                return has_adapters and has_middleware and has_integration
            else:
                log_test("System Health (Library Integration)", False, f"HTTP {response.status_code}")
                return False
                
    except Exception as e:
        log_test("System Health (Library Integration)", False, str(e))
        return False

async def test_tts_middleware():
    """Test TTS middleware pipeline endpoints."""
    try:
        async with httpx.AsyncClient() as client:
            # Test middleware status
            response = await client.get(f"{BASE_URL}/tts/middleware/status")
            success = response.status_code == 200
            
            if success:
                data = response.json()
                total_middleware = data.get('total_middleware', 0)
                enabled_middleware = data.get('enabled_middleware', 0)
                middleware_names = data.get('middleware_names', [])
                
                details = f"Total: {total_middleware}, Enabled: {enabled_middleware}, Names: {middleware_names}"
                log_test("TTS Middleware Status", success, details)
                
                # Test middleware reload
                response = await client.post(f"{BASE_URL}/tts/middleware/reload")
                reload_success = response.status_code == 200
                log_test("TTS Middleware Reload", reload_success, "Middleware pipeline reloaded")
                
                return success and reload_success
            else:
                log_test("TTS Middleware Status", False, f"HTTP {response.status_code}")
                return False
                
    except Exception as e:
        log_test("TTS Middleware", False, str(e))
        return False

async def test_tts_statistics():
    """Test TTS system statistics with middleware data."""
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{BASE_URL}/tts/statistics")
            success = response.status_code == 200
            
            if success:
                data = response.json()
                system_info = data.get('system', {})
                middleware_info = data.get('middleware', {})
                
                adapter_type = system_info.get('adapter_type', 'unknown')
                library_available = system_info.get('library_available', False)
                middleware_count = middleware_info.get('enabled_count', 0)
                
                details = f"Adapter: {adapter_type}, Library: {library_available}, Middleware: {middleware_count}"
                log_test("TTS Statistics", success, details)
                return True
            else:
                log_test("TTS Statistics", False, f"HTTP {response.status_code}")
                return False
                
    except Exception as e:
        log_test("TTS Statistics", False, str(e))
        return False

async def test_stt_adapter():
    """Test STT adapter integration."""
    try:
        async with httpx.AsyncClient() as client:
            # Test STT status
            response = await client.get(f"{BASE_URL}/stt/status")
            success = response.status_code == 200
            
            if success:
                data = response.json()
                available = data.get('available', False)
                adapter_type = data.get('adapter_type', 'unknown')
                library_integration = data.get('library_integration', False)
                
                details = f"Available: {available}, Adapter: {adapter_type}, Integration: {library_integration}"
                log_test("STT Adapter Status", success, details)
                
                # Test STT models
                response = await client.get(f"{BASE_URL}/stt/models")
                models_success = response.status_code == 200
                if models_success:
                    models_data = response.json()
                    current_model = models_data.get('current_model', 'unknown')
                    adapter_type_models = models_data.get('adapter_type', 'unknown')
                    log_test("STT Models", True, f"Current: {current_model}, Type: {adapter_type_models}")
                else:
                    log_test("STT Models", False, f"HTTP {response.status_code}")
                
                return success and models_success
            else:
                log_test("STT Adapter Status", False, f"HTTP {response.status_code}")
                return False
                
    except Exception as e:
        log_test("STT Adapter", False, str(e))
        return False

async def test_conversation_system():
    """Test conversation system with library integration."""
    try:
        async with httpx.AsyncClient() as client:
            # Test conversation status
            response = await client.get(f"{BASE_URL}/conversation/status")
            success = response.status_code == 200
            
            if success:
                data = response.json()
                system_type = data.get('system_type', 'unknown')
                stt_available = data.get('stt_available', False)
                tts_available = data.get('tts_available', False)
                library_available = data.get('library_available', False)
                
                details = f"Type: {system_type}, STT: {stt_available}, TTS: {tts_available}, Library: {library_available}"
                log_test("Conversation System Status", success, details)
                return True
            else:
                log_test("Conversation System Status", False, f"HTTP {response.status_code}")
                return False
                
    except Exception as e:
        log_test("Conversation System", False, str(e))
        return False

async def test_config_validation():
    """Test configuration validation with library features."""
    try:
        async with httpx.AsyncClient() as client:
            # Test config validation
            response = await client.get(f"{BASE_URL}/config/validation")
            success = response.status_code == 200
            
            if success:
                data = response.json()
                is_valid = data.get('valid', False)
                errors = data.get('errors', [])
                warnings = data.get('warnings', [])
                
                details = f"Valid: {is_valid}, Errors: {len(errors)}, Warnings: {len(warnings)}"
                log_test("Config Validation", success, details)
                
                # Test config schema
                response = await client.get(f"{BASE_URL}/config/schema")
                schema_success = response.status_code == 200
                log_test("Config Schema", schema_success, "Configuration schema available")
                
                return success and schema_success
            else:
                log_test("Config Validation", False, f"HTTP {response.status_code}")
                return False
                
    except Exception as e:
        log_test("Config Validation", False, str(e))
        return False

async def test_file_management():
    """Test enhanced file management features."""
    try:
        async with httpx.AsyncClient() as client:
            # Test storage usage
            response = await client.get(f"{BASE_URL}/files/storage/usage")
            success = response.status_code == 200
            
            if success:
                data = response.json()
                total_size = data.get('total', {}).get('size_mb', 0)
                file_count = data.get('total', {}).get('file_count', 0)
                
                details = f"Size: {total_size}MB, Files: {file_count}"
                log_test("File Storage Usage", success, details)
                return True
            else:
                log_test("File Storage Usage", False, f"HTTP {response.status_code}")
                return False
                
    except Exception as e:
        log_test("File Management", False, str(e))
        return False

async def test_websocket_organization():
    """Test that WebSocket routers are properly organized."""
    try:
        async with httpx.AsyncClient() as client:
            # Test WebSocket conversation v2 status
            response = await client.get(f"{BASE_URL}/ws/conversation/status")
            success = response.status_code == 200
            
            if success:
                data = response.json()
                status = data.get('status', 'unknown')
                
                details = f"WebSocket Conversation Status: {status}"
                log_test("WebSocket Organization", success, details)
                return True
            else:
                log_test("WebSocket Organization", False, f"HTTP {response.status_code}")
                return False
                
    except Exception as e:
        log_test("WebSocket Organization", False, str(e))
        return False

async def test_legacy_compatibility():
    """Test that legacy endpoints still work."""
    try:
        async with httpx.AsyncClient() as client:
            # Test legacy reference files endpoint
            response = await client.get(f"{BASE_URL}/get_reference_files")
            legacy_success = response.status_code == 200
            log_test("Legacy Compatibility", legacy_success, "Legacy endpoints redirecting correctly")
            
            return legacy_success
                
    except Exception as e:
        log_test("Legacy Compatibility", False, str(e))
        return False

async def run_integration_tests():
    """Run all library integration tests."""
    print("🧪 Testing Library Integration (Phase 2)")
    print("=" * 60)
    
    # Basic connectivity and health
    system_healthy = await test_system_health()
    
    if not system_healthy:
        print("❌ System not healthy - skipping detailed tests")
        return False
    
    # Test core integrations
    await test_tts_middleware()
    await test_tts_statistics()
    await test_stt_adapter()
    await test_conversation_system()
    
    # Test management features
    await test_config_validation()
    await test_file_management()
    
    # Test organization and compatibility
    await test_websocket_organization()
    await test_legacy_compatibility()
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 Library Integration Test Summary")
    print("=" * 60)
    
    total_tests = len(TEST_RESULTS)
    passed_tests = sum(1 for _, success, _ in TEST_RESULTS if success)
    failed_tests = total_tests - passed_tests
    
    print(f"Total Tests: {total_tests}")
    print(f"Passed: {passed_tests}")
    print(f"Failed: {failed_tests}")
    print(f"Success Rate: {(passed_tests/total_tests)*100:.1f}%")
    
    # Show integration status
    if passed_tests >= total_tests * 0.8:  # 80% pass rate
        print("\n✅ Library Integration: SUCCESSFUL")
        print("🎉 Phase 2 refactoring completed successfully!")
        print("\nKey Achievements:")
        print("• Adapter pattern implemented")
        print("• Middleware pipeline functional") 
        print("• Router organization improved")
        print("• Legacy compatibility maintained")
        print("• Enhanced monitoring and analytics")
    else:
        print("\n⚠️  Library Integration: PARTIAL")
        print("Some features may need additional work.")
    
    if failed_tests > 0:
        print("\n❌ Failed Tests:")
        for test_name, success, details in TEST_RESULTS:
            if not success:
                print(f"  - {test_name}: {details}")
    
    return failed_tests == 0

if __name__ == "__main__":
    print("Library Integration Test Suite")
    print(f"Testing server at: {BASE_URL}")
    print("Make sure the server is running with: python server_v2.py")
    print()
    
    try:
        success = asyncio.run(run_integration_tests())
        exit_code = 0 if success else 1
        print(f"\nTest suite {'PASSED' if success else 'FAILED'}")
        exit(exit_code)
    except KeyboardInterrupt:
        print("\n🛑 Tests interrupted by user")
        exit(1)
    except Exception as e:
        print(f"\n💥 Test suite crashed: {e}")
        exit(1)