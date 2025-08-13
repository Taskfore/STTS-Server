#!/usr/bin/env python3
"""
Test script to verify that the new router structure works correctly.
Tests all major endpoints to ensure backward compatibility and functionality.
"""

import asyncio
import httpx
import json
import time
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

async def test_health_endpoint():
    """Test system health endpoint."""
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{BASE_URL}/health")
            success = response.status_code == 200
            data = response.json() if success else {}
            details = f"Status: {data.get('status', 'unknown')}" if success else f"HTTP {response.status_code}"
            log_test("Health Check", success, details)
            return success
    except Exception as e:
        log_test("Health Check", False, str(e))
        return False

async def test_info_endpoint():
    """Test system info endpoint."""
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{BASE_URL}/info")
            success = response.status_code == 200
            data = response.json() if success else {}
            details = f"Version: {data.get('server', {}).get('version', 'unknown')}" if success else f"HTTP {response.status_code}"
            log_test("System Info", success, details)
            return success
    except Exception as e:
        log_test("System Info", False, str(e))
        return False

async def test_config_endpoints():
    """Test configuration management endpoints."""
    try:
        async with httpx.AsyncClient() as client:
            # Test get current config
            response = await client.get(f"{BASE_URL}/config/current")
            success = response.status_code == 200
            log_test("Config - Get Current", success, f"HTTP {response.status_code}")
            
            # Test get defaults
            response = await client.get(f"{BASE_URL}/config/defaults")
            success = response.status_code == 200
            log_test("Config - Get Defaults", success, f"HTTP {response.status_code}")
            
            # Test validation
            response = await client.get(f"{BASE_URL}/config/validation")
            success = response.status_code == 200
            data = response.json() if success else {}
            valid = data.get('valid', False) if success else False
            log_test("Config - Validation", success, f"Valid: {valid}" if success else f"HTTP {response.status_code}")
            
            # Test schema
            response = await client.get(f"{BASE_URL}/config/schema")
            success = response.status_code == 200
            log_test("Config - Schema", success, f"HTTP {response.status_code}")
            
            return True
    except Exception as e:
        log_test("Config Endpoints", False, str(e))
        return False

async def test_file_endpoints():
    """Test file management endpoints."""
    try:
        async with httpx.AsyncClient() as client:
            # Test get reference files
            response = await client.get(f"{BASE_URL}/files/reference-audio")
            success = response.status_code == 200
            data = response.json() if success else []
            count = len(data) if success else 0
            log_test("Files - Reference Audio List", success, f"Count: {count}" if success else f"HTTP {response.status_code}")
            
            # Test get predefined voices
            response = await client.get(f"{BASE_URL}/files/predefined-voices")
            success = response.status_code == 200
            data = response.json() if success else []
            count = len(data) if success else 0
            log_test("Files - Predefined Voices List", success, f"Count: {count}" if success else f"HTTP {response.status_code}")
            
            # Test storage usage
            response = await client.get(f"{BASE_URL}/files/storage/usage")
            success = response.status_code == 200
            data = response.json() if success else {}
            total_mb = data.get('total', {}).get('size_mb', 0) if success else 0
            log_test("Files - Storage Usage", success, f"Total: {total_mb}MB" if success else f"HTTP {response.status_code}")
            
            return True
    except Exception as e:
        log_test("File Endpoints", False, str(e))
        return False

async def test_tts_endpoints():
    """Test TTS endpoints."""
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            # Test voices list
            response = await client.get(f"{BASE_URL}/tts/voices")
            success = response.status_code == 200
            data = response.json() if success else []
            count = len(data) if success else 0
            log_test("TTS - List Voices", success, f"Count: {count}" if success else f"HTTP {response.status_code}")
            
            # Test TTS synthesis (if voices available)
            if count > 0:
                voice = data[0]
                tts_payload = {
                    "text": "Hello, this is a test.",
                    "voice_mode": voice["type"], 
                    "predefined_voice_id" if voice["type"] == "predefined" else "reference_audio_filename": voice["id"],
                    "output_format": "wav"
                }
                
                response = await client.post(f"{BASE_URL}/tts", json=tts_payload)
                success = response.status_code == 200
                size = len(response.content) if success else 0
                log_test("TTS - Synthesis", success, f"Audio size: {size} bytes" if success else f"HTTP {response.status_code}")
            else:
                log_test("TTS - Synthesis", False, "No voices available for testing")
            
            return True
    except Exception as e:
        log_test("TTS Endpoints", False, str(e))
        return False

async def test_legacy_endpoints():
    """Test legacy compatibility endpoints."""
    try:
        async with httpx.AsyncClient() as client:
            # Test legacy reference files endpoint
            response = await client.get(f"{BASE_URL}/get_reference_files")
            success = response.status_code == 200
            log_test("Legacy - Get Reference Files", success, f"HTTP {response.status_code}")
            
            # Test legacy predefined voices endpoint
            response = await client.get(f"{BASE_URL}/get_predefined_voices")
            success = response.status_code == 200
            log_test("Legacy - Get Predefined Voices", success, f"HTTP {response.status_code}")
            
            return True
    except Exception as e:
        log_test("Legacy Endpoints", False, str(e))
        return False

async def test_ui_endpoints():
    """Test UI and static endpoints."""
    try:
        async with httpx.AsyncClient() as client:
            # Test main UI
            response = await client.get(f"{BASE_URL}/")
            success = response.status_code == 200
            log_test("UI - Main Page", success, f"HTTP {response.status_code}")
            
            # Test initial data
            response = await client.get(f"{BASE_URL}/api/ui/initial-data")
            success = response.status_code == 200
            log_test("UI - Initial Data", success, f"HTTP {response.status_code}")
            
            return True
    except Exception as e:
        log_test("UI Endpoints", False, str(e))
        return False

async def run_all_tests():
    """Run all tests and print summary."""
    print("🧪 Testing New Router Structure")
    print("=" * 50)
    
    # Basic connectivity
    await test_health_endpoint()
    await test_info_endpoint()
    
    # Core functionality
    await test_config_endpoints()
    await test_file_endpoints() 
    await test_tts_endpoints()
    
    # Compatibility
    await test_legacy_endpoints()
    await test_ui_endpoints()
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 Test Summary")
    print("=" * 50)
    
    total_tests = len(TEST_RESULTS)
    passed_tests = sum(1 for _, success, _ in TEST_RESULTS if success)
    failed_tests = total_tests - passed_tests
    
    print(f"Total Tests: {total_tests}")
    print(f"Passed: {passed_tests}")
    print(f"Failed: {failed_tests}")
    print(f"Success Rate: {(passed_tests/total_tests)*100:.1f}%")
    
    if failed_tests > 0:
        print("\n❌ Failed Tests:")
        for test_name, success, details in TEST_RESULTS:
            if not success:
                print(f"  - {test_name}: {details}")
    
    return failed_tests == 0

if __name__ == "__main__":
    print("Starting test suite...")
    print(f"Testing server at: {BASE_URL}")
    print("Make sure the server is running with: python server_clean.py")
    print()
    
    try:
        success = asyncio.run(run_all_tests())
        exit_code = 0 if success else 1
        print(f"\nTest suite {'PASSED' if success else 'FAILED'}")
        exit(exit_code)
    except KeyboardInterrupt:
        print("\n🛑 Tests interrupted by user")
        exit(1)
    except Exception as e:
        print(f"\n💥 Test suite crashed: {e}")
        exit(1)