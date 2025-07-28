#!/usr/bin/env python3
"""
Test script for API endpoints without requiring models to be loaded.
Tests endpoint availability and request validation.
"""

import requests
import json
from pathlib import Path


BASE_URL = "http://localhost:8004"


def test_health_check():
    """Test if server is running."""
    try:
        response = requests.get(f"{BASE_URL}/api/ui/initial-data", timeout=5)
        if response.status_code == 200:
            print("✅ Server is running and responsive")
            return True
        else:
            print(f"⚠️  Server responded with status: {response.status_code}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"❌ Server is not accessible: {e}")
        return False


def test_api_docs():
    """Test API documentation endpoint."""
    try:
        response = requests.get(f"{BASE_URL}/docs", timeout=5)
        if response.status_code == 200:
            print("✅ API documentation is accessible at /docs")
            return True
        else:
            print(f"⚠️  API docs responded with status: {response.status_code}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"❌ API docs not accessible: {e}")
        return False


def test_stt_endpoint_validation():
    """Test STT endpoint parameter validation (without model)."""
    try:
        # Test without file (should fail validation)
        response = requests.post(f"{BASE_URL}/stt")
        
        # Should return 422 (validation error) since no file provided
        if response.status_code == 422:
            print("✅ STT endpoint validation working correctly")
            return True
        elif response.status_code == 503:
            print("⚠️  STT endpoint available but model not loaded (expected)")
            return True
        else:
            print(f"⚠️  STT endpoint unexpected status: {response.status_code}")
            return False
            
    except requests.exceptions.RequestException as e:
        print(f"❌ STT endpoint test failed: {e}")
        return False


def test_conversation_endpoint_validation():
    """Test conversation endpoint parameter validation (without model)."""
    try:
        # Test without file (should fail validation)
        response = requests.post(f"{BASE_URL}/conversation")
        
        # Should return 422 (validation error) since no file provided
        if response.status_code == 422:
            print("✅ Conversation endpoint validation working correctly")
            return True
        elif response.status_code == 503:
            print("⚠️  Conversation endpoint available but models not loaded (expected)")
            return True
        else:
            print(f"⚠️  Conversation endpoint unexpected status: {response.status_code}")
            return False
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Conversation endpoint test failed: {e}")
        return False


def test_tts_endpoint_validation():
    """Test existing TTS endpoint validation."""
    try:
        # Test with minimal invalid request
        response = requests.post(
            f"{BASE_URL}/tts",
            json={"text": ""}  # Empty text should fail validation
        )
        
        if response.status_code == 422:
            print("✅ TTS endpoint validation working correctly")
            return True
        elif response.status_code == 503:
            print("⚠️  TTS endpoint available but model not loaded (expected)")
            return True
        else:
            print(f"⚠️  TTS endpoint unexpected status: {response.status_code}")
            return False
            
    except requests.exceptions.RequestException as e:
        print(f"❌ TTS endpoint test failed: {e}")
        return False


def main():
    """Run all API tests."""
    print("API Endpoints Test Suite")
    print("========================")
    
    tests = [
        ("Server Health Check", test_health_check),
        ("API Documentation", test_api_docs),
        ("STT Endpoint Validation", test_stt_endpoint_validation),
        ("Conversation Endpoint Validation", test_conversation_endpoint_validation),
        ("TTS Endpoint Validation", test_tts_endpoint_validation),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n🧪 Running: {test_name}")
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ Test '{test_name}' crashed: {e}")
            results.append((test_name, False))
    
    # Summary
    print(f"\n{'='*50}")
    print("Test Results Summary:")
    print(f"{'='*50}")
    
    passed = 0
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} {test_name}")
        if result:
            passed += 1
    
    print(f"\n📊 {passed}/{len(results)} tests passed")
    
    if passed == len(results):
        print("🎉 All tests passed! API endpoints are working correctly.")
    else:
        print("⚠️  Some tests failed. Check server status and model loading.")


if __name__ == "__main__":
    main()