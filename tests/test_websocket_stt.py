#!/usr/bin/env python3
"""
WebSocket STT client for testing real-time transcription.
This demonstrates how to connect to the WebSocket STT endpoint.
"""

import asyncio
import json
import websockets
import numpy as np
import wave
from pathlib import Path


async def test_websocket_connection():
    """Test basic WebSocket STT connection."""
    uri = "ws://localhost:8004/ws/transcribe"
    
    try:
        async with websockets.connect(uri) as websocket:
            print("✅ Connected to WebSocket STT endpoint")
            
            # Wait for ready message
            ready_msg = await websocket.recv()
            print(f"📡 Server: {ready_msg}")
            
            # Send ping command
            await websocket.send(json.dumps({"action": "ping"}))
            pong_msg = await websocket.recv()
            print(f"🏓 Ping response: {pong_msg}")
            
            # Send clear command
            await websocket.send(json.dumps({"action": "clear"}))
            clear_msg = await websocket.recv()
            print(f"🧹 Clear response: {clear_msg}")
            
            print("✅ WebSocket connection test completed successfully!")
            
    except Exception as e:
        print(f"❌ WebSocket test failed: {e}")


def generate_test_audio():
    """Generate a simple test audio file (1 second of 440Hz tone)."""
    sample_rate = 16000
    duration = 1.0
    frequency = 440.0
    
    t = np.linspace(0, duration, int(sample_rate * duration), False)
    audio = np.sin(2 * np.pi * frequency * t)
    
    # Convert to 16-bit PCM
    audio_int16 = (audio * 32767).astype(np.int16)
    
    # Save in tests directory
    test_file = Path(__file__).parent / "test_audio.wav"
    
    with wave.open(str(test_file), "wb") as wav_file:
        wav_file.setnchannels(1)  # Mono
        wav_file.setsampwidth(2)  # 2 bytes (16-bit)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(audio_int16.tobytes())
    
    print(f"📁 Generated {test_file}")
    return audio_int16.tobytes()


async def test_audio_transcription():
    """Test sending audio data to WebSocket STT."""
    uri = "ws://localhost:8004/ws/transcribe"
    
    # Generate test audio
    test_audio_bytes = generate_test_audio()
    
    try:
        async with websockets.connect(uri) as websocket:
            print("🎤 Testing audio transcription...")
            
            # Wait for ready message
            await websocket.recv()
            
            # Send audio data in chunks
            chunk_size = 1024
            for i in range(0, len(test_audio_bytes), chunk_size):
                chunk = test_audio_bytes[i:i + chunk_size]
                await websocket.send(chunk)
                
                # Check for transcription response (with timeout)
                try:
                    response = await asyncio.wait_for(websocket.recv(), timeout=1.0)
                    response_data = json.loads(response)
                    if response_data.get("type") == "transcription":
                        print(f"🎯 Transcription: {response_data.get('text')}")
                except asyncio.TimeoutError:
                    pass  # No response yet, continue
            
            print("✅ Audio test completed")
            
    except Exception as e:
        print(f"❌ Audio test failed: {e}")


if __name__ == "__main__":
    print("WebSocket STT Test Client")
    print("=========================")
    
    # Test basic connection
    asyncio.run(test_websocket_connection())
    
    print("\n" + "="*50)
    
    # Test audio transcription
    asyncio.run(test_audio_transcription())
    
    print(f"\n📋 Usage Instructions:")
    print(f"1. Start the server: python server.py")
    print(f"2. Run this test: python tests/test_websocket_stt.py")
    print(f"3. Or connect directly to: ws://localhost:8004/ws/transcribe")
    print(f"4. Send audio data as binary frames (16-bit PCM, 16kHz)")
    print(f"5. Receive JSON transcription results")