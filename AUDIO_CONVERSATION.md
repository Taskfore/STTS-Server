# Audio Conversation Feature

This document describes the real-time audio conversation feature implemented in the Chatterbox TTS Server. This feature enables bidirectional voice communication with pause detection, speech-to-text transcription, intelligent response generation, and text-to-speech synthesis.

## Overview

The audio conversation feature provides a complete voice-to-voice interaction system:

1. **Real-time Audio Input** - Streams PCM audio from client
2. **Pause Detection** - Uses WebRTC VAD to detect when speech ends
3. **Speech-to-Text** - Transcribes user speech when pauses are detected
4. **Response Generation** - Generates intelligent text responses (echo/template modes)
5. **Text-to-Speech** - Converts responses back to audio using voice cloning
6. **Audio Streaming** - Returns synthesized speech to client

## Architecture

### Core Components

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────────┐
│   Client App    │    │  WebSocket       │    │  Conversation       │
│                 │◄──►│  /ws/conversation│◄──►│  Processor          │
│  PCM Audio      │    │                  │    │                     │
│  Microphone     │    │  JSON Messages   │    │  • Pause Detection  │
└─────────────────┘    └──────────────────┘    │  • STT Processing   │
                                               │  • Response Gen     │
                                               │  • TTS Synthesis    │
                                               └─────────────────────┘
```

### Data Flow

```
PCM Audio Stream → WebRTC VAD → Speech End Detection → 
STT Transcription → Response Generation → TTS Synthesis → 
Base64 Audio Response → Client Playback
```

## Implementation Files

### New Components

1. **`pause_detection.py`** - Voice Activity Detection and pause detection
   - WebRTC VAD integration with fallback to energy-based detection
   - Configurable sensitivity and timing thresholds
   - Real-time speech/silence state tracking

2. **`conversation_engine.py`** - Text response generation
   - Echo mode: Simple repetition with variations
   - Template mode: Keyword-based responses with context
   - Extensible for future LLM integration

3. **`routers/websocket_conversation.py`** - WebSocket endpoint implementation
   - Real-time audio processing pipeline
   - Async response generation and streaming
   - State management and error handling

### Modified Components

1. **`requirements.txt`** - Added `webrtcvad` dependency
2. **`server.py`** - Integrated conversation router

## API Reference

### WebSocket Endpoint

**Primary Endpoint:** `ws://localhost:8004/ws/conversation`

**Query Parameters:**
- `language` (optional): STT language code (e.g., "en", "es") or null for auto-detection
- `voice_mode` (default: "predefined"): TTS voice mode ("predefined" or "clone")
- `predefined_voice_id` (optional): Predefined voice filename
- `reference_audio_filename` (optional): Reference audio for voice cloning
- `response_mode` (default: "echo"): Response generation mode ("echo" or "template")
- `pause_aggressiveness` (default: 2): WebRTC VAD sensitivity (0-3, higher = more aggressive)

**Example Connection:**
```javascript
const ws = new WebSocket('ws://localhost:8004/ws/conversation?response_mode=template&voice_mode=predefined');
```

### Message Format

#### Audio Input (Binary)
Send raw PCM audio data as binary WebSocket frames:
- **Format**: 16-bit little-endian PCM
- **Sample Rate**: 16kHz
- **Channels**: Mono
- **Frame Size**: Any size (recommended ~480 samples for 30ms frames)

#### Server Responses (JSON)

**Ready Message:**
```json
{
  "type": "ready",
  "message": "Real-time conversation ready",
  "config": {
    "voice_mode": "predefined",
    "response_mode": "template",
    "language": null,
    "pause_aggressiveness": 2
  }
}
```

**Conversation State Updates:**
```json
{
  "type": "conversation_state",
  "state": "listening",
  "is_speaking": true,
  "silence_duration_ms": 150,
  "events": ["speech_start"],
  "pause_detection": {
    "is_speaking": true,
    "speech_frames": 8,
    "silence_frames": 0,
    "speech_confidence": 1.0,
    "pause_confidence": 0.0
  }
}
```

**Transcription Results:**
```json
{
  "type": "transcription",
  "text": "Hello, how are you today?",
  "language": "en",
  "partial": false,
  "timing": {
    "start": 1.2,
    "end": 3.1,
    "duration": 1.9
  }
}
```

**Audio Response:**
```json
{
  "type": "response_audio",
  "audio_data": "UklGRjIAAABXQVZFZm10IBAAAA...",
  "format": "wav",
  "sample_rate": 16000,
  "duration_ms": 2150,
  "response_text": "I'm doing great, thanks for asking!"
}
```

#### Client Commands (Text)

**Reset Conversation:**
```json
{"action": "reset"}
```

**Change Response Mode:**
```json
{"action": "change_response_mode", "mode": "template"}
```

**Get Statistics:**
```json
{"action": "stats"}
```

**Ping Test:**
```json
{"action": "ping"}
```

### Test Endpoint

**Test Endpoint:** `ws://localhost:8004/ws/conversation/test`

Simple echo endpoint for testing WebSocket connectivity and basic functionality.

## Configuration

### Pause Detection Settings

The pause detector can be configured for different use cases:

**Conversation Mode (Default):**
- Aggressiveness: 2
- Min speech frames: 8 (~240ms)
- Min pause frames: 20 (~600ms)
- Good for natural conversation flow

**Formal Speech Mode:**
- Aggressiveness: 1
- Min speech frames: 10 (~300ms)
- Min pause frames: 35 (~1050ms)
- Less sensitive, waits longer for pauses

**Responsive Mode:**
- Aggressiveness: 3
- Min speech frames: 5 (~150ms)
- Min pause frames: 15 (~450ms)
- Very responsive, good for quick interactions

### Response Generation Modes

#### Echo Mode
Simple repetition with variations:
- "I heard you say: [input]"
- "You said: [input]"
- "Did you say: [input]?"

#### Template Mode
Intelligent keyword-based responses organized by categories:

**Supported Categories:**
- **Greetings**: hello, hi, hey → personalized greetings
- **Farewells**: bye, goodbye, see you → polite farewells
- **Questions**: how are you, what is → contextual responses
- **Affirmations**: yes, sure, okay → encouraging responses
- **Negations**: no, not really → understanding responses
- **Compliments**: thank you, great job → grateful responses
- **Confusion**: confused, don't understand → clarifying responses
- **Time**: what time, current time → time-based responses
- **Generic**: fallback responses for unmatched input

**Template Variables:**
- `{input}`: Original transcribed text
- `{time}`: Current time (HH:MM format)
- `{date}`: Current date (YYYY-MM-DD format)
- `{user}`: User name (default: "there")
- `{count}`: Number of conversation exchanges

## Installation and Setup

### Dependencies

Install the WebRTC VAD library:
```bash
pip install webrtcvad
```

Or install all dependencies:
```bash
pip install -r requirements.txt
```

### Starting the Server

```bash
python server.py
```

The conversation endpoint will be available at `ws://localhost:8004/ws/conversation`

### Testing the Feature

1. **Install dependencies** including `webrtcvad`
2. **Start the server** with `python server.py`
3. **Connect to WebSocket** endpoint with a WebSocket client
4. **Send PCM audio data** as binary frames
5. **Receive conversation state** and audio responses

## Client Implementation Example

### JavaScript WebSocket Client

```javascript
class AudioConversationClient {
    constructor(serverUrl = 'ws://localhost:8004/ws/conversation') {
        this.serverUrl = serverUrl;
        this.websocket = null;
        this.mediaRecorder = null;
        this.audioContext = null;
    }
    
    async connect() {
        this.websocket = new WebSocket(this.serverUrl);
        
        this.websocket.onopen = () => {
            console.log('Connected to conversation server');
        };
        
        this.websocket.onmessage = (event) => {
            const message = JSON.parse(event.data);
            this.handleServerMessage(message);
        };
        
        this.websocket.onclose = () => {
            console.log('Disconnected from conversation server');
        };
    }
    
    handleServerMessage(message) {
        switch (message.type) {
            case 'ready':
                console.log('Server ready:', message.config);
                break;
                
            case 'conversation_state':
                console.log('State:', message.state, 
                           'Speaking:', message.is_speaking);
                break;
                
            case 'response_audio':
                this.playAudioResponse(message.audio_data);
                console.log('Response:', message.response_text);
                break;
                
            case 'error':
                console.error('Server error:', message.message);
                break;
        }
    }
    
    async startRecording() {
        const stream = await navigator.mediaDevices.getUserMedia({
            audio: {
                sampleRate: 16000,
                channelCount: 1,
                echoCancellation: true,
                noiseSuppression: true
            }
        });
        
        this.audioContext = new AudioContext({ sampleRate: 16000 });
        const source = this.audioContext.createMediaStreamSource(stream);
        
        // Process audio and send PCM data to server
        await this.audioContext.audioWorklet.addModule('pcm-processor.js');
        const processor = new AudioWorkletNode(this.audioContext, 'pcm-processor');
        
        processor.port.onmessage = (event) => {
            const pcmData = event.data;
            if (this.websocket.readyState === WebSocket.OPEN) {
                this.websocket.send(pcmData);
            }
        };
        
        source.connect(processor);
    }
    
    playAudioResponse(base64Audio) {
        const audioData = atob(base64Audio);
        const audioBuffer = new ArrayBuffer(audioData.length);
        const view = new Uint8Array(audioBuffer);
        
        for (let i = 0; i < audioData.length; i++) {
            view[i] = audioData.charCodeAt(i);
        }
        
        const audioBlob = new Blob([audioBuffer], { type: 'audio/wav' });
        const audioUrl = URL.createObjectURL(audioBlob);
        const audio = new Audio(audioUrl);
        audio.play();
    }
    
    sendCommand(command) {
        if (this.websocket.readyState === WebSocket.OPEN) {
            this.websocket.send(JSON.stringify(command));
        }
    }
    
    disconnect() {
        if (this.websocket) {
            this.websocket.close();
        }
        if (this.audioContext) {
            this.audioContext.close();
        }
    }
}

// Usage
const client = new AudioConversationClient();
await client.connect();
await client.startRecording();
```

### PCM Audio Processor (pcm-processor.js)

```javascript
class PCMProcessor extends AudioWorkletProcessor {
    process(inputs, outputs, parameters) {
        const input = inputs[0];
        if (input.length > 0) {
            const channelData = input[0];
            
            // Convert float32 to int16 PCM
            const pcmData = new Int16Array(channelData.length);
            for (let i = 0; i < channelData.length; i++) {
                pcmData[i] = Math.max(-32768, Math.min(32767, channelData[i] * 32768));
            }
            
            this.port.postMessage(pcmData.buffer);
        }
        
        return true;
    }
}

registerProcessor('pcm-processor', PCMProcessor);
```

## Performance Considerations

### Latency Optimization
- **WebRTC VAD**: ~1ms processing time per frame
- **Pause Detection**: Configurable timing (600ms default)
- **STT Processing**: ~100-500ms depending on audio length
- **Response Generation**: ~1-10ms for template mode
- **TTS Synthesis**: ~200-1000ms depending on text length
- **Total Round-trip**: ~1-2 seconds typical

### Resource Usage
- **CPU**: Moderate (VAD + STT + TTS processing)
- **Memory**: ~100-500MB depending on model sizes
- **Network**: PCM audio streams (~32KB/s per connection)

### Scalability
- **Concurrent Connections**: Limited by STT/TTS model capacity
- **Threading**: Dedicated thread pools for CPU-intensive tasks
- **Memory Management**: Automatic cleanup of audio buffers

## Troubleshooting

### Common Issues

**WebRTC VAD Installation:**
```bash
# If webrtcvad installation fails
pip install --upgrade pip
pip install webrtcvad

# On some systems, you may need
sudo apt-get install python3-dev  # Linux
# or
xcode-select --install  # macOS
```

**Audio Format Issues:**
- Ensure client sends 16-bit PCM at 16kHz
- Check audio is mono (single channel)
- Verify little-endian byte order

**Pause Detection Sensitivity:**
- Too sensitive: Increase `pause_aggressiveness` (0-3)
- Not sensitive enough: Decrease `pause_aggressiveness`
- Adjust `min_pause_frames` for timing

**Response Generation:**
- Empty responses: Check input text cleaning
- No template matches: Add custom templates or use echo mode
- Template variables not substituted: Check template syntax

### Debug Commands

**Test WebSocket Connection:**
```bash
# Using wscat (npm install -g wscat)
wscat -c "ws://localhost:8004/ws/conversation/test"
```

**Check Server Logs:**
```bash
# Server logs show detailed conversation processing
tail -f logs/chatterbox_tts_server.log
```

**Test Individual Components:**
```python
# Test pause detection
from pause_detection import PauseDetector
detector = PauseDetector()

# Test response generation
from conversation_engine import ConversationResponseGenerator
generator = ConversationResponseGenerator("template")
response = generator.generate_response("Hello there!")
```

## Future Enhancements

### Planned Features
1. **LLM Integration** - OpenAI GPT, local LLMs, custom models
2. **Conversation Memory** - Multi-turn context and personality
3. **Advanced Voice Control** - Interrupt handling, voice commands
4. **Analytics** - Conversation metrics and insights
5. **Multi-language Support** - Automatic language switching
6. **Custom Wake Words** - Activation phrase detection

### Extension Points
- **Response Generators**: Implement custom response logic
- **Audio Processing**: Add noise reduction, audio effects
- **Voice Selection**: Dynamic voice changing during conversation
- **Context Integration**: Connect to external APIs and databases

## API Documentation

For complete API documentation including all endpoints, parameters, and response formats, visit:
- **Swagger UI**: `http://localhost:8004/docs`
- **ReDoc**: `http://localhost:8004/redoc`

## Support

For issues, feature requests, or contributions related to the audio conversation feature:
1. Check the troubleshooting section above
2. Review server logs for detailed error information
3. Test individual components to isolate issues
4. Report bugs with conversation flow details and audio format specifications
