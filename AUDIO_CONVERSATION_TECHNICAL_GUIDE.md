# Audio Conversation Feature: Technical Deep Dive

## Overview

The Audio Conversation feature in this TTS server provides real-time, bidirectional voice communication through WebSocket connections. It combines Speech-to-Text (STT), Text-to-Speech (TTS), pause detection, and response generation into a seamless conversation pipeline. This document provides technical details of the implementation for engineers.

## Architecture Overview

The system follows a client-server WebSocket architecture with three main modes:

1. **Record Mode**: Traditional file upload STT processing
2. **Real-time STT Mode**: Live transcription with WebSocket streaming
3. **Conversation Mode**: Full bidirectional conversation with pause detection and TTS responses

```
┌─────────────────┐    WebSocket    ┌─────────────────┐
│     Frontend    │ ←──────────────→ │     Backend     │
│                 │    PCM Audio     │                 │
│ • Audio Capture │    Streaming     │ • STT Engine    │
│ • WebSocket     │                  │ • TTS Engine    │
│ • UI Management │                  │ • Pause Detect  │
│ • Audio Playback│                  │ • Response Gen  │
└─────────────────┘                  └─────────────────┘
```

## Frontend Implementation Deep Dive

### Audio Capture Pipeline

The frontend uses the Web Audio API for real-time audio processing:

**File: `ui/script.js:370-500`**

```javascript
class AudioConversationManager {
    constructor() {
        this.websocket = null;
        this.connected = false;
        this.audioContext = null;
        this.mediaStream = null;
        this.pcmProcessor = null;
        this.conversationHistory = [];
    }
}
```

#### 1. Media Stream Acquisition

```javascript
// Get user media with specific constraints
const stream = await navigator.mediaDevices.getUserMedia({
    audio: {
        sampleRate: 16000,        // WebRTC VAD requirement
        channelCount: 1,          // Mono audio
        echoCancellation: true,   // Improve quality
        noiseSuppression: true,   // Reduce background noise
        autoGainControl: true     // Normalize volume
    }
});
```

#### 2. Web Audio API Processing Chain

The audio processing chain converts microphone input to PCM data:

```javascript
// Create audio processing context
this.audioContext = new AudioContext({ sampleRate: 16000 });
const source = this.audioContext.createMediaStreamSource(stream);

// Create ScriptProcessor for PCM extraction
this.pcmProcessor = this.audioContext.createScriptProcessor(4096, 1, 1);

this.pcmProcessor.onaudioprocess = (event) => {
    const inputBuffer = event.inputBuffer;
    const inputData = inputBuffer.getChannelData(0); // Get mono channel
    
    // Convert Float32 to Int16 PCM
    const pcmData = new Int16Array(inputData.length);
    for (let i = 0; i < inputData.length; i++) {
        pcmData[i] = Math.max(-32768, Math.min(32767, inputData[i] * 32768));
    }
    
    // Send PCM data via WebSocket
    if (this.websocket && this.websocket.readyState === WebSocket.OPEN) {
        this.websocket.send(pcmData.buffer);
    }
};

// Connect the processing chain
source.connect(this.pcmProcessor);
this.pcmProcessor.connect(this.audioContext.destination);
```

#### 3. WebSocket Connection Management

**Connection Establishment:**

```javascript
async connect() {
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const host = window.location.host || 'localhost:8004';
    const baseWsUrl = IS_LOCAL_FILE ? 'ws://localhost:8004' : `${protocol}//${host}`;
    
    // Build parameters for conversation mode
    const params = new URLSearchParams({
        response_mode: this.responseMode,
        pause_aggressiveness: this.pauseSensitivity,
        voice_mode: voiceMode
    });
    
    const wsUrl = `${baseWsUrl}/ws/conversation?${params.toString()}`;
    this.websocket = new WebSocket(wsUrl);
    this.setupWebSocketEventHandlers();
}
```

**Message Handling:**

```javascript
this.websocket.onmessage = (event) => {
    const message = JSON.parse(event.data);
    
    switch (message.type) {
        case 'transcription':
            this.handleTranscription(message);
            break;
        case 'response_audio':
            this.handleAudioResponse(message);
            break;
        case 'conversation_state':
            this.handleConversationStateUpdate(message);
            break;
    }
};
```

#### 4. Audio Response Playback

```javascript
handleAudioResponse(message) {
    // Decode base64 audio data
    const audioData = atob(message.audio_data);
    const audioBytes = new Uint8Array(audioData.length);
    for (let i = 0; i < audioData.length; i++) {
        audioBytes[i] = audioData.charCodeAt(i);
    }
    
    // Create audio blob and play
    const audioBlob = new Blob([audioBytes], { type: 'audio/wav' });
    const audioUrl = URL.createObjectURL(audioBlob);
    const audio = new Audio(audioUrl);
    audio.play();
}
```

### UI State Management

**Transcription Display with Timing:**

```javascript
// Frontend handles timing-based overlap detection
currentTranscriptionTiming = message.timing;
if (currentTranscriptionTiming && lastFinalizedTiming) {
    const overlapThreshold = 0.5; // seconds
    const isOverlapping = currentTranscriptionTiming.start < 
                         (lastFinalizedTiming.end + overlapThreshold);
    
    if (isOverlapping) {
        // Replace overlapping transcription instead of appending
        updateTranscriptionLine(message.text, true);
    } else {
        // Add new line for non-overlapping transcription
        addNewTranscriptionLine(message.text);
    }
}
```

## Backend Implementation Deep Dive

### WebSocket STT Router (`routers/websocket_stt.py`)

#### 1. PCM Audio Decoder

```python
class PCMAudioDecoder:
    @staticmethod
    async def decode_pcm_to_numpy(pcm_data: bytes) -> Optional[np.ndarray]:
        """Convert raw PCM data to numpy array."""
        try:
            # Frontend sends 16-bit signed integers in little-endian format
            audio_np = np.frombuffer(pcm_data, dtype=np.int16).astype(np.float32) / 32768.0
            return audio_np
        except Exception as e:
            logger.error(f"Error converting PCM to numpy: {e}")
            return None
```

#### 2. Optimized Audio Buffer

```python
class OptimizedAudioBuffer:
    def __init__(self, max_duration_seconds: float = 30.0, sample_rate: int = 16000):
        self.max_duration = max_duration_seconds
        self.sample_rate = sample_rate
        self.max_samples = int(max_duration_seconds * sample_rate)
        self.chunks = deque()  # Efficient append/pop operations
        self.total_samples = 0
        self.lock = threading.Lock()
    
    def add_audio(self, audio_data: np.ndarray):
        """Add audio data with automatic buffer management."""
        with self.lock:
            self.chunks.append(audio_data)
            self.total_samples += len(audio_data)
            
            # Remove old chunks if exceeding max duration
            while self.total_samples > self.max_samples and self.chunks:
                removed_chunk = self.chunks.popleft()
                self.total_samples -= len(removed_chunk)
    
    def get_recent_audio(self, duration_seconds: float = 5.0) -> np.ndarray:
        """Get recent audio for transcription."""
        target_samples = int(duration_seconds * self.sample_rate)
        
        with self.lock:
            if not self.chunks:
                return np.array([], dtype=np.float32)
            
            # Collect recent chunks up to target duration
            collected_chunks = []
            collected_samples = 0
            
            # Work backwards from most recent chunks
            for chunk in reversed(self.chunks):
                collected_chunks.insert(0, chunk)
                collected_samples += len(chunk)
                if collected_samples >= target_samples:
                    break
            
            return np.concatenate(collected_chunks) if collected_chunks else np.array([], dtype=np.float32)
```

#### 3. Real-time STT Processor

```python
class OptimizedRealtimeSTT:
    def __init__(self, stt_engine: STTEngine, language: Optional[str] = None):
        self.stt_engine = stt_engine
        self.language = language
        self.audio_buffer = OptimizedAudioBuffer()
        self.processing = False
        self.process_every_n_chunks = 4  # Process every 4 chunks (~256ms)
        
    async def process_audio_chunk(self, audio_data: bytes) -> Optional[TranscriptionResult]:
        """Process incoming PCM audio chunk."""
        audio_np = await self.decoder.decode_pcm_to_numpy(audio_data)
        if audio_np is None:
            return None
        
        self.audio_buffer.add_audio(audio_np)
        self.chunk_count += 1

        # Process every N chunks to accumulate enough audio
        if self.chunk_count % self.process_every_n_chunks != 0:
            return None
        
        if self.processing:
            return None
            
        self.processing = True
        
        try:
            # Get recent audio for transcription (2 seconds worth)
            recent_audio = self.audio_buffer.get_recent_audio(duration_seconds=2.0)
            
            # Transcribe using dedicated thread pool
            loop = asyncio.get_event_loop()
            transcription_result = await loop.run_in_executor(
                TRANSCRIPTION_THREAD_POOL,
                self._transcribe_numpy_audio,
                recent_audio
            )
            
            return transcription_result
        finally:
            self.processing = False
```

### WebSocket Conversation Router (`routers/websocket_conversation.py`)

#### 1. Conversation Processor

```python
class ConversationProcessor:
    def __init__(self, stt_engine: STTEngine, websocket: WebSocket, 
                 voice_mode: str = "predefined", response_mode: str = "echo",
                 pause_aggressiveness: int = 2):
        self.stt_engine = stt_engine
        self.websocket = websocket
        self.voice_mode = voice_mode
        
        # Initialize audio processing
        self.audio_buffer = OptimizedAudioBuffer(max_duration_seconds=10.0)
        self.decoder = PCMAudioDecoder()
        
        # Initialize pause detection
        if WEBRTC_AVAILABLE:
            self.pause_detector = PauseDetector(
                aggressiveness=pause_aggressiveness,
                min_speech_frames=3,   # ~90ms - sensitive
                min_pause_frames=10,   # ~300ms - short pauses
            )
        else:
            self.pause_detector = EnergyFallbackDetector()
        
        # Initialize response generation
        self.response_generator = ConversationResponseGenerator(response_mode)
```

#### 2. Speech-to-Response Pipeline

```python
async def _process_speech_to_response(self, audio_data: np.ndarray):
    """Complete speech-to-response processing pipeline."""
    try:
        # Step 1: STT - Transcribe speech
        transcription_result = await self._transcribe_audio(audio_data)
        if not transcription_result or not transcription_result.text.strip():
            return None
        
        transcribed_text = transcription_result.text.strip()
        
        # Step 2: Generate text response
        response_text = self.response_generator.generate_response(transcribed_text)
        
        # Step 3: Send transcription to client with timing
        await self.websocket.send_json({
            "type": "transcription",
            "text": transcribed_text,
            "language": transcription_result.language,
            "partial": False,
            "timing": {
                "start": transcription_result.segments[0].start if transcription_result.segments else 0.0,
                "end": transcription_result.segments[-1].end if transcription_result.segments else 0.0,
                "duration": duration
            },
            "segments": [
                {
                    "text": seg.text,
                    "start": seg.start,
                    "end": seg.end
                } for seg in transcription_result.segments
            ]
        })
        
        # Step 4: TTS - Convert response to speech
        audio_response = await self._text_to_speech(response_text)
        
        if audio_response:
            # Send audio response to client
            await self.websocket.send_json({
                "type": "response_audio",
                "audio_data": audio_response['audio_data'],  # base64 encoded
                "format": audio_response['format'],
                "sample_rate": audio_response['sample_rate'],
                "duration_ms": audio_response['duration_ms'],
                "response_text": response_text
            })
            
    except Exception as e:
        logger.error(f"Error in speech-to-response processing: {e}")
    finally:
        self.processing_transcription = False
        self.conversation_state = "listening"
```

### STT Engine (`stt_engine.py`)

#### Core Transcription Methods

```python
class STTEngine:
    def transcribe_numpy_with_timing(self, audio_array: 'np.ndarray', 
                                   language: Optional[str] = None) -> Optional[TranscriptionResult]:
        """Transcribe with full timing information."""
        try:
            # Use configured language or auto-detection
            detect_language = language or get_stt_language()
            language_param = None if detect_language == "auto" else detect_language
            
            # Get raw Whisper result
            raw_result = self.model.transcribe(audio_array, language=language_param)
            
            # Convert to typed model with segments and timing
            transcription_result = TranscriptionResult(**raw_result)
            return transcription_result
            
        except Exception as e:
            logger.error(f"Error during numpy transcription with timing: {e}")
            return None
```

### Pause Detection (`pause_detection.py`)

#### WebRTC VAD Integration

```python
class PauseDetector:
    def __init__(self, aggressiveness: int = 2, frame_duration_ms: int = 30,
                 min_speech_frames: int = 10, min_pause_frames: int = 25):
        self.vad = webrtcvad.Vad(aggressiveness)
        self.frame_duration_ms = frame_duration_ms
        self.frame_size = int(sample_rate * frame_duration_ms / 1000)
        self.min_speech_frames = min_speech_frames
        self.min_pause_frames = min_pause_frames
        
        # State tracking
        self.speech_frames = 0
        self.silence_frames = 0
        self.is_speaking = False
        
    def process_pcm_chunk(self, pcm_data: bytes) -> Dict[str, Any]:
        """Process PCM chunk and detect speech/pause events."""
        audio_np = np.frombuffer(pcm_data, dtype=np.int16)
        events = []
        
        # Process audio in WebRTC-compatible frames
        for i in range(0, len(audio_np), self.frame_size):
            frame = audio_np[i:i + self.frame_size]
            if len(frame) < self.frame_size:
                continue
                
            # Convert frame back to bytes for WebRTC VAD
            frame_bytes = frame.tobytes()
            
            # WebRTC VAD detection
            is_speech = self.vad.is_speech(frame_bytes, self.sample_rate)
            
            if is_speech:
                self.speech_frames += 1
                self.silence_frames = 0
                
                # Trigger speech start
                if not self.is_speaking and self.speech_frames >= self.min_speech_frames:
                    self.is_speaking = True
                    events.append('speech_start')
            else:
                self.silence_frames += 1
                self.speech_frames = 0
                
                # Trigger speech end (pause detected)
                if self.is_speaking and self.silence_frames >= self.min_pause_frames:
                    self.is_speaking = False
                    events.append('speech_end')
        
        return {
            'events': events,
            'is_speaking': self.is_speaking,
            'speech_confidence': self.speech_frames / max(1, self.min_speech_frames),
            'silence_duration_ms': self.silence_frames * self.frame_duration_ms
        }
```

### Response Generation (`conversation_engine.py`)

#### Template-based Response System

```python
class ConversationResponseGenerator:
    def __init__(self, response_mode: str = "echo"):
        self.response_mode = response_mode
        self.conversation_history = []
        self.template_responses = self._load_default_templates()
    
    def generate_response(self, input_text: str) -> str:
        """Generate response based on configured mode."""
        cleaned_input = self._clean_text(input_text)
        
        # Add to conversation history
        self.conversation_history.append({
            'timestamp': datetime.now(),
            'input': cleaned_input
        })
        
        if self.response_mode == "echo":
            return self._generate_echo_response(cleaned_input)
        elif self.response_mode == "template":
            return self._generate_template_response(cleaned_input)
        else:
            return self._get_fallback_response()
    
    def _generate_echo_response(self, input_text: str) -> str:
        """Generate echo-style responses with variety."""
        echo_patterns = [
            f"I heard you say: {input_text}",
            f"You said: {input_text}",
            f"Did you say: {input_text}?",
            f"I understand you said: {input_text}",
        ]
        return random.choice(echo_patterns)
```

## Audio Processing Pipeline

### TTS Synthesis in Conversations

```python
async def _text_to_speech(self, text: str) -> Optional[Dict[str, Any]]:
    """Convert text to speech for WebSocket transmission."""
    # Get TTS parameters from config
    temperature = get_gen_default_temperature()
    exaggeration = get_gen_default_exaggeration()
    cfg_weight = get_gen_default_cfg_weight()
    seed = get_gen_default_seed()
    speed_factor = get_gen_default_speed_factor()
    
    # Synthesize speech using TTS engine
    loop = asyncio.get_event_loop()
    audio_tensor, sample_rate = await loop.run_in_executor(
        CONVERSATION_THREAD_POOL,
        engine.synthesize,
        text, self.voice_path, temperature, exaggeration, cfg_weight, seed
    )
    
    # Convert to numpy and apply speed factor
    audio_np = audio_tensor.cpu().numpy().squeeze()
    if speed_factor != 1.0:
        audio_np, _ = utils.apply_speed_factor(audio_np, sample_rate, speed_factor)
    
    # Encode to WAV format for WebSocket transmission
    target_sample_rate = get_audio_sample_rate()
    encoded_audio = utils.encode_audio(
        audio_array=audio_np,
        sample_rate=sample_rate,
        output_format="wav",
        target_sample_rate=target_sample_rate
    )
    
    # Encode to base64 for JSON transmission
    audio_base64 = base64.b64encode(encoded_audio).decode('utf-8')
    duration_ms = int(len(audio_np) / sample_rate * 1000)
    
    return {
        'audio_data': audio_base64,
        'format': 'wav',
        'sample_rate': target_sample_rate,
        'duration_ms': duration_ms
    }
```

## Performance Optimizations

### 1. Thread Pool Management

```python
# Dedicated thread pools prevent blocking
TRANSCRIPTION_THREAD_POOL = ThreadPoolExecutor(max_workers=2, thread_name_prefix="STT-Worker")
CONVERSATION_THREAD_POOL = ThreadPoolExecutor(max_workers=3, thread_name_prefix="Conversation-Worker")
```

### 2. Audio Buffer Optimization

- **Deque-based chunks**: O(1) append/pop operations
- **Automatic cleanup**: Prevents memory leaks
- **Configurable duration**: Balance memory vs. context

### 3. Processing Throttling

```python
# Process every N chunks to accumulate sufficient audio
self.process_every_n_chunks = 4  # ~256ms of audio at 16kHz
if self.chunk_count % self.process_every_n_chunks != 0:
    return None
```

### 4. Concurrent Processing Prevention

```python
# Prevent overlapping transcription requests
if self.processing:
    return None
self.processing = True
try:
    # Process audio
finally:
    self.processing = False
```

## Data Flow Summary

### Real-time Conversation Flow

1. **Audio Capture**: Frontend captures microphone → Float32 samples
2. **PCM Conversion**: Convert Float32 → Int16 PCM → WebSocket bytes
3. **WebSocket Transmission**: Stream PCM chunks to backend
4. **Audio Buffering**: Backend accumulates chunks in optimized buffer
5. **Pause Detection**: WebRTC VAD analyzes frames for speech/silence
6. **Speech End Trigger**: Pause detected → trigger transcription pipeline
7. **STT Processing**: Recent audio buffer → Whisper → transcription with timing
8. **Response Generation**: Transcribed text → response generator → response text
9. **TTS Synthesis**: Response text → Chatterbox TTS → audio tensor
10. **Audio Encoding**: Audio tensor → WAV → base64 → WebSocket JSON
11. **Client Playback**: Frontend decodes base64 → Audio blob → browser playback

### Message Types

**Frontend → Backend:**
- **PCM Audio Chunks**: Raw binary audio data (16-bit, 16kHz, mono)
- **Control Commands**: JSON commands (`{"action": "reset"}`, `{"action": "stop"}`)

**Backend → Frontend:**
- **Ready**: `{"type": "ready", "message": "..."}`
- **Transcription**: `{"type": "transcription", "text": "...", "timing": {...}, "segments": [...]}`
- **Response Audio**: `{"type": "response_audio", "audio_data": "base64...", "format": "wav", ...}`
- **Conversation State**: `{"type": "conversation_state", "state": "listening", "events": [...]}`
- **Error**: `{"type": "error", "message": "..."}`

## Configuration Parameters

### Audio Settings
- **Sample Rate**: 16kHz (WebRTC VAD requirement)
- **Channels**: Mono (1 channel)
- **Bit Depth**: 16-bit PCM
- **Frame Duration**: 30ms (WebRTC standard)

### Pause Detection
- **Aggressiveness**: 0-3 (WebRTC VAD sensitivity)
- **Min Speech Frames**: 3 frames (~90ms) to confirm speech start
- **Min Pause Frames**: 10 frames (~300ms) to confirm speech end

### Processing Intervals
- **Chunk Processing**: Every 4 chunks (~256ms)
- **Buffer Duration**: 2-5 seconds for transcription context
- **Max Buffer**: 10-30 seconds before cleanup

This technical implementation provides a robust, scalable foundation for real-time voice conversations with precise timing control and efficient resource management.