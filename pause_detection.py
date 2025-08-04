# File: pause_detection.py
# Voice Activity Detection and pause detection for real-time audio streams

import logging
import time
from typing import Optional, Dict, Any
from collections import deque

try:
    import webrtcvad
    WEBRTC_AVAILABLE = True
except ImportError:
    WEBRTC_AVAILABLE = False
    webrtcvad = None

import numpy as np

logger = logging.getLogger(__name__)


class PauseDetector:
    """
    Voice Activity Detection and pause detection using WebRTC VAD.
    
    Detects when speech starts and ends in real-time audio streams,
    with configurable sensitivity and timing thresholds.
    """
    
    def __init__(self, 
                 aggressiveness: int = 2,
                 frame_duration_ms: int = 30,
                 min_speech_frames: int = 10,
                 min_pause_frames: int = 25,
                 sample_rate: int = 16000):
        """
        Initialize pause detector.
        
        Args:
            aggressiveness: WebRTC VAD aggressiveness (0-3, higher = more aggressive)
            frame_duration_ms: Frame duration in milliseconds (10, 20, or 30)
            min_speech_frames: Minimum frames to confirm speech start
            min_pause_frames: Minimum frames to confirm speech end (pause)
            sample_rate: Audio sample rate in Hz
        """
        if not WEBRTC_AVAILABLE:
            raise ImportError("webrtcvad library not available. Install with: pip install webrtcvad")
        
        # Validate parameters
        if aggressiveness not in range(4):
            raise ValueError("aggressiveness must be 0-3")
        if frame_duration_ms not in [10, 20, 30]:
            raise ValueError("frame_duration_ms must be 10, 20, or 30")
        if sample_rate not in [8000, 16000, 32000, 48000]:
            raise ValueError("sample_rate must be 8000, 16000, 32000, or 48000")
        
        self.vad = webrtcvad.Vad(aggressiveness)
        self.frame_duration_ms = frame_duration_ms
        self.sample_rate = sample_rate
        self.frame_size = int(sample_rate * frame_duration_ms / 1000)
        self.min_speech_frames = min_speech_frames
        self.min_pause_frames = min_pause_frames
        
        # State tracking
        self.speech_frames = 0
        self.silence_frames = 0
        self.is_speaking = False
        self.speech_start_time = None
        self.last_speech_time = None
        
        # Frame buffer for batch processing
        self.frame_buffer = deque()
        
        logger.info(f"PauseDetector initialized: aggressiveness={aggressiveness}, "
                   f"frame_duration={frame_duration_ms}ms, sample_rate={sample_rate}Hz")
    
    def process_pcm_chunk(self, pcm_data: bytes) -> Dict[str, Any]:
        """
        Process a chunk of PCM audio data and detect speech/pause events.
        
        Args:
            pcm_data: Raw PCM audio bytes (16-bit little-endian)
            
        Returns:
            Dict containing detection results and state information
        """
        if len(pcm_data) == 0:
            return self._get_current_state()
        
        # Log basic info about the chunk (every 100 calls to avoid spam)
        if not hasattr(self, '_chunk_count'):
            self._chunk_count = 0
        self._chunk_count += 1
        
        if self._chunk_count % 100 == 0:
            logger.info(f"PauseDetector processing chunk #{self._chunk_count}: {len(pcm_data)} bytes")
        
        # Convert PCM bytes to numpy array for frame extraction
        try:
            audio_np = np.frombuffer(pcm_data, dtype=np.int16)
            if self._chunk_count % 100 == 0:
                logger.info(f"Converted PCM to numpy: {len(audio_np)} samples, dtype={audio_np.dtype}, range=[{audio_np.min()}, {audio_np.max()}]")
        except Exception as e:
            logger.error(f"Error converting PCM data: {e}")
            return self._get_current_state()
        
        # Extract frames and process each one
        events = []
        for i in range(0, len(audio_np), self.frame_size):
            frame_data = audio_np[i:i + self.frame_size]
            
            # Only process complete frames
            if len(frame_data) == self.frame_size:
                frame_events = self._process_frame(frame_data.tobytes())
                events.extend(frame_events)
        
        # Return consolidated state with all events from this chunk
        state = self._get_current_state()
        state['events'] = events
        return state
    
    def _process_frame(self, frame_bytes: bytes) -> list:
        """
        Process a single audio frame and update state.
        
        Args:
            frame_bytes: Raw PCM frame bytes
            
        Returns:
            List of events that occurred during this frame
        """
        events = []
        current_time = time.time()
        
        try:
            # Use WebRTC VAD to detect speech in this frame
            is_speech = self.vad.is_speech(frame_bytes, self.sample_rate)
            
            # Log VAD results periodically
            if hasattr(self, '_frame_count'):
                self._frame_count += 1
            else:
                self._frame_count = 1
                
            if self._frame_count % 50 == 0:
                logger.info(f"WebRTC VAD frame #{self._frame_count}: speech={is_speech}, frame_size={len(frame_bytes)} bytes, sample_rate={self.sample_rate}")
                
        except Exception as e:
            logger.warning(f"WebRTC VAD error on frame #{getattr(self, '_frame_count', 0)}: {e}")
            is_speech = False
        
        # Update counters
        if is_speech:
            self.speech_frames += 1
            self.silence_frames = 0
            self.last_speech_time = current_time
        else:
            self.silence_frames += 1
            self.speech_frames = 0
        
        # Log counter updates periodically
        if self._frame_count % 50 == 0:
            logger.info(f"VAD counters: speech_frames={self.speech_frames}, silence_frames={self.silence_frames}, is_speaking={self.is_speaking}, min_speech={self.min_speech_frames}, min_pause={self.min_pause_frames}")
        
        # Check for state transitions
        prev_speaking = self.is_speaking
        
        # Speech start detection
        if not self.is_speaking and self.speech_frames >= self.min_speech_frames:
            self.is_speaking = True
            self.speech_start_time = current_time
            events.append('speech_start')
            logger.info(f"SPEECH START detected: speech_frames={self.speech_frames} >= min_speech_frames={self.min_speech_frames}")
        
        # Speech end detection (pause)
        elif self.is_speaking and self.silence_frames >= self.min_pause_frames:
            self.is_speaking = False
            speech_duration = current_time - self.speech_start_time if self.speech_start_time else 0
            events.append('speech_end')
            logger.info(f"SPEECH END detected: silence_frames={self.silence_frames} >= min_pause_frames={self.min_pause_frames}, speech_duration={speech_duration:.2f}s")
        
        return events
    
    def _get_current_state(self) -> Dict[str, Any]:
        """Get current detector state."""
        current_time = time.time()
        
        # Calculate durations
        silence_duration_ms = self.silence_frames * self.frame_duration_ms
        speech_duration_ms = self.speech_frames * self.frame_duration_ms
        
        # Calculate time since last speech
        time_since_speech = None
        if self.last_speech_time:
            time_since_speech = current_time - self.last_speech_time
        
        return {
            'is_speaking': self.is_speaking,
            'speech_frames': self.speech_frames,
            'silence_frames': self.silence_frames,
            'silence_duration_ms': silence_duration_ms,
            'speech_duration_ms': speech_duration_ms,
            'time_since_speech': time_since_speech,
            'speech_confidence': min(self.speech_frames / self.min_speech_frames, 1.0),
            'pause_confidence': min(self.silence_frames / self.min_pause_frames, 1.0),
            'events': []  # Will be populated by process_pcm_chunk
        }
    
    def reset(self):
        """Reset detector state."""
        self.speech_frames = 0
        self.silence_frames = 0
        self.is_speaking = False
        self.speech_start_time = None
        self.last_speech_time = None
        self.frame_buffer.clear()
        logger.debug("PauseDetector state reset")
    
    def get_recommended_settings(self) -> Dict[str, Any]:
        """Get recommended settings for different use cases."""
        return {
            'conversation': {
                'aggressiveness': 2,
                'min_speech_frames': 8,   # ~240ms at 30ms frames
                'min_pause_frames': 20,   # ~600ms at 30ms frames
                'description': 'Balanced for natural conversation flow'
            },
            'formal_speech': {
                'aggressiveness': 1,
                'min_speech_frames': 10,  # ~300ms
                'min_pause_frames': 35,   # ~1050ms
                'description': 'Less sensitive, waits longer for pauses'
            },
            'responsive': {
                'aggressiveness': 3,
                'min_speech_frames': 5,   # ~150ms
                'min_pause_frames': 15,   # ~450ms
                'description': 'Very responsive, good for quick interactions'
            }
        }


class EnergyFallbackDetector:
    """
    Fallback pause detector using energy-based detection.
    Used when WebRTC VAD is not available or fails.
    """
    
    def __init__(self, 
                 silence_threshold_db: float = -40.0,
                 min_pause_duration_ms: int = 750,
                 min_speech_duration_ms: int = 250,
                 frame_duration_ms: int = 30):
        """
        Initialize energy-based pause detector.
        
        Args:
            silence_threshold_db: RMS energy threshold in dB for silence
            min_pause_duration_ms: Minimum pause duration to confirm speech end
            min_speech_duration_ms: Minimum speech duration to confirm speech start
            frame_duration_ms: Frame processing duration
        """
        self.silence_threshold_db = silence_threshold_db
        self.min_pause_frames = int(min_pause_duration_ms / frame_duration_ms)
        self.min_speech_frames = int(min_speech_duration_ms / frame_duration_ms)
        self.frame_duration_ms = frame_duration_ms
        
        # State tracking
        self.energy_history = deque(maxlen=100)
        self.speech_frames = 0
        self.silence_frames = 0
        self.is_speaking = False
        
        logger.info(f"EnergyFallbackDetector initialized: threshold={silence_threshold_db}dB")
    
    def process_pcm_chunk(self, pcm_data: bytes) -> Dict[str, Any]:
        """Process PCM data using energy-based detection."""
        if len(pcm_data) == 0:
            return self._get_current_state()
        
        # Convert to numpy and calculate RMS energy
        try:
            audio_np = np.frombuffer(pcm_data, dtype=np.int16).astype(np.float32) / 32768.0
            rms_energy = np.sqrt(np.mean(audio_np ** 2))
            energy_db = 20 * np.log10(rms_energy + 1e-7)
        except Exception as e:
            logger.error(f"Error calculating energy: {e}")
            return self._get_current_state()
        
        self.energy_history.append(energy_db)
        
        # Simple thresholding
        is_speech = energy_db > self.silence_threshold_db
        events = []
        
        if is_speech:
            self.speech_frames += 1
            self.silence_frames = 0
        else:
            self.silence_frames += 1
            self.speech_frames = 0
        
        # State transitions
        prev_speaking = self.is_speaking
        
        if not self.is_speaking and self.speech_frames >= self.min_speech_frames:
            self.is_speaking = True
            events.append('speech_start')
        elif self.is_speaking and self.silence_frames >= self.min_pause_frames:
            self.is_speaking = False
            events.append('speech_end')
        
        state = self._get_current_state()
        state['events'] = events
        state['energy_db'] = energy_db
        
        return state
    
    def _get_current_state(self) -> Dict[str, Any]:
        """Get current detector state."""
        return {
            'is_speaking': self.is_speaking,
            'speech_frames': self.speech_frames,
            'silence_frames': self.silence_frames,
            'silence_duration_ms': self.silence_frames * self.frame_duration_ms,
            'speech_duration_ms': self.speech_frames * self.frame_duration_ms,
            'events': []
        }
    
    def reset(self):
        """Reset detector state."""
        self.speech_frames = 0
        self.silence_frames = 0
        self.is_speaking = False
        self.energy_history.clear()