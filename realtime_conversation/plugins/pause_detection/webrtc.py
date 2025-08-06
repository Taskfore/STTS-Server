"""
WebRTC VAD-based pause detection.

Provides voice activity detection using the WebRTC VAD algorithm.
"""

import logging
import time
from typing import Dict, Any
from collections import deque

try:
    import webrtcvad
    WEBRTC_AVAILABLE = True
except ImportError:
    WEBRTC_AVAILABLE = False
    webrtcvad = None

import numpy as np

from .base import BasePauseDetector
from ...core.interfaces import AudioData

logger = logging.getLogger(__name__)


class WebRTCPauseDetector(BasePauseDetector):
    """WebRTC VAD-based pause detector."""
    
    def __init__(
        self,
        aggressiveness: int = 2,
        frame_duration_ms: int = 30,
        min_speech_frames: int = 3,
        min_pause_frames: int = 10,
        sample_rate: int = 16000
    ):
        """
        Initialize WebRTC pause detector.
        
        Args:
            aggressiveness: WebRTC VAD aggressiveness (0-3, higher = more aggressive)
            frame_duration_ms: Frame duration in milliseconds (10, 20, or 30)
            min_speech_frames: Minimum frames to confirm speech start
            min_pause_frames: Minimum frames to confirm speech end (pause)
            sample_rate: Audio sample rate in Hz
        """
        if not WEBRTC_AVAILABLE:
            raise ImportError("webrtcvad library not available. Install with: pip install webrtcvad")
        
        super().__init__()
        
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
        self.speech_start_time = None
        self._frame_buffer = deque()
        
        logger.info(
            f"WebRTC pause detector initialized: aggressiveness={aggressiveness}, "
            f"frame_duration={frame_duration_ms}ms, sample_rate={sample_rate}Hz"
        )
    
    def _process_chunk_sync(self, audio: AudioData) -> Dict[str, Any]:
        """Process audio chunk using WebRTC VAD."""
        events = []
        
        try:
            # Convert audio data to frames
            frames = self._extract_frames(audio.data)
            
            if not frames:
                return self._create_state_result(events)
            
            # Process each frame through VAD
            for frame in frames:
                is_speech = self._process_frame(frame)
                
                if is_speech:
                    self._silence_frames = 0
                    self._speech_frames += 1
                    
                    # Check for speech start
                    if not self._is_speaking and self._speech_frames >= self.min_speech_frames:
                        self._is_speaking = True
                        self.speech_start_time = time.time()
                        events.append("speech_start")
                        logger.debug("Speech start detected")
                
                else:
                    self._speech_frames = 0
                    self._silence_frames += 1
                    
                    # Check for speech end
                    if self._is_speaking and self._silence_frames >= self.min_pause_frames:
                        self._is_speaking = False
                        events.append("speech_end")
                        logger.debug("Speech end detected")
            
            # Calculate silence duration
            silence_duration_ms = int(self._silence_frames * self.frame_duration_ms)
            
            return self._create_state_result(
                events=events,
                silence_duration_ms=silence_duration_ms
            )
            
        except Exception as e:
            logger.error(f"Error in WebRTC VAD processing: {e}")
            return self._create_state_result(events)
    
    def _extract_frames(self, audio_data: bytes) -> list:
        """Extract fixed-size frames from audio data."""
        try:
            # Convert bytes to numpy array (assuming 16-bit PCM)
            audio_np = np.frombuffer(audio_data, dtype=np.int16)
            
            frames = []
            frame_bytes = self.frame_size * 2  # 2 bytes per sample for 16-bit
            
            # Add to frame buffer
            self._frame_buffer.extend(audio_data)
            
            # Extract complete frames
            while len(self._frame_buffer) >= frame_bytes:
                frame_data = bytes(list(self._frame_buffer)[:frame_bytes])
                frames.append(frame_data)
                
                # Remove processed samples from buffer
                for _ in range(frame_bytes):
                    self._frame_buffer.popleft()
            
            return frames
            
        except Exception as e:
            logger.error(f"Error extracting frames: {e}")
            return []
    
    def _process_frame(self, frame_data: bytes) -> bool:
        """Process a single frame through WebRTC VAD."""
        try:
            # WebRTC VAD expects exactly the right frame size
            if len(frame_data) != self.frame_size * 2:
                return False
            
            return self.vad.is_speech(frame_data, self.sample_rate)
            
        except Exception as e:
            logger.error(f"Error processing frame with WebRTC VAD: {e}")
            return False
    
    def reset(self) -> None:
        """Reset WebRTC detector state."""
        super().reset()
        self.speech_start_time = None
        self._frame_buffer.clear()
        logger.debug("WebRTC pause detector state reset")
    
    # Configuration methods
    
    def set_aggressiveness(self, aggressiveness: int) -> None:
        """Set VAD aggressiveness level."""
        if aggressiveness not in range(4):
            raise ValueError("aggressiveness must be 0-3")
        
        self.vad.set_mode(aggressiveness)
        logger.info(f"WebRTC VAD aggressiveness set to: {aggressiveness}")
    
    def get_config(self) -> Dict[str, Any]:
        """Get current detector configuration."""
        return {
            "type": "webrtc",
            "aggressiveness": self.vad.mode if hasattr(self.vad, 'mode') else 2,
            "frame_duration_ms": self.frame_duration_ms,
            "min_speech_frames": self.min_speech_frames,
            "min_pause_frames": self.min_pause_frames,
            "sample_rate": self.sample_rate,
            "frame_size": self.frame_size
        }


# Factory functions

def create_webrtc_detector(
    aggressiveness: int = 2,
    sample_rate: int = 16000
) -> WebRTCPauseDetector:
    """
    Create a WebRTC pause detector with standard settings.
    
    Args:
        aggressiveness: VAD aggressiveness level (0-3)
        sample_rate: Audio sample rate
        
    Returns:
        Configured WebRTC pause detector
    """
    return WebRTCPauseDetector(
        aggressiveness=aggressiveness,
        sample_rate=sample_rate
    )


def create_webrtc_detector_from_config(config: Dict[str, Any]) -> WebRTCPauseDetector:
    """
    Create WebRTC pause detector from configuration.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Configured WebRTC pause detector
    """
    return WebRTCPauseDetector(
        aggressiveness=config.get("aggressiveness", 2),
        frame_duration_ms=config.get("frame_duration_ms", 30),
        min_speech_frames=config.get("min_speech_frames", 3),
        min_pause_frames=config.get("min_pause_frames", 10),
        sample_rate=config.get("sample_rate", 16000)
    )