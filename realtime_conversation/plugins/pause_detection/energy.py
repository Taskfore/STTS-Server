"""
Energy-based pause detection.

Provides voice activity detection using energy/RMS-based analysis as a fallback
when WebRTC VAD is not available.
"""

import logging
import time
from typing import Dict, Any
from collections import deque
import numpy as np

from .base import BasePauseDetector
from ...core.interfaces import AudioData
from ...audio.codecs import decode_pcm

logger = logging.getLogger(__name__)


class EnergyPauseDetector(BasePauseDetector):
    """Energy/RMS-based pause detector."""
    
    def __init__(
        self,
        energy_threshold: float = 0.01,
        frame_duration_ms: int = 30,
        min_speech_frames: int = 3,
        min_pause_frames: int = 10,
        sample_rate: int = 16000,
        adaptive_threshold: bool = True,
        adaptation_rate: float = 0.95
    ):
        """
        Initialize energy-based pause detector.
        
        Args:
            energy_threshold: Initial energy threshold for speech detection
            frame_duration_ms: Frame duration in milliseconds
            min_speech_frames: Minimum frames to confirm speech start
            min_pause_frames: Minimum frames to confirm speech end
            sample_rate: Audio sample rate in Hz
            adaptive_threshold: Enable adaptive threshold adjustment
            adaptation_rate: Rate of threshold adaptation (0.0-1.0)
        """
        super().__init__()
        
        self.energy_threshold = energy_threshold
        self.frame_duration_ms = frame_duration_ms
        self.sample_rate = sample_rate
        self.frame_size = int(sample_rate * frame_duration_ms / 1000)
        self.min_speech_frames = min_speech_frames
        self.min_pause_frames = min_pause_frames
        self.adaptive_threshold = adaptive_threshold
        self.adaptation_rate = adaptation_rate
        
        # Adaptive threshold state
        self.background_energy = energy_threshold
        self.energy_history = deque(maxlen=100)  # Keep last 100 energy values
        
        # Frame processing state
        self.speech_start_time = None
        self._audio_buffer = deque()
        
        logger.info(
            f"Energy pause detector initialized: threshold={energy_threshold}, "
            f"frame_duration={frame_duration_ms}ms, adaptive={adaptive_threshold}"
        )
    
    def _process_chunk_sync(self, audio: AudioData) -> Dict[str, Any]:
        """Process audio chunk using energy-based detection."""
        events = []
        
        try:
            # Decode audio data to numpy array
            audio_np = decode_pcm(audio.data, audio.sample_rate, audio.channels)
            if audio_np is None or len(audio_np) == 0:
                return self._create_state_result(events)
            
            # Add to buffer
            self._audio_buffer.extend(audio_np)
            
            # Process frames
            while len(self._audio_buffer) >= self.frame_size:
                # Extract frame
                frame = np.array(list(self._audio_buffer)[:self.frame_size])
                
                # Remove processed samples
                for _ in range(self.frame_size):
                    self._audio_buffer.popleft()
                
                # Process frame
                is_speech = self._process_frame(frame)
                
                if is_speech:
                    self._silence_frames = 0
                    self._speech_frames += 1
                    
                    # Check for speech start
                    if not self._is_speaking and self._speech_frames >= self.min_speech_frames:
                        self._is_speaking = True
                        self.speech_start_time = time.time()
                        events.append("speech_start")
                        logger.debug("Speech start detected (energy)")
                
                else:
                    self._speech_frames = 0
                    self._silence_frames += 1
                    
                    # Check for speech end
                    if self._is_speaking and self._silence_frames >= self.min_pause_frames:
                        self._is_speaking = False
                        events.append("speech_end")
                        logger.debug("Speech end detected (energy)")
            
            # Calculate silence duration
            silence_duration_ms = int(self._silence_frames * self.frame_duration_ms)
            
            return self._create_state_result(
                events=events,
                silence_duration_ms=silence_duration_ms,
                confidence=self._get_confidence()
            )
            
        except Exception as e:
            logger.error(f"Error in energy-based processing: {e}")
            return self._create_state_result(events)
    
    def _process_frame(self, frame: np.ndarray) -> bool:
        """Process a single frame using energy analysis."""
        try:
            # Calculate RMS energy
            rms_energy = np.sqrt(np.mean(frame ** 2))
            
            # Add to history
            self.energy_history.append(rms_energy)
            
            # Update adaptive threshold
            if self.adaptive_threshold:
                self._update_threshold(rms_energy)
            
            # Determine if speech
            is_speech = rms_energy > self.energy_threshold
            
            # Log occasionally for debugging
            if self._chunk_count % 100 == 0:
                logger.debug(
                    f"Energy analysis: RMS={rms_energy:.4f}, threshold={self.energy_threshold:.4f}, "
                    f"is_speech={is_speech}"
                )
            
            return is_speech
            
        except Exception as e:
            logger.error(f"Error processing frame: {e}")
            return False
    
    def _update_threshold(self, current_energy: float) -> None:
        """Update adaptive threshold based on background energy."""
        try:
            if len(self.energy_history) < 10:
                return  # Need some history first
            
            # Calculate background energy (lower percentile of recent history)
            recent_energies = list(self.energy_history)[-50:]  # Last 50 values
            background = np.percentile(recent_energies, 20)  # 20th percentile
            
            # Smooth update
            self.background_energy = (
                self.adaptation_rate * self.background_energy +
                (1 - self.adaptation_rate) * background
            )
            
            # Set threshold as multiple of background energy
            self.energy_threshold = max(
                self.background_energy * 3.0,  # At least 3x background
                0.005  # Minimum threshold
            )
            
        except Exception as e:
            logger.error(f"Error updating adaptive threshold: {e}")
    
    def _get_confidence(self) -> float:
        """Get confidence measure based on energy levels."""
        try:
            if len(self.energy_history) < 5:
                return 0.5  # Default confidence
            
            recent_energy = np.mean(list(self.energy_history)[-5:])
            confidence = min(recent_energy / (self.energy_threshold * 2), 1.0)
            return confidence
            
        except Exception as e:
            logger.error(f"Error calculating confidence: {e}")
            return 0.5
    
    def reset(self) -> None:
        """Reset energy detector state."""
        super().reset()
        self.speech_start_time = None
        self._audio_buffer.clear()
        self.energy_history.clear()
        # Don't reset background_energy to preserve adaptation
        logger.debug("Energy pause detector state reset")
    
    # Configuration methods
    
    def set_energy_threshold(self, threshold: float) -> None:
        """Set energy threshold manually."""
        if threshold <= 0:
            raise ValueError("Energy threshold must be positive")
        
        self.energy_threshold = threshold
        logger.info(f"Energy threshold set to: {threshold}")
    
    def enable_adaptive_threshold(self, enabled: bool) -> None:
        """Enable or disable adaptive threshold."""
        self.adaptive_threshold = enabled
        logger.info(f"Adaptive threshold {'enabled' if enabled else 'disabled'}")
    
    def get_config(self) -> Dict[str, Any]:
        """Get current detector configuration."""
        return {
            "type": "energy",
            "energy_threshold": self.energy_threshold,
            "background_energy": self.background_energy,
            "frame_duration_ms": self.frame_duration_ms,
            "min_speech_frames": self.min_speech_frames,
            "min_pause_frames": self.min_pause_frames,
            "sample_rate": self.sample_rate,
            "adaptive_threshold": self.adaptive_threshold,
            "energy_history_length": len(self.energy_history)
        }
    
    def get_stats(self) -> Dict[str, Any]:
        """Get detector statistics."""
        try:
            if not self.energy_history:
                return {"error": "No energy history available"}
            
            energies = list(self.energy_history)
            
            return {
                "current_energy": energies[-1] if energies else 0,
                "mean_energy": np.mean(energies),
                "std_energy": np.std(energies),
                "min_energy": np.min(energies),
                "max_energy": np.max(energies),
                "background_energy": self.background_energy,
                "threshold": self.energy_threshold,
                "snr_estimate": np.mean(energies) / (self.background_energy + 1e-8)
            }
            
        except Exception as e:
            logger.error(f"Error getting stats: {e}")
            return {"error": str(e)}


# Factory functions

def create_energy_detector(
    energy_threshold: float = 0.01,
    sample_rate: int = 16000,
    adaptive: bool = True
) -> EnergyPauseDetector:
    """
    Create an energy-based pause detector.
    
    Args:
        energy_threshold: Initial energy threshold
        sample_rate: Audio sample rate
        adaptive: Enable adaptive threshold
        
    Returns:
        Configured energy pause detector
    """
    return EnergyPauseDetector(
        energy_threshold=energy_threshold,
        sample_rate=sample_rate,
        adaptive_threshold=adaptive
    )


def create_energy_detector_from_config(config: Dict[str, Any]) -> EnergyPauseDetector:
    """
    Create energy pause detector from configuration.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Configured energy pause detector
    """
    return EnergyPauseDetector(
        energy_threshold=config.get("energy_threshold", 0.01),
        frame_duration_ms=config.get("frame_duration_ms", 30),
        min_speech_frames=config.get("min_speech_frames", 3),
        min_pause_frames=config.get("min_pause_frames", 10),
        sample_rate=config.get("sample_rate", 16000),
        adaptive_threshold=config.get("adaptive_threshold", True),
        adaptation_rate=config.get("adaptation_rate", 0.95)
    )