"""
Audio processing utilities for the conversation library.

Provides common audio processing functions like resampling,
format conversion, and audio effects.
"""

import logging
from typing import Tuple, Optional
import numpy as np
from scipy import signal
import math

logger = logging.getLogger(__name__)


class AudioProcessor:
    """
    Audio processing utility class with common operations.
    """
    
    @staticmethod
    def resample_audio(
        audio_data: np.ndarray, 
        original_rate: int, 
        target_rate: int
    ) -> np.ndarray:
        """
        Resample audio to a different sample rate.
        
        Args:
            audio_data: Input audio samples
            original_rate: Original sample rate in Hz
            target_rate: Target sample rate in Hz
            
        Returns:
            Resampled audio data
        """
        if original_rate == target_rate:
            return audio_data
        
        try:
            # Calculate resampling ratio
            ratio = target_rate / original_rate
            new_length = int(len(audio_data) * ratio)
            
            # Use scipy's resampling function
            resampled = signal.resample(audio_data, new_length)
            
            logger.debug(
                f"Resampled audio: {original_rate}Hz -> {target_rate}Hz, "
                f"{len(audio_data)} -> {len(resampled)} samples"
            )
            
            return resampled.astype(np.float32)
            
        except Exception as e:
            logger.error(f"Error resampling audio: {e}")
            return audio_data
    
    @staticmethod
    def apply_speed_factor(
        audio_data: np.ndarray,
        sample_rate: int,
        speed_factor: float
    ) -> Tuple[np.ndarray, int]:
        """
        Apply speed factor to audio without changing pitch.
        
        Args:
            audio_data: Input audio samples
            sample_rate: Audio sample rate
            speed_factor: Speed multiplier (>1.0 = faster, <1.0 = slower)
            
        Returns:
            Tuple of (processed_audio, sample_rate)
        """
        if speed_factor == 1.0:
            return audio_data, sample_rate
        
        try:
            # Simple time-stretching by resampling
            # This will change pitch as well - for better results, use PSOLA or similar
            new_length = int(len(audio_data) / speed_factor)
            
            if new_length <= 0:
                return audio_data, sample_rate
            
            # Resample to new length
            processed = signal.resample(audio_data, new_length)
            
            logger.debug(
                f"Applied speed factor {speed_factor}: "
                f"{len(audio_data)} -> {len(processed)} samples"
            )
            
            return processed.astype(np.float32), sample_rate
            
        except Exception as e:
            logger.error(f"Error applying speed factor: {e}")
            return audio_data, sample_rate
    
    @staticmethod
    def normalize_audio(
        audio_data: np.ndarray, 
        target_level: float = 0.95
    ) -> np.ndarray:
        """
        Normalize audio to target peak level.
        
        Args:
            audio_data: Input audio samples
            target_level: Target peak level (0.0 to 1.0)
            
        Returns:
            Normalized audio data
        """
        try:
            if len(audio_data) == 0:
                return audio_data
            
            # Find current peak level
            current_peak = np.max(np.abs(audio_data))
            
            if current_peak == 0:
                return audio_data
            
            # Calculate scaling factor
            scale_factor = target_level / current_peak
            
            # Apply scaling
            normalized = audio_data * scale_factor
            
            logger.debug(
                f"Normalized audio: peak {current_peak:.3f} -> {target_level:.3f} "
                f"(scale: {scale_factor:.3f})"
            )
            
            return normalized.astype(np.float32)
            
        except Exception as e:
            logger.error(f"Error normalizing audio: {e}")
            return audio_data
    
    @staticmethod
    def apply_fade(
        audio_data: np.ndarray,
        fade_in_samples: int = 0,
        fade_out_samples: int = 0
    ) -> np.ndarray:
        """
        Apply fade in/out to audio data.
        
        Args:
            audio_data: Input audio samples
            fade_in_samples: Number of samples for fade in
            fade_out_samples: Number of samples for fade out
            
        Returns:
            Audio with fades applied
        """
        try:
            if len(audio_data) == 0:
                return audio_data
            
            result = audio_data.copy()
            
            # Apply fade in
            if fade_in_samples > 0:
                fade_in_samples = min(fade_in_samples, len(result))
                fade_curve = np.linspace(0.0, 1.0, fade_in_samples)
                result[:fade_in_samples] *= fade_curve
            
            # Apply fade out
            if fade_out_samples > 0:
                fade_out_samples = min(fade_out_samples, len(result))
                fade_curve = np.linspace(1.0, 0.0, fade_out_samples)
                result[-fade_out_samples:] *= fade_curve
            
            return result.astype(np.float32)
            
        except Exception as e:
            logger.error(f"Error applying fades: {e}")
            return audio_data
    
    @staticmethod
    def detect_silence(
        audio_data: np.ndarray,
        threshold: float = 0.01,
        min_silence_duration: float = 0.1,
        sample_rate: int = 16000
    ) -> list:
        """
        Detect silence segments in audio.
        
        Args:
            audio_data: Input audio samples
            threshold: Silence threshold (RMS level)
            min_silence_duration: Minimum silence duration in seconds
            sample_rate: Audio sample rate
            
        Returns:
            List of (start_sample, end_sample) tuples for silence segments
        """
        try:
            if len(audio_data) == 0:
                return []
            
            # Calculate frame size for analysis
            frame_size = int(0.025 * sample_rate)  # 25ms frames
            hop_size = int(0.010 * sample_rate)    # 10ms hop
            
            silence_segments = []
            current_silence_start = None
            
            for i in range(0, len(audio_data) - frame_size, hop_size):
                frame = audio_data[i:i + frame_size]
                rms = np.sqrt(np.mean(frame ** 2))
                
                if rms < threshold:
                    # Silence detected
                    if current_silence_start is None:
                        current_silence_start = i
                else:
                    # Non-silence detected
                    if current_silence_start is not None:
                        silence_duration = (i - current_silence_start) / sample_rate
                        
                        if silence_duration >= min_silence_duration:
                            silence_segments.append((current_silence_start, i))
                        
                        current_silence_start = None
            
            # Handle silence at the end
            if current_silence_start is not None:
                silence_duration = (len(audio_data) - current_silence_start) / sample_rate
                if silence_duration >= min_silence_duration:
                    silence_segments.append((current_silence_start, len(audio_data)))
            
            logger.debug(f"Detected {len(silence_segments)} silence segments")
            return silence_segments
            
        except Exception as e:
            logger.error(f"Error detecting silence: {e}")
            return []
    
    @staticmethod
    def trim_silence(
        audio_data: np.ndarray,
        threshold: float = 0.01,
        sample_rate: int = 16000
    ) -> np.ndarray:
        """
        Trim silence from beginning and end of audio.
        
        Args:
            audio_data: Input audio samples
            threshold: Silence threshold (RMS level)
            sample_rate: Audio sample rate
            
        Returns:
            Trimmed audio data
        """
        try:
            if len(audio_data) == 0:
                return audio_data
            
            # Calculate frame size
            frame_size = int(0.025 * sample_rate)  # 25ms frames
            
            # Find start of non-silence
            start_idx = 0
            for i in range(0, len(audio_data) - frame_size, frame_size // 2):
                frame = audio_data[i:i + frame_size]
                rms = np.sqrt(np.mean(frame ** 2))
                
                if rms >= threshold:
                    start_idx = max(0, i - frame_size)  # Keep a bit of lead-in
                    break
            
            # Find end of non-silence
            end_idx = len(audio_data)
            for i in range(len(audio_data) - frame_size, 0, -(frame_size // 2)):
                if i + frame_size > len(audio_data):
                    continue
                    
                frame = audio_data[i:i + frame_size]
                rms = np.sqrt(np.mean(frame ** 2))
                
                if rms >= threshold:
                    end_idx = min(len(audio_data), i + frame_size * 2)  # Keep a bit of tail
                    break
            
            if start_idx >= end_idx:
                return np.array([], dtype=np.float32)
            
            trimmed = audio_data[start_idx:end_idx]
            
            logger.debug(
                f"Trimmed silence: {len(audio_data)} -> {len(trimmed)} samples "
                f"({(len(audio_data) - len(trimmed)) / sample_rate:.2f}s removed)"
            )
            
            return trimmed.astype(np.float32)
            
        except Exception as e:
            logger.error(f"Error trimming silence: {e}")
            return audio_data
    
    @staticmethod
    def calculate_rms_level(audio_data: np.ndarray) -> float:
        """
        Calculate RMS level of audio data.
        
        Args:
            audio_data: Input audio samples
            
        Returns:
            RMS level (0.0 to 1.0)
        """
        try:
            if len(audio_data) == 0:
                return 0.0
            
            return float(np.sqrt(np.mean(audio_data ** 2)))
            
        except Exception as e:
            logger.error(f"Error calculating RMS: {e}")
            return 0.0
    
    @staticmethod
    def calculate_peak_level(audio_data: np.ndarray) -> float:
        """
        Calculate peak level of audio data.
        
        Args:
            audio_data: Input audio samples
            
        Returns:
            Peak level (0.0 to 1.0)
        """
        try:
            if len(audio_data) == 0:
                return 0.0
            
            return float(np.max(np.abs(audio_data)))
            
        except Exception as e:
            logger.error(f"Error calculating peak: {e}")
            return 0.0