"""
Audio buffer implementations for the conversation library.

Provides thread-safe audio buffering with various strategies
for real-time audio stream processing.
"""

import threading
import time
import logging
from typing import Optional, List
from collections import deque
import numpy as np

logger = logging.getLogger(__name__)


class OptimizedAudioBuffer:
    """
    Thread-safe circular audio buffer optimized for real-time processing.
    
    Uses deque for efficient append/pop operations and maintains
    configurable duration limits.
    """
    
    def __init__(
        self, 
        max_duration_seconds: float = 10.0, 
        sample_rate: int = 16000,
        channels: int = 1
    ):
        """
        Initialize audio buffer.
        
        Args:
            max_duration_seconds: Maximum buffer duration in seconds
            sample_rate: Audio sample rate in Hz
            channels: Number of audio channels
        """
        self.max_duration_seconds = max_duration_seconds
        self.sample_rate = sample_rate
        self.channels = channels
        
        # Thread-safe storage using deque
        self._buffer: deque = deque()
        self._lock = threading.RLock()
        self._total_samples = 0
        
        logger.debug(
            f"OptimizedAudioBuffer initialized: {max_duration_seconds}s max, "
            f"{sample_rate}Hz, {channels}ch"
        )
    
    def add_audio(self, audio_data: np.ndarray) -> None:
        """
        Add audio data to the buffer.
        
        Args:
            audio_data: Audio samples as numpy array
        """
        if audio_data is None or len(audio_data) == 0:
            return
        
        with self._lock:
            self._buffer.append({
                'data': audio_data.copy(),
                'timestamp': time.time(),
                'samples': len(audio_data)
            })
            
            self._total_samples += len(audio_data)
            
            # Maintain maximum duration
            self._trim_to_duration()
    
    def get_recent_audio(self, duration_seconds: float = 5.0) -> np.ndarray:
        """
        Get recent audio data from the buffer.
        
        Args:
            duration_seconds: Duration of audio to retrieve in seconds
            
        Returns:
            Concatenated audio data as numpy array
        """
        with self._lock:
            if not self._buffer:
                return np.array([])
            
            target_samples = int(duration_seconds * self.sample_rate)
            collected_samples = 0
            audio_chunks = []
            
            # Collect from newest to oldest
            for chunk in reversed(self._buffer):
                audio_chunks.append(chunk['data'])
                collected_samples += chunk['samples']
                
                if collected_samples >= target_samples:
                    break
            
            if not audio_chunks:
                return np.array([])
            
            # Reverse to get chronological order and concatenate
            audio_chunks.reverse()
            combined_audio = np.concatenate(audio_chunks)
            
            # Trim to exact duration if we collected too much
            if len(combined_audio) > target_samples:
                combined_audio = combined_audio[-target_samples:]
            
            logger.debug(
                f"Retrieved {len(combined_audio)} samples "
                f"({len(combined_audio)/self.sample_rate:.2f}s) from buffer"
            )
            
            return combined_audio
    
    def get_all_audio(self) -> np.ndarray:
        """
        Get all audio data from the buffer.
        
        Returns:
            All buffered audio data as numpy array
        """
        with self._lock:
            if not self._buffer:
                return np.array([])
            
            audio_chunks = [chunk['data'] for chunk in self._buffer]
            return np.concatenate(audio_chunks) if audio_chunks else np.array([])
    
    def clear(self) -> None:
        """Clear all audio data from the buffer."""
        with self._lock:
            self._buffer.clear()
            self._total_samples = 0
            logger.debug("Audio buffer cleared")
    
    def _trim_to_duration(self) -> None:
        """Remove old audio data to maintain maximum duration."""
        max_samples = int(self.max_duration_seconds * self.sample_rate)
        
        while self._total_samples > max_samples and len(self._buffer) > 1:
            removed_chunk = self._buffer.popleft()
            self._total_samples -= removed_chunk['samples']
    
    @property
    def current_duration(self) -> float:
        """Get current buffer duration in seconds."""
        with self._lock:
            return self._total_samples / self.sample_rate
    
    @property
    def chunk_count(self) -> int:
        """Get number of audio chunks in buffer."""
        with self._lock:
            return len(self._buffer)
    
    @property
    def is_empty(self) -> bool:
        """Check if buffer is empty."""
        with self._lock:
            return len(self._buffer) == 0
    
    def get_buffer_stats(self) -> dict:
        """Get detailed buffer statistics."""
        with self._lock:
            if not self._buffer:
                return {
                    'chunk_count': 0,
                    'total_samples': 0,
                    'duration_seconds': 0.0,
                    'oldest_timestamp': None,
                    'newest_timestamp': None
                }
            
            timestamps = [chunk['timestamp'] for chunk in self._buffer]
            
            return {
                'chunk_count': len(self._buffer),
                'total_samples': self._total_samples,
                'duration_seconds': self.current_duration,
                'oldest_timestamp': min(timestamps),
                'newest_timestamp': max(timestamps),
                'buffer_span_seconds': max(timestamps) - min(timestamps)
            }


class RingAudioBuffer:
    """
    Fixed-size circular audio buffer for constant memory usage.
    
    Overwrites oldest data when buffer is full.
    """
    
    def __init__(
        self, 
        max_samples: int,
        sample_rate: int = 16000,
        channels: int = 1
    ):
        """
        Initialize ring buffer.
        
        Args:
            max_samples: Maximum number of audio samples to store
            sample_rate: Audio sample rate in Hz
            channels: Number of audio channels
        """
        self.max_samples = max_samples
        self.sample_rate = sample_rate
        self.channels = channels
        
        # Pre-allocate buffer
        self._buffer = np.zeros(max_samples, dtype=np.float32)
        self._write_index = 0
        self._samples_written = 0
        self._lock = threading.RLock()
        
        logger.debug(
            f"RingAudioBuffer initialized: {max_samples} samples "
            f"({max_samples/sample_rate:.2f}s)"
        )
    
    def add_audio(self, audio_data: np.ndarray) -> None:
        """
        Add audio data to the ring buffer.
        
        Args:
            audio_data: Audio samples as numpy array
        """
        if audio_data is None or len(audio_data) == 0:
            return
        
        with self._lock:
            samples_to_write = len(audio_data)
            
            # Handle case where new data is larger than buffer
            if samples_to_write > self.max_samples:
                # Take only the most recent samples
                audio_data = audio_data[-self.max_samples:]
                samples_to_write = self.max_samples
                self._write_index = 0
                self._samples_written = self.max_samples
                self._buffer[:] = audio_data
                return
            
            # Write data, wrapping around if necessary
            end_index = self._write_index + samples_to_write
            
            if end_index <= self.max_samples:
                # Simple write - no wrap around
                self._buffer[self._write_index:end_index] = audio_data
            else:
                # Wrap around
                first_part_size = self.max_samples - self._write_index
                second_part_size = samples_to_write - first_part_size
                
                self._buffer[self._write_index:] = audio_data[:first_part_size]
                self._buffer[:second_part_size] = audio_data[first_part_size:]
            
            self._write_index = end_index % self.max_samples
            self._samples_written = min(self._samples_written + samples_to_write, self.max_samples)
    
    def get_recent_audio(self, duration_seconds: float = 5.0) -> np.ndarray:
        """
        Get recent audio data from the ring buffer.
        
        Args:
            duration_seconds: Duration of audio to retrieve in seconds
            
        Returns:
            Recent audio data as numpy array
        """
        with self._lock:
            if self._samples_written == 0:
                return np.array([])
            
            target_samples = min(
                int(duration_seconds * self.sample_rate),
                self._samples_written
            )
            
            if self._samples_written < self.max_samples:
                # Buffer not yet full - simple slice
                start_index = max(0, self._samples_written - target_samples)
                return self._buffer[start_index:self._samples_written].copy()
            else:
                # Buffer is full - may need to wrap around
                start_index = (self._write_index - target_samples) % self.max_samples
                
                if start_index + target_samples <= self.max_samples:
                    # No wrap around
                    return self._buffer[start_index:start_index + target_samples].copy()
                else:
                    # Wrap around
                    first_part = self._buffer[start_index:].copy()
                    second_part = self._buffer[:target_samples - len(first_part)].copy()
                    return np.concatenate([first_part, second_part])
    
    def clear(self) -> None:
        """Clear the ring buffer."""
        with self._lock:
            self._write_index = 0
            self._samples_written = 0
            self._buffer.fill(0.0)
            logger.debug("Ring buffer cleared")
    
    @property
    def current_samples(self) -> int:
        """Get current number of samples in buffer."""
        with self._lock:
            return self._samples_written
    
    @property
    def current_duration(self) -> float:
        """Get current buffer duration in seconds."""
        with self._lock:
            return self._samples_written / self.sample_rate
    
    @property
    def is_full(self) -> bool:
        """Check if buffer is full."""
        with self._lock:
            return self._samples_written >= self.max_samples