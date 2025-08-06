"""
Audio codec utilities for the conversation library.

Provides encoding/decoding utilities for various audio formats
including PCM, WAV, MP3, and Opus.
"""

import asyncio
import logging
import io
import wave
import struct
from typing import Optional, Union, Dict, Any
import numpy as np

logger = logging.getLogger(__name__)


class PCMAudioDecoder:
    """
    Utility class for decoding PCM audio data to numpy arrays.
    
    Handles common PCM formats used in real-time audio streaming.
    """
    
    def __init__(self, sample_rate: int = 16000, channels: int = 1, sample_width: int = 2):
        """
        Initialize PCM decoder.
        
        Args:
            sample_rate: Audio sample rate in Hz
            channels: Number of audio channels
            sample_width: Sample width in bytes (2 for 16-bit)
        """
        self.sample_rate = sample_rate
        self.channels = channels
        self.sample_width = sample_width
        
        # Determine numpy dtype based on sample width
        if sample_width == 1:
            self.dtype = np.uint8
        elif sample_width == 2:
            self.dtype = np.int16
        elif sample_width == 4:
            self.dtype = np.int32
        else:
            raise ValueError(f"Unsupported sample width: {sample_width}")
        
        logger.debug(
            f"PCMAudioDecoder initialized: {sample_rate}Hz, "
            f"{channels}ch, {sample_width*8}-bit"
        )
    
    async def decode_pcm_to_numpy(self, pcm_data: bytes) -> Optional[np.ndarray]:
        """
        Decode PCM bytes to numpy array asynchronously.
        
        Args:
            pcm_data: Raw PCM audio bytes
            
        Returns:
            Numpy array containing decoded audio, or None on error
        """
        if not pcm_data or len(pcm_data) == 0:
            return None
        
        try:
            # Run decoding in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(
                None, self._decode_pcm_sync, pcm_data
            )
        except Exception as e:
            logger.error(f"Error decoding PCM audio: {e}")
            return None
    
    def _decode_pcm_sync(self, pcm_data: bytes) -> np.ndarray:
        """Synchronous PCM decoding implementation."""
        # Convert bytes to numpy array
        audio_array = np.frombuffer(pcm_data, dtype=self.dtype)
        
        # Handle multi-channel audio
        if self.channels > 1:
            # Reshape to (samples, channels) and take first channel
            try:
                audio_array = audio_array.reshape(-1, self.channels)
                audio_array = audio_array[:, 0]  # Take first channel
            except ValueError:
                # If reshape fails, truncate to make it divisible by channels
                truncated_length = (len(audio_array) // self.channels) * self.channels
                audio_array = audio_array[:truncated_length].reshape(-1, self.channels)
                audio_array = audio_array[:, 0]
        
        # Normalize to float32 range [-1.0, 1.0]
        if self.dtype == np.int16:
            audio_array = audio_array.astype(np.float32) / 32768.0
        elif self.dtype == np.int32:
            audio_array = audio_array.astype(np.float32) / 2147483648.0
        elif self.dtype == np.uint8:
            audio_array = (audio_array.astype(np.float32) - 128.0) / 128.0
        
        return audio_array


def decode_pcm(pcm_data: bytes, sample_rate: int = 16000, channels: int = 1) -> Optional[np.ndarray]:
    """
    Synchronous PCM decoding function.
    
    Args:
        pcm_data: Raw PCM audio bytes
        sample_rate: Audio sample rate in Hz
        channels: Number of audio channels
        
    Returns:
        Numpy array containing decoded audio, or None on error
    """
    try:
        # Assume 16-bit PCM
        audio_array = np.frombuffer(pcm_data, dtype=np.int16)
        
        # Handle multi-channel audio
        if channels > 1:
            try:
                audio_array = audio_array.reshape(-1, channels)
                audio_array = audio_array[:, 0]  # Take first channel
            except ValueError:
                # If reshape fails, truncate to make it divisible by channels
                truncated_length = (len(audio_array) // channels) * channels
                audio_array = audio_array[:truncated_length].reshape(-1, channels)
                audio_array = audio_array[:, 0]
        
        # Normalize to float32 range [-1.0, 1.0]
        audio_array = audio_array.astype(np.float32) / 32768.0
        
        return audio_array
    except Exception as e:
        logger.error(f"Error decoding PCM audio: {e}")
        return None


def encode_wav(
    audio_data: Union[np.ndarray, bytes], 
    sample_rate: int = 16000, 
    channels: int = 1,
    sample_width: int = 2
) -> bytes:
    """
    Encode audio data as WAV format.
    
    Args:
        audio_data: Audio data as numpy array or bytes
        sample_rate: Audio sample rate in Hz
        channels: Number of audio channels
        sample_width: Sample width in bytes
        
    Returns:
        WAV-encoded audio bytes
    """
    try:
        # Convert numpy array to bytes if needed
        if isinstance(audio_data, np.ndarray):
            # Convert float to int16
            if audio_data.dtype != np.int16:
                # Assume float in range [-1.0, 1.0]
                audio_data = (audio_data * 32767).astype(np.int16)
            audio_bytes = audio_data.tobytes()
        else:
            audio_bytes = audio_data
        
        # Create WAV file in memory
        wav_buffer = io.BytesIO()
        
        with wave.open(wav_buffer, 'wb') as wav_file:
            wav_file.setnchannels(channels)
            wav_file.setsampwidth(sample_width)
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(audio_bytes)
        
        return wav_buffer.getvalue()
    
    except Exception as e:
        logger.error(f"Error encoding WAV audio: {e}")
        return b""


def encode_opus(
    audio_data: np.ndarray,
    sample_rate: int = 16000,
    bitrate: int = 64000
) -> Optional[bytes]:
    """
    Encode audio data as Opus format.
    
    Args:
        audio_data: Audio data as numpy array
        sample_rate: Audio sample rate in Hz
        bitrate: Opus bitrate in bits per second
        
    Returns:
        Opus-encoded audio bytes, or None if encoding fails
    """
    try:
        # This would require opus encoder library (like opuslib)
        # For now, return None to indicate Opus encoding is not available
        logger.warning("Opus encoding not implemented - opuslib required")
        return None
    
    except Exception as e:
        logger.error(f"Error encoding Opus audio: {e}")
        return None


def encode_mp3(
    audio_data: np.ndarray,
    sample_rate: int = 16000,
    bitrate: str = "64k"
) -> Optional[bytes]:
    """
    Encode audio data as MP3 format.
    
    Args:
        audio_data: Audio data as numpy array
        sample_rate: Audio sample rate in Hz
        bitrate: MP3 bitrate (e.g., "64k", "128k")
        
    Returns:
        MP3-encoded audio bytes, or None if encoding fails
    """
    try:
        # This would require ffmpeg or lame encoder
        # For now, return None to indicate MP3 encoding is not available
        logger.warning("MP3 encoding not implemented - ffmpeg/lame required")
        return None
    
    except Exception as e:
        logger.error(f"Error encoding MP3 audio: {e}")
        return None


def get_audio_format_info(audio_data: bytes) -> Dict[str, Any]:
    """
    Analyze audio data to determine format information.
    
    Args:
        audio_data: Raw audio bytes
        
    Returns:
        Dictionary containing format information
    """
    info = {
        "size_bytes": len(audio_data),
        "format": "unknown",
        "sample_rate": None,
        "channels": None,
        "duration": None
    }
    
    try:
        # Check for WAV format
        if audio_data.startswith(b'RIFF') and b'WAVE' in audio_data[:12]:
            info["format"] = "wav"
            # Parse WAV header for more info
            try:
                with wave.open(io.BytesIO(audio_data), 'rb') as wav_file:
                    info["sample_rate"] = wav_file.getframerate()
                    info["channels"] = wav_file.getnchannels()
                    info["duration"] = wav_file.getnframes() / wav_file.getframerate()
            except:
                pass
        
        # Check for other formats (simplified detection)
        elif audio_data.startswith(b'ID3') or audio_data.startswith(b'\xff\xfb'):
            info["format"] = "mp3"
        elif audio_data.startswith(b'OggS'):
            info["format"] = "ogg"
        else:
            # Assume raw PCM
            info["format"] = "pcm"
            
    except Exception as e:
        logger.error(f"Error analyzing audio format: {e}")
    
    return info