"""
Configuration for the ARIA library.

Uses a simple dataclass - no complex config management.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class Config:
    """Configuration for ARIA dataset operations.
    
    Attributes:
        sample_rate: Target sample rate for audio (default: 48000).
        default_codec: Default AI codec to use (default: "encodec_3").
        audio_format: Default audio format for saving (default: "flac").
    """

    sample_rate: int = 48000
    default_codec: str = "encodec_3"
    audio_format: str = "flac"

    def __post_init__(self):
        # Validate sample rate
        if self.sample_rate not in (44100, 48000):
            raise ValueError(f"Unsupported sample rate: {self.sample_rate}")
        
        # Validate audio format
        valid_formats = ("wav", "flac", "mp3")
        if self.audio_format not in valid_formats:
            raise ValueError(
                f"Unsupported format: {self.audio_format}. "
                f"Choose from: {valid_formats}"
            )


# Default configuration
DEFAULT_CONFIG = Config()


