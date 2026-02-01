"""
Audio I/O utilities.

Pure functions for loading, saving, and processing audio.
Uses soundfile for I/O (consistent with existing codebase).
"""

from pathlib import Path
from typing import Tuple

import numpy as np
import soundfile as sf

from aria.exceptions import AudioLoadError


def load_audio(
    path: Path | str,
    target_sr: int | None = None,
) -> Tuple[np.ndarray, int]:
    """Load audio from file.

    Args:
        path: Path to audio file.
        target_sr: Target sample rate. If provided and different from file's
            sample rate, audio will be resampled.

    Returns:
        Tuple of (audio, sample_rate).
        Audio shape: (channels, samples) - typically (2, N) for stereo.

    Raises:
        AudioLoadError: If file cannot be loaded.
    """
    path = Path(path)

    if not path.exists():
        raise AudioLoadError(str(path), "File does not exist")

    try:
        # soundfile loads as (samples, channels)
        audio, sr = sf.read(path, always_2d=True)
        # Transpose to (channels, samples)
        audio = audio.T
    except Exception as e:
        raise AudioLoadError(str(path), str(e)) from e

    # Resample if needed
    if target_sr is not None and sr != target_sr:
        audio = resample(audio, sr, target_sr)
        sr = target_sr

    return audio, sr


def save_audio(
    path: Path | str,
    audio: np.ndarray,
    sample_rate: int,
    format: str = "flac",
    subtype: str | None = None,
) -> None:
    """Save audio to file.

    Args:
        path: Output path.
        audio: Audio array of shape (channels, samples).
        sample_rate: Sample rate.
        format: Audio format ("wav", "flac", "mp3").
        subtype: Subtype for format (e.g., "PCM_16", "PCM_24").
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Transpose to (samples, channels) for soundfile
    audio_to_save = audio.T

    # Default subtypes
    if subtype is None:
        subtype = "PCM_24" if format in ("wav", "flac") else None

    sf.write(path, audio_to_save, sample_rate, subtype=subtype)


def resample(audio: np.ndarray, from_sr: int, to_sr: int) -> np.ndarray:
    """Resample audio to target sample rate.

    Uses scipy for resampling (simple and dependency-light).

    Args:
        audio: Audio array of shape (channels, samples).
        from_sr: Source sample rate.
        to_sr: Target sample rate.

    Returns:
        Resampled audio with shape (channels, new_samples).
    """
    if from_sr == to_sr:
        return audio

    from scipy import signal

    # Calculate new length
    new_length = int(audio.shape[-1] * to_sr / from_sr)

    # Resample each channel
    resampled = signal.resample(audio, new_length, axis=-1)

    return resampled.astype(audio.dtype)


def pad_to_length(audio: np.ndarray, target_length: int) -> np.ndarray:
    """Pad audio to target length with zeros.

    Args:
        audio: Audio array of shape (channels, samples).
        target_length: Target length in samples.

    Returns:
        Padded audio array.
    """
    current_length = audio.shape[-1]

    if current_length >= target_length:
        return audio

    pad_width = ((0, 0), (0, target_length - current_length))
    return np.pad(audio, pad_width, mode="constant", constant_values=0)


def sum_stems(stems: list[np.ndarray]) -> np.ndarray:
    """Pad all stems to same length and sum them.

    This follows the MoisesDB mixing approach.

    Args:
        stems: List of audio arrays, each with shape (channels, samples).

    Returns:
        Mixed audio with shape (channels, samples).

    Raises:
        ValueError: If stems list is empty.
    """
    if not stems:
        raise ValueError("Cannot sum empty list of stems")

    # Find max length
    max_length = max(stem.shape[-1] for stem in stems)

    # Pad and sum
    padded = [pad_to_length(stem, max_length) for stem in stems]
    mixed = np.sum(padded, axis=0)

    return mixed


