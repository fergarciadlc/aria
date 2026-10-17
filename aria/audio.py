"""
Audio I/O utilities.

Pure functions for loading, saving, and processing audio.
Uses soundfile for I/O; MP3 encoding goes through the ffmpeg binary.
"""

import shutil
import subprocess
import tempfile
from math import gcd
from pathlib import Path

import numpy as np
import soundfile as sf

from aria.constants import MP3_BITRATE_KBPS
from aria.exceptions import AudioLoadError


def load_audio(
    path: Path | str,
    target_sr: int | None = None,
    stereo: bool = False,
) -> tuple[np.ndarray, int]:
    """Load audio from file.

    Args:
        path: Path to audio file.
        target_sr: Target sample rate. If provided and different from file's
            sample rate, audio will be resampled.
        stereo: If True, return exactly two channels (see to_stereo).

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
        audio, sr = sf.read(path, always_2d=True, dtype="float32")
        # Transpose to (channels, samples)
        audio = audio.T
    except Exception as e:
        raise AudioLoadError(str(path), str(e)) from e

    # Resample if needed
    if target_sr is not None and sr != target_sr:
        audio = resample(audio, sr, target_sr)
        sr = target_sr

    if stereo:
        audio = to_stereo(audio)

    return audio, sr


def save_audio(
    path: Path | str,
    audio: np.ndarray,
    sample_rate: int,
    format: str | None = None,
    subtype: str | None = None,
    mp3_bitrate: int = MP3_BITRATE_KBPS,
) -> None:
    """Save audio to file.

    Args:
        path: Output path. Its extension must match ``format``.
        audio: Audio array of shape (channels, samples).
        sample_rate: Sample rate.
        format: Audio format ("wav", "flac", "mp3"). Inferred from the path
            extension when omitted.
        subtype: Subtype for wav/flac (e.g., "PCM_16", "PCM_24").
        mp3_bitrate: Constant bitrate in kbps for MP3 output.

    Raises:
        ValueError: If format and path extension disagree.
    """
    path = Path(path)
    suffix = path.suffix.lstrip(".").lower()
    format = (format or suffix).lower()
    if format != suffix:
        raise ValueError(f"Format {format!r} does not match file extension {path.suffix!r}")

    path.parent.mkdir(parents=True, exist_ok=True)

    if format == "mp3":
        _save_mp3(path, audio, sample_rate, mp3_bitrate)
        return

    # Default subtypes
    if subtype is None:
        subtype = "PCM_24" if format in ("wav", "flac") else None

    # Transpose to (samples, channels) for soundfile
    sf.write(path, audio.T, sample_rate, subtype=subtype)


def ffmpeg_available() -> bool:
    """Whether the ffmpeg binary (needed for MP3 output) is on PATH."""
    return shutil.which("ffmpeg") is not None


def _save_mp3(path: Path, audio: np.ndarray, sample_rate: int, bitrate_kbps: int) -> None:
    """Encode audio to CBR MP3 via ffmpeg, using a float WAV intermediary."""
    if not ffmpeg_available():
        raise RuntimeError("ffmpeg is required to write MP3 files but was not found in PATH")

    bitrate_kbps = min(int(bitrate_kbps), 320)

    with tempfile.NamedTemporaryFile(suffix=".wav") as tmp_file:
        sf.write(tmp_file.name, audio.T, sample_rate, subtype="FLOAT")
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-hide_banner",
                "-loglevel",
                "error",
                "-i",
                tmp_file.name,
                "-c:a",
                "mp3",
                "-b:a",
                f"{bitrate_kbps}k",
                str(path),
            ],
            check=True,
        )


def resample(audio: np.ndarray, from_sr: int, to_sr: int) -> np.ndarray:
    """Resample audio to target sample rate.

    Uses polyphase filtering (scipy.signal.resample_poly).

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

    divisor = gcd(from_sr, to_sr)
    resampled = signal.resample_poly(audio, to_sr // divisor, from_sr // divisor, axis=-1)

    return resampled.astype(audio.dtype)


def to_stereo(audio: np.ndarray) -> np.ndarray:
    """Return exactly two channels: mono is duplicated, extra channels dropped.

    Args:
        audio: Audio array of shape (channels, samples).

    Returns:
        Audio array of shape (2, samples).
    """
    if audio.shape[0] == 1:
        return np.concatenate([audio, audio], axis=0)
    return audio[:2]


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
