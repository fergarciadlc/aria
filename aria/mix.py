"""
Mix specification and computation.

A mix is a combination of stems where some are real and some are AI-generated.
MixSpec is a pure data container; compute_mix is a function.
"""

from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np

from aria.audio import load_audio, sum_stems
from aria.constants import AUDIO_EXTENSIONS, MIX_FORMAT
from aria.track import Track, get_stem_path


@dataclass(frozen=True)
class MixSpec:
    """Specification for a mix.

    This is a pure data container describing which stems should be AI-generated.
    Immutable (frozen) for use as dict keys, etc.

    Attributes:
        ai_indices: Tuple of stem indices that are AI-generated (0-indexed
            into Track.stem_names).
        codec: Codec used for AI stems (e.g., "encodec_3").
    """

    ai_indices: tuple[int, ...]
    codec: str

    @property
    def num_ai_stems(self) -> int:
        """Number of AI-generated stems in this mix."""
        return len(self.ai_indices)

    @property
    def u_folder(self) -> str:
        """Folder name for this mix (e.g., "U2" for 2 AI stems)."""
        return f"U{self.num_ai_stems}"

    @property
    def filename(self) -> str:
        """Filename stem for this mix, without extension (e.g., "mix_S-0-2")."""
        if not self.ai_indices:
            return "mix_S-empty"
        indices_str = "-".join(str(i) for i in sorted(self.ai_indices))
        return f"mix_S-{indices_str}"

    def is_all_real(self) -> bool:
        """Check if this mix has no AI stems."""
        return len(self.ai_indices) == 0

    def is_all_ai(self, num_stems: int) -> bool:
        """Check if all stems are AI-generated."""
        return len(self.ai_indices) == num_stems

    @classmethod
    def from_filename(cls, filename: str, codec: str) -> "MixSpec":
        """Parse a mix filename (with or without extension).

        Args:
            filename: e.g. "mix_S-0-2.mp3" or "mix_S-empty".
            codec: Codec of the mix.

        Returns:
            Matching MixSpec.

        Raises:
            ValueError: If the name doesn't follow the mix naming convention.
        """
        name = Path(filename).stem
        prefix = "mix_S-"
        if not name.startswith(prefix):
            raise ValueError(f"Not a mix filename: {filename}")
        indices_part = name[len(prefix):]
        if indices_part == "empty":
            return cls(ai_indices=(), codec=codec)
        return cls(ai_indices=tuple(int(i) for i in indices_part.split("-")), codec=codec)

    def __repr__(self) -> str:
        if not self.ai_indices:
            return f"MixSpec(all_real, codec={self.codec!r})"
        return f"MixSpec(ai={self.ai_indices}, codec={self.codec!r})"


def generate_all_mix_specs(num_stems: int, codec: str) -> list[MixSpec]:
    """Generate all possible mix specifications for a track.

    For n stems, generates 2^n combinations.

    Args:
        num_stems: Number of stems in the track.
        codec: Codec to use for AI stems.

    Returns:
        List of MixSpec objects, one for each combination.
    """
    all_specs = []

    for k in range(num_stems + 1):
        for combo in combinations(range(num_stems), k):
            spec = MixSpec(ai_indices=tuple(combo), codec=codec)
            all_specs.append(spec)

    return all_specs


def compute_mix(
    track: Track,
    mix_spec: MixSpec,
    target_sr: int | None = None,
) -> np.ndarray:
    """Compute a mix by loading and summing stems.

    Loads each stem (real or AI depending on mix_spec) as stereo, resamples
    every stem to a common rate and sums them.

    Args:
        track: Track to mix.
        mix_spec: Specification of which stems are AI.
        target_sr: Output sample rate. Defaults to the first stem's rate.

    Returns:
        Mixed audio array of shape (2, samples).
    """
    stems = []

    for i, stem_name in enumerate(track.stem_names):
        # Determine source for this stem
        if i in mix_spec.ai_indices:
            source = "ai_generated"
            codec = mix_spec.codec
        else:
            source = "real"
            codec = None

        # Get path and load audio
        stem_path = get_stem_path(track, stem_name, source, codec)
        audio, sr = load_audio(stem_path, target_sr=target_sr, stereo=True)
        if target_sr is None:
            target_sr = sr
        stems.append(audio)

    # Sum all stems
    return sum_stems(stems)


def mix_metadata(track: Track, mix_spec: MixSpec) -> dict[str, Any]:
    """Build the sidecar JSON written next to each pre-computed mix.

    Args:
        track: Track the mix belongs to.
        mix_spec: Mix specification.

    Returns:
        JSON-serialisable dict.
    """
    return {
        "mix_name": mix_spec.filename,
        "num_stems": track.num_stems,
        "num_ai_stems": mix_spec.num_ai_stems,
        "ai_stem_indices": sorted(mix_spec.ai_indices),
        "ai_codec": mix_spec.codec,
        "stem_names": track.stem_names,
    }


def get_mix_dir(track: Track, mix_spec: MixSpec) -> Path:
    """Get the directory where a pre-computed mix is stored."""
    return track.path / "mixes" / mix_spec.codec / mix_spec.u_folder


def get_mix_path(track: Track, mix_spec: MixSpec, format: str = MIX_FORMAT) -> Path:
    """Get the path where a pre-computed mix would be stored.

    Args:
        track: Track instance.
        mix_spec: Mix specification.
        format: Audio format / file extension without dot.

    Returns:
        Path to the mix file (may not exist).
    """
    return get_mix_dir(track, mix_spec) / f"{mix_spec.filename}.{format}"


def find_mix_path(track: Track, mix_spec: MixSpec) -> Path | None:
    """Find an existing pre-computed mix in any supported format.

    Args:
        track: Track instance.
        mix_spec: Mix specification.

    Returns:
        Path to the mix file, or None if no pre-computed mix exists.
    """
    mix_dir = get_mix_dir(track, mix_spec)
    for extension in AUDIO_EXTENSIONS:
        mix_path = mix_dir / f"{mix_spec.filename}{extension}"
        if mix_path.exists():
            return mix_path
    return None


def load_precomputed_mix(
    track: Track,
    mix_spec: MixSpec,
    target_sr: int | None = None,
) -> np.ndarray | None:
    """Load a pre-computed mix if it exists.

    Args:
        track: Track instance.
        mix_spec: Mix specification.
        target_sr: Optional target sample rate.

    Returns:
        Audio array if mix exists, None otherwise.
    """
    mix_path = find_mix_path(track, mix_spec)

    if mix_path is None:
        return None

    audio, _ = load_audio(mix_path, target_sr=target_sr)
    return audio


def get_or_compute_mix(
    track: Track,
    mix_spec: MixSpec,
    target_sr: int | None = None,
    prefer_precomputed: bool = True,
) -> np.ndarray:
    """Get a mix, loading pre-computed if available or computing on-the-fly.

    Args:
        track: Track instance.
        mix_spec: Mix specification.
        target_sr: Optional target sample rate.
        prefer_precomputed: If True, try to load pre-computed mix first.

    Returns:
        Audio array of shape (channels, samples).
    """
    if prefer_precomputed:
        audio = load_precomputed_mix(track, mix_spec, target_sr)
        if audio is not None:
            return audio

    return compute_mix(track, mix_spec, target_sr)
