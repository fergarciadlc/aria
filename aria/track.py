"""
Track representation for ARIA dataset.

Track is a dataclass containing metadata and path information.
Audio is loaded by module-level functions (separation of concerns).

On-disk layout of a track:
    {root}/{track_id}/metadata.json
    {root}/{track_id}/stems/real/{stem_name}.{ext}
    {root}/{track_id}/stems/ai_generated/{codec}/{stem_name}.{ext}
    {root}/{track_id}/mixes/{codec}/U{k}/mix_S-{indices}.{ext}
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from aria.constants import AUDIO_EXTENSIONS
from aria.exceptions import MissingStemError, TrackNotFoundError


@dataclass
class Track:
    """A single track in the ARIA dataset.

    This is a data container - audio loading is done via functions.

    Attributes:
        id: Unique track identifier (UUID).
        artist: Artist name.
        name: Track name.
        genre: Genre classification.
        stem_names: Stem names sorted alphabetically (e.g., ["bass", "drums",
            "vocals"]). A stem's position in this list is its index in mix
            names (mix_S-0-2 = stems 0 and 2 are AI-generated).
        path: Path to the track directory.
        provider: Data provider (e.g., "moisesdb").
        additional_info: Any extra metadata.
    """

    id: str
    artist: str
    name: str
    genre: str
    stem_names: list[str]
    path: Path
    provider: str = ""
    additional_info: dict[str, Any] = field(default_factory=dict)

    @property
    def num_stems(self) -> int:
        """Number of stems in this track."""
        return len(self.stem_names)

    @property
    def display_name(self) -> str:
        """Human-readable track name."""
        return f"{self.artist} - {self.name}"

    def __repr__(self) -> str:
        return f"Track({self.display_name!r}, stems={self.num_stems})"


def is_track_dir(path: Path) -> bool:
    """Whether path is a track directory (contains metadata.json)."""
    return path.is_dir() and (path / "metadata.json").exists()


def iter_track_dirs(dataset_path: Path | str) -> list[Path]:
    """List track directories under a dataset root, sorted by directory name.

    Args:
        dataset_path: Path to dataset root.

    Returns:
        Sorted list of directories containing a metadata.json.
    """
    return sorted(d for d in Path(dataset_path).iterdir() if is_track_dir(d))


def load_track(track_path: Path) -> Track:
    """Load a track from its directory.

    Stem names are sorted alphabetically so that stem indices match the
    pre-computed mixes and their sidecar JSON files.

    Args:
        track_path: Path to the track directory (containing metadata.json).

    Returns:
        Track instance.

    Raises:
        TrackNotFoundError: If track directory or metadata doesn't exist.
    """
    track_path = Path(track_path)
    metadata_path = track_path / "metadata.json"

    if not metadata_path.exists():
        raise TrackNotFoundError(str(track_path))

    try:
        with open(metadata_path) as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        raise TrackNotFoundError(f"{track_path} (invalid JSON: {e})") from e

    return Track(
        id=data.get("id", ""),
        artist=data.get("artist", "Unknown"),
        name=data.get("name", "Unknown"),
        genre=data.get("genre", "unknown"),
        stem_names=sorted(data.get("stems", [])),
        path=track_path,
        provider=data.get("provider", ""),
        additional_info=data.get("additional_info", {}),
    )


def get_stem_dir(track: Track, source: str = "real", codec: str | None = None) -> Path:
    """Get the directory holding a track's stems for a source.

    Args:
        track: Track instance.
        source: Either "real" or "ai_generated".
        codec: Required if source is "ai_generated" (e.g., "encodec_3").

    Returns:
        Path to the stems directory (may not exist).

    Raises:
        ValueError: If codec not provided for ai_generated source.
    """
    if source == "real":
        return track.path / "stems" / "real"
    if source == "ai_generated":
        if codec is None:
            raise ValueError("codec must be specified for ai_generated stems")
        return track.path / "stems" / "ai_generated" / codec
    raise ValueError(f"Invalid source: {source}. Use 'real' or 'ai_generated'")


def get_stem_path(
    track: Track,
    stem_name: str,
    source: str = "real",
    codec: str | None = None,
) -> Path:
    """Get the file path for a specific stem.

    Args:
        track: Track instance.
        stem_name: Name of the stem (e.g., "vocals").
        source: Either "real" or "ai_generated".
        codec: Required if source is "ai_generated" (e.g., "encodec_3").

    Returns:
        Path to the stem audio file.

    Raises:
        ValueError: If codec not provided for ai_generated source.
        MissingStemError: If the stem file doesn't exist.
    """
    stem_dir = get_stem_dir(track, source, codec)

    for extension in AUDIO_EXTENSIONS:
        stem_path = stem_dir / f"{stem_name}{extension}"
        if stem_path.exists():
            return stem_path

    raise MissingStemError(stem_name, source, track.id)


def get_available_codecs(track: Track) -> list[str]:
    """Get list of available AI codecs for a track.

    Args:
        track: Track instance.

    Returns:
        List of codec names (e.g., ["encodec_3", "encodec_6"]).
    """
    ai_gen_path = track.path / "stems" / "ai_generated"

    if not ai_gen_path.exists():
        return []

    return sorted([d.name for d in ai_gen_path.iterdir() if d.is_dir()])


def has_all_stems(track: Track, source: str, codec: str | None = None) -> bool:
    """Check if track has all stems for a given source.

    Args:
        track: Track instance.
        source: Either "real" or "ai_generated".
        codec: Required if source is "ai_generated".

    Returns:
        True if all stems exist.
    """
    for stem_name in track.stem_names:
        try:
            get_stem_path(track, stem_name, source, codec)
        except MissingStemError:
            return False
    return True
