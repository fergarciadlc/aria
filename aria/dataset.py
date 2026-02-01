"""
ARIA Dataset class.

Thin wrapper providing ergonomic access to tracks and mixes.
Core logic is in functions; this class provides a nice API.
"""

from pathlib import Path
from typing import Iterator, Sequence

from aria.config import Config, DEFAULT_CONFIG
from aria.exceptions import DatasetNotFoundError, TrackNotFoundError
from aria.track import Track, load_track, get_available_codecs


class ARIADataset:
    """ARIA dataset for loading tracks and generating mixes.

    This is a thin wrapper around functional operations, providing:
    - Track discovery and iteration
    - Filtering by genre, num_stems, etc.
    - Easy access by index or track ID

    Example:
        >>> dataset = ARIADataset("/path/to/aria")
        >>> len(dataset)
        240
        >>> track = dataset[0]
        >>> track.display_name
        'Artist - Song'

    Attributes:
        path: Path to the ARIA dataset root.
        config: Configuration instance.
    """

    def __init__(
        self,
        path: str | Path,
        config: Config | None = None,
    ):
        """Initialize the dataset.

        Args:
            path: Path to the ARIA dataset root directory.
            config: Optional configuration. Uses defaults if not provided.

        Raises:
            DatasetNotFoundError: If path doesn't exist or has no tracks.
        """
        self.path = Path(path)
        self.config = config or DEFAULT_CONFIG

        if not self.path.exists():
            raise DatasetNotFoundError(str(self.path))

        # Discover tracks (lazy-ish: done once at init)
        self._tracks = _discover_tracks(self.path)

        if not self._tracks:
            raise DatasetNotFoundError(f"{self.path} (no tracks found)")

        # Build ID index for fast lookup
        self._id_to_idx = {t.id: i for i, t in enumerate(self._tracks)}

    def __len__(self) -> int:
        """Number of tracks in dataset."""
        return len(self._tracks)

    def __getitem__(self, key: int | str) -> Track:
        """Get track by index or ID.

        Args:
            key: Integer index or string track ID.

        Returns:
            Track instance.

        Raises:
            TrackNotFoundError: If track not found.
            IndexError: If index out of range.
        """
        if isinstance(key, int):
            if key < 0 or key >= len(self._tracks):
                raise IndexError(f"Track index out of range: {key}")
            return self._tracks[key]

        if isinstance(key, str):
            idx = self._id_to_idx.get(key)
            if idx is None:
                raise TrackNotFoundError(key)
            return self._tracks[idx]

        raise TypeError(f"Key must be int or str, got {type(key)}")

    def __iter__(self) -> Iterator[Track]:
        """Iterate over all tracks."""
        return iter(self._tracks)

    def __repr__(self) -> str:
        return f"ARIADataset({self.path}, tracks={len(self)})"

    @property
    def tracks(self) -> Sequence[Track]:
        """All tracks as a sequence (read-only view)."""
        return self._tracks

    def filter(
        self,
        genre: str | None = None,
        min_stems: int | None = None,
        max_stems: int | None = None,
        has_codec: str | None = None,
    ) -> list[Track]:
        """Filter tracks by criteria.

        Args:
            genre: Filter by genre (case-insensitive substring match).
            min_stems: Minimum number of stems.
            max_stems: Maximum number of stems.
            has_codec: Only tracks with this codec available.

        Returns:
            List of matching tracks.
        """
        result = list(self._tracks)

        if genre is not None:
            genre_lower = genre.lower()
            result = [t for t in result if genre_lower in t.genre.lower()]

        if min_stems is not None:
            result = [t for t in result if t.num_stems >= min_stems]

        if max_stems is not None:
            result = [t for t in result if t.num_stems <= max_stems]

        if has_codec is not None:
            result = [t for t in result if has_codec in get_available_codecs(t)]

        return result

    def get_by_name(self, name: str) -> Track | None:
        """Find track by artist/name (case-insensitive).

        Args:
            name: Search string to match against display_name.

        Returns:
            First matching track, or None.
        """
        name_lower = name.lower()
        for track in self._tracks:
            if name_lower in track.display_name.lower():
                return track
        return None

    @property
    def genres(self) -> list[str]:
        """List of unique genres in the dataset."""
        return sorted(set(t.genre for t in self._tracks))

    @property
    def codecs(self) -> list[str]:
        """List of available codecs (from first track with AI stems)."""
        for track in self._tracks:
            codecs = get_available_codecs(track)
            if codecs:
                return codecs
        return []

    def stats(self) -> dict:
        """Get dataset statistics.

        Returns:
            Dictionary with dataset statistics.
        """
        genres = {}
        stem_counts = []

        for track in self._tracks:
            genres[track.genre] = genres.get(track.genre, 0) + 1
            stem_counts.append(track.num_stems)

        return {
            "num_tracks": len(self._tracks),
            "genres": genres,
            "total_stems": sum(stem_counts),
            "avg_stems": sum(stem_counts) / len(stem_counts) if stem_counts else 0,
            "min_stems": min(stem_counts) if stem_counts else 0,
            "max_stems": max(stem_counts) if stem_counts else 0,
            "codecs": self.codecs,
        }


def _discover_tracks(dataset_path: Path) -> list[Track]:
    """Discover all tracks in the dataset.

    Scans for directories containing metadata.json.

    Args:
        dataset_path: Path to dataset root.

    Returns:
        List of Track instances, sorted by display_name.
    """
    tracks = []

    for item in dataset_path.iterdir():
        if not item.is_dir():
            continue

        metadata_path = item / "metadata.json"
        if not metadata_path.exists():
            continue

        try:
            track = load_track(item)
            tracks.append(track)
        except Exception:
            # Skip invalid tracks
            continue

    # Sort by display name for consistent ordering
    tracks.sort(key=lambda t: t.display_name.lower())

    return tracks


