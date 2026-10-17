"""
ARIA Dataset class.

Thin wrapper providing ergonomic access to tracks and mixes.
Core logic is in functions; this class provides a nice API.
"""

import logging
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any

from aria.exceptions import DatasetNotFoundError, TrackNotFoundError
from aria.track import Track, get_available_codecs, iter_track_dirs, load_track

logger = logging.getLogger(__name__)


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
    """

    def __init__(self, path: str | Path):
        """Initialize the dataset.

        Args:
            path: Path to the ARIA dataset root directory.

        Raises:
            DatasetNotFoundError: If path doesn't exist or has no tracks.
        """
        self.path = Path(path)

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
            key: Integer index (negative indices allowed) or string track ID.

        Returns:
            Track instance.

        Raises:
            TrackNotFoundError: If track not found.
            IndexError: If index out of range.
        """
        if isinstance(key, str):
            idx = self._id_to_idx.get(key)
            if idx is None:
                raise TrackNotFoundError(key)
            return self._tracks[idx]

        try:
            return self._tracks[key]
        except IndexError:
            raise IndexError(f"Track index out of range: {key}") from None
        except TypeError:
            raise TypeError(f"Key must be int or str, got {type(key)}") from None

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
        """List of codecs available in any track."""
        return sorted({c for t in self._tracks for c in get_available_codecs(t)})

    def stats(self) -> dict[str, Any]:
        """Get dataset statistics.

        Returns:
            Dictionary with dataset statistics.
        """
        genres: dict[str, int] = {}
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

    Scans for directories containing metadata.json. Tracks whose metadata
    can't be loaded are skipped with a warning.

    Args:
        dataset_path: Path to dataset root.

    Returns:
        List of Track instances, sorted by display_name.
    """
    tracks = []

    for track_dir in iter_track_dirs(dataset_path):
        try:
            tracks.append(load_track(track_dir))
        except TrackNotFoundError as e:
            logger.warning("Skipping invalid track %s: %s", track_dir.name, e)

    # Sort by display name for consistent ordering
    tracks.sort(key=lambda t: t.display_name.lower())

    return tracks
