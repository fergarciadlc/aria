"""
Split manifest for ARIA dataset.

A manifest is a JSONL file where each line is a mix sample with its
pre-computed path, label (ai_ratio), and split assignment (train/val/test).

Tracks are the atomic split unit — all mixes from a track land in the same
split to prevent leakage.
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator


@dataclass(frozen=True)
class ManifestRecord:
    """A single mix sample entry in a split manifest.

    Attributes:
        track_id: Unique track identifier.
        artist: Artist name.
        track_name: Track title.
        genre: Genre classification.
        codec: AI codec used (e.g. "encodec_3").
        num_stems: Total number of stems in the track.
        num_ai_stems: Number of AI-generated stems in this mix.
        ai_indices: Which stem indices are AI-generated (0-indexed).
        ai_ratio: Fraction of AI stems — the regression target (0.0 to 1.0).
        split: Dataset partition ("train", "val", or "test").
        duration: Audio duration in seconds.
        mix_path: Path to the pre-computed mix file, relative to dataset root.
    """

    track_id: str
    artist: str
    track_name: str
    genre: str
    codec: str
    num_stems: int
    num_ai_stems: int
    ai_indices: tuple[int, ...]
    ai_ratio: float
    split: str
    duration: float
    mix_path: str


class SplitManifest:
    """Loads and queries an ARIA split manifest (JSONL format).

    Each line in the manifest file is a JSON object describing one mix sample.
    Tracks are split at the track level — all mixes from a given track
    belong to the same partition.

    Example:
        >>> manifest = SplitManifest("/path/to/aria/splits/canonical_v1.jsonl",
        ...                          dataset_root="/path/to/aria")
        >>> train_records = manifest.get_split("train")
        >>> len(train_records)
        3124

    Attributes:
        path: Path to the JSONL manifest file.
        dataset_root: Optional root path used to resolve relative mix_paths.
    """

    def __init__(
        self,
        path: str | Path,
        dataset_root: str | Path | None = None,
    ):
        """Load a manifest from a JSONL file.

        Args:
            path: Path to the JSONL manifest file.
            dataset_root: Dataset root directory. Required if you call
                resolve_mix_path(). Not needed for label/metadata access.

        Raises:
            FileNotFoundError: If the manifest file does not exist.
        """
        self.path = Path(path)
        self.dataset_root = Path(dataset_root) if dataset_root is not None else None

        if not self.path.exists():
            raise FileNotFoundError(f"Manifest not found: {self.path}")

        self._records = self._load()

    def _load(self) -> list[ManifestRecord]:
        records = []
        with open(self.path) as f:
            for lineno, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                except json.JSONDecodeError as e:
                    raise ValueError(f"Invalid JSON on line {lineno} of {self.path}: {e}") from e

                records.append(ManifestRecord(
                    track_id=data["track_id"],
                    artist=data["artist"],
                    track_name=data["track_name"],
                    genre=data["genre"],
                    codec=data["codec"],
                    num_stems=data["num_stems"],
                    num_ai_stems=data["num_ai_stems"],
                    ai_indices=tuple(data["ai_indices"]),
                    ai_ratio=data["ai_ratio"],
                    split=data["split"],
                    duration=data["duration"],
                    mix_path=data["mix_path"],
                ))
        return records

    def get_split(self, split: str) -> list[ManifestRecord]:
        """Return all records belonging to the given split.

        Args:
            split: One of "train", "val", or "test".

        Returns:
            List of ManifestRecord for that split.
        """
        return [r for r in self._records if r.split == split]

    def resolve_mix_path(self, record: ManifestRecord) -> Path:
        """Resolve a record's relative mix_path to an absolute Path.

        Args:
            record: A ManifestRecord from this manifest.

        Returns:
            Absolute path to the mix file.

        Raises:
            ValueError: If dataset_root was not set at construction time.
        """
        if self.dataset_root is None:
            raise ValueError(
                "dataset_root must be provided to SplitManifest to resolve mix paths."
            )
        return self.dataset_root / record.mix_path

    def __iter__(self) -> Iterator[ManifestRecord]:
        return iter(self._records)

    def __len__(self) -> int:
        return len(self._records)

    def __repr__(self) -> str:
        split_counts: dict[str, int] = {}
        for r in self._records:
            split_counts[r.split] = split_counts.get(r.split, 0) + 1
        return f"SplitManifest({self.path.name}, total={len(self)}, splits={split_counts})"
