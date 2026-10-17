"""
Split manifests for the ARIA dataset.

A manifest is a JSONL file where each line is one pre-computed mix with its
labels and split assignment (train/val/test). Tracks are the atomic split
unit: all mixes from a track land in the same split to prevent leakage.

Record schema (one JSON object per line):
    track_id, artist, track_name, genre, codec,
    num_stems, num_ai_stems, ai_indices,
    ai_stem_ratio     num_ai_stems / num_stems
    ai_energy_ratio   energy share of AI stems in the mix, or null
    split             "train" | "val" | "test"
    duration          seconds, or null
    mix_path          relative to the dataset root

scripts/create_splits.py is the CLI around build_manifest().
"""

import csv
import json
import logging
import random
from collections import defaultdict
from collections.abc import Iterable, Iterator
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path
from typing import Any

import soundfile as sf

from aria.mix import find_mix_path, generate_all_mix_specs
from aria.track import Track

logger = logging.getLogger(__name__)

SPLITS = ("train", "val", "test")
MANIFEST_VERSION = "3.0"


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
        ai_indices: Which stem indices are AI-generated (0-indexed into the
            track's alphabetically sorted stem names).
        ai_stem_ratio: Fraction of stems that are AI-generated (0.0 to 1.0).
        ai_energy_ratio: Fraction of mix energy coming from AI stems, if
            computed (None otherwise).
        split: Dataset partition ("train", "val", or "test").
        duration: Mix duration in seconds, if known.
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
    ai_stem_ratio: float
    ai_energy_ratio: float | None
    split: str
    duration: float | None
    mix_path: str

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ManifestRecord":
        """Build a record from a manifest line.

        Manifests written before v3.0 lack ai_stem_ratio and duration; the
        stem ratio is derived from the stem counts and duration is None.

        Raises:
            KeyError: If a required field is missing.
        """
        num_stems = data["num_stems"]
        num_ai_stems = data["num_ai_stems"]
        return cls(
            track_id=data["track_id"],
            artist=data["artist"],
            track_name=data["track_name"],
            genre=data["genre"],
            codec=data["codec"],
            num_stems=num_stems,
            num_ai_stems=num_ai_stems,
            ai_indices=tuple(data["ai_indices"]),
            ai_stem_ratio=data.get("ai_stem_ratio", num_ai_stems / num_stems if num_stems else 0.0),
            ai_energy_ratio=data.get("ai_energy_ratio"),
            split=data["split"],
            duration=data.get("duration"),
            mix_path=data["mix_path"],
        )

    def to_dict(self) -> dict[str, Any]:
        """JSON-serialisable representation (one manifest line)."""
        data = asdict(self)
        data["ai_indices"] = list(self.ai_indices)
        return data


class SplitManifest:
    """Loads and queries an ARIA split manifest (JSONL format).

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
            ValueError: If a line is not valid JSON or misses a field.
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
                    records.append(ManifestRecord.from_dict(json.loads(line)))
                except json.JSONDecodeError as e:
                    raise ValueError(f"Invalid JSON on line {lineno} of {self.path}: {e}") from e
                except KeyError as e:
                    raise ValueError(f"Missing field {e} on line {lineno} of {self.path}") from e
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


# ---------------------------------------------------------------------------
# Manifest creation
# ---------------------------------------------------------------------------


def stratified_split_by_num_stems(
    tracks: Iterable[Track],
    train_ratio: float,
    val_ratio: float,
    seed: int,
) -> dict[str, str]:
    """Split tracks into train/val/test, stratified by num_stems.

    Groups tracks by their stem count and applies the split ratio within each
    group independently, so each partition has a similar distribution of stem
    counts (and therefore achievable ai_stem_ratio values).

    Args:
        tracks: All tracks to split.
        train_ratio: Fraction of tracks for training.
        val_ratio: Fraction of tracks for validation. Test gets the rest.
        seed: Random seed for reproducibility.

    Returns:
        Mapping from track_id to split name.

    Raises:
        ValueError: If train_ratio + val_ratio leaves no room for test.
    """
    if train_ratio + val_ratio >= 1.0:
        raise ValueError("train_ratio + val_ratio must be < 1.0 (test set needs some tracks).")

    rng = random.Random(seed)

    groups: dict[int, list[Track]] = defaultdict(list)
    for track in tracks:
        groups[track.num_stems].append(track)

    assignment: dict[str, str] = {}

    for stem_count in sorted(groups):
        group = list(groups[stem_count])
        rng.shuffle(group)
        n_train = round(len(group) * train_ratio)
        n_val = round(len(group) * val_ratio)

        # test gets the remainder so rounding never drops tracks
        for i, track in enumerate(group):
            if i < n_train:
                assignment[track.id] = "train"
            elif i < n_train + n_val:
                assignment[track.id] = "val"
            else:
                assignment[track.id] = "test"

    return assignment


def load_energy_metrics(metrics_path: Path | str) -> dict[str, float]:
    """Load ai_energy_ratio from a mix metrics CSV keyed by relative mix path.

    The CSV needs ``audio_path`` and ``ai_energy_ratio`` columns. Keys are the
    last 5 path components of audio_path ({track_id}/mixes/{codec}/U{k}/{file}),
    which matches mix_path relative to the dataset root.
    """
    lookup: dict[str, float] = {}
    with open(metrics_path, newline="") as f:
        for row in csv.DictReader(f):
            key = "/".join(Path(row["audio_path"]).parts[-5:])
            raw = row.get("ai_energy_ratio", "")
            lookup[key] = float(raw) if raw else 0.0
    return lookup


def build_manifest(
    tracks: Iterable[Track],
    dataset_root: Path | str,
    track_to_split: dict[str, str],
    codec: str,
    energy_lookup: dict[str, float] | None = None,
) -> tuple[list[ManifestRecord], int]:
    """Expand tracks to one record per existing pre-computed mix.

    Args:
        tracks: Tracks to include.
        dataset_root: Dataset root, for relative mix paths.
        track_to_split: Mapping from track_id to split name.
        codec: Codec whose mixes to include (e.g. "encodec_3").
        energy_lookup: Optional mapping from relative mix path to
            ai_energy_ratio. When given, mixes missing from it are skipped.

    Returns:
        Tuple of (records, number of skipped mixes).
    """
    dataset_root = Path(dataset_root)
    records = []
    skipped = 0

    for track in tracks:
        for spec in generate_all_mix_specs(track.num_stems, codec):
            mix_path = find_mix_path(track, spec)
            if mix_path is None:
                logger.warning("Missing mix %s for track %s", spec.filename, track.id)
                skipped += 1
                continue

            rel_path = mix_path.relative_to(dataset_root).as_posix()

            energy_ratio = None
            if energy_lookup is not None:
                if rel_path not in energy_lookup:
                    logger.warning("Missing energy metrics, skipping: %s", rel_path)
                    skipped += 1
                    continue
                energy_ratio = round(energy_lookup[rel_path], 6)

            records.append(ManifestRecord(
                track_id=track.id,
                artist=track.artist,
                track_name=track.name,
                genre=track.genre,
                codec=codec,
                num_stems=track.num_stems,
                num_ai_stems=spec.num_ai_stems,
                ai_indices=spec.ai_indices,
                ai_stem_ratio=round(spec.num_ai_stems / track.num_stems, 6),
                ai_energy_ratio=energy_ratio,
                split=track_to_split[track.id],
                duration=round(sf.info(mix_path).duration, 3),
                mix_path=rel_path,
            ))

    return records, skipped


def write_manifest(records: Iterable[ManifestRecord], path: Path | str) -> None:
    """Write records as a JSONL manifest."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for record in records:
            f.write(json.dumps(record.to_dict()) + "\n")


def manifest_summary(
    records: list[ManifestRecord],
    codec: str,
    seed: int,
    train_ratio: float,
    val_ratio: float,
) -> dict[str, Any]:
    """Build the sidecar metadata written next to a manifest."""

    def split_stats(split: str) -> dict[str, Any]:
        split_records = [r for r in records if r.split == split]
        genre_counts: dict[str, int] = defaultdict(int)
        stem_dist: dict[int, int] = defaultdict(int)
        ai_count_dist: dict[int, int] = defaultdict(int)
        for r in split_records:
            genre_counts[r.genre] += 1
            stem_dist[r.num_stems] += 1
            ai_count_dist[r.num_ai_stems] += 1
        return {
            "num_tracks": len({r.track_id for r in split_records}),
            "num_mixes": len(split_records),
            "genre_counts": dict(sorted(genre_counts.items())),
            "stem_count_dist": {str(k): v for k, v in sorted(stem_dist.items())},
            "num_ai_stems_dist": {str(k): v for k, v in sorted(ai_count_dist.items())},
        }

    return {
        "version": MANIFEST_VERSION,
        "created_at": date.today().isoformat(),
        "codec": codec,
        "seed": seed,
        "split_ratios": {
            "train": train_ratio,
            "val": val_ratio,
            "test": round(1.0 - train_ratio - val_ratio, 4),
        },
        "stratify_by": "num_stems",
        "stem_order": "alphabetical",
        "has_energy_ratio": any(r.ai_energy_ratio is not None for r in records),
        "total_records": len(records),
        "splits": {split: split_stats(split) for split in SPLITS},
    }
