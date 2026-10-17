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

Window manifests
----------------
A second manifest flavour has one record per (mix, window) instead of one per
mix. It is the format the window-level detector was trained on (5 s and 1 s
windows). To avoid a clash with the schema versions above, this module calls
it a "window manifest" and never uses the version number to tell the two
apart: a file is a window manifest when its first row has "window_start_sec"
(see is_window_manifest()). Window record schema (WindowRecord):

    track_id, artist, track_name, genre, codec,
    num_stems, num_ai_stems, ai_indices, split, mix_path,
    window_index        0-based position of the window within its mix
    window_start_sec    window start on the mix timeline, seconds
    window_end_sec      window end on the mix timeline, seconds
    ai_energy_ratio     energy share of AI stems inside this window, never null
    is_partial          True for a trailing window shorter than the window size

The window records keep the field set of the original training-time manifests
so the files stay byte-compatible with them. They do not carry the per-mix
ai_stem_ratio and duration fields.
"""

import csv
import json
import logging
import math
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


def write_manifest(
    records: Iterable["ManifestRecord | WindowRecord"], path: Path | str
) -> None:
    """Write records (per-mix or window) as a JSONL manifest."""
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


# ---------------------------------------------------------------------------
# Window manifests (one record per mix window)
# ---------------------------------------------------------------------------

WINDOW_MANIFEST_VERSION = "3.0"

_NAN_TOKENS = ("", "nan", "NaN")


@dataclass(frozen=True)
class WindowRecord:
    """One (mix, window) entry in a window manifest.

    Field order is the on-disk key order and matches the manifests the
    detector was trained on. ``ai_energy_ratio`` is never None: a window with
    no energy in the metrics CSV (silence) is labelled 0.0, see
    load_window_metrics().

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
        split: Dataset partition ("train", "val", or "test").
        mix_path: Path to the pre-computed mix file, relative to dataset root.
        window_index: 0-based position of the window in the mix, in order of
            window_start_sec.
        window_start_sec: Window start in seconds, rounded to 3 decimals.
        window_end_sec: Window end in seconds, rounded to 3 decimals.
        ai_energy_ratio: Energy share of the AI stems in this window,
            rounded to 6 decimals.
        is_partial: Whether the window is a short trailing window.
    """

    track_id: str
    artist: str
    track_name: str
    genre: str
    codec: str
    num_stems: int
    num_ai_stems: int
    ai_indices: tuple[int, ...]
    split: str
    mix_path: str
    window_index: int
    window_start_sec: float
    window_end_sec: float
    ai_energy_ratio: float
    is_partial: bool

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "WindowRecord":
        """Build a record from a window manifest line.

        Raises:
            KeyError: If a required field is missing. ``is_partial`` is
                optional and defaults to False, as in the detector's reader.
        """
        return cls(
            track_id=data["track_id"],
            artist=data["artist"],
            track_name=data["track_name"],
            genre=data["genre"],
            codec=data["codec"],
            num_stems=data["num_stems"],
            num_ai_stems=data["num_ai_stems"],
            ai_indices=tuple(data["ai_indices"]),
            split=data["split"],
            mix_path=data["mix_path"],
            window_index=int(data["window_index"]),
            window_start_sec=float(data["window_start_sec"]),
            window_end_sec=float(data["window_end_sec"]),
            ai_energy_ratio=float(data["ai_energy_ratio"]),
            is_partial=bool(data.get("is_partial", False)),
        )

    def to_dict(self) -> dict[str, Any]:
        """JSON-serialisable representation (one manifest line)."""
        data = asdict(self)
        data["ai_indices"] = list(self.ai_indices)
        return data


@dataclass(frozen=True)
class EnergyWindow:
    """One row of a windowed mix metrics CSV, after parsing."""

    start_sec: float
    end_sec: float
    ai_energy_ratio: float
    is_partial: bool


@dataclass(frozen=True)
class WindowMetrics:
    """Parsed windowed mix metrics CSV.

    Attributes:
        windows: Relative mix path to its windows, sorted by start time.
        is_windowed: True when the CSV has a window_start_sec column and at
            least one mix has more than one row. A windowed CSV where every
            mix has a single row (all tracks shorter than one window) is
            reported as not windowed, as the original script did.
        window_sec: Most common length of the non-partial windows, or None
            when the CSV is not windowed.
        num_silent: Rows whose ai_energy_ratio cell was empty (silent window)
            and were labelled 0.0.
    """

    windows: dict[str, list[EnergyWindow]]
    is_windowed: bool
    window_sec: float | None
    num_silent: int


def _float_or_none(raw: str | None) -> float | None:
    """Parse a CSV cell, treating empty and NaN cells as missing."""
    if raw is None or raw.strip() in _NAN_TOKENS:
        return None
    value = float(raw)
    return None if math.isnan(value) else value


def load_window_metrics(metrics_path: Path | str) -> WindowMetrics:
    """Read a windowed mix_metrics.csv into per-mix window lists.

    Needs ``audio_path`` and ``ai_energy_ratio`` columns; the window columns
    are ``window_start_sec``, ``window_end_sec`` and optionally ``is_partial``
    ("True"/"False"; absent means False). Keys are the last 5 path components
    of audio_path, as in load_energy_metrics().

    Semantics that the released window manifests depend on, kept on purpose:

    * An empty (NaN) ``ai_energy_ratio`` cell becomes 0.0. pandas writes NaN
      as an empty cell, and the metrics script emits NaN for windows with no
      signal energy, so silent windows are labelled 0.0 (and counted in
      ``num_silent``). The value is never None.
    * Empty or NaN window bounds become 0.0 / 0.0.
    * Each mix's windows are sorted by start time (stable for ties).
    * The window size is the most common non-partial window length, rounded
      to 3 decimals.
    """
    windows: dict[str, list[EnergyWindow]] = defaultdict(list)
    length_counts: dict[float, int] = defaultdict(int)
    num_silent = 0

    with open(metrics_path, newline="") as f:
        reader = csv.DictReader(f)
        has_window_cols = bool(
            reader.fieldnames and "window_start_sec" in reader.fieldnames
        )

        for row in reader:
            key = "/".join(Path(row["audio_path"]).parts[-5:])

            ratio = _float_or_none(row.get("ai_energy_ratio", ""))
            if ratio is None:
                ratio = 0.0
                num_silent += 1

            start = end = 0.0
            is_partial = False
            if has_window_cols:
                start = _float_or_none(row.get("window_start_sec", "")) or 0.0
                end = _float_or_none(row.get("window_end_sec", "")) or 0.0
                is_partial = (row.get("is_partial") or "").strip().lower() == "true"
                if end > start and not is_partial:
                    length_counts[round(end - start, 3)] += 1

            windows[key].append(EnergyWindow(start, end, ratio, is_partial))

    for entries in windows.values():
        entries.sort(key=lambda w: w.start_sec)

    is_windowed = has_window_cols and any(len(v) > 1 for v in windows.values())
    window_sec = None
    if is_windowed and length_counts:
        window_sec = max(length_counts, key=lambda k: length_counts[k])

    if num_silent:
        logger.info("%d window(s) with empty ai_energy_ratio labelled 0.0", num_silent)
    logger.info(
        "Loaded window metrics for %d mixes from %s (%s%s)",
        len(windows),
        metrics_path,
        "windowed" if is_windowed else "per-mix",
        f", window={window_sec}s" if window_sec is not None else "",
    )
    return WindowMetrics(dict(windows), is_windowed, window_sec, num_silent)


def build_window_manifest(
    tracks: Iterable[Track],
    dataset_root: Path | str,
    track_to_split: dict[str, str],
    codec: str,
    window_lookup: dict[str, list[EnergyWindow]],
) -> tuple[list[WindowRecord], int]:
    """Expand tracks to one record per existing mix window.

    Mixes missing on disk, or missing from ``window_lookup``, are skipped.
    ``window_index`` counts the windows of each mix in start-time order.

    Args:
        tracks: Tracks to include.
        dataset_root: Dataset root, for relative mix paths.
        track_to_split: Mapping from track_id to split name.
        codec: Codec whose mixes to include (e.g. "encodec_3").
        window_lookup: WindowMetrics.windows (relative mix path to windows).

    Returns:
        Tuple of (records, number of skipped mixes).
    """
    dataset_root = Path(dataset_root)
    records: list[WindowRecord] = []
    skipped = 0

    for track in tracks:
        for spec in generate_all_mix_specs(track.num_stems, codec):
            mix_path = find_mix_path(track, spec)
            if mix_path is None:
                logger.warning("Missing mix %s for track %s", spec.filename, track.id)
                skipped += 1
                continue

            rel_path = mix_path.relative_to(dataset_root).as_posix()
            entries = window_lookup.get(rel_path)
            if not entries:
                logger.warning("Missing window metrics, skipping: %s", rel_path)
                skipped += 1
                continue

            for index, window in enumerate(entries):
                records.append(WindowRecord(
                    track_id=track.id,
                    artist=track.artist,
                    track_name=track.name,
                    genre=track.genre,
                    codec=codec,
                    num_stems=track.num_stems,
                    num_ai_stems=spec.num_ai_stems,
                    ai_indices=spec.ai_indices,
                    split=track_to_split[track.id],
                    mix_path=rel_path,
                    window_index=index,
                    window_start_sec=round(window.start_sec, 3),
                    window_end_sec=round(window.end_sec, 3),
                    ai_energy_ratio=round(window.ai_energy_ratio, 6),
                    is_partial=window.is_partial,
                ))

    return records, skipped


def is_window_manifest(path: Path | str) -> bool:
    """Whether a manifest is window-level, judged by its first row.

    Uses the same test as the detector's loaders: the first non-empty line
    has a ``window_start_sec`` key.
    """
    with open(path) as f:
        for line in f:
            if line.strip():
                return "window_start_sec" in json.loads(line)
    return False


def load_window_manifest(path: Path | str) -> list[WindowRecord]:
    """Load every record of a window manifest.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If a line is invalid JSON or misses a field.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Manifest not found: {path}")
    records = []
    with open(path) as f:
        for lineno, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                records.append(WindowRecord.from_dict(json.loads(line)))
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON on line {lineno} of {path}: {e}") from e
            except KeyError as e:
                raise ValueError(f"Missing field {e} on line {lineno} of {path}") from e
    return records


def window_manifest_summary(
    records: list[WindowRecord],
    codec: str,
    seed: int,
    train_ratio: float,
    val_ratio: float,
    window_sec: float | None,
) -> dict[str, Any]:
    """Build the sidecar metadata written next to a window manifest.

    The version is "3.0", as in the original window manifests. Per-mix
    manifests from manifest_summary() also say "3.0", so readers tell the two
    apart with ``is_windowed`` / ``record_unit``.
    """

    def split_stats(split: str) -> dict[str, Any]:
        split_records = [r for r in records if r.split == split]
        genre_counts: dict[str, int] = defaultdict(int)
        stem_dist: dict[int, int] = defaultdict(int)
        for r in split_records:
            genre_counts[r.genre] += 1
            stem_dist[r.num_stems] += 1
        return {
            "num_tracks": len({r.track_id for r in split_records}),
            "num_windows": len(split_records),
            "num_partial_windows": sum(r.is_partial for r in split_records),
            "genre_counts": dict(sorted(genre_counts.items())),
            "stem_count_dist": {str(k): v for k, v in sorted(stem_dist.items())},
        }

    summary: dict[str, Any] = {
        "version": WINDOW_MANIFEST_VERSION,
        "created_at": date.today().isoformat(),
        "codec": codec,
        "seed": seed,
        "split_ratios": {
            "train": train_ratio,
            "val": val_ratio,
            "test": round(1.0 - train_ratio - val_ratio, 4),
        },
        "stratify_by": "num_stems",
        "split_unit": "track",
        "record_unit": "window",
        "is_windowed": True,
        "total_records": len(records),
        "splits": {split: split_stats(split) for split in SPLITS},
    }
    if window_sec is not None:
        summary["window_sec"] = window_sec
    return summary
