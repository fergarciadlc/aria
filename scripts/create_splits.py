"""
Create train/val/test split manifests for the ARIA dataset.

Splits are performed at the track level (stratified by num_stems) so that
all mixes from a given track land in the same partition — preventing any
leakage of audio content across splits.

Output:
    <output>.jsonl          — JSONL manifest, one mix record per line
    <output>.meta.json      — split statistics and provenance metadata

Usage:
    python scripts/create_splits.py --aria-path /path/to/aria --output splits/canonical_v2.jsonl
    python scripts/create_splits.py --aria-path /path/to/aria --output splits/canonical_v2.jsonl --seed 42
"""

import argparse
import csv
import json
import sys
from collections import defaultdict
from datetime import date
from pathlib import Path
import random

from loguru import logger
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from aria.dataset import ARIADataset
from aria.mix import generate_all_mix_specs, get_mix_path
from aria.track import Track


CODEC = "encodec_3"
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
# test ratio = 1 - TRAIN_RATIO - VAL_RATIO = 0.1
DEFAULT_METRICS_CSV = "metrics/mix_metrics.csv"


# ---------------------------------------------------------------------------
# Energy metrics loader
# ---------------------------------------------------------------------------

def load_energy_metrics(metrics_path: Path) -> dict[str, float]:
    """Load ai_energy_ratio from mix_metrics.csv keyed by relative mix path.

    Key format: {uuid}/mixes/{codec}/{stem_folder}/{filename} — the last 5
    path components of the absolute audio_path in the CSV. This matches the
    mix_path stored relative to aria_root in the manifest records.
    """
    lookup: dict[str, float] = {}
    with open(metrics_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = "/".join(Path(row["audio_path"]).parts[-5:])
            raw = row.get("ai_energy_ratio", "")
            lookup[key] = float(raw) if raw else 0.0
    logger.info(f"Loaded energy metrics for {len(lookup)} mixes from {metrics_path}")
    return lookup


# ---------------------------------------------------------------------------
# Track-level stratified split
# ---------------------------------------------------------------------------

def stratified_split_by_num_stems(
    tracks: list[Track],
    train_ratio: float,
    val_ratio: float,
    seed: int,
) -> tuple[list[Track], list[Track], list[Track]]:
    """Split tracks into train/val/test, stratified by num_stems.

    Groups tracks by their stem count and applies the split ratio within each
    group independently, ensuring each partition has a similar distribution of
    stem counts (and therefore achievable ai_ratio values).

    Args:
        tracks: All tracks to split.
        train_ratio: Fraction of tracks for training.
        val_ratio: Fraction of tracks for validation.
        seed: Random seed for reproducibility.

    Returns:
        Tuple of (train_tracks, val_tracks, test_tracks).
    """
    rng = random.Random(seed)

    # Group by num_stems
    groups: dict[int, list[Track]] = defaultdict(list)
    for track in tracks:
        groups[track.num_stems].append(track)

    train, val, test = [], [], []

    for stem_count in sorted(groups.keys()):
        group = list(groups[stem_count])
        rng.shuffle(group)
        n = len(group)

        n_train = round(n * train_ratio)
        n_val = round(n * val_ratio)
        # test gets the remainder to avoid rounding dropping any tracks
        n_train = max(n_train, 0)
        n_val = max(n_val, 0)

        train.extend(group[:n_train])
        val.extend(group[n_train : n_train + n_val])
        test.extend(group[n_train + n_val :])

        logger.debug(
            f"  num_stems={stem_count}: {n} tracks → "
            f"{len(group[:n_train])} train / "
            f"{len(group[n_train:n_train+n_val])} val / "
            f"{len(group[n_train+n_val:])} test"
        )

    return train, val, test


# ---------------------------------------------------------------------------
# Mix record building
# ---------------------------------------------------------------------------

def build_records(
    tracks: list[Track],
    track_to_split: dict[str, str],
    codec: str,
    aria_root: Path,
    energy_lookup: dict[str, float],
) -> tuple[list[dict], int]:
    """Expand tracks to individual mix records, verifying each path exists.

    Args:
        tracks: All tracks (all splits).
        track_to_split: Mapping from track_id to split name.
        codec: Encodec codec string (e.g. "encodec_3").
        aria_root: Dataset root for resolving relative paths.
        energy_lookup: Mapping from relative mix path to ai_energy_ratio.

    Returns:
        Tuple of (records list, skipped_count).
    """
    records = []
    skipped_count = 0

    for track in tqdm(tracks, desc="Building manifest"):
        split = track_to_split[track.id]
        mix_specs = generate_all_mix_specs(track.num_stems, codec)

        for spec in mix_specs:
            mix_path = get_mix_path(track, spec, format="mp3")

            if not mix_path.exists():
                logger.warning(f"Missing mix file: {mix_path}")
                skipped_count += 1
                continue

            rel_key = "/".join(mix_path.relative_to(aria_root).parts)
            if rel_key not in energy_lookup:
                logger.warning(f"Missing energy metrics, skipping: {rel_key}")
                skipped_count += 1
                continue

            records.append({
                "track_id": track.id,
                "artist": track.artist,
                "track_name": track.name,
                "genre": track.genre,
                "codec": codec,
                "num_stems": track.num_stems,
                "num_ai_stems": spec.num_ai_stems,
                "ai_indices": list(spec.ai_indices),
                "ai_energy_ratio": round(energy_lookup[rel_key], 6),
                "split": split,
                "mix_path": str(mix_path.relative_to(aria_root)),
            })

    return records, skipped_count


# ---------------------------------------------------------------------------
# Sidecar metadata
# ---------------------------------------------------------------------------

def build_sidecar(
    records: list[dict],
    codec: str,
    seed: int,
    train_ratio: float,
    val_ratio: float,
    train_tracks: list[Track],
    val_tracks: list[Track],
    test_tracks: list[Track],
) -> dict:
    """Build the sidecar metadata dict."""

    def split_stats(split_name: str, split_tracks: list[Track]) -> dict:
        split_records = [r for r in records if r["split"] == split_name]
        genre_counts: dict[str, int] = defaultdict(int)
        stem_dist: dict[str, int] = defaultdict(int)
        label_dist: dict[str, int] = defaultdict(int)

        for r in split_records:
            genre_counts[r["genre"]] += 1
            stem_dist[str(r["num_stems"])] += 1
            label_dist[str(r["ai_energy_ratio"])] += 1

        return {
            "num_tracks": len(split_tracks),
            "num_mixes": len(split_records),
            "genre_counts": dict(sorted(genre_counts.items())),
            "stem_count_dist": dict(sorted(stem_dist.items(), key=lambda x: int(x[0]))),
            "label_distribution": dict(sorted(label_dist.items(), key=lambda x: float(x[0]))),
        }

    return {
        "version": "2.0",
        "created_at": date.today().isoformat(),
        "codec": codec,
        "seed": seed,
        "split_ratios": {
            "train": train_ratio,
            "val": val_ratio,
            "test": round(1.0 - train_ratio - val_ratio, 4),
        },
        "stratify_by": "num_stems",
        "total_records": len(records),
        "splits": {
            "train": split_stats("train", train_tracks),
            "val": split_stats("val", val_tracks),
            "test": split_stats("test", test_tracks),
        },
    }


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def create_splits(
    aria_path: str,
    output_path: str,
    codec: str = CODEC,
    train_ratio: float = TRAIN_RATIO,
    val_ratio: float = VAL_RATIO,
    seed: int = 42,
    metrics_csv: str | None = None,
) -> None:
    aria_root = Path(aria_path)
    output_file = Path(output_path)
    metrics_path = Path(metrics_csv) if metrics_csv else aria_root / DEFAULT_METRICS_CSV

    logger.info("=" * 70)
    logger.info("ARIA Split Manifest Generation")
    logger.info("=" * 70)
    logger.info(f"Dataset:      {aria_root}")
    logger.info(f"Output:       {output_file}")
    logger.info(f"Codec:        {codec}")
    logger.info(f"Metrics CSV:  {metrics_path}")
    logger.info(f"Split ratios: {train_ratio:.0%} / {val_ratio:.0%} / {1-train_ratio-val_ratio:.0%}")
    logger.info(f"Seed:         {seed}")
    logger.info(f"Stratify by:  num_stems")

    if not metrics_path.exists():
        logger.error(f"Metrics CSV not found: {metrics_path}")
        sys.exit(1)
    energy_lookup = load_energy_metrics(metrics_path)

    dataset = ARIADataset(aria_root)
    tracks = list(dataset)
    logger.info(f"Loaded {len(tracks)} tracks")

    # --- Track-level split ---
    train_tracks, val_tracks, test_tracks = stratified_split_by_num_stems(
        tracks, train_ratio, val_ratio, seed
    )
    logger.info(
        f"Track split: {len(train_tracks)} train / "
        f"{len(val_tracks)} val / {len(test_tracks)} test"
    )

    track_to_split: dict[str, str] = {}
    for track in train_tracks:
        track_to_split[track.id] = "train"
    for track in val_tracks:
        track_to_split[track.id] = "val"
    for track in test_tracks:
        track_to_split[track.id] = "test"

    # --- Expand to mix records ---
    records, missing_count = build_records(tracks, track_to_split, codec, aria_root, energy_lookup)

    if missing_count > 0:
        logger.warning(
            f"Skipped {missing_count} mixes (missing files or unreadable). "
            f"Run generate_mixes.py to pre-compute all mixes before creating the manifest."
        )

    # --- Write JSONL manifest ---
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")

    logger.info(f"Wrote {len(records)} records → {output_file}")

    # --- Write sidecar ---
    sidecar = build_sidecar(
        records, codec, seed, train_ratio, val_ratio,
        train_tracks, val_tracks, test_tracks,
    )
    sidecar_path = output_file.with_suffix("").with_suffix(".meta.json")
    # Handle double suffix like "canonical_v1.jsonl" → "canonical_v1.meta.json"
    sidecar_path = output_file.parent / (output_file.stem.split(".")[0] + ".meta.json")
    with open(sidecar_path, "w") as f:
        json.dump(sidecar, f, indent=2)

    logger.info(f"Wrote metadata  → {sidecar_path}")

    # --- Summary ---
    logger.info("=" * 70)
    for split_name in ("train", "val", "test"):
        s = sidecar["splits"][split_name]
        logger.info(
            f"  {split_name:5s}: {s['num_tracks']:3d} tracks, {s['num_mixes']:5d} mixes"
        )
    logger.info("=" * 70)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate train/val/test split manifests for the ARIA dataset.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/create_splits.py --aria-path /data/aria --output /data/aria/splits/canonical_v1.jsonl
  python scripts/create_splits.py --aria-path /data/aria --output splits/canonical_v1.jsonl --seed 123
        """,
    )
    parser.add_argument(
        "--aria-path",
        required=True,
        help="Path to the ARIA dataset root directory.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output path for the JSONL manifest (e.g. splits/canonical_v1.jsonl).",
    )
    parser.add_argument(
        "--codec",
        default=CODEC,
        help=f"Codec for AI stems (default: {CODEC}).",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=TRAIN_RATIO,
        help=f"Fraction of tracks for training (default: {TRAIN_RATIO}).",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=VAL_RATIO,
        help=f"Fraction of tracks for validation (default: {VAL_RATIO}).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42).",
    )
    parser.add_argument(
        "--metrics-csv",
        default=None,
        help=f"Path to mix_metrics.csv (default: {{aria-path}}/{DEFAULT_METRICS_CSV}).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.train_ratio + args.val_ratio >= 1.0:
        logger.error("train_ratio + val_ratio must be < 1.0 (test set needs some tracks).")
        sys.exit(1)

    create_splits(
        aria_path=args.aria_path,
        output_path=args.output,
        codec=args.codec,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed,
        metrics_csv=args.metrics_csv,
    )
