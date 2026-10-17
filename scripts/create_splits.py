"""
Create train/val/test split manifests for the ARIA dataset.

Splits are performed at the track level (stratified by num_stems) so that
all mixes from a given track land in the same partition, which prevents
leakage of audio content across splits. The manifest schema lives in
aria/splits.py.

Output:
    <output>.jsonl          : JSONL manifest, one mix record per line
    <output>.meta.json      : split statistics and provenance metadata

Usage:
    python scripts/create_splits.py --aria-path /path/to/aria --output splits/canonical_v2.jsonl
    python scripts/create_splits.py --aria-path /path/to/aria --output splits/canonical_v2.jsonl --seed 42
"""

import argparse
import json
import sys
from pathlib import Path

from loguru import logger

from aria.constants import DEFAULT_CODEC
from aria.dataset import ARIADataset
from aria.splits import (
    SPLITS,
    build_manifest,
    load_energy_metrics,
    manifest_summary,
    stratified_split_by_num_stems,
    write_manifest,
)

TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
# test ratio = 1 - TRAIN_RATIO - VAL_RATIO = 0.1
DEFAULT_METRICS_CSV = "metrics/mix_metrics.csv"


def create_splits(
    aria_path: str,
    output_path: str,
    codec: str = DEFAULT_CODEC,
    train_ratio: float = TRAIN_RATIO,
    val_ratio: float = VAL_RATIO,
    seed: int = 42,
    metrics_csv: str | None = None,
) -> None:
    aria_root = Path(aria_path)
    output_file = Path(output_path)

    # Energy metrics are optional: an explicit path must exist, the default
    # location is used only if present.
    if metrics_csv:
        metrics_path: Path | None = Path(metrics_csv)
        if not metrics_path.exists():
            logger.error(f"Metrics CSV not found: {metrics_path}")
            sys.exit(1)
    else:
        metrics_path = aria_root / DEFAULT_METRICS_CSV
        if not metrics_path.exists():
            metrics_path = None

    logger.info("=" * 70)
    logger.info("ARIA Split Manifest Generation")
    logger.info("=" * 70)
    logger.info(f"Dataset:      {aria_root}")
    logger.info(f"Output:       {output_file}")
    logger.info(f"Codec:        {codec}")
    logger.info(f"Metrics CSV:  {metrics_path or 'none (ai_energy_ratio will be null)'}")
    logger.info(f"Split ratios: {train_ratio:.0%} / {val_ratio:.0%} / {1-train_ratio-val_ratio:.0%}")
    logger.info(f"Seed:         {seed}")
    logger.info("Stratify by:  num_stems")

    energy_lookup = None
    if metrics_path is not None:
        energy_lookup = load_energy_metrics(metrics_path)
        logger.info(f"Loaded energy metrics for {len(energy_lookup)} mixes")

    tracks = list(ARIADataset(aria_root))
    logger.info(f"Loaded {len(tracks)} tracks")

    track_to_split = stratified_split_by_num_stems(tracks, train_ratio, val_ratio, seed)

    records, skipped = build_manifest(tracks, aria_root, track_to_split, codec, energy_lookup)
    if skipped > 0:
        logger.warning(
            f"Skipped {skipped} mixes (missing files or metrics). "
            f"Run generate_mixes.py to pre-compute all mixes before creating the manifest."
        )

    write_manifest(records, output_file)
    logger.info(f"Wrote {len(records)} records → {output_file}")

    # "canonical_v1.jsonl" → "canonical_v1.meta.json"
    summary = manifest_summary(records, codec, seed, train_ratio, val_ratio)
    sidecar_path = output_file.parent / (output_file.stem.split(".")[0] + ".meta.json")
    with open(sidecar_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Wrote metadata  → {sidecar_path}")

    logger.info("=" * 70)
    for split in SPLITS:
        s = summary["splits"][split]
        logger.info(f"  {split:5s}: {s['num_tracks']:3d} tracks, {s['num_mixes']:5d} mixes")
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
    parser.add_argument("--aria-path", required=True, help="Path to the ARIA dataset root directory.")
    parser.add_argument(
        "--output",
        required=True,
        help="Output path for the JSONL manifest (e.g. splits/canonical_v1.jsonl).",
    )
    parser.add_argument("--codec", default=DEFAULT_CODEC, help=f"Codec for AI stems (default: {DEFAULT_CODEC}).")
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
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility (default: 42).")
    parser.add_argument(
        "--metrics-csv",
        default=None,
        help=f"Path to mix_metrics.csv with ai_energy_ratio (default: {{aria-path}}/{DEFAULT_METRICS_CSV} if present).",
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
