"""
Create train/val/test split manifests for the ARIA dataset.

Splits are performed at the track level (stratified by num_stems) so that
all mixes, and all windows of a mix, from a given track land in the same
partition, which prevents leakage of audio content across splits. The manifest
schemas live in aria/splits.py.

Two manifest flavours, chosen from the metrics CSV:

    * per-mix manifest: one record per mix (default).
    * window manifest: one record per (mix, window), used when the CSV has
      window_start_sec / window_end_sec columns and several rows per mix.
      Produced from the output of a windowed metrics run.

Track assignment:

    * By default tracks are split with a seeded stratified split, fed in
      sorted track-id order so the result is the same on every filesystem.
    * --split-from PATH copies the track to split mapping of an existing
      manifest instead. Use it to rebuild a manifest on the published split
      (for example the paper's canonical_v2.jsonl) or to make a window
      manifest with the same split as a per-mix one.

Output:
    <output>.jsonl          : JSONL manifest, one record per line
    <output>.meta.json      : split statistics and provenance metadata

Usage:
    python scripts/create_splits.py --aria-path /path/to/aria --output splits/canonical_v2.jsonl
    python scripts/create_splits.py --aria-path /path/to/aria --output splits/windows_5s.jsonl \\
        --metrics-csv metrics/5s/mix_metrics.csv --split-from splits/canonical_v2.jsonl
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
    assign_reference_splits,
    build_manifest,
    build_window_manifest,
    load_energy_metrics,
    load_window_metrics,
    manifest_summary,
    read_track_splits,
    split_tracks,
    window_manifest_summary,
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
    split_from: str | None = None,
    allow_unassigned: bool = False,
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
    if split_from:
        logger.info(f"Split from:   {split_from}")
    else:
        logger.info(f"Split ratios: {train_ratio:.0%} / {val_ratio:.0%} / {1-train_ratio-val_ratio:.0%}")
        logger.info(f"Seed:         {seed}")
        logger.info("Stratify by:  num_stems (tracks in sorted id order)")

    # Per-mix versus window manifest is decided by the CSV contents.
    window_metrics = None
    energy_lookup = None
    if metrics_path is not None:
        loaded = load_window_metrics(metrics_path)
        if loaded.is_windowed:
            window_metrics = loaded
            logger.info(f"Mode:         window manifest (window={loaded.window_sec}s)")
        else:
            energy_lookup = load_energy_metrics(metrics_path)
            logger.info(f"Mode:         per-mix manifest ({len(energy_lookup)} mixes with metrics)")
    else:
        logger.info("Mode:         per-mix manifest")

    tracks = list(ARIADataset(aria_root))
    logger.info(f"Loaded {len(tracks)} tracks")

    if split_from:
        track_to_split = assign_reference_splits(
            tracks, read_track_splits(split_from), allow_missing=allow_unassigned
        )
        tracks = [t for t in tracks if t.id in track_to_split]
    else:
        track_to_split = split_tracks(tracks, train_ratio, val_ratio, seed)

    if window_metrics is not None:
        records, skipped = build_window_manifest(
            tracks, aria_root, track_to_split, codec, window_metrics.windows
        )
    else:
        records, skipped = build_manifest(tracks, aria_root, track_to_split, codec, energy_lookup)
    if skipped > 0:
        logger.warning(
            f"Skipped {skipped} mixes (missing files or metrics). "
            f"Run generate_mixes.py to pre-compute all mixes before creating the manifest."
        )

    write_manifest(records, output_file)
    logger.info(f"Wrote {len(records)} records → {output_file}")

    if window_metrics is not None:
        n_partial = sum(r.is_partial for r in records)
        if n_partial:
            logger.warning(
                f"{n_partial} window(s) are short trailing windows flagged "
                f"is_partial=true. Filter them at training time if needed."
            )
        summary = window_manifest_summary(
            records, codec, seed, train_ratio, val_ratio, window_metrics.window_sec
        )
        count_key = "num_windows"
    else:
        summary = manifest_summary(records, codec, seed, train_ratio, val_ratio)
        count_key = "num_mixes"

    if split_from:
        # Seed and ratios describe a split that was not made here.
        summary["split_from"] = Path(split_from).name
        summary["seed"] = None
        counts = [list(track_to_split.values()).count(s) for s in SPLITS]
        summary["split_ratios"] = {
            s: round(c / max(len(track_to_split), 1), 4) for s, c in zip(SPLITS, counts, strict=True)
        }

    # "canonical_v1.jsonl" → "canonical_v1.meta.json"
    sidecar_path = output_file.parent / (output_file.stem.split(".")[0] + ".meta.json")
    with open(sidecar_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Wrote metadata  → {sidecar_path}")

    logger.info("=" * 70)
    for split in SPLITS:
        s = summary["splits"][split]
        logger.info(f"  {split:5s}: {s['num_tracks']:3d} tracks, {s[count_key]:6d} {count_key[4:]}")
    logger.info("=" * 70)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate train/val/test split manifests for the ARIA dataset.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/create_splits.py --aria-path /data/aria --output /data/aria/splits/canonical_v1.jsonl
  python scripts/create_splits.py --aria-path /data/aria --output splits/canonical_v1.jsonl --seed 123

  # Window manifest on the same tracks-to-split assignment as an existing manifest
  python scripts/create_splits.py --aria-path /data/aria --output splits/windows_5s.jsonl \\
      --metrics-csv metrics/5s/mix_metrics.csv --split-from splits/canonical_v1.jsonl
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
        help=(
            "Path to mix_metrics.csv with ai_energy_ratio. A CSV with several "
            "window_start_sec rows per mix yields a window manifest, any other "
            f"CSV a per-mix manifest (default: {{aria-path}}/{DEFAULT_METRICS_CSV} if present)."
        ),
    )
    parser.add_argument(
        "--split-from",
        default=None,
        help=(
            "Reuse the track to split assignment of an existing manifest (per-mix or "
            "window) instead of splitting. Ignores --seed and the ratios. "
            "Tracks missing from it are an error unless --allow-unassigned is set."
        ),
    )
    parser.add_argument(
        "--allow-unassigned",
        action="store_true",
        help="With --split-from, leave out tracks absent from the reference instead of failing.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.train_ratio + args.val_ratio >= 1.0:
        logger.error("train_ratio + val_ratio must be < 1.0 (test set needs some tracks).")
        sys.exit(1)

    try:
        create_splits(
            aria_path=args.aria_path,
            output_path=args.output,
            codec=args.codec,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            seed=args.seed,
            metrics_csv=args.metrics_csv,
            split_from=args.split_from,
            allow_unassigned=args.allow_unassigned,
        )
    except (ValueError, FileNotFoundError) as e:
        logger.error(str(e))
        sys.exit(1)
