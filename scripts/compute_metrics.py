"""Compute energy metrics (RMS and AI energy ratio) for ARIA stems and mixes.

Writes two CSVs to --output-dir (default {aria_path}/metrics):
    stem_metrics.csv : RMS of every real and AI stem, per file or per window
    mix_metrics.csv  : per mix (and window), AI power, real power and
                       ai_energy_ratio = AI power / total power

Mix metrics are derived from the stem RMS values (sum of stem powers), so no
mix audio is decoded. Stem indices follow the library convention (alphabetical
stem names), which is what mix_S-* file names use.

With --window N, each stem is cut into non-overlapping N-second windows from
t=0; the trailing partial window is dropped, and a stem shorter than N seconds
gives one full-duration row with is_partial=True. Windows where every stem is
silent get an empty ai_energy_ratio (NaN).

Needs the ``metrics`` extra (librosa). The loader is fixed to librosa so the
numbers match the ones used to train the regression model.

Usage:
    python scripts/compute_metrics.py --aria-path /path/to/aria
    python scripts/compute_metrics.py --aria-path /path/to/aria --window 5 --workers 8
    python scripts/compute_metrics.py --aria-path /path/to/aria --codecs encodec_3 encodec_6
"""

import argparse
import os
import sys
from pathlib import Path

from loguru import logger

from aria.constants import DEFAULT_CODEC
from aria.metrics import (
    MIX_COLUMNS,
    STEM_COLUMNS,
    compute_mix_metrics,
    compute_stem_metrics,
    count_stem_errors,
    discover_mix_entries,
    discover_stem_jobs,
    read_stem_csv,
    write_metrics_csv,
)


def compute_metrics(
    aria_path: str | Path,
    output_dir: str | Path | None = None,
    codecs: list[str] | None = None,
    window_sec: float | None = None,
    workers: int = 1,
    overwrite: bool = False,
) -> int:
    """Run both stages and write the CSVs. Returns a process exit code."""
    if window_sec is not None and window_sec <= 0:
        logger.error("--window must be positive.")
        return 2
    if workers < 1:
        logger.error("--workers must be >= 1.")
        return 2
    aria_root = Path(aria_path).expanduser().resolve()
    if not aria_root.is_dir():
        logger.error(f"Dataset not found: {aria_root}")
        return 2

    codecs = list(codecs or [DEFAULT_CODEC])
    out_dir = Path(output_dir) if output_dir else aria_root / "metrics"
    out_dir.mkdir(parents=True, exist_ok=True)
    stem_csv = out_dir / "stem_metrics.csv"
    mix_csv = out_dir / "mix_metrics.csv"

    # Stem stage: the only one that reads audio.
    if stem_csv.exists() and not overwrite:
        logger.info(f"{stem_csv} exists; reusing it (--overwrite to recompute).")
        stem_rows = read_stem_csv(stem_csv)
    else:
        jobs = discover_stem_jobs(aria_root, codecs)
        logger.info(f"Found {len(jobs)} stem files")
        if not jobs:
            logger.error("No stem files found; check --aria-path and --codecs.")
            return 1
        stem_rows = compute_stem_metrics(jobs, window_sec, workers)
        n_errors = count_stem_errors(stem_rows)
        logger.info(f"Measured {len(stem_rows)} stem rows ({n_errors} read errors)")
        write_metrics_csv(stem_csv, stem_rows, STEM_COLUMNS)
        logger.info(f"Wrote {stem_csv}")

    # Mix stage: pure arithmetic on the stem rows.
    mixes = discover_mix_entries(aria_root, codecs)
    logger.info(f"Found {len(mixes)} mixes")
    if not mixes:
        logger.warning("No mixes found; check --codecs.")
    mix_rows = compute_mix_metrics(mixes, stem_rows)
    n_missing = sum(1 for r in mix_rows if r["missing_stems"])
    logger.info(f"Computed {len(mix_rows)} mix rows ({n_missing} with missing stems)")
    write_metrics_csv(mix_csv, mix_rows, MIX_COLUMNS)
    logger.info(f"Wrote {mix_csv}")
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute RMS and AI energy ratio metrics for ARIA stems and mixes",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/compute_metrics.py --aria-path /path/to/aria
  python scripts/compute_metrics.py --aria-path /path/to/aria --window 5 --workers 8
  python scripts/compute_metrics.py --aria-path /path/to/aria --codecs encodec_3 encodec_6
        """,
    )
    parser.add_argument(
        "--aria-path",
        default=os.environ.get("ARIA_ROOT"),
        required="ARIA_ROOT" not in os.environ,
        help="Path to the ARIA dataset (default: $ARIA_ROOT)",
    )
    parser.add_argument(
        "--codecs",
        nargs="+",
        default=[DEFAULT_CODEC],
        help=f"Codec folders to measure (default: {DEFAULT_CODEC})",
    )
    parser.add_argument(
        "--window",
        type=float,
        default=None,
        help="Non-overlapping window length in seconds (default: one row per file)",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Where to write stem_metrics.csv and mix_metrics.csv (default: {aria_path}/metrics)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Worker processes for the stem stage (default: 1)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Recompute stem_metrics.csv even if it exists (default: reuse it)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    return compute_metrics(
        aria_path=args.aria_path,
        output_dir=args.output_dir,
        codecs=args.codecs,
        window_sec=args.window,
        workers=args.workers,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    sys.exit(main())
