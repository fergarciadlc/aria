"""
Generate all possible mixes combining real and AI-generated stems.

This script processes tracks from the ARIA dataset (after prepare_moisesdb.py and generate_ai_stems.py)
and creates all 2^n combinations of real and AI-generated stems, where n is the number of stems per track.
Mixing, naming and stem order come from the aria library (aria.mix).

The output structure follows the ARIA specification:
    {aria_path}/{track_id}/mixes/{codec}/Uk/mix_S-{indices}.mp3
    {aria_path}/{track_id}/mixes/{codec}/Uk/mix_S-{indices}.json

Where:
    - codec = AI model used (e.g., encodec_3, encodec_6)
    - Uk = folder for mixes with exactly k AI-generated stems
    - indices = hyphen-separated indices into the track's alphabetically sorted stem names
    - "empty" = special case for all-real stems (U0)

For a track with 4 stems using encodec_3, this generates 2^4 = 16 mixes:
    - mixes/encodec_3/U0/mix_S-empty.mp3 (all real)
    - mixes/encodec_3/U1/mix_S-0.mp3, mix_S-1.mp3, mix_S-2.mp3, mix_S-3.mp3
    - mixes/encodec_3/U2/mix_S-0-1.mp3, mix_S-0-2.mp3, ...
    - mixes/encodec_3/U3/mix_S-0-1-2.mp3, ...
    - mixes/encodec_3/U4/mix_S-0-1-2-3.mp3 (all AI-generated)

Usage:
    python scripts/generate_mixes.py --aria-path PATH [--limit N] [--codec encodec_3] [--workers N]
"""

import argparse
import json
import os
from dataclasses import dataclass
from multiprocessing import Pool, cpu_count
from pathlib import Path

from loguru import logger
from tqdm import tqdm

from aria.audio import ffmpeg_available, save_audio
from aria.constants import DEFAULT_CODEC, MIX_FORMAT, MP3_BITRATE_KBPS, SAMPLE_RATE
from aria.mix import compute_mix, generate_all_mix_specs, get_mix_path, mix_metadata
from aria.track import get_available_codecs, has_all_stems, iter_track_dirs, load_track


@dataclass
class ProcessingStats:
    """Track processing statistics."""
    total_tracks: int = 0
    processed_tracks: int = 0
    skipped_tracks: int = 0
    total_mixes: int = 0
    generated_mixes: int = 0
    skipped_mixes: int = 0
    failed_mixes: int = 0

    def merge(self, other: "ProcessingStats") -> None:
        """Accumulate per-track statistics into these totals."""
        self.processed_tracks += other.processed_tracks
        self.skipped_tracks += other.skipped_tracks
        self.total_mixes += other.total_mixes
        self.generated_mixes += other.generated_mixes
        self.skipped_mixes += other.skipped_mixes
        self.failed_mixes += other.failed_mixes


def process_track(track_path: Path, codec: str, force_regenerate: bool = False) -> ProcessingStats:
    """
    Generate all possible mixes for one track.

    Args:
        track_path: Path to the track directory
        codec: Codec identifier (e.g., "encodec_3")
        force_regenerate: If True, regenerate even if mix already exists
    """
    stats = ProcessingStats()
    track = load_track(track_path)

    if not has_all_stems(track, "real"):
        logger.warning(f"Missing real stems for {track.display_name}")
        stats.skipped_tracks += 1
        return stats

    if codec not in get_available_codecs(track) or not has_all_stems(track, "ai_generated", codec):
        logger.warning(f"Missing AI-generated stems ({codec}) for {track.display_name}")
        stats.skipped_tracks += 1
        return stats

    specs = generate_all_mix_specs(track.num_stems, codec)
    stats.total_mixes += len(specs)

    logger.debug(f"Processing {track.display_name}: {track.num_stems} stems, {len(specs)} mixes")

    for spec in specs:
        output_path = get_mix_path(track, spec, format=MIX_FORMAT)

        if output_path.exists() and not force_regenerate:
            stats.skipped_mixes += 1
            continue

        try:
            mixed_audio = compute_mix(track, spec, target_sr=SAMPLE_RATE)
            save_audio(output_path, mixed_audio, SAMPLE_RATE, mp3_bitrate=MP3_BITRATE_KBPS)
            with open(output_path.with_suffix(".json"), "w") as f:
                json.dump(mix_metadata(track, spec), f, indent=2)
            stats.generated_mixes += 1
        except Exception as e:
            logger.error(f"Failed to generate {spec.filename} for {track.display_name}: {e}")
            stats.failed_mixes += 1

    stats.processed_tracks += 1
    return stats


def process_track_worker(args: tuple[Path, str, bool]) -> ProcessingStats:
    """Worker wrapper for multiprocessing."""
    return process_track(*args)


def generate_mixes(
    aria_path: str,
    limit: int | None = None,
    codec: str = DEFAULT_CODEC,
    force_regenerate: bool = False,
    num_workers: int = 1,
) -> None:
    """
    Generate all mixes for every track.

    Args:
        aria_path: Path to the ARIA dataset
        limit: Optional limit on number of tracks to process (for testing)
        codec: Codec of the AI-generated stems (e.g., "encodec_3", "encodec_6")
        force_regenerate: If True, regenerate mixes even if they already exist
        num_workers: Number of worker processes (one track per task)
    """
    if not ffmpeg_available():
        raise RuntimeError("ffmpeg is required to export MP3 mixes but was not found in PATH")

    if num_workers > cpu_count():
        logger.warning(f"Requested {num_workers} workers but only {cpu_count()} CPUs available.")
        num_workers = cpu_count()

    logger.info("=" * 80)
    logger.info("ARIA Mix Generation")
    logger.info("=" * 80)
    logger.info(f"Dataset path: {aria_path}")
    logger.info(f"AI codec: {codec}")
    logger.info(f"Sample rate: {SAMPLE_RATE} Hz")
    logger.info(f"Output format: {MIX_FORMAT} @ {MP3_BITRATE_KBPS} kbps")
    logger.info(f"Force regenerate: {force_regenerate}")
    logger.info(f"Workers: {num_workers}")

    track_dirs = iter_track_dirs(aria_path)
    if limit is not None:
        logger.info(f"Limiting processing to first {limit} tracks")
        track_dirs = track_dirs[:limit]
    if not track_dirs:
        logger.error("No tracks found. Exiting.")
        return

    stats = ProcessingStats(total_tracks=len(track_dirs))
    work_items = [(track_path, codec, force_regenerate) for track_path in track_dirs]

    logger.info(f"Processing {len(track_dirs)} tracks...")
    if num_workers == 1:
        for item in tqdm(work_items, desc="Generating mixes"):
            stats.merge(process_track_worker(item))
    else:
        with Pool(processes=num_workers) as pool:
            for track_stats in tqdm(
                pool.imap_unordered(process_track_worker, work_items),
                total=len(work_items),
                desc="Generating mixes",
            ):
                stats.merge(track_stats)

    logger.info("=" * 80)
    logger.info("Processing Complete!")
    logger.info("=" * 80)
    logger.info(f"Tracks processed: {stats.processed_tracks}/{stats.total_tracks}")
    logger.info(f"Tracks skipped: {stats.skipped_tracks}")
    logger.info(f"Mixes generated: {stats.generated_mixes}/{stats.total_mixes}")
    logger.info(f"Mixes skipped (already exist): {stats.skipped_mixes}")
    logger.info(f"Mixes failed: {stats.failed_mixes}")
    logger.info("=" * 80)


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate all possible mixes combining real and AI-generated stems",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process all tracks with the default codec (encodec_3)
  python scripts/generate_mixes.py --aria-path /path/to/aria

  # Process first 10 tracks only (testing)
  python scripts/generate_mixes.py --aria-path /path/to/aria --limit 10

  # Use a specific codec, 8 worker processes
  python scripts/generate_mixes.py --aria-path /path/to/aria --codec encodec_6 --workers 8

  # Force regeneration of existing mixes
  python scripts/generate_mixes.py --aria-path /path/to/aria --force
        """,
    )

    parser.add_argument(
        "--aria-path",
        default=os.environ.get("ARIA_ROOT"),
        required="ARIA_ROOT" not in os.environ,
        help="Path to the ARIA dataset (default: $ARIA_ROOT)",
    )
    parser.add_argument("--limit", type=int, default=None, help="Limit number of tracks to process (useful for testing)")
    parser.add_argument(
        "--codec",
        "--model",
        default=DEFAULT_CODEC,
        help=f"Codec of the AI stems (e.g., encodec_3, encodec_6, encodec_24) (default: {DEFAULT_CODEC})",
    )
    parser.add_argument("--force", action="store_true", help="Force regeneration of mixes even if they already exist")
    parser.add_argument("--workers", type=int, default=1, help="Number of parallel worker processes (default: 1)")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    generate_mixes(
        aria_path=args.aria_path,
        limit=args.limit,
        codec=args.codec,
        force_regenerate=args.force,
        num_workers=args.workers,
    )
