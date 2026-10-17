"""Generate AI-reconstructed ARIA stems with Encodec.

For every real stem of every track, encodes the stem once and writes one
reconstruction per bandwidth to:
    {aria_path}/{track_id}/stems/ai_generated/encodec_{bandwidth}/{stem_name}.mp3

Processing mode, sample rate and bitrate defaults reproduce the released
dataset; see aria/generators/encodec.py for what "replicate" means.

Requires the ``generators`` extra (torch, transformers) and ffmpeg.
"""

import argparse
import os
from dataclasses import dataclass
from multiprocessing import Pool, cpu_count
from pathlib import Path

from loguru import logger
from tqdm import tqdm

from aria.audio import ffmpeg_available, load_audio, save_audio
from aria.constants import ENCODEC_BANDWIDTHS, MP3_BITRATE_KBPS, SAMPLE_RATE
from aria.exceptions import MissingStemError
from aria.generators.encodec import EncodecGenerator, detect_device
from aria.track import get_stem_dir, get_stem_path, iter_track_dirs, load_track


@dataclass
class ProcessingStats:
    total_tracks: int = 0
    total_stems: int = 0
    written_files: int = 0
    skipped_existing: int = 0
    skipped_short: int = 0
    errors: int = 0


@dataclass(frozen=True)
class StemJob:
    stem_path: Path
    out_paths: dict[int, Path]


@dataclass(frozen=True)
class JobSettings:
    save_sr: int
    bitrate: int
    min_duration: float
    max_duration: float
    overwrite: bool


@dataclass
class JobResult:
    written: int = 0
    skipped_existing: int = 0
    skipped_short: int = 0
    error: str = ""


def collect_jobs(track_dirs: list[Path], bandwidths: list[int], stats: ProcessingStats) -> list[StemJob]:
    """One job per real stem, with its output path for every bandwidth."""
    jobs = []
    for track_dir in track_dirs:
        track = load_track(track_dir)
        for stem_name in track.stem_names:
            try:
                stem_path = get_stem_path(track, stem_name, "real")
            except MissingStemError as err:
                logger.warning(str(err))
                continue
            out_paths = {
                bandwidth: get_stem_dir(track, "ai_generated", f"encodec_{bandwidth}") / f"{stem_name}.mp3"
                for bandwidth in bandwidths
            }
            jobs.append(StemJob(stem_path, out_paths))
            stats.total_stems += 1
    return jobs


# Worker state: one model per process, loaded by init_worker.
_generator: EncodecGenerator | None = None
_settings: JobSettings | None = None


def init_worker(mode: str, device: str, settings: JobSettings) -> None:
    global _generator, _settings
    _generator = EncodecGenerator(mode=mode, device=device)
    _settings = settings


def process_stem(job: StemJob) -> JobResult:
    assert _generator is not None and _settings is not None
    settings = _settings
    result = JobResult()

    pending = {
        bandwidth: path
        for bandwidth, path in job.out_paths.items()
        if settings.overwrite or not path.exists()
    }
    result.skipped_existing = len(job.out_paths) - len(pending)
    if not pending:
        return result

    try:
        audio, sr = load_audio(job.stem_path, target_sr=settings.save_sr, stereo=True)

        if settings.min_duration > 0 and audio.shape[-1] < int(sr * settings.min_duration):
            result.skipped_short = 1
            return result
        if settings.max_duration > 0:
            audio = audio[:, : int(sr * settings.max_duration)]

        rebuilt = _generator.process_multi(audio, sr, list(pending))
        for bandwidth, path in pending.items():
            save_audio(path, rebuilt[bandwidth], sr, mp3_bitrate=settings.bitrate)
            result.written += 1
    except Exception as err:
        result.error = f"{job.stem_path}: {err}"

    return result


def generate_ai_stems(
    aria_path: str,
    limit: int | None = None,
    bandwidths: list[int] | None = None,
    save_sr: int = SAMPLE_RATE,
    mode: str = "replicate",
    bitrate: int = MP3_BITRATE_KBPS,
    overwrite: bool = False,
    min_duration: float = 0.0,
    max_duration: float = 0.0,
    num_workers: int = 1,
) -> None:
    if not ffmpeg_available():
        raise RuntimeError("ffmpeg is required to export MP3 stems but was not found in PATH")

    bandwidths = list(bandwidths or ENCODEC_BANDWIDTHS)
    device = detect_device()
    if num_workers > cpu_count():
        logger.warning(f"Requested {num_workers} workers but only {cpu_count()} CPUs available.")
        num_workers = cpu_count()

    logger.info("=" * 80)
    logger.info("ARIA AI Stem Generation - Encodec")
    logger.info("=" * 80)
    logger.info(f"Dataset path: {aria_path}")
    logger.info(f"Device: {device}")
    logger.info(f"Workers: {num_workers}")
    logger.info(f"Bandwidths: {bandwidths}")
    logger.info(f"Mode: {mode}")
    logger.info(f"Save sample rate: {save_sr} Hz")
    logger.info(f"Output format: mp3 @ {bitrate} kbps")
    logger.info(f"Min duration: {min_duration} s")
    logger.info(f"Max duration: {max_duration if max_duration > 0 else 'disabled'}")

    track_dirs = iter_track_dirs(aria_path)
    if limit is not None:
        track_dirs = track_dirs[:limit]
    if not track_dirs:
        logger.error("No tracks found. Exiting.")
        return

    stats = ProcessingStats(total_tracks=len(track_dirs))
    jobs = collect_jobs(track_dirs, bandwidths, stats)
    settings = JobSettings(save_sr, bitrate, min_duration, max_duration, overwrite)

    logger.info(f"Processing {len(jobs)} stems from {len(track_dirs)} tracks...")
    if num_workers == 1:
        init_worker(mode, device, settings)
        results = (process_stem(job) for job in jobs)
        results = list(tqdm(results, total=len(jobs), desc="Processing stems"))
    else:
        with Pool(num_workers, initializer=init_worker, initargs=(mode, device, settings)) as pool:
            results = list(tqdm(pool.imap(process_stem, jobs), total=len(jobs), desc="Processing stems"))

    for result in results:
        stats.written_files += result.written
        stats.skipped_existing += result.skipped_existing
        stats.skipped_short += result.skipped_short
        if result.error:
            stats.errors += 1
            logger.error(f"Failed: {result.error}")

    logger.info("=" * 80)
    logger.info("Processing Complete!")
    logger.info("=" * 80)
    logger.info(f"Tracks: {stats.total_tracks}")
    logger.info(f"Source stems seen: {stats.total_stems}")
    logger.info(f"Files written: {stats.written_files}")
    logger.info(f"Skipped existing outputs: {stats.skipped_existing}")
    logger.info(f"Skipped short stems: {stats.skipped_short}")
    logger.info(f"Errors: {stats.errors}")
    logger.info("=" * 80)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate AI-reconstructed stems for ARIA with Encodec",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/generate_ai_stems.py --aria-path /path/to/aria
  python scripts/generate_ai_stems.py --aria-path /path/to/aria --bandwidths 3 6 24
  python scripts/generate_ai_stems.py --aria-path /path/to/aria --limit 10 --mode corrected
  python scripts/generate_ai_stems.py --aria-path /path/to/aria --workers 8   # CPU only
        """,
    )

    parser.add_argument(
        "--aria-path",
        default=os.environ.get("ARIA_ROOT"),
        required="ARIA_ROOT" not in os.environ,
        help="Path to the ARIA dataset (default: $ARIA_ROOT)",
    )
    parser.add_argument("--limit", type=int, default=None, help="Limit number of tracks to process")
    parser.add_argument(
        "--bandwidths",
        nargs="+",
        type=int,
        default=list(ENCODEC_BANDWIDTHS),
        choices=list(EncodecGenerator.SUPPORTED_BANDWIDTHS),
        help=f"Encodec bandwidths to use in kbps (default: {list(ENCODEC_BANDWIDTHS)})",
    )
    parser.add_argument("--save-sr", type=int, default=SAMPLE_RATE, help=f"Output sample rate (default: {SAMPLE_RATE})")
    parser.add_argument(
        "--mode",
        choices=["replicate", "corrected"],
        default="replicate",
        help="Encodec processing mode (default: replicate, as in the released dataset)",
    )
    parser.add_argument(
        "--bitrate",
        "--default-bitrate",
        type=int,
        default=MP3_BITRATE_KBPS,
        help=f"MP3 bitrate in kbps (default: {MP3_BITRATE_KBPS})",
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing generated stems")
    parser.add_argument(
        "--min-duration",
        type=float,
        default=0.0,
        help="Minimum duration in seconds; shorter stems are skipped (default: disabled)",
    )
    parser.add_argument(
        "--max-duration",
        type=float,
        default=0.0,
        help="Maximum duration in seconds; longer stems are truncated (default: disabled)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Worker processes, each with its own model (default: 1). Use >1 only on CPU.",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    generate_ai_stems(
        aria_path=args.aria_path,
        limit=args.limit,
        bandwidths=args.bandwidths,
        save_sr=args.save_sr,
        mode=args.mode,
        bitrate=args.bitrate,
        overwrite=args.overwrite,
        min_duration=args.min_duration,
        max_duration=args.max_duration,
        num_workers=args.workers,
    )
