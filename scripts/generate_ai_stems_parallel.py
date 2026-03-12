"""Generate AI-reconstructed ARIA stems in parallel using the reference Encodec CLI flow."""

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from multiprocessing import Pool, cpu_count
from pathlib import Path
from typing import List, Optional, Tuple

import torch
import torchaudio
from loguru import logger
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from ae_models.encodec import Encodec


ARIA_PATH = "/Users/fernando/datasets/aria"
SAVE_SAMPLE_RATE = 44100
ENCODEC_BANDWIDTHS = [3, 6, 24]
DEFAULT_BITRATE_KBPS = 256
NUM_WORKERS = 4
REAL_STEM_EXTENSIONS = (".wav", ".mp3")

DEVICE = "cpu"


@dataclass
class ProcessingStats:
    total_tracks: int = 0
    processed_tracks: int = 0
    skipped_tracks: int = 0
    total_stems: int = 0
    written_files: int = 0
    skipped_existing: int = 0
    skipped_short: int = 0
    errors: int = 0


def load_track_metadata(track_path: Path) -> Optional[dict]:
    metadata_path = track_path / "metadata.json"
    if not metadata_path.exists():
        logger.warning(f"Metadata not found: {metadata_path}")
        return None

    try:
        with open(metadata_path, "r") as handle:
            return json.load(handle)
    except Exception as err:
        logger.error(f"Failed to load metadata from {metadata_path}: {err}")
        return None


def get_track_directories(aria_path: Path) -> List[Path]:
    if not aria_path.exists():
        logger.error(f"ARIA path does not exist: {aria_path}")
        return []

    track_dirs = [
        entry for entry in aria_path.iterdir() if entry.is_dir() and (entry / "metadata.json").exists()
    ]
    logger.info(f"Found {len(track_dirs)} tracks in {aria_path}")
    return sorted(track_dirs)


def get_real_stems(track_path: Path) -> List[Path]:
    real_stems_path = track_path / "stems" / "real"
    if not real_stems_path.exists():
        logger.warning(f"Real stems directory not found: {real_stems_path}")
        return []

    stem_files = []
    for extension in REAL_STEM_EXTENSIONS:
        stem_files.extend(real_stems_path.glob(f"*{extension}"))
    return sorted(stem_files)


def ensure_ffmpeg_available() -> None:
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg is required to export MP3 stems but was not found in PATH")


def load_audio(path: Path) -> tuple[torch.Tensor, int]:
    file_format = path.suffix.lstrip(".") or None
    return torchaudio.load(str(path), format=file_format)


def resample_audio(audio: torch.Tensor, source_sr: int, target_sr: int) -> torch.Tensor:
    if source_sr == target_sr:
        return audio
    return torchaudio.functional.resample(audio, source_sr, target_sr)


def normalize_audio_shape(audio: torch.Tensor) -> torch.Tensor:
    audio = torch.squeeze(audio)
    if audio.ndim == 1:
        audio = torch.stack([audio, audio], 0)
    elif audio.ndim == 2 and audio.shape[0] > 2:
        audio = audio[:2, :]
    return audio.contiguous()


def preprocess_audio(
    path: Path,
    save_sr: int,
    min_duration: float,
    max_duration: float,
) -> Optional[torch.Tensor]:
    audio, source_sr = load_audio(path)
    audio = resample_audio(audio, source_sr, save_sr)
    audio = normalize_audio_shape(audio)

    min_len = int(save_sr * min_duration) if min_duration > 0 else 0
    max_len = int(save_sr * max_duration) if max_duration > 0 else 0

    if min_len and audio.shape[-1] < min_len:
        return None
    if max_len and audio.shape[-1] > max_len:
        audio = audio[:, :max_len]
    return audio.contiguous()


def save_mp3(audio: torch.Tensor, output_path: Path, sample_rate: int, bitrate_kbps: int) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    bitrate_kbps = min(int(bitrate_kbps), 320)

    with tempfile.NamedTemporaryFile(suffix=".wav") as tmp_file:
        torchaudio.save(tmp_file.name, audio, sample_rate=sample_rate)
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-hide_banner",
                "-loglevel",
                "error",
                "-i",
                tmp_file.name,
                "-c:a",
                "mp3",
                "-b:a",
                f"{bitrate_kbps}k",
                str(output_path),
            ],
            check=True,
        )


def build_output_paths(stem_path: Path, track_path: Path, bandwidths: List[int]) -> dict[int, Path]:
    stem_name = stem_path.stem
    return {
        bandwidth: track_path / "stems" / "ai_generated" / f"encodec_{bandwidth}" / f"{stem_name}.mp3"
        for bandwidth in bandwidths
    }


def process_stem_with_encodec_worker(
    args: Tuple[Path, Path, List[int], int, str, int, float, float, bool]
) -> Tuple[int, int, int, int, str]:
    stem_path, track_path, bandwidths, save_sr, mode, default_bitrate, min_duration, max_duration, overwrite = args

    out_paths = build_output_paths(stem_path, track_path, bandwidths)
    existing_outputs = sum(1 for output_path in out_paths.values() if output_path.exists())
    if existing_outputs == len(out_paths) and not overwrite:
        return (0, existing_outputs, 0, 0, "")

    try:
        audio = preprocess_audio(
            stem_path,
            save_sr=save_sr,
            min_duration=min_duration,
            max_duration=max_duration,
        )
        if audio is None:
            return (0, 0, 1, 0, "")

        model = Encodec(max(bandwidths), save_sr, DEVICE)
        model_sr = model.processor.sampling_rate

        model_input = audio
        if mode == "corrected":
            model_input = resample_audio(audio, save_sr, model_sr)

        with torch.no_grad():
            rebuilt_list = model.autoencode_multi(model_input, bandwidths)

        written_files = 0
        skipped_existing = existing_outputs

        for bandwidth, rebuilt in zip(bandwidths, rebuilt_list):
            output_path = out_paths[bandwidth]
            if output_path.exists() and not overwrite:
                continue

            rebuilt = normalize_audio_shape(rebuilt.cpu())
            if mode == "corrected":
                rebuilt = resample_audio(rebuilt, model_sr, save_sr)
            rebuilt = rebuilt[:, : audio.shape[-1]].contiguous()
            save_mp3(rebuilt, output_path, save_sr, default_bitrate)
            written_files += 1

        return (written_files, skipped_existing, 0, 0, "")
    except Exception as err:
        return (0, 0, 0, 1, f"{stem_path.name}: {err}")


def collect_all_work_items(
    track_dirs: List[Path],
    bandwidths: List[int],
    save_sr: int,
    mode: str,
    default_bitrate: int,
    min_duration: float,
    max_duration: float,
    overwrite: bool,
) -> List[Tuple[Path, Path, List[int], int, str, int, float, float, bool]]:
    work_items = []

    for track_path in track_dirs:
        metadata = load_track_metadata(track_path)
        if metadata is None:
            continue

        real_stems = get_real_stems(track_path)
        for stem_path in real_stems:
            work_items.append(
                (
                    stem_path,
                    track_path,
                    bandwidths,
                    save_sr,
                    mode,
                    default_bitrate,
                    min_duration,
                    max_duration,
                    overwrite,
                )
            )

    return work_items


def generate_ai_stems_parallel(
    aria_path: str = ARIA_PATH,
    limit: Optional[int] = None,
    bandwidths: Optional[List[int]] = None,
    save_sr: int = SAVE_SAMPLE_RATE,
    mode: str = "replicate",
    default_bitrate: int = DEFAULT_BITRATE_KBPS,
    overwrite: bool = False,
    min_duration: float = 0.0,
    max_duration: float = 0.0,
    num_workers: int = NUM_WORKERS,
) -> None:
    aria_path = Path(aria_path)
    ensure_ffmpeg_available()
    target_bandwidths = bandwidths if bandwidths is not None else ENCODEC_BANDWIDTHS

    max_workers = cpu_count()
    if num_workers > max_workers:
        logger.warning(
            f"Requested {num_workers} workers but only {max_workers} CPUs available. Using {max_workers}."
        )
        num_workers = max_workers

    logger.info("=" * 80)
    logger.info("ARIA AI Stem Generation - Encodec (PARALLEL)")
    logger.info("=" * 80)
    logger.info(f"Dataset path: {aria_path}")
    logger.info(f"Device: CPU (parallel processing)")
    logger.info(f"Workers: {num_workers} (max available: {max_workers})")
    logger.info(f"Bandwidths: {target_bandwidths}")
    logger.info(f"Mode: {mode}")
    logger.info(f"Save sample rate: {save_sr} Hz")
    logger.info(f"Output format: mp3 @ {default_bitrate} kbps")
    logger.info(f"Min duration: {min_duration} s")
    logger.info(f"Max duration: {max_duration if max_duration > 0 else 'disabled'}")

    track_dirs = get_track_directories(aria_path)
    if not track_dirs:
        logger.error("No tracks found. Exiting.")
        return

    if limit is not None:
        logger.info(f"Limiting processing to first {limit} tracks")
        track_dirs = track_dirs[:limit]

    logger.info("Collecting work items...")
    work_items = collect_all_work_items(
        track_dirs,
        target_bandwidths,
        save_sr,
        mode,
        default_bitrate,
        min_duration,
        max_duration,
        overwrite,
    )

    logger.info(f"Total source stems queued: {len(work_items)}")
    if not work_items:
        logger.info("Nothing to process. All stems already generated.")
        return

    stats = ProcessingStats(
        total_tracks=len(track_dirs),
        total_stems=len(work_items),
    )

    logger.info(f"Processing {len(work_items)} source stems with {num_workers} workers...")

    with Pool(processes=num_workers) as pool:
        results = list(tqdm(
            pool.imap(process_stem_with_encodec_worker, work_items),
            total=len(work_items),
            desc="Processing stems"
        ))

    processed_tracks = set()
    skipped_tracks = 0
    for track_path in track_dirs:
        if load_track_metadata(track_path) is None or not get_real_stems(track_path):
            skipped_tracks += 1
        else:
            processed_tracks.add(track_path)

    stats.processed_tracks = len(processed_tracks)
    stats.skipped_tracks = skipped_tracks

    for written_files, skipped_existing, skipped_short, errors, error_msg in results:
        stats.written_files += written_files
        stats.skipped_existing += skipped_existing
        stats.skipped_short += skipped_short
        stats.errors += errors
        if error_msg:
            logger.error(f"Failed: {error_msg}")

    logger.info("=" * 80)
    logger.info("Processing Complete!")
    logger.info("=" * 80)
    logger.info(f"Tracks processed: {stats.processed_tracks}/{stats.total_tracks}")
    logger.info(f"Tracks skipped: {stats.skipped_tracks}")
    logger.info(f"Source stems seen: {stats.total_stems}")
    logger.info(f"Files written: {stats.written_files}")
    logger.info(f"Skipped existing outputs: {stats.skipped_existing}")
    logger.info(f"Skipped short stems: {stats.skipped_short}")
    logger.info(f"Errors: {stats.errors}")
    logger.info("=" * 80)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate AI-reconstructed stems for ARIA in parallel using the reference Encodec CLI flow",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/generate_ai_stems_parallel.py
  python scripts/generate_ai_stems_parallel.py --aria-path /path/to/aria --bandwidths 3 6 24
  python scripts/generate_ai_stems_parallel.py --limit 10 --mode corrected --workers 8
  python scripts/generate_ai_stems_parallel.py --overwrite --save-sr 44100
        """
    )

    parser.add_argument("--aria-path", type=str, default=ARIA_PATH, help=f"Path to the ARIA dataset (default: {ARIA_PATH})")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of tracks to process")
    parser.add_argument(
        "--bandwidths",
        nargs="+",
        type=int,
        default=ENCODEC_BANDWIDTHS,
        choices=[3, 6, 12, 24],
        help=f"Encodec bandwidths to use in kbps (default: {ENCODEC_BANDWIDTHS})",
    )
    parser.add_argument("--save-sr", type=int, default=SAVE_SAMPLE_RATE, help=f"Output sample rate (default: {SAVE_SAMPLE_RATE})")
    parser.add_argument(
        "--mode",
        choices=["replicate", "corrected"],
        default="replicate",
        help="Reference processing mode to use (default: replicate)",
    )
    parser.add_argument(
        "--default-bitrate",
        type=int,
        default=DEFAULT_BITRATE_KBPS,
        help=f"MP3 bitrate to use when writing outputs (default: {DEFAULT_BITRATE_KBPS})",
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
        default=NUM_WORKERS,
        help=f"Number of parallel worker processes (default: {NUM_WORKERS}, max: CPU count)",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    
    generate_ai_stems_parallel(
        aria_path=args.aria_path,
        limit=args.limit,
        bandwidths=args.bandwidths,
        save_sr=args.save_sr,
        mode=args.mode,
        default_bitrate=args.default_bitrate,
        overwrite=args.overwrite,
        min_duration=args.min_duration,
        max_duration=args.max_duration,
        num_workers=args.workers,
    )
