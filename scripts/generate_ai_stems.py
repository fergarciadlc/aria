"""Generate AI-reconstructed ARIA stems using the reference Encodec CLI flow."""

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

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
REAL_STEM_EXTENSIONS = (".wav", ".mp3")


if torch.cuda.is_available():
    DEVICE = "cuda"
    logger.info("Using CUDA GPU")
elif torch.backends.mps.is_available():
    DEVICE = "cpu"
    logger.warning("MPS detected but using CPU (Encodec has MPS compatibility issues)")
else:
    DEVICE = "cpu"
    logger.info("Using CPU")


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


def initialize_encodec_model(bandwidths: List[int], save_sr: int) -> Encodec:
    logger.info("Initializing Encodec model...")
    max_bandwidth = max(bandwidths)
    logger.info(f"Loading Encodec model (max bandwidth={max_bandwidth} kbps)")
    return Encodec(max_bandwidth, save_sr, DEVICE)


def process_track(
    track_path: Path,
    model: Encodec,
    bandwidths: List[int],
    save_sr: int,
    mode: str,
    default_bitrate: int,
    overwrite: bool,
    min_duration: float,
    max_duration: float,
    stats: ProcessingStats,
) -> None:
    metadata = load_track_metadata(track_path)
    if metadata is None:
        stats.skipped_tracks += 1
        return

    real_stems = get_real_stems(track_path)
    if not real_stems:
        logger.warning(f"No real stems found for track {track_path.name}")
        stats.skipped_tracks += 1
        return

    stats.total_stems += len(real_stems)
    stats.processed_tracks += 1

    track_name = f"{metadata.get('artist', 'Unknown')} - {metadata.get('name', 'Unknown')}"
    logger.debug(
        f"Processing track [{metadata.get('id', 'NONE')}]: {track_name} with {len(real_stems)} stems"
    )

    model_sr = model.processor.sampling_rate

    for stem_path in real_stems:
        out_paths = build_output_paths(stem_path, track_path, bandwidths)
        if not overwrite and all(path.exists() for path in out_paths.values()):
            stats.skipped_existing += len(out_paths)
            continue

        try:
            audio = preprocess_audio(
                stem_path,
                save_sr=save_sr,
                min_duration=min_duration,
                max_duration=max_duration,
            )
            if audio is None:
                stats.skipped_short += 1
                continue

            model_input = audio
            if mode == "corrected":
                model_input = resample_audio(audio, save_sr, model_sr)

            with torch.no_grad():
                rebuilt_list = model.autoencode_multi(model_input, bandwidths)

            for bandwidth, rebuilt in zip(bandwidths, rebuilt_list):
                output_path = out_paths[bandwidth]
                if output_path.exists() and not overwrite:
                    stats.skipped_existing += 1
                    continue

                rebuilt = normalize_audio_shape(rebuilt.cpu())
                if mode == "corrected":
                    rebuilt = resample_audio(rebuilt, model_sr, save_sr)
                rebuilt = rebuilt[:, : audio.shape[-1]].contiguous()
                save_mp3(rebuilt, output_path, save_sr, default_bitrate)
                stats.written_files += 1
        except Exception as err:
            stats.errors += 1
            logger.error(f"Failed to process {stem_path.name}: {err}")


def generate_ai_stems(
    aria_path: str = ARIA_PATH,
    limit: Optional[int] = None,
    bandwidths: Optional[List[int]] = None,
    save_sr: int = SAVE_SAMPLE_RATE,
    mode: str = "replicate",
    default_bitrate: int = DEFAULT_BITRATE_KBPS,
    overwrite: bool = False,
    min_duration: float = 0.0,
    max_duration: float = 0.0,
) -> None:
    aria_root = Path(aria_path)
    ensure_ffmpeg_available()
    target_bandwidths = bandwidths if bandwidths is not None else ENCODEC_BANDWIDTHS

    logger.info("=" * 80)
    logger.info("ARIA AI Stem Generation - Encodec")
    logger.info("=" * 80)
    logger.info(f"Dataset path: {aria_root}")
    logger.info(f"Device: {DEVICE}")
    logger.info(f"Bandwidths: {target_bandwidths}")
    logger.info(f"Mode: {mode}")
    logger.info(f"Save sample rate: {save_sr} Hz")
    logger.info(f"Output format: mp3 @ {default_bitrate} kbps")
    logger.info(f"Min duration: {min_duration} s")
    logger.info(f"Max duration: {max_duration if max_duration > 0 else 'disabled'}")

    track_dirs = get_track_directories(aria_root)
    if not track_dirs:
        logger.error("No tracks found. Exiting.")
        return

    if limit is not None:
        logger.info(f"Limiting processing to first {limit} tracks")
        track_dirs = track_dirs[:limit]

    stats = ProcessingStats(total_tracks=len(track_dirs))
    model = initialize_encodec_model(target_bandwidths, save_sr)

    logger.info(f"Processing {len(track_dirs)} tracks...")
    progress = tqdm(track_dirs, desc="Processing tracks")

    for track_path in progress:
        metadata = load_track_metadata(track_path)
        if metadata:
            track_name = f"{metadata.get('artist', 'Unknown')} - {metadata.get('name', 'Unknown')}"
            progress.set_postfix_str(track_name[:50])

        process_track(
            track_path=track_path,
            model=model,
            bandwidths=target_bandwidths,
            save_sr=save_sr,
            mode=mode,
            default_bitrate=default_bitrate,
            overwrite=overwrite,
            min_duration=min_duration,
            max_duration=max_duration,
            stats=stats,
        )

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
        description="Generate AI-reconstructed stems for ARIA using the reference Encodec CLI flow",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/generate_ai_stems.py
  python scripts/generate_ai_stems.py --aria-path /path/to/aria --bandwidths 3 6 24
  python scripts/generate_ai_stems.py --limit 10 --mode corrected
  python scripts/generate_ai_stems.py --overwrite --save-sr 44100
        """,
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

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    generate_ai_stems(
        aria_path=args.aria_path,
        limit=args.limit,
        bandwidths=args.bandwidths,
        save_sr=args.save_sr,
        mode=args.mode,
        default_bitrate=args.default_bitrate,
        overwrite=args.overwrite,
        min_duration=args.min_duration,
        max_duration=args.max_duration,
    )

