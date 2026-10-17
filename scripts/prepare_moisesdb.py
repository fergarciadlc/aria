"""
Prepare MoisesDB dataset for ARIA by exporting stems and creating metadata.

This script processes tracks from the MoisesDB dataset and:
1. Exports real stems for each track
2. Creates metadata files with track information

The output structure follows the ARIA specification:
    {aria_path}/{track_id}/stems/real/{stem_name}.wav
    {aria_path}/{track_id}/metadata.json

Usage:
    python scripts/prepare_moisesdb.py --aria-path PATH --moisesdb-path PATH [--limit N]

Paths default to the ARIA_ROOT and MOISESDB_ROOT environment variables.
"""

import argparse
import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from uuid import uuid4

from loguru import logger
from moisesdb.dataset import MoisesDB
from moisesdb.track import MoisesDBTrack as Track
from moisesdb.utils import save_audio as save_wav
from tqdm import tqdm

from aria.audio import save_audio
from aria.constants import MP3_BITRATE_KBPS, SAMPLE_RATE

ARIA_STEMS_PATH = "{aria_path}/{track_id}/stems/real"


def save_json(filepath: str, data: dict) -> None:
    with open(filepath, "w") as f:
        json.dump(data, f)


@dataclass
class AriaTrackMetadata:
    id: str
    artist: str
    name: str
    genre: str
    provider: str
    number_of_stems: int = 0
    stems: list[str] = field(default_factory=list)
    additional_info: dict = field(default_factory=dict)


def get_moisesdb(data_path: str, sample_rate: int = SAMPLE_RATE) -> MoisesDB:
    return MoisesDB(data_path=data_path, sample_rate=sample_rate)


def get_aria_track_metadata(track: Track) -> AriaTrackMetadata:
    stem_names = [s["stemName"] for s in track.json_data.get("stems", [])]
    unique_stem_names = list(dict.fromkeys(stem_names))

    return AriaTrackMetadata(
        id=str(uuid4()),
        artist=track.artist,
        name=track.name,
        genre=track.genre,
        provider=track.provider,
        number_of_stems=len(unique_stem_names),
        stems=unique_stem_names,
        additional_info={
            "moisesdb_id": track.id,
        },
    )


def save_stems_with_format(track: Track, stems_path: str, format: str = "wav") -> None:
    """Save track stems in the specified format.
    
    Args:
        track: MoisesDB track to export
        stems_path: Directory path where stems will be saved
        format: Audio format ('wav' or 'mp3')
    """
    os.makedirs(stems_path, exist_ok=True)
    
    for stem_name, audio in track.stems.items():
        if audio is None:
            continue
        if format == "wav":
            save_wav(os.path.join(stems_path, f"{stem_name}.wav"), audio, sr=track.sr)
        elif format == "mp3":
            # audio shape is (channels, samples)
            save_audio(Path(stems_path) / f"{stem_name}.mp3", audio, track.sr, mp3_bitrate=MP3_BITRATE_KBPS)
        else:
            raise ValueError(f"Unsupported format: {format}. Use 'wav' or 'mp3'.")


def process_moisesdb(aria_path: str, moisesdb_path: str, limit: int | None = None, sample_rate: int = SAMPLE_RATE, format: str = "wav"):
    """Process MoisesDB tracks and export to ARIA format.
    
    Args:
        aria_path: Path to the ARIA dataset output directory
        moisesdb_path: Path to the MoisesDB dataset
        limit: Maximum number of tracks to process (None for all tracks)
        sample_rate: Sample rate for exported stems (default: 44100)
        format: Audio format for exported stems ('wav' or 'mp3', default: 'wav')
    """
    db = get_moisesdb(data_path=moisesdb_path, sample_rate=sample_rate)
    total_tracks = len(db)
    tracks_to_process = min(limit, total_tracks) if limit else total_tracks
    
    logger.info(f"Preparing MoisesDB dataset: processing {tracks_to_process} of {total_tracks} tracks.")
    logger.info(f"Sample rate: {sample_rate} Hz")
    logger.info(f"Output format: {format.upper()}" + (f" @ {MP3_BITRATE_KBPS} kbps" if format == "mp3" else ""))

    counter = 0
    pbar = tqdm(db, total=tracks_to_process, desc="Processing MoisesDB tracks")
    for track in pbar:
        if limit and counter >= limit:
            break
        
        metadata = get_aria_track_metadata(track)
        pbar.set_postfix_str(f"{metadata.artist} - {metadata.name}")

        # Export stems
        stems_export_path = ARIA_STEMS_PATH.format(
            aria_path=aria_path, track_id=metadata.id
        )
        save_stems_with_format(track, stems_export_path, format=format)

        # Save track metadata
        save_json(f"{aria_path}/{metadata.id}/metadata.json", asdict(metadata))
        counter += 1
    
    logger.info(f"MoisesDB dataset preparation completed. Processed {counter} tracks.")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Prepare MoisesDB dataset for ARIA",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process all tracks (default WAV format)
  python scripts/prepare_moisesdb.py --aria-path /path/to/aria --moisesdb-path /path/to/moisesdb

  # Process first 10 tracks only (testing), MP3 output
  python scripts/prepare_moisesdb.py --aria-path /path/to/aria --moisesdb-path /path/to/moisesdb --limit 10 --format mp3
        """
    )
    
    parser.add_argument(
        "--aria-path",
        default=os.environ.get("ARIA_ROOT"),
        required="ARIA_ROOT" not in os.environ,
        help="Path to the ARIA dataset output directory (default: $ARIA_ROOT)"
    )
    
    parser.add_argument(
        "--moisesdb-path",
        default=os.environ.get("MOISESDB_ROOT"),
        required="MOISESDB_ROOT" not in os.environ,
        help="Path to the MoisesDB dataset (default: $MOISESDB_ROOT)"
    )
    
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of tracks to process (useful for testing)"
    )
    
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=SAMPLE_RATE,
        help=f"Sample rate for exported stems in Hz (default: {SAMPLE_RATE})"
    )
    
    parser.add_argument(
        "--format",
        type=str,
        choices=["wav", "mp3"],
        default="wav",
        help=f"Audio format for exported stems: 'wav' (original) or 'mp3' ({MP3_BITRATE_KBPS} kbps, default: wav)"
    )
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    
    process_moisesdb(
        aria_path=args.aria_path,
        moisesdb_path=args.moisesdb_path,
        limit=args.limit,
        sample_rate=args.sample_rate,
        format=args.format
    )
