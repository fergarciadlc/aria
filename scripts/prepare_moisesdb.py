"""
Prepare MoisesDB dataset for ARIA by exporting stems and creating metadata.

This script processes tracks from the MoisesDB dataset and:
1. Exports real stems for each track
2. Creates metadata files with track information

The output structure follows the ARIA specification:
    {aria_path}/{track_id}/stems/real/{stem_name}.wav
    {aria_path}/{track_id}/metadata.json

Usage:
    python prepare_moisesdb.py [--aria-path PATH] [--moisesdb-path PATH] [--limit N]
    
Examples:
    # Process all tracks
    python prepare_moisesdb.py
    
    # Process first 10 tracks only (testing)
    python prepare_moisesdb.py --limit 10
    
    # Use custom dataset paths
    python prepare_moisesdb.py --aria-path /path/to/aria --moisesdb-path /path/to/moisesdb
"""

import argparse
import json
from dataclasses import dataclass, asdict
from typing import List, Optional
from uuid import uuid4

from loguru import logger
from moisesdb.dataset import MoisesDB
from moisesdb.track import MoisesDBTrack as Track
from tqdm import tqdm

MOISESDB_PATH = "/Users/fernando/datasets/moisesdb"
ARIA_PATH = "/Users/fernando/datasets/aria"
ARIA_STEMS_PATH = "{aria_path}/{track_id}/stems/real"
SAMPLE_RATE = 44100


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
    stems: List[str] = None
    additional_info: dict = None


def get_moisesdb(data_path: str = MOISESDB_PATH, sample_rate: int = SAMPLE_RATE) -> MoisesDB:
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


def export_moisesdb_track_stems(track: Track, stems_path: str) -> None:
    track.save_stems(stems_path)


def process_mosiesdb(aria_path: str = ARIA_PATH, moisesdb_path: str = MOISESDB_PATH, limit: Optional[int] = None, sample_rate: int = SAMPLE_RATE):
    """Process MoisesDB tracks and export to ARIA format.
    
    Args:
        aria_path: Path to the ARIA dataset output directory
        moisesdb_path: Path to the MoisesDB dataset
        limit: Maximum number of tracks to process (None for all tracks)
        sample_rate: Sample rate for exported stems (default: 44100)
    """
    db = get_moisesdb(data_path=moisesdb_path, sample_rate=sample_rate)
    total_tracks = len(db)
    tracks_to_process = min(limit, total_tracks) if limit else total_tracks
    
    logger.info(f"Preparing MoisesDB dataset: processing {tracks_to_process} of {total_tracks} tracks.")
    logger.info(f"Sample rate: {sample_rate} Hz")

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
        export_moisesdb_track_stems(track, stems_export_path)

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
  # Process all tracks
  python prepare_moisesdb.py
  
  # Process first 10 tracks only (testing)
  python prepare_moisesdb.py --limit 10
  
  # Use custom ARIA dataset path
  python prepare_moisesdb.py --aria-path /path/to/aria
  
  # Use custom MoisesDB path
  python prepare_moisesdb.py --moisesdb-path /path/to/moisesdb
        """
    )
    
    parser.add_argument(
        "--aria-path",
        type=str,
        default=ARIA_PATH,
        help=f"Path to the ARIA dataset output directory (default: {ARIA_PATH})"
    )
    
    parser.add_argument(
        "--moisesdb-path",
        type=str,
        default=MOISESDB_PATH,
        help=f"Path to the MoisesDB dataset (default: {MOISESDB_PATH})"
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
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    
    process_mosiesdb(
        aria_path=args.aria_path,
        moisesdb_path=args.moisesdb_path,
        limit=args.limit,
        sample_rate=args.sample_rate
    )
