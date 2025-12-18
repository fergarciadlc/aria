"""
Create the main metadata CSV file for the ARIA dataset.

This script scans the processed ARIA dataset directory and generates a consolidated
CSV file with track metadata from individual metadata.json files.

Output CSV format:
    id,artist,name,genre,number_of_stems,stems

Usage:
    python create_metadata_csv.py [--aria-path PATH] [--output PATH]
    
Examples:
    # Generate tracks.csv from default ARIA dataset location
    python create_metadata_csv.py
    
    # Use custom paths
    python create_metadata_csv.py --aria-path /path/to/aria --output /path/to/tracks.csv
"""

import argparse
import csv
import json
from pathlib import Path
from typing import Generator

from loguru import logger
from tqdm import tqdm

ARIA_PATH = "/Users/fernando/datasets/aria"
DEFAULT_OUTPUT = "tracks.csv"


def load_metadata(metadata_path: Path) -> dict | None:
    """Load metadata from a JSON file."""
    try:
        with open(metadata_path, "r") as f:
            return json.load(f)
    except (json.JSONDecodeError, FileNotFoundError) as e:
        logger.warning(f"Failed to load metadata from {metadata_path}: {e}")
        return None


def discover_tracks(aria_path: Path) -> Generator[dict, None, None]:
    """
    Discover all tracks in the ARIA dataset by scanning for metadata.json files.
    
    Yields:
        dict: Track metadata for each discovered track
    """
    if not aria_path.exists():
        logger.error(f"ARIA dataset path does not exist: {aria_path}")
        return
    
    # Find all metadata.json files in track directories
    metadata_files = list(aria_path.glob("*/metadata.json"))
    logger.info(f"Found {len(metadata_files)} tracks in {aria_path}")
    
    for metadata_path in tqdm(metadata_files, desc="Loading track metadata"):
        metadata = load_metadata(metadata_path)
        if metadata:
            yield metadata


def format_stems(stems: list) -> str:
    """Format stems list as comma-separated string."""
    if not stems:
        return ""
    return ",".join(stems)


def write_csv(tracks: list[dict], output_path: Path) -> None:
    """Write track metadata to CSV file."""
    fieldnames = ["id", "artist", "name", "genre", "number_of_stems", "stems"]
    
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, quoting=csv.QUOTE_ALL)
        writer.writeheader()
        
        for track in tracks:
            row = {
                "id": track.get("id", ""),
                "artist": track.get("artist", ""),
                "name": track.get("name", ""),
                "genre": track.get("genre", ""),
                "number_of_stems": track.get("number_of_stems", 0),
                "stems": format_stems(track.get("stems", [])),
            }
            writer.writerow(row)
    
    logger.info(f"Wrote {len(tracks)} tracks to {output_path}")


def create_metadata_csv(aria_path: str = ARIA_PATH, output_path: str = DEFAULT_OUTPUT) -> None:
    """
    Create the main metadata CSV file from the ARIA dataset.
    
    Args:
        aria_path: Path to the ARIA dataset directory
        output_path: Path to the output CSV file
    """
    aria_path = Path(aria_path)
    output_path = Path(output_path)
    
    # Discover and collect all tracks
    tracks = list(discover_tracks(aria_path))
    
    if not tracks:
        logger.warning("No tracks found in the ARIA dataset")
        return
    
    # Sort tracks by artist and name for consistent output
    tracks.sort(key=lambda t: (t.get("artist", "").lower(), t.get("name", "").lower()))
    
    # Write to CSV
    write_csv(tracks, output_path)
    
    # Print summary statistics
    genres = {}
    total_stems = 0
    for track in tracks:
        genre = track.get("genre", "unknown")
        genres[genre] = genres.get(genre, 0) + 1
        total_stems += track.get("number_of_stems", 0)
    
    logger.info(f"\nDataset Summary:")
    logger.info(f"  Total tracks: {len(tracks)}")
    logger.info(f"  Total stems: {total_stems}")
    logger.info(f"  Average stems per track: {total_stems / len(tracks):.1f}")
    logger.info(f"  Genres: {len(genres)}")
    for genre, count in sorted(genres.items(), key=lambda x: -x[1]):
        logger.info(f"    - {genre}: {count} tracks")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Create main metadata CSV file for ARIA dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate tracks.csv from default ARIA dataset location
  python create_metadata_csv.py
  
  # Use custom ARIA dataset path
  python create_metadata_csv.py --aria-path /path/to/aria
  
  # Specify custom output file
  python create_metadata_csv.py --output my_tracks.csv
  
  # Both custom paths
  python create_metadata_csv.py --aria-path /path/to/aria --output /path/to/tracks.csv
        """
    )
    
    parser.add_argument(
        "--aria-path",
        type=str,
        default=ARIA_PATH,
        help=f"Path to the ARIA dataset directory (default: {ARIA_PATH})"
    )
    
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=DEFAULT_OUTPUT,
        help=f"Path to the output CSV file (default: {DEFAULT_OUTPUT})"
    )
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    
    create_metadata_csv(
        aria_path=args.aria_path,
        output_path=args.output
    )

