"""
Create the main metadata CSV file for the ARIA dataset.

Loads every track through the aria library and writes one row per track.

Output CSV format:
    id,artist,name,genre,number_of_stems,stems

``stems`` lists the stem names in index order (alphabetical), so position i
is stem index i in mix names.

Usage:
    python scripts/create_metadata_csv.py --aria-path /path/to/aria --output /path/to/tracks.csv
"""

import argparse
import csv
import os
from pathlib import Path

from loguru import logger

from aria.dataset import ARIADataset

DEFAULT_OUTPUT = "tracks.csv"
FIELDNAMES = ["id", "artist", "name", "genre", "number_of_stems", "stems"]


def create_metadata_csv(aria_path: str, output_path: str = DEFAULT_OUTPUT) -> None:
    """
    Create the main metadata CSV file from the ARIA dataset.

    Args:
        aria_path: Path to the ARIA dataset directory
        output_path: Path to the output CSV file
    """
    dataset = ARIADataset(aria_path)
    tracks = sorted(dataset, key=lambda t: (t.artist.lower(), t.name.lower()))

    with open(Path(output_path), "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES, quoting=csv.QUOTE_ALL)
        writer.writeheader()
        for track in tracks:
            writer.writerow({
                "id": track.id,
                "artist": track.artist,
                "name": track.name,
                "genre": track.genre,
                "number_of_stems": track.num_stems,
                "stems": ",".join(track.stem_names),
            })

    logger.info(f"Wrote {len(tracks)} tracks to {output_path}")

    stats = dataset.stats()
    logger.info("Dataset Summary:")
    logger.info(f"  Total tracks: {stats['num_tracks']}")
    logger.info(f"  Total stems: {stats['total_stems']}")
    logger.info(f"  Average stems per track: {stats['avg_stems']:.1f}")
    logger.info(f"  Genres: {len(stats['genres'])}")
    for genre, count in sorted(stats["genres"].items(), key=lambda x: -x[1]):
        logger.info(f"    - {genre}: {count} tracks")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Create main metadata CSV file for ARIA dataset")
    parser.add_argument(
        "--aria-path",
        default=os.environ.get("ARIA_ROOT"),
        required="ARIA_ROOT" not in os.environ,
        help="Path to the ARIA dataset directory (default: $ARIA_ROOT)",
    )
    parser.add_argument(
        "--output", "-o",
        default=DEFAULT_OUTPUT,
        help=f"Path to the output CSV file (default: {DEFAULT_OUTPUT})",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    create_metadata_csv(aria_path=args.aria_path, output_path=args.output)
