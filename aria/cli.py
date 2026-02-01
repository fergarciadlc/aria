"""
Command-line interface for ARIA dataset tools.

Usage:
    python -m aria.cli package --source /path/to/aria --output /path/to/output --tier tiny
    python -m aria.cli validate /path/to/aria
    python -m aria.cli info /path/to/aria
"""

import argparse
import sys

from aria.dataset import ARIADataset
from aria.validate import validate_dataset, get_dataset_summary
from aria.packaging import package_dataset, PackageConfig, list_tiers


def cmd_package(args):
    """Package dataset command."""
    config = None
    tier = args.tier
    
    if not tier:
        # Build custom config
        config = PackageConfig(
            num_tracks=args.num_tracks,
            audio_format=args.format,
            include_mixes=args.include_mixes,
            codecs=args.codecs.split(",") if args.codecs else None,
            seed=args.seed,
        )
    
    stats = package_dataset(
        source_path=args.source,
        output_path=args.output,
        config=config,
        tier=tier,
        verbose=True,
    )
    
    if stats.errors:
        sys.exit(1)


def cmd_validate(args):
    """Validate dataset command."""
    print(f"Validating: {args.path}")
    
    result = validate_dataset(
        args.path,
        check_real_stems=True,
        check_ai_stems=not args.skip_ai,
        codecs=args.codecs.split(",") if args.codecs else None,
        limit=args.limit,
    )
    
    if result.valid:
        print("✓ Dataset is valid!")
    else:
        print(f"✗ Dataset has {len(result.errors)} errors:")
        for err in result.errors[:20]:
            print(f"  - {err}")
        if len(result.errors) > 20:
            print(f"  ... and {len(result.errors) - 20} more")
        sys.exit(1)
    
    if result.warnings:
        print(f"\nWarnings ({len(result.warnings)}):")
        for warn in result.warnings[:10]:
            print(f"  - {warn}")


def cmd_info(args):
    """Show dataset info command."""
    summary = get_dataset_summary(args.path)
    
    if not summary["exists"]:
        print(f"✗ No dataset found at: {args.path}")
        sys.exit(1)
    
    print(f"ARIA Dataset: {args.path}")
    print(f"  Tracks: {summary['num_tracks']}")
    print(f"  Codecs: {', '.join(summary['codecs']) if summary['codecs'] else 'None'}")
    
    if args.detailed:
        dataset = ARIADataset(args.path)
        stats = dataset.stats()
        
        print(f"\nDetailed Statistics:")
        print(f"  Total stems: {stats['total_stems']}")
        print(f"  Average stems/track: {stats['avg_stems']:.1f}")
        print(f"  Stem range: {stats['min_stems']} - {stats['max_stems']}")
        
        print(f"\nGenres:")
        for genre, count in sorted(stats['genres'].items(), key=lambda x: -x[1]):
            print(f"  - {genre}: {count} tracks")


def cmd_tiers(args):
    """List available tiers command."""
    tiers = list_tiers()
    
    print("Available dataset tiers:\n")
    for name, info in tiers.items():
        tracks = info["num_tracks"] or "all"
        print(f"  {name:12} {tracks:>6} tracks - {info['description']}")


def main():
    parser = argparse.ArgumentParser(
        prog="aria",
        description="ARIA Dataset Tools",
    )
    subparsers = parser.add_subparsers(dest="command", help="Commands")
    
    # Package command
    pkg_parser = subparsers.add_parser("package", help="Package dataset for distribution")
    pkg_parser.add_argument("--source", "-s", required=True, help="Source dataset path")
    pkg_parser.add_argument("--output", "-o", required=True, help="Output path")
    pkg_parser.add_argument("--tier", "-t", choices=["tiny", "small", "medium", "full"],
                           help="Use predefined tier")
    pkg_parser.add_argument("--num-tracks", "-n", type=int, help="Number of tracks")
    pkg_parser.add_argument("--format", "-f", choices=["wav", "flac", "mp3"], default="flac",
                           help="Audio format (default: flac)")
    pkg_parser.add_argument("--include-mixes", "-m", action="store_true",
                           help="Pre-generate all mixes")
    pkg_parser.add_argument("--codecs", "-c", help="Comma-separated codecs (e.g., encodec_3,encodec_6)")
    pkg_parser.add_argument("--seed", type=int, default=42, help="Random seed for track selection")
    pkg_parser.set_defaults(func=cmd_package)
    
    # Validate command
    val_parser = subparsers.add_parser("validate", help="Validate dataset integrity")
    val_parser.add_argument("path", help="Dataset path")
    val_parser.add_argument("--skip-ai", action="store_true", help="Skip AI stem validation")
    val_parser.add_argument("--codecs", help="Comma-separated codecs to validate")
    val_parser.add_argument("--limit", type=int, help="Limit number of tracks to check")
    val_parser.set_defaults(func=cmd_validate)
    
    # Info command
    info_parser = subparsers.add_parser("info", help="Show dataset information")
    info_parser.add_argument("path", help="Dataset path")
    info_parser.add_argument("--detailed", "-d", action="store_true", help="Show detailed stats")
    info_parser.set_defaults(func=cmd_info)
    
    # Tiers command
    tiers_parser = subparsers.add_parser("tiers", help="List available dataset tiers")
    tiers_parser.set_defaults(func=cmd_tiers)
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    args.func(args)


if __name__ == "__main__":
    main()


