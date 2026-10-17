"""
Dataset packaging utilities.

Create distribution variants of the ARIA dataset with different:
- Sizes (tiny, small, medium, full)
- Formats (WAV, FLAC, MP3)
- Content (stems only, stems + mixes)
"""

import json
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from tqdm import tqdm

from aria.audio import load_audio, save_audio
from aria.constants import MP3_BITRATE_KBPS, SAMPLE_RATE
from aria.dataset import ARIADataset
from aria.mix import compute_mix, find_mix_path, generate_all_mix_specs, mix_metadata
from aria.track import Track, get_available_codecs, get_stem_path

AudioFormat = Literal["wav", "flac", "mp3"]


# Predefined dataset tiers
TIERS = {
    "tiny": {"num_tracks": 10, "description": "10 tracks for quick testing"},
    "small": {"num_tracks": 50, "description": "50 tracks for development"},
    "medium": {"num_tracks": 100, "description": "100 tracks for experiments"},
    "full": {"num_tracks": None, "description": "All tracks"},
}


@dataclass
class PackageConfig:
    """Configuration for dataset packaging.
    
    Attributes:
        num_tracks: Number of tracks to include (None = all).
        audio_format: Output audio format.
        include_mixes: Whether to include mixes (copied when pre-computed,
            computed from stems otherwise).
        codecs: Which AI codecs to include (None = all available).
        mp3_bitrate: Bitrate for MP3 encoding (kbps).
    """
    num_tracks: int | None = None
    audio_format: AudioFormat = "flac"
    include_mixes: bool = False
    codecs: list[str] | None = None
    mp3_bitrate: int = MP3_BITRATE_KBPS
    
    @classmethod
    def from_tier(cls, tier: str, **overrides) -> "PackageConfig":
        """Create config from predefined tier."""
        if tier not in TIERS:
            raise ValueError(f"Unknown tier: {tier}. Choose from: {list(TIERS.keys())}")
        
        config = cls(num_tracks=TIERS[tier]["num_tracks"])
        
        # Apply overrides
        for key, value in overrides.items():
            if hasattr(config, key):
                setattr(config, key, value)
        
        return config


@dataclass
class PackageStats:
    """Statistics from packaging operation."""
    tracks_processed: int = 0
    stems_copied: int = 0
    mixes_generated: int = 0
    errors: list[str] = field(default_factory=list)


def select_tracks(dataset: ARIADataset, num_tracks: int | None) -> list[Track]:
    """Select tracks for packaging: the first N in display-name order.
    
    Args:
        dataset: Source dataset (already sorted by display_name).
        num_tracks: Number of tracks to select (None = all).
    
    Returns:
        List of selected tracks.
    """
    return list(dataset.tracks[:num_tracks])


def copy_audio(src_path: Path, dst_path: Path, config: PackageConfig) -> None:
    """Copy an audio file, converting it if its format differs.
    
    Args:
        src_path: Source audio file.
        dst_path: Destination path (extension = target format).
        config: Package configuration (for MP3 bitrate).
    
    Raises:
        Exception: If loading or encoding fails.
    """
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    if src_path.suffix == dst_path.suffix:
        shutil.copy2(src_path, dst_path)
        return
    audio, sr = load_audio(src_path)
    save_audio(dst_path, audio, sr, mp3_bitrate=config.mp3_bitrate)


def package_track(
    track: Track,
    output_path: Path,
    config: PackageConfig,
    stats: PackageStats,
) -> None:
    """Package a single track.
    
    Args:
        track: Source track.
        output_path: Output directory for this track.
        config: Package configuration.
        stats: Statistics to update.
    """
    ext = f".{config.audio_format}"
    
    # Create track directory
    track_output = output_path / track.id
    track_output.mkdir(parents=True, exist_ok=True)
    
    # Copy metadata
    src_metadata = track.path / "metadata.json"
    if src_metadata.exists():
        shutil.copy2(src_metadata, track_output / "metadata.json")
    
    # Determine which codecs to include
    available_codecs = get_available_codecs(track)
    if config.codecs:
        codecs_to_copy = [c for c in config.codecs if c in available_codecs]
    else:
        codecs_to_copy = available_codecs
    
    # Copy real stems, then AI stems per codec
    sources: list[tuple[str, str | None]] = [("real", None)]
    sources += [("ai_generated", codec) for codec in codecs_to_copy]
    
    for source, codec in sources:
        out_dir = track_output / "stems" / source
        if codec is not None:
            out_dir = out_dir / codec
        
        for stem_name in track.stem_names:
            try:
                src = get_stem_path(track, stem_name, source, codec)
                copy_audio(src, out_dir / f"{stem_name}{ext}", config)
                stats.stems_copied += 1
            except Exception as e:
                label = source if codec is None else f"{source} {codec}"
                stats.errors.append(f"Failed to copy {label} stem {stem_name} for {track.id}: {e}")
    
    # Copy pre-computed mixes, computing the missing ones
    if config.include_mixes:
        for codec in codecs_to_copy:
            for spec in generate_all_mix_specs(track.num_stems, codec):
                try:
                    mix_dir = track_output / "mixes" / codec / spec.u_folder
                    mix_path = mix_dir / f"{spec.filename}{ext}"
                    
                    src = find_mix_path(track, spec)
                    if src is not None:
                        copy_audio(src, mix_path, config)
                    else:
                        audio = compute_mix(track, spec, target_sr=SAMPLE_RATE)
                        save_audio(mix_path, audio, SAMPLE_RATE, mp3_bitrate=config.mp3_bitrate)
                    
                    with open(mix_dir / f"{spec.filename}.json", "w") as f:
                        json.dump(mix_metadata(track, spec), f, indent=2)
                    
                    stats.mixes_generated += 1
                    
                except Exception as e:
                    stats.errors.append(f"Failed to package mix {spec.filename} for {track.id}: {e}")
    
    stats.tracks_processed += 1


def package_dataset(
    source_path: str | Path,
    output_path: str | Path,
    config: PackageConfig | None = None,
    tier: str | None = None,
    verbose: bool = True,
) -> PackageStats:
    """Package ARIA dataset for distribution.
    
    Creates a copy of the dataset with specified configuration.
    
    Args:
        source_path: Path to full ARIA dataset.
        output_path: Where to write the packaged dataset.
        config: Package configuration (use this or tier, not both).
        tier: Predefined tier name ("tiny", "small", "medium", "full").
        verbose: Show progress bar.
    
    Returns:
        PackageStats with operation results.
    
    Example:
        # Package tiny version
        stats = package_dataset(
            "/full/aria",
            "/releases/aria-tiny",
            tier="tiny",
        )
        
        # Custom configuration
        config = PackageConfig(
            num_tracks=50,
            audio_format="flac",
            include_mixes=True,
            codecs=["encodec_3"],
        )
        stats = package_dataset("/full/aria", "/releases/custom", config=config)
    """
    source_path = Path(source_path)
    output_path = Path(output_path)
    
    # Resolve config
    if tier and config:
        raise ValueError("Specify either 'tier' or 'config', not both")
    
    if tier:
        config = PackageConfig.from_tier(tier)
    elif config is None:
        config = PackageConfig()
    
    # Load source dataset
    dataset = ARIADataset(source_path)
    
    # Select tracks
    tracks = select_tracks(dataset, config.num_tracks)
    
    # Initialize stats
    stats = PackageStats()
    
    # Create output directory
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Write package metadata
    package_meta = {
        "source": str(source_path),
        "num_tracks": len(tracks),
        "audio_format": config.audio_format,
        "include_mixes": config.include_mixes,
        "codecs": config.codecs,
        "tier": tier,
    }
    with open(output_path / "package_info.json", "w") as f:
        json.dump(package_meta, f, indent=2)
    
    # Package each track
    iterator = tqdm(tracks, desc="Packaging tracks") if verbose else tracks
    
    for track in iterator:
        if verbose:
            iterator.set_postfix_str(track.display_name[:40])
        package_track(track, output_path, config, stats)
    
    # Summary
    if verbose:
        print(f"\n✓ Packaged {stats.tracks_processed} tracks")
        print(f"  Stems copied: {stats.stems_copied}")
        if config.include_mixes:
            print(f"  Mixes packaged: {stats.mixes_generated}")
        if stats.errors:
            print(f"  Errors: {len(stats.errors)}")
            for err in stats.errors[:5]:
                print(f"    - {err}")
    
    return stats


def list_tiers() -> dict[str, dict[str, Any]]:
    """List available predefined tiers."""
    return TIERS.copy()


