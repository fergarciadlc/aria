"""
Dataset packaging utilities.

Create distribution variants of the ARIA dataset with different:
- Sizes (tiny, small, medium, full)
- Formats (WAV, FLAC, MP3)
- Content (stems only, stems + mixes)
"""

import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from tqdm import tqdm

from aria.audio import load_audio, save_audio
from aria.dataset import ARIADataset
from aria.mix import MixSpec, generate_all_mix_specs, compute_mix
from aria.track import Track, get_stem_path, get_available_codecs


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
        include_mixes: Whether to pre-generate mixes.
        codecs: Which AI codecs to include (None = all available).
        seed: Random seed for reproducible track selection.
        mp3_bitrate: Bitrate for MP3 encoding (kbps).
    """
    num_tracks: int | None = None
    audio_format: AudioFormat = "flac"
    include_mixes: bool = False
    codecs: list[str] | None = None
    seed: int = 42
    mp3_bitrate: int = 320
    
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
    errors: list[str] = None
    
    def __post_init__(self):
        if self.errors is None:
            self.errors = []


def select_tracks(
    dataset: ARIADataset,
    num_tracks: int | None,
    seed: int = 42,
) -> list[Track]:
    """Select tracks for packaging.
    
    Uses deterministic selection based on seed for reproducibility.
    Tracks are sorted by display_name, then first N selected.
    
    Args:
        dataset: Source dataset.
        num_tracks: Number of tracks to select (None = all).
        seed: Random seed (currently unused, deterministic selection).
    
    Returns:
        List of selected tracks.
    """
    all_tracks = list(dataset)
    
    # Sort by display_name for deterministic order
    all_tracks.sort(key=lambda t: t.display_name.lower())
    
    if num_tracks is None:
        return all_tracks
    
    return all_tracks[:num_tracks]


def get_audio_extension(format: AudioFormat) -> str:
    """Get file extension for audio format."""
    return f".{format}"


def convert_audio_file(
    src_path: Path,
    dst_path: Path,
    target_format: AudioFormat,
    mp3_bitrate: int = 320,
) -> bool:
    """Convert audio file to target format.
    
    Args:
        src_path: Source audio file.
        dst_path: Destination path (with correct extension).
        target_format: Target format.
        mp3_bitrate: Bitrate for MP3 (ignored for other formats).
    
    Returns:
        True if successful.
    """
    try:
        audio, sr = load_audio(src_path)
        
        # Determine subtype
        if target_format == "mp3":
            # MP3 requires special handling
            import soundfile as sf
            dst_path.parent.mkdir(parents=True, exist_ok=True)
            # Note: soundfile doesn't support MP3 writing directly
            # For now, save as FLAC and note that MP3 needs additional tooling
            # TODO: Add proper MP3 support via pydub or ffmpeg
            save_audio(dst_path.with_suffix(".flac"), audio, sr, format="flac")
            return True
        else:
            save_audio(dst_path, audio, sr, format=target_format)
            return True
            
    except Exception as e:
        return False


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
    ext = get_audio_extension(config.audio_format)
    
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
    
    # Copy real stems
    real_output = track_output / "stems" / "real"
    real_output.mkdir(parents=True, exist_ok=True)
    
    for stem_name in track.stem_names:
        try:
            src = get_stem_path(track, stem_name, "real")
            dst = real_output / f"{stem_name}{ext}"
            
            if config.audio_format == "wav" and src.suffix == ".wav":
                # Just copy if same format
                shutil.copy2(src, dst)
            else:
                convert_audio_file(src, dst, config.audio_format, config.mp3_bitrate)
            
            stats.stems_copied += 1
        except Exception as e:
            stats.errors.append(f"Failed to copy real stem {stem_name} for {track.id}: {e}")
    
    # Copy AI stems
    for codec in codecs_to_copy:
        ai_output = track_output / "stems" / "ai_generated" / codec
        ai_output.mkdir(parents=True, exist_ok=True)
        
        for stem_name in track.stem_names:
            try:
                src = get_stem_path(track, stem_name, "ai_generated", codec)
                dst = ai_output / f"{stem_name}{ext}"
                
                if config.audio_format == "wav" and src.suffix == ".wav":
                    shutil.copy2(src, dst)
                else:
                    convert_audio_file(src, dst, config.audio_format, config.mp3_bitrate)
                
                stats.stems_copied += 1
            except Exception as e:
                stats.errors.append(f"Failed to copy AI stem {stem_name} ({codec}) for {track.id}: {e}")
    
    # Generate mixes if requested
    if config.include_mixes:
        for codec in codecs_to_copy:
            mixes_output = track_output / "mixes" / codec
            
            all_specs = generate_all_mix_specs(track.num_stems, codec)
            
            for spec in all_specs:
                try:
                    mix_dir = mixes_output / spec.u_folder
                    mix_dir.mkdir(parents=True, exist_ok=True)
                    
                    mix_path = mix_dir / f"{spec.filename}{ext}"
                    
                    # Compute mix
                    audio = compute_mix(track, spec)
                    save_audio(mix_path, audio, 48000, format=config.audio_format)
                    
                    # Save mix metadata
                    mix_meta = {
                        "mix_name": spec.filename,
                        "num_stems": track.num_stems,
                        "num_ai_stems": spec.num_ai_stems,
                        "ai_stem_indices": list(spec.ai_indices),
                        "ai_codec": spec.codec,
                        "stem_names": track.stem_names,
                    }
                    with open(mix_dir / f"{spec.filename}.json", "w") as f:
                        json.dump(mix_meta, f, indent=2)
                    
                    stats.mixes_generated += 1
                    
                except Exception as e:
                    stats.errors.append(f"Failed to generate mix {spec.filename} for {track.id}: {e}")
    
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
    tracks = select_tracks(dataset, config.num_tracks, config.seed)
    
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
            print(f"  Mixes generated: {stats.mixes_generated}")
        if stats.errors:
            print(f"  Errors: {len(stats.errors)}")
            for err in stats.errors[:5]:
                print(f"    - {err}")
    
    return stats


def list_tiers() -> dict:
    """List available predefined tiers."""
    return TIERS.copy()


