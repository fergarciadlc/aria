"""                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     
Generate all possible mixes combining real and AI-generated stems.

This script processes tracks from the ARIA dataset (after prepare_moisesdb.py and generate_ai_stems.py)
and creates all 2^n combinations of real and AI-generated stems, where n is the number of stems per track.

The output structure follows the ARIA specification:
    {aria_path}/{track_id}/mixes/{codec}/Uk/mix_S-{indices}.wav

Where:
    - codec = AI model used (e.g., encodec_3, encodec_6)
    - Uk = folder for mixes with exactly k AI-generated stems
    - indices = hyphen-separated list of which stem indices are AI-generated
    - "empty" = special case for all-real stems (U0)

For a track with 4 stems using encodec_3, this generates 2^4 = 16 mixes:
    - mixes/encodec_3/U0/mix_S-empty.wav (all real)
    - mixes/encodec_3/U1/mix_S-0.wav, mix_S-1.wav, mix_S-2.wav, mix_S-3.wav
    - mixes/encodec_3/U2/mix_S-0-1.wav, mix_S-0-2.wav, ...
    - mixes/encodec_3/U3/mix_S-0-1-2.wav, ...
    - mixes/encodec_3/U4/mix_S-0-1-2-3.wav (all AI-generated)

Usage:
    python generate_mixes.py [--aria-path PATH] [--limit N] [--model encodec_3]
    
Examples:
    # Process all tracks with default model (encodec_3)
    python generate_mixes.py
    
    # Process first 10 tracks only (testing)
    python generate_mixes.py --limit 10
    
    # Use custom ARIA dataset path
    python generate_mixes.py --aria-path /path/to/aria
    
    # Use specific model/codec
    python generate_mixes.py --model encodec_6
"""

import argparse
import json
import sys
from dataclasses import dataclass, asdict
from itertools import combinations
from pathlib import Path
from typing import List, Optional, Set, Tuple

import numpy as np
import soundfile as sf
from loguru import logger
from tqdm import tqdm

# Configuration
ARIA_PATH = "/Users/fernando/datasets/aria"
SAMPLE_RATE = 48000
DEFAULT_CODEC = "encodec_3"


@dataclass
class MixMetadata:
    """Metadata for a generated mix."""
    mix_name: str
    num_stems: int
    num_ai_stems: int
    ai_stem_indices: List[int]
    ai_codec: str
    stem_names: List[str]
    
    def to_dict(self):
        return asdict(self)


@dataclass
class ProcessingStats:
    """Track processing statistics."""
    total_tracks: int = 0
    processed_tracks: int = 0
    skipped_tracks: int = 0
    total_mixes: int = 0
    generated_mixes: int = 0
    skipped_mixes: int = 0
    failed_mixes: int = 0


def load_track_metadata(track_path: Path) -> Optional[dict]:
    """Load metadata.json for a track."""
    metadata_path = track_path / "metadata.json"
    if not metadata_path.exists():
        logger.warning(f"Metadata not found: {metadata_path}")
        return None
    
    try:
        with open(metadata_path, "r") as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Failed to load metadata from {metadata_path}: {e}")
        return None


def get_track_directories(aria_path: Path) -> List[Path]:
    """Get all track directories in the ARIA dataset."""
    if not aria_path.exists():
        logger.error(f"ARIA path does not exist: {aria_path}")
        return []
    
    # Get all directories that contain a metadata.json file
    track_dirs = [d for d in aria_path.iterdir() 
                  if d.is_dir() and (d / "metadata.json").exists()]
    
    logger.info(f"Found {len(track_dirs)} tracks in {aria_path}")
    return sorted(track_dirs)


def get_stem_files(track_path: Path, stem_type: str, codec: Optional[str] = None) -> List[Path]:
    """
    Get stem files for a track.
    
    Args:
        track_path: Path to the track directory
        stem_type: Either "real" or "ai_generated"
        codec: For ai_generated stems, the codec subdirectory (e.g., "encodec_3")
    
    Returns:
        Sorted list of stem file paths
    """
    if stem_type == "real":
        stems_path = track_path / "stems" / "real"
    elif stem_type == "ai_generated":
        if codec is None:
            raise ValueError("codec must be specified for ai_generated stems")
        stems_path = track_path / "stems" / "ai_generated" / codec
    else:
        raise ValueError(f"Invalid stem_type: {stem_type}")
    
    if not stems_path.exists():
        logger.warning(f"Stems directory not found: {stems_path}")
        return []
    
    # Get all .wav files
    stem_files = sorted(stems_path.glob("*.wav"))
    return stem_files


def load_audio(file_path: Path) -> Tuple[np.ndarray, int]:
    """
    Load audio file.
    
    Returns:
        Tuple of (audio_array, sample_rate)
        Audio array shape: (channels, samples)
    """
    audio, sr = sf.read(file_path, always_2d=True)
    # soundfile loads as (samples, channels), transpose to (channels, samples)
    audio = audio.T
    return audio, sr


def pad_to_length(audio: np.ndarray, target_length: int) -> np.ndarray:
    """
    Pad audio to target length with zeros.
    
    Args:
        audio: Audio array of shape (channels, samples)
        target_length: Target length in samples
    
    Returns:
        Padded audio array
    """
    if audio.shape[-1] >= target_length:
        return audio
    
    pad_width = ((0, 0), (0, target_length - audio.shape[-1]))
    return np.pad(audio, pad_width, mode='constant', constant_values=0)


def pad_and_mix(audio_list: List[np.ndarray]) -> np.ndarray:
    """
    Pad all audio to the same length and mix (sum) them.
    
    This follows the moises-db approach for mixing stems.
    
    Args:
        audio_list: List of audio arrays, each with shape (channels, samples)
    
    Returns:
        Mixed audio array with shape (channels, samples)
    """
    if not audio_list:
        raise ValueError("Cannot mix empty audio list")
    
    # Find maximum length
    max_length = max(audio.shape[-1] for audio in audio_list)
    
    # Pad all audio to max length
    padded_audio = [pad_to_length(audio, max_length) for audio in audio_list]
    
    # Stack and sum
    mixed = np.stack(padded_audio).sum(axis=0)
    
    return mixed


def generate_mix_name(ai_indices: Set[int]) -> str:
    """
    Generate mix filename according to ARIA naming convention.
    
    Args:
        ai_indices: Set of stem indices that are AI-generated
    
    Returns:
        Mix filename (without .wav extension)
    """
    if not ai_indices:
        return "mix_S-empty"
    
    # Sort indices and join with hyphen
    sorted_indices = sorted(ai_indices)
    indices_str = "-".join(str(i) for i in sorted_indices)
    return f"mix_S-{indices_str}"


def generate_all_combinations(n_stems: int) -> List[Set[int]]:
    """
    Generate all possible combinations of AI-generated stem indices.
    
    For n stems, this generates 2^n combinations.
    
    Args:
        n_stems: Number of stems
    
    Returns:
        List of sets, where each set contains the indices that should be AI-generated
    """
    all_combinations = []
    
    # Generate combinations of size k for k = 0 to n
    for k in range(n_stems + 1):
        for combo in combinations(range(n_stems), k):
            all_combinations.append(set(combo))
    
    return all_combinations


def save_audio(file_path: Path, audio: np.ndarray, sample_rate: int):
    """
    Save audio to file.
    
    Args:
        file_path: Path to save the audio
        audio: Audio array of shape (channels, samples)
        sample_rate: Sample rate
    """
    # Ensure output directory exists
    file_path.parent.mkdir(parents=True, exist_ok=True)
    
    # soundfile expects (samples, channels), so transpose
    audio_to_save = audio.T
    
    sf.write(file_path, audio_to_save, sample_rate, subtype='PCM_24')


def save_mix_metadata(file_path: Path, metadata: MixMetadata):
    """Save mix metadata to JSON file."""
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, 'w') as f:
        json.dump(metadata.to_dict(), f, indent=2)


def process_track(
    track_path: Path,
    codec: str,
    stats: ProcessingStats,
    force_regenerate: bool = False
) -> None:
    """
    Process a single track and generate all possible mixes.
    
    Args:
        track_path: Path to the track directory
        codec: Codec identifier (e.g., "encodec_3")
        stats: ProcessingStats object to update
        force_regenerate: If True, regenerate even if mix already exists
    """
    # Load metadata
    metadata = load_track_metadata(track_path)
    if metadata is None:
        stats.skipped_tracks += 1
        return
    
    track_name = f"{metadata.get('artist', 'Unknown')} - {metadata.get('name', 'Unknown')}"
    
    # Get real stems
    real_stems = get_stem_files(track_path, "real")
    if not real_stems:
        logger.warning(f"No real stems found for {track_name}")
        stats.skipped_tracks += 1
        return
    
    # Get AI-generated stems
    ai_stems = get_stem_files(track_path, "ai_generated", codec)
    if not ai_stems:
        logger.warning(f"No AI-generated stems ({codec}) found for {track_name}")
        stats.skipped_tracks += 1
        return
    
    # Verify we have matching stems
    n_stems = len(real_stems)
    if len(ai_stems) != n_stems:
        logger.error(f"Mismatch in stem counts for {track_name}: {n_stems} real vs {len(ai_stems)} AI")
        stats.skipped_tracks += 1
        return
    
    # Extract stem names (without extension)
    stem_names = [stem.stem for stem in real_stems]
    
    # Generate all combinations
    all_combinations = generate_all_combinations(n_stems)
    stats.total_mixes += len(all_combinations)
    
    logger.debug(f"Processing {track_name}: {n_stems} stems, {len(all_combinations)} mixes to generate")
    
    # Process each combination
    for ai_indices in all_combinations:
        num_ai_stems = len(ai_indices)
        mix_name = generate_mix_name(ai_indices)
        
        # Define output path with codec folder
        output_dir = track_path / "mixes" / codec / f"U{num_ai_stems}"
        output_path = output_dir / f"{mix_name}.wav"
        metadata_path = output_dir / f"{mix_name}.json"
        
        # Skip if already exists (unless force_regenerate)
        if output_path.exists() and not force_regenerate:
            logger.debug(f"Skipping {mix_name} - already exists")
            stats.skipped_mixes += 1
            continue
        
        try:
            # Load stems for this mix
            audio_list = []
            for i in range(n_stems):
                if i in ai_indices:
                    # Use AI-generated stem
                    stem_path = ai_stems[i]
                else:
                    # Use real stem
                    stem_path = real_stems[i]
                
                audio, sr = load_audio(stem_path)
                
                # Resample if needed (shouldn't happen, but just in case)
                if sr != SAMPLE_RATE:
                    logger.warning(f"Unexpected sample rate {sr} for {stem_path.name}, expected {SAMPLE_RATE}")
                    # For now, we'll skip resampling as all stems should be at SAMPLE_RATE
                
                audio_list.append(audio)
            
            # Mix stems
            mixed_audio = pad_and_mix(audio_list)
            
            # Save mix
            save_audio(output_path, mixed_audio, SAMPLE_RATE)
            
            # Save metadata
            mix_metadata = MixMetadata(
                mix_name=mix_name,
                num_stems=n_stems,
                num_ai_stems=num_ai_stems,
                ai_stem_indices=sorted(ai_indices),
                ai_codec=codec,
                stem_names=stem_names
            )
            save_mix_metadata(metadata_path, mix_metadata)
            
            stats.generated_mixes += 1
            
        except Exception as e:
            logger.error(f"Failed to generate {mix_name} for {track_name}: {e}")
            stats.failed_mixes += 1
    
    stats.processed_tracks += 1


def generate_mixes(
    aria_path: str = ARIA_PATH,
    limit: Optional[int] = None,
    model: str = DEFAULT_CODEC,
    force_regenerate: bool = False
) -> None:
    """
    Main function to generate all mixes.
    
    Args:
        aria_path: Path to the ARIA dataset
        limit: Optional limit on number of tracks to process (for testing)
        model: Model/codec identifier for AI-generated stems (e.g., "encodec_3", "encodec_6")
        force_regenerate: If True, regenerate mixes even if they already exist
    """
    aria_path = Path(aria_path)
    
    logger.info("="*80)
    logger.info("ARIA Mix Generation")
    logger.info("="*80)
    logger.info(f"Dataset path: {aria_path}")
    logger.info(f"AI model/codec: {model}")
    logger.info(f"Sample rate: {SAMPLE_RATE} Hz")
    logger.info(f"Force regenerate: {force_regenerate}")
    
    # Get track directories
    track_dirs = get_track_directories(aria_path)
    if not track_dirs:
        logger.error("No tracks found. Exiting.")
        return
    
    # Apply limit if specified
    if limit is not None:
        logger.info(f"Limiting processing to first {limit} tracks")
        track_dirs = track_dirs[:limit]
    
    # Initialize statistics
    stats = ProcessingStats(total_tracks=len(track_dirs))
    
    # Process each track
    logger.info(f"Processing {len(track_dirs)} tracks...")
    pbar = tqdm(track_dirs, desc="Generating mixes")
    
    for track_path in pbar:
        metadata = load_track_metadata(track_path)
        if metadata:
            track_name = f"{metadata.get('artist', 'Unknown')} - {metadata.get('name', 'Unknown')}"
            pbar.set_postfix_str(track_name[:50])
        
        process_track(track_path, model, stats, force_regenerate)
    
    # Print final statistics
    logger.info("="*80)
    logger.info("Processing Complete!")
    logger.info("="*80)
    logger.info(f"Tracks processed: {stats.processed_tracks}/{stats.total_tracks}")
    logger.info(f"Tracks skipped: {stats.skipped_tracks}")
    logger.info(f"Mixes generated: {stats.generated_mixes}/{stats.total_mixes}")
    logger.info(f"Mixes skipped (already exist): {stats.skipped_mixes}")
    logger.info(f"Mixes failed: {stats.failed_mixes}")
    logger.info("="*80)


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate all possible mixes combining real and AI-generated stems",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process all tracks with default model (encodec_3)
  python generate_mixes.py
  
  # Process first 10 tracks only (testing)
  python generate_mixes.py --limit 10
  
  # Use custom ARIA dataset path
  python generate_mixes.py --aria-path /path/to/aria
  
  # Use specific model/codec
  python generate_mixes.py --model encodec_6
  
  # Force regeneration of existing mixes
  python generate_mixes.py --force
        """
    )
    
    parser.add_argument(
        "--aria-path",
        type=str,
        default=ARIA_PATH,
        help=f"Path to the ARIA dataset (default: {ARIA_PATH})"
    )
    
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of tracks to process (useful for testing)"
    )
    
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_CODEC,
        help=f"AI model/codec identifier (e.g., encodec_3, encodec_6, encodec_24) (default: {DEFAULT_CODEC})"
    )
    
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force regeneration of mixes even if they already exist"
    )
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    
    generate_mixes(
        aria_path=args.aria_path,
        limit=args.limit,
        model=args.model,
        force_regenerate=args.force
    )

