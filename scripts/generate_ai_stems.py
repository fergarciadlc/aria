"""
Generate AI-reconstructed stems using Encodec at different bandwidths.

This script processes the real stems from the ARIA dataset (prepared by prepare_moisesdb.py)
and generates AI-reconstructed versions using Encodec at bandwidths 3, 6, and 24 kbps.

The output structure will be:
    {aria_path}/{track_id}/stems/ai_generated/encodec_{bandwidth}/{stem_name}.wav

Usage:
    python generate_ai_stems.py [--aria-path PATH] [--limit N] [--bandwidths 3 6 24]

Examples:
    # Process all tracks
    python generate_ai_stems.py
    
    # Process first 10 tracks only (testing)
    python generate_ai_stems.py --limit 10
    
    # Use custom ARIA dataset path
    python generate_ai_stems.py --aria-path /path/to/aria
    
    # Process only specific bandwidths
    python generate_ai_stems.py --bandwidths 3 6
"""

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import torch
import torchaudio
from loguru import logger
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

# Import Encodec model from ae_models
from ae_models.encodec import Encodec

# Configuration
ARIA_PATH = "/Users/fernando/datasets/aria"
SAMPLE_RATE = 44100  # Original sample rate of ARIA stems
ENCODEC_SAMPLE_RATE = 48000  # Encodec requires 48kHz
ENCODEC_BANDWIDTHS = [3, 6, 24]  # kbps
KEEP_ORIGINAL_SR = False  # Whether to resample back to original SR (default: save at 48kHz)

# Device detection - handle MPS issues with Encodec
if torch.cuda.is_available():
    DEVICE = "cuda"
    logger.info("Using CUDA GPU")
elif torch.backends.mps.is_available():
    # Encodec has known issues with MPS (quantization operations produce noise)
    DEVICE = "cpu"
    logger.warning("MPS detected but using CPU (Encodec has MPS compatibility issues)")
else:
    DEVICE = "cpu"
    logger.info("Using CPU")


@dataclass
class ProcessingStats:
    """Track processing statistics."""
    total_tracks: int = 0
    processed_tracks: int = 0
    skipped_tracks: int = 0
    total_stems: int = 0
    processed_stems: int = 0
    failed_stems: int = 0


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


def get_real_stems(track_path: Path) -> List[Path]:
    """Get all real stem files for a track."""
    real_stems_path = track_path / "stems" / "real"
    
    if not real_stems_path.exists():
        logger.warning(f"Real stems directory not found: {real_stems_path}")
        return []
    
    # Get all .wav files in the real stems directory
    stem_files = sorted(real_stems_path.glob("*.wav"))
    return stem_files


def process_stem_with_encodec(
    stem_path: Path,
    output_path: Path,
    encodec_model: Encodec,
    bandwidth: int,
    keep_original_sr: bool = False
) -> bool:
    """
    Process a single stem through Encodec and save the result.
    
    This function replicates the exact behavior of pipeline.py's run_loop method:
    1. Load audio and resample to ENCODEC_SAMPLE_RATE (48kHz)
    2. Squeeze and handle mono by stacking channels
    3. Move to device (CPU preprocessing handled by Encodec internally)
    4. Autoencode (trimming is done inside Encodec.autoencode)
    5. Move back to CPU
    6. Optionally resample back to original SR
    7. Save with sox backend
    
    Args:
        stem_path: Path to the input stem audio file
        output_path: Path where the processed stem should be saved
        encodec_model: Initialized Encodec model
        bandwidth: Target bandwidth for Encodec compression
        keep_original_sr: If True, resample back to original SR; if False, save at 48kHz
    
    Returns:
        True if successful, False otherwise
    """
    try:
        # Load audio (matches pipeline.py line 74)
        audio_raw, sr = torchaudio.load(stem_path)
        
        # Resample to 48kHz if needed (matches pipeline.py lines 75-78)
        if sr != ENCODEC_SAMPLE_RATE:
            logger.debug(f"Resampling {stem_path.name}: {sr}Hz -> {ENCODEC_SAMPLE_RATE}Hz")
            # Using torchaudio resample instead of soxr for consistency
            resampler = torchaudio.transforms.Resample(sr, ENCODEC_SAMPLE_RATE)
            audio_raw = resampler(audio_raw)
        
        # Squeeze before checking dimensions (matches pipeline.py line 80)
        audio_raw = torch.squeeze(audio_raw)
        
        # Handle mono audio by stacking to stereo (matches pipeline.py lines 84-85)
        if len(audio_raw.shape) == 1:  # mono
            audio_raw = torch.stack([audio_raw, audio_raw], 0)
        
        # Note: Pipeline uses has_cpu_preprocess flag. For Encodec with CPU device,
        # the audio stays on CPU (encodec.py was called with has_cpu_preprocess=True)
        # The Encodec model handles moving data internally via processor
        
        # Process through Encodec (matches pipeline.py lines 101-104)
        # Note: Encodec.autoencode() already trims output to input length
        # via: return audio_values[:,:,:x.shape[1]] (see encodec.py line 67)
        with torch.no_grad():
            audio_rebuilt = encodec_model.autoencode(audio_raw)
            audio_rebuilt = audio_rebuilt.to("cpu")
        
        # Squeeze batch dimension - autoencode returns (batch, channels, time)
        # but torchaudio.save expects (channels, time)
        audio_rebuilt = audio_rebuilt.squeeze(0)
        
        # Optionally resample back to original sample rate
        output_sr = ENCODEC_SAMPLE_RATE
        if keep_original_sr and ENCODEC_SAMPLE_RATE != SAMPLE_RATE:
            logger.debug(f"Resampling back: {ENCODEC_SAMPLE_RATE}Hz -> {SAMPLE_RATE}Hz")
            resampler = torchaudio.transforms.Resample(ENCODEC_SAMPLE_RATE, SAMPLE_RATE)
            audio_rebuilt = resampler(audio_rebuilt)
            output_sr = SAMPLE_RATE
        
        # Ensure output directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save the reconstructed audio (matches pipeline.py lines 110-115)
        torchaudio.save(
            str(output_path),
            audio_rebuilt,
            sample_rate=output_sr,
            backend="sox"
        )
        
        return True
        
    except Exception as e:
        logger.error(f"Failed to process {stem_path.name} with Encodec (BW={bandwidth}): {e}")
        return False


def process_track(
    track_path: Path,
    encodec_models: dict,
    stats: ProcessingStats,
    keep_original_sr: bool = False
) -> None:
    """
    Process all stems of a track through Encodec at different bandwidths.
    
    Args:
        track_path: Path to the track directory
        encodec_models: Dictionary mapping bandwidth to Encodec model instances
        stats: ProcessingStats object to update
        keep_original_sr: If True, resample outputs back to original SR
    """
    # Load metadata
    metadata = load_track_metadata(track_path)
    if metadata is None:
        stats.skipped_tracks += 1
        return
    
    # Get real stems
    real_stems = get_real_stems(track_path)
    if not real_stems:
        logger.warning(f"No real stems found for track {track_path.name}")
        stats.skipped_tracks += 1
        return
    
    stats.total_stems += len(real_stems)
    track_name = f"{metadata.get('artist', 'Unknown')} - {metadata.get('name', 'Unknown')}"
    logger.debug(f"Processing track [{metadata.get('id', 'NONE')}]: {track_name} with {len(real_stems)} stems")
    # Get target bandwidths from encodec_models keys
    target_bandwidths = list(encodec_models.keys())
    
    # Process each stem with each bandwidth
    for stem_path in real_stems:
        stem_name = stem_path.name
        
        for bandwidth in target_bandwidths:
            # Define output path
            output_dir = track_path / "stems" / "ai_generated" / f"encodec_{bandwidth}"
            output_path = output_dir / stem_name
            
            # Skip if already processed
            if output_path.exists():
                logger.debug(f"Skipping {stem_name} (BW={bandwidth}) - already exists")
                continue
            
            # Process the stem
            logger.debug(f"Processing {track_name} - {stem_name} (BW={bandwidth})")
            success = process_stem_with_encodec(
                stem_path,
                output_path,
                encodec_models[bandwidth],
                bandwidth,
                keep_original_sr
            )
            
            if success:
                stats.processed_stems += 1
            else:
                stats.failed_stems += 1
    
    stats.processed_tracks += 1


def initialize_encodec_models(bandwidths: List[int]) -> dict:
    """
    Initialize Encodec models for each bandwidth.
    
    Args:
        bandwidths: List of bandwidths to initialize models for
    
    Returns:
        Dictionary mapping bandwidth to Encodec model instance
    """
    models = {}
    
    logger.info("Initializing Encodec models...")
    for bandwidth in bandwidths:
        logger.info(f"Loading Encodec model (bandwidth={bandwidth} kbps)")
        models[bandwidth] = Encodec(
            bandwidth=bandwidth,
            sr=ENCODEC_SAMPLE_RATE,
            device=DEVICE
        )
    
    return models


def generate_ai_stems(
    aria_path: str = ARIA_PATH,
    limit: Optional[int] = None,
    bandwidths: Optional[List[int]] = None,
    keep_original_sr: bool = KEEP_ORIGINAL_SR
) -> None:
    """
    Main function to generate AI-reconstructed stems.
    
    Args:
        aria_path: Path to the ARIA dataset
        limit: Optional limit on number of tracks to process (for testing)
        bandwidths: List of bandwidths to use (defaults to [3, 6, 24])
        keep_original_sr: If True, resample outputs to original SR (44.1kHz); if False, save at 48kHz
    """
    aria_path = Path(aria_path)
    
    # Use provided bandwidths or default
    target_bandwidths = bandwidths if bandwidths is not None else ENCODEC_BANDWIDTHS
    
    logger.info("="*80)
    logger.info("ARIA AI Stem Generation - Encodec")
    logger.info("="*80)
    logger.info(f"Dataset path: {aria_path}")
    logger.info(f"Device: {DEVICE}")
    logger.info(f"Bandwidths: {target_bandwidths}")
    output_sr = SAMPLE_RATE if keep_original_sr else ENCODEC_SAMPLE_RATE
    logger.info(f"Output sample rate: {output_sr} Hz (Encodec internal: {ENCODEC_SAMPLE_RATE} Hz)")
    logger.info(f"Keep original SR: {keep_original_sr}")
    
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
    
    # Initialize Encodec models
    encodec_models = initialize_encodec_models(target_bandwidths)
    
    # Process each track
    logger.info(f"Processing {len(track_dirs)} tracks...")
    pbar = tqdm(track_dirs, desc="Processing tracks")
    
    for track_path in pbar:
        metadata = load_track_metadata(track_path)
        if metadata:
            track_name = f"{metadata.get('artist', 'Unknown')} - {metadata.get('name', 'Unknown')}"
            pbar.set_postfix_str(track_name[:50])
        
        process_track(track_path, encodec_models, stats, keep_original_sr)
    
    # Print final statistics
    logger.info("="*80)
    logger.info("Processing Complete!")
    logger.info("="*80)
    logger.info(f"Tracks processed: {stats.processed_tracks}/{stats.total_tracks}")
    logger.info(f"Tracks skipped: {stats.skipped_tracks}")
    logger.info(f"Stems processed: {stats.processed_stems}/{stats.total_stems}")
    logger.info(f"Stems failed: {stats.failed_stems}")
    logger.info("="*80)


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate AI-reconstructed stems using Encodec",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process all tracks
  python generate_ai_stems.py
  
  # Process first 10 tracks only (testing)
  python generate_ai_stems.py --limit 10
  
  # Use custom ARIA dataset path
  python generate_ai_stems.py --aria-path /path/to/aria
  
  # Process only specific bandwidths
  python generate_ai_stems.py --bandwidths 3 6
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
        "--bandwidths",
        type=int,
        nargs="+",
        default=ENCODEC_BANDWIDTHS,
        choices=[3, 6, 12, 24],
        help=f"Encodec bandwidths to use in kbps (default: {ENCODEC_BANDWIDTHS})"
    )
    
    parser.add_argument(
        "--keep-original-sr",
        action="store_true",
        default=KEEP_ORIGINAL_SR,
        help="Resample output back to original sample rate (44.1kHz) instead of keeping at 48kHz. "
             "Recommended: False (48kHz) to avoid double resampling artifacts and match original implementation."
    )
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    
    generate_ai_stems(
        aria_path=args.aria_path,
        limit=args.limit,
        bandwidths=args.bandwidths,
        keep_original_sr=args.keep_original_sr
    )

