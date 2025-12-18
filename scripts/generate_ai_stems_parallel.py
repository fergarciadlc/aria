"""
Generate AI-reconstructed stems using Encodec at different bandwidths (PARALLEL VERSION).

This script processes the real stems from the ARIA dataset using multiple CPU processes
to parallelize the encoding work across stems.

The output structure will be:
    {aria_path}/{track_id}/stems/ai_generated/encodec_{bandwidth}/{stem_name}.wav

Usage:
    python generate_ai_stems_parallel.py [--aria-path PATH] [--limit N] [--bandwidths 3 6 24] [--workers 4]

Examples:
    # Process with 4 parallel workers (default)
    python generate_ai_stems_parallel.py
    
    # Use 8 workers for faster processing
    python generate_ai_stems_parallel.py --workers 8
    
    # Process first 10 tracks only (testing)
    python generate_ai_stems_parallel.py --limit 10
    
    # Use custom ARIA dataset path
    python generate_ai_stems_parallel.py --aria-path /path/to/aria
"""

import argparse
import json
import sys
from dataclasses import dataclass
from multiprocessing import Pool, cpu_count
from pathlib import Path
from typing import List, Optional, Tuple

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
NUM_WORKERS = 4  # Default number of parallel workers

# Force CPU device for all workers
DEVICE = "cpu"


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
        return None
    
    try:
        with open(metadata_path, "r") as f:
            return json.load(f)
    except Exception as e:
        return None


def get_track_directories(aria_path: Path) -> List[Path]:
    """Get all track directories in the ARIA dataset."""
    if not aria_path.exists():
        logger.error(f"ARIA path does not exist: {aria_path}")
        return []
    
    track_dirs = [d for d in aria_path.iterdir() 
                  if d.is_dir() and (d / "metadata.json").exists()]
    
    logger.info(f"Found {len(track_dirs)} tracks in {aria_path}")
    return sorted(track_dirs)


def get_real_stems(track_path: Path) -> List[Path]:
    """Get all real stem files for a track."""
    real_stems_path = track_path / "stems" / "real"
    
    if not real_stems_path.exists():
        return []
    
    stem_files = sorted(real_stems_path.glob("*.wav"))
    return stem_files


def process_stem_with_encodec_worker(args: Tuple[Path, Path, int, int, bool]) -> Tuple[bool, str]:
    """
    Worker function to process a single stem through Encodec.
    
    This function is called by multiprocessing workers. Each worker:
    1. Creates its own Encodec model instance
    2. Processes one stem
    3. Returns success status
    
    Args:
        args: Tuple of (stem_path, output_path, bandwidth, encodec_sr, keep_original_sr)
    
    Returns:
        Tuple of (success, error_message)
    """
    stem_path, output_path, bandwidth, keep_original_sr = args
    
    # Skip if already processed
    if output_path.exists():
        return (True, "")
    
    try:
        # Each worker creates its own model instance (not shared across processes)
        encodec_model = Encodec(
            bandwidth=bandwidth,
            sr=ENCODEC_SAMPLE_RATE,
            device=DEVICE
        )
        
        # Load audio
        audio_raw, sr = torchaudio.load(stem_path)
        
        # Resample to 48kHz if needed
        if sr != ENCODEC_SAMPLE_RATE:
            resampler = torchaudio.transforms.Resample(sr, ENCODEC_SAMPLE_RATE)
            audio_raw = resampler(audio_raw)
        
        # Squeeze before checking dimensions
        audio_raw = torch.squeeze(audio_raw)
        
        # Handle mono audio by stacking to stereo
        if len(audio_raw.shape) == 1:  # mono
            audio_raw = torch.stack([audio_raw, audio_raw], 0)
        
        # Process through Encodec
        with torch.no_grad():
            audio_rebuilt = encodec_model.autoencode(audio_raw)
            audio_rebuilt = audio_rebuilt.to("cpu")
        
        # Squeeze batch dimension
        audio_rebuilt = audio_rebuilt.squeeze(0)
        
        # Optionally resample back to original sample rate
        output_sr = ENCODEC_SAMPLE_RATE
        if keep_original_sr and ENCODEC_SAMPLE_RATE != SAMPLE_RATE:
            resampler = torchaudio.transforms.Resample(ENCODEC_SAMPLE_RATE, SAMPLE_RATE)
            audio_rebuilt = resampler(audio_rebuilt)
            output_sr = SAMPLE_RATE
        
        # Ensure output directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save the reconstructed audio
        torchaudio.save(
            str(output_path),
            audio_rebuilt,
            sample_rate=output_sr,
            backend="sox"
        )
        
        return (True, "")
        
    except Exception as e:
        return (False, f"{stem_path.name}: {str(e)}")


def collect_all_work_items(
    track_dirs: List[Path],
    bandwidths: List[int],
    keep_original_sr: bool
) -> List[Tuple[Path, Path, int, bool]]:
    """
    Collect all stem processing tasks across all tracks and bandwidths.
    
    Returns:
        List of tuples: (stem_path, output_path, bandwidth, keep_original_sr)
    """
    work_items = []
    
    for track_path in track_dirs:
        real_stems = get_real_stems(track_path)
        
        for stem_path in real_stems:
            for bandwidth in bandwidths:
                output_dir = track_path / "stems" / "ai_generated" / f"encodec_{bandwidth}"
                output_path = output_dir / stem_path.name
                
                # Add to work queue
                work_items.append((stem_path, output_path, bandwidth, keep_original_sr))
    
    return work_items


def generate_ai_stems_parallel(
    aria_path: str = ARIA_PATH,
    limit: Optional[int] = None,
    bandwidths: Optional[List[int]] = None,
    keep_original_sr: bool = KEEP_ORIGINAL_SR,
    num_workers: int = NUM_WORKERS
) -> None:
    """
    Main function to generate AI-reconstructed stems using parallel processing.
    
    Args:
        aria_path: Path to the ARIA dataset
        limit: Optional limit on number of tracks to process (for testing)
        bandwidths: List of bandwidths to use (defaults to [3, 6, 24])
        keep_original_sr: If True, resample outputs to original SR (44.1kHz); if False, save at 48kHz
        num_workers: Number of parallel worker processes
    """
    aria_path = Path(aria_path)
    
    # Use provided bandwidths or default
    target_bandwidths = bandwidths if bandwidths is not None else ENCODEC_BANDWIDTHS
    
    # Determine optimal number of workers
    max_workers = cpu_count()
    if num_workers > max_workers:
        logger.warning(f"Requested {num_workers} workers but only {max_workers} CPUs available. Using {max_workers}.")
        num_workers = max_workers
    
    logger.info("="*80)
    logger.info("ARIA AI Stem Generation - Encodec (PARALLEL)")
    logger.info("="*80)
    logger.info(f"Dataset path: {aria_path}")
    logger.info(f"Device: CPU (parallel processing)")
    logger.info(f"Workers: {num_workers} (max available: {max_workers})")
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
    
    # Collect all work items (stem processing tasks)
    logger.info("Collecting work items...")
    work_items = collect_all_work_items(track_dirs, target_bandwidths, keep_original_sr)
    
    # Filter out already processed items
    work_items_todo = [item for item in work_items if not item[1].exists()]
    
    logger.info(f"Total work items: {len(work_items)}")
    logger.info(f"Already processed: {len(work_items) - len(work_items_todo)}")
    logger.info(f"To process: {len(work_items_todo)}")
    
    if len(work_items_todo) == 0:
        logger.info("Nothing to process. All stems already generated.")
        return
    
    # Initialize statistics
    stats = ProcessingStats(
        total_tracks=len(track_dirs),
        total_stems=len(work_items_todo)
    )
    
    # Process stems in parallel
    logger.info(f"Processing {len(work_items_todo)} stems with {num_workers} workers...")
    
    with Pool(processes=num_workers) as pool:
        # Use imap for progress tracking
        results = list(tqdm(
            pool.imap(process_stem_with_encodec_worker, work_items_todo),
            total=len(work_items_todo),
            desc="Processing stems"
        ))
    
    # Collect statistics
    for success, error_msg in results:
        if success:
            stats.processed_stems += 1
        else:
            stats.failed_stems += 1
            if error_msg:
                logger.error(f"Failed: {error_msg}")
    
    # Count processed tracks (any track with at least one processed stem)
    processed_tracks = set()
    for stem_path, _, _, _ in work_items_todo:
        processed_tracks.add(stem_path.parent.parent.parent)
    stats.processed_tracks = len(processed_tracks)
    
    # Print final statistics
    logger.info("="*80)
    logger.info("Processing Complete!")
    logger.info("="*80)
    logger.info(f"Tracks processed: {stats.processed_tracks}/{stats.total_tracks}")
    logger.info(f"Stems processed: {stats.processed_stems}/{stats.total_stems}")
    logger.info(f"Stems failed: {stats.failed_stems}")
    logger.info("="*80)


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate AI-reconstructed stems using Encodec (parallel version)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process with 4 parallel workers (default)
  python generate_ai_stems_parallel.py
  
  # Use 8 workers for faster processing
  python generate_ai_stems_parallel.py --workers 8
  
  # Process first 10 tracks only (testing)
  python generate_ai_stems_parallel.py --limit 10
  
  # Use custom ARIA dataset path
  python generate_ai_stems_parallel.py --aria-path /path/to/aria
  
  # Process only specific bandwidths
  python generate_ai_stems_parallel.py --bandwidths 3 6
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
    
    parser.add_argument(
        "--workers",
        type=int,
        default=NUM_WORKERS,
        help=f"Number of parallel worker processes (default: {NUM_WORKERS}, max: CPU count)"
    )
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    
    generate_ai_stems_parallel(
        aria_path=args.aria_path,
        limit=args.limit,
        bandwidths=args.bandwidths,
        keep_original_sr=args.keep_original_sr,
        num_workers=args.workers
    )
