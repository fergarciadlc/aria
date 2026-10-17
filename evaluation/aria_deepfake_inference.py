"""
Deepfake detection inference on ARIA dataset mixes

This script runs the Deezer deepfake detector model on mixes from the ARIA dataset,
which contain combinations of real and AI-generated stems. It stores detailed metadata
about each mix including which stems are AI-generated and what codec was used.

Requirements:
    - A checkout of the Deezer deepfake detector, passed with --detector-path
      or the DEEZER_DETECTOR_PATH environment variable
    - TensorFlow 2.x
    - torchaudio, soxr
    - loguru

Usage (--aria-path / --detector-path default to $ARIA_ROOT / $DEEZER_DETECTOR_PATH):
    # Process all mixes from all tracks
    python -m evaluation.aria_deepfake_inference --aria-path /path/to/aria --detector-path /path/to/deepfake-detector

    # Process mixes from first 10 tracks only
    python -m evaluation.aria_deepfake_inference --limit 10
    
    # Process specific codec
    python -m evaluation.aria_deepfake_inference --codec encodec_6
    
    # Custom output file
    python -m evaluation.aria_deepfake_inference --output results.jsonl --limit 5
    
    # Use CPU only
    python -m evaluation.aria_deepfake_inference --gpu -1

Output format (JSONL):
    Each line contains:
    - file: filename of the mix
    - path: full path to the mix file
    - track_id: ARIA track ID
    - num_stems: total number of stems in mix
    - num_ai_stems: number of AI-generated stems (0 to num_stems)
    - ai_stem_indices: list of which stem indices are AI
    - ai_codec: codec used for AI stems (e.g., "encodec_3")
    - stem_names: names of all stems, in index order (alphabetical)
    - real_probability: mean probability of being real (0-1)
    - real_probability_std: std dev across slices
    - real_probability_min/max: min/max across slices
    - deepfake_probability: 1 - real_probability
    - prediction: "real" or "deepfake"
    - confidence: max(real_prob, fake_prob)
    - encoder_distribution: probability distribution over encoder types
    - most_likely_encoder: encoder with highest probability
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import tensorflow as tf
from tensorflow import keras
import torchaudio
import soxr
from loguru import logger
from tqdm import tqdm

from aria.mix import MixSpec, find_mix_path, generate_all_mix_specs
from aria.track import Track, iter_track_dirs, load_track

SAMPLE_RATE_TARGET= 44100  # Deezer model expects 44.1kHz

# Encoder names for multi-task model
ENCODER_NAMES = [
    "real", "encodec24", "encodec3", "encodec6", 
    "griffin256", "griffin512", "lac14", "lac2", "lac7", "musika"
]


def parse_args():
    parser = argparse.ArgumentParser(description="ARIA dataset deepfake detection inference")
    parser.add_argument(
        "--aria-path",
        help="Path to ARIA dataset (default: $ARIA_ROOT)",
        default=os.environ.get("ARIA_ROOT"),
        required="ARIA_ROOT" not in os.environ,
    )
    parser.add_argument(
        "--detector-path",
        help="Path to the Deezer deepfake-detector checkout (default: $DEEZER_DETECTOR_PATH)",
        default=os.environ.get("DEEZER_DETECTOR_PATH"),
        required="DEEZER_DETECTOR_PATH" not in os.environ,
    )
    parser.add_argument("--limit", help="Limit to N tracks (for testing)", type=int, default=None)
    parser.add_argument("--codec", help="Process only this codec (e.g., encodec_3)", type=str, default=None)
    parser.add_argument("--config", help="Deezer model config name", type=str, default="interpretable")
    parser.add_argument("--weights", help="Weights path (default: dzr_interpretable)", type=str, default="")
    parser.add_argument("--output", help="Output JSONL file", type=str, default="")
    parser.add_argument("--gpu", help="GPU to use (-1 for CPU)", type=int, default=-1)
    parser.add_argument("--skip-existing", help="Skip tracks with existing results", action="store_true")
    return parser.parse_args()


def setup_gpu(gpu_id):
    """Configure GPU/CPU"""
    if gpu_id >= 0:
        gpus = tf.config.list_physical_devices("GPU")
        if len(gpus) > gpu_id:
            tf.config.set_visible_devices(gpus[gpu_id], "GPU")
            tf.config.experimental.set_memory_growth(gpus[gpu_id], True)
            logger.info(f"Using GPU {gpu_id}")
        else:
            logger.warning(f"GPU {gpu_id} not found, using CPU")
    else:
        logger.info("Using CPU")


def get_mix_files(track: Track, codec: Optional[str] = None) -> List[Tuple[Path, MixSpec]]:
    """
    Get all pre-computed mix files (any audio format) for a track.

    Returns:
        List of (mix_path, mix_spec) tuples
    """
    mixes_path = track.path / "mixes"
    if not mixes_path.exists():
        logger.debug(f"Mixes directory not found: {mixes_path}")
        return []

    if codec:
        codecs = [codec]
    else:
        codecs = sorted(d.name for d in mixes_path.iterdir() if d.is_dir())

    mix_files = []
    for codec_name in codecs:
        for spec in generate_all_mix_specs(track.num_stems, codec_name):
            mix_path = find_mix_path(track, spec)
            if mix_path is not None:
                mix_files.append((mix_path, spec))

    return mix_files


def load_and_preprocess_audio(audio_path: Path, target_sr: int, audio_slice_sec: float):
    """Load and preprocess audio file for Deezer model."""
    try:
        audio_raw, sr = torchaudio.load(str(audio_path))
    except Exception as e:
        raise ValueError(f"Failed to load audio: {e}")
    
    # Resample to target SR
    if sr != target_sr:
        audio_raw = soxr.resample(audio_raw.numpy().T, sr, target_sr).T
        audio_raw = tf.constant(audio_raw, dtype=tf.float32)
    else:
        audio_raw = tf.constant(audio_raw.numpy(), dtype=tf.float32)
    
    # Ensure stereo
    if len(audio_raw.shape) == 1 or audio_raw.shape[0] == 1:
        if len(audio_raw.shape) == 1:
            audio_raw = tf.stack([audio_raw, audio_raw], axis=0)
        else:
            audio_raw = tf.concat([audio_raw, audio_raw], axis=0)
    
    # Extract slices
    target_len = int(audio_slice_sec * target_sr)
    total_len = audio_raw.shape[1]
    
    if total_len < target_len:
        pad_len = target_len - total_len
        audio_raw = tf.pad(audio_raw, [[0, 0], [0, pad_len]], mode="CONSTANT")
    
    # Extract up to 5 slices
    num_slices = max(1, total_len // target_len)
    slices = []
    
    for i in range(min(num_slices, 5)):
        start = i * target_len if i < num_slices - 1 else total_len - target_len
        audio_slice = audio_raw[:, start:start + target_len]
        audio_slice = tf.transpose(audio_slice)
        slices.append(audio_slice)
    
    return tf.stack(slices)


def process_mix(
    mix_path: Path, 
    mix_spec: MixSpec,
    track: Track,
model, 
    augmenter, 
    global_conf: Dict,
) -> Dict:
    """Run inference on a single mix file."""
    try:
        # Load and preprocess
        audio_slices = load_and_preprocess_audio(
            mix_path, 
            global_conf["sr"], 
            global_conf["audio_slice"]
        )
        
        # Apply augmentation
        processed_slices = []
        for i in range(audio_slices.shape[0]):
            processed = augmenter.transform(audio_slices[i])
            processed_slices.append(processed)
        
        audio_batch = tf.stack(processed_slices)
        
        # Handle patches
        if "patch_size_t" in global_conf and "patch_size_f" in global_conf:
            patch_size_t = global_conf['patch_size_t']
            patch_size_f = global_conf['patch_size_f']
            
            patched_slices = []
            for spec in audio_batch:
                spec_shape = tf.shape(spec)
                t_offset = (spec_shape[0] - patch_size_t) // 2
                f_offset = (spec_shape[1] - patch_size_f) // 2
                patch = tf.slice(spec, [t_offset, f_offset, 0], [patch_size_t, patch_size_f, -1])
                patched_slices.append(patch)
            
            audio_batch = tf.stack(patched_slices)
        
        # Add channel dimension if needed
        if "use_raw" in global_conf:
            audio_batch = tf.expand_dims(audio_batch, -1)
        
        # Predict
        predictions = model.predict(audio_batch, verbose=0)
        
        # Parse predictions
        if isinstance(predictions, list):
            # Multi-output: [deepfake, encoder]
            deepfake_probs = predictions[0].flatten()
            encoder_probs = predictions[1]
            avg_encoder_probs = np.mean(encoder_probs, axis=0).tolist()
        else:
            # Single output
            deepfake_probs = predictions.flatten()
            avg_encoder_probs = None
        
        # Calculate statistics
        real_prob_mean = float(np.mean(deepfake_probs))
        real_prob_std = float(np.std(deepfake_probs))
        real_prob_min = float(np.min(deepfake_probs))
        real_prob_max = float(np.max(deepfake_probs))
        
        # Build result with ARIA metadata
        result = {
            "file": mix_path.name,
            "path": str(mix_path),
            "track_id": track.id,
            "num_stems": track.num_stems,
            "stem_names": track.stem_names,

            # Mix metadata
            "num_ai_stems": mix_spec.num_ai_stems,
            "ai_stem_indices": list(mix_spec.ai_indices),
            "ai_codec": mix_spec.codec,
            
            # Detection results
            "real_probability": real_prob_mean,
            "real_probability_std": real_prob_std,
            "real_probability_min": real_prob_min,
            "real_probability_max": real_prob_max,
            "real_probability_per_slice": deepfake_probs.tolist(),
            "deepfake_probability": 1 - real_prob_mean,
            "prediction": "real" if real_prob_mean > 0.5 else "deepfake",
            "confidence": max(real_prob_mean, 1 - real_prob_mean),
            "num_slices": int(audio_slices.shape[0]),
        }
        
        # Add encoder distribution if available
        if avg_encoder_probs is not None:
            encoder_dist = {
                name: float(prob) 
                for name, prob in zip(ENCODER_NAMES, avg_encoder_probs)
            }
            result["encoder_distribution"] = encoder_dist
            result["most_likely_encoder"] = ENCODER_NAMES[np.argmax(avg_encoder_probs)]
        
        return result
        
    except Exception as e:
        return {
            "file": mix_path.name,
            "path": str(mix_path),
            "track_id": track.id,
            "error": str(e),
            "prediction": "error"
        }


def load_model(detector_path: str, config_name: str, weights_path: str):
    """Load the Deezer deepfake detector model from its repository checkout."""
    sys.path.insert(0, detector_path)
    from loader.audio import EvalAugmenter
    from loader.config import ConfLoader
    from loader.global_variables import CONF_PATH, WEIGHTS_PATH
    from model.simple_cnn import SimpleCNN, SimpleSpectrogramCNN

# Load config
    configuration = ConfLoader(CONF_PATH)
    configuration.load_model(config_name)
    global_conf = configuration.conf
    
    # Setup paths - use Deezer pre-trained weights for interpretable config by default
    if not weights_path:
        if config_name == "interpretable":
            weights_path = os.path.join(WEIGHTS_PATH, "dzr_interpretable")
        else:
            weights_path = os.path.join(WEIGHTS_PATH, config_name)
    
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Weights not found: {weights_path}")
    
    # Create augmenter
    augmenter = EvalAugmenter(global_conf)
    
    # Build model architecture (need a dummy input to get shape)
    dummy_audio = tf.zeros([1, int(global_conf["audio_slice"] * global_conf["sr"]), 2])
    dummy_processed = augmenter.transform(dummy_audio[0])
    
    if "patch_size_t" in global_conf and "patch_size_f" in global_conf:
        patch_size_t = global_conf['patch_size_t']
        patch_size_f = global_conf['patch_size_f']
        spec_shape = tf.shape(dummy_processed)
        t_offset = (spec_shape[0] - patch_size_t) // 2
        f_offset = (spec_shape[1] - patch_size_f) // 2
        dummy_processed = tf.slice(dummy_processed, [t_offset, f_offset, 0], [patch_size_t, patch_size_f, -1])
    
    if "use_raw" in global_conf:
        dummy_processed = tf.expand_dims(dummy_processed, -1)
    
    input_shape = dummy_processed.shape
    
    # Build model
    if "use_raw" in global_conf:
        model_wrapper = SimpleCNN(input_shape, global_conf, detect_encoder=10)
    else:
        model_wrapper = SimpleSpectrogramCNN(input_shape, global_conf, detect_encoder=10)
    
    # Load weights
    try:
        model_wrapper.m = keras.models.load_model(weights_path)
        logger.info("Loaded complete model")
    except:
        model_wrapper.m.load_weights(weights_path)
        logger.info("Loaded model weights")
    
    return model_wrapper.m, augmenter, global_conf


def main():
    args = parse_args()
    
    # Setup
    setup_gpu(args.gpu)
    
    aria_path = Path(args.aria_path)
    if not aria_path.exists():
        logger.error(f"ARIA path does not exist: {aria_path}")
        sys.exit(1)
    
    # Setup output file
    if args.output:
        output_file = args.output
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        codec_suffix = f"_{args.codec}" if args.codec else ""
        limit_suffix = f"_limit{args.limit}" if args.limit else ""
        output_file = f"aria_inference{codec_suffix}{limit_suffix}_{timestamp}.jsonl"
    
    logger.info("="*60)
    logger.info("ARIA Dataset Deepfake Detection Inference")
    logger.info("="*60)
    logger.info(f"ARIA path: {aria_path}")
    logger.info(f"Config: {args.config}")
    logger.info(f"Output: {output_file}")
    if args.limit:
        logger.info(f"Track limit: {args.limit}")
    if args.codec:
        logger.info(f"Codec filter: {args.codec}")
    logger.info("="*60)
    
    # Load model
    logger.info("\nLoading model...")
    model, augmenter, global_conf = load_model(args.detector_path, args.config, args.weights)
    
    logger.info(f"Model configuration:")
    logger.info(f"  Sample rate: {global_conf['sr']} Hz")
    logger.info(f"  Audio slice: {global_conf['audio_slice']} seconds")
    if "patch_size_t" in global_conf and "patch_size_f" in global_conf:
        logger.info(f"  Patch size: {global_conf['patch_size_t']}x{global_conf['patch_size_f']}")
    logger.info(f"  Effects: {global_conf.get('effects', 'N/A')}")
    
    # Get track directories
    track_dirs = iter_track_dirs(aria_path)
    if args.limit:
        track_dirs = track_dirs[:args.limit]
    logger.info(f"Processing {len(track_dirs)} tracks in {aria_path}")
    
    if not track_dirs:
        logger.error("No tracks found!")
        sys.exit(1)
    
    # Process tracks
    total_mixes = 0
    processed_mixes = 0
    error_mixes = 0
    
    # Create/open output file
    with open(output_file, 'w') as f:
        for track_dir in tqdm(track_dirs, desc="Processing tracks"):
            track = load_track(track_dir)
            mix_files = get_mix_files(track, args.codec)
            
            if not mix_files:
                logger.debug(f"No mixes found for track: {track_dir.name}")
                continue
            
            total_mixes += len(mix_files)
            
            # Process each mix
            for mix_path, mix_spec in tqdm(mix_files, desc=f"  {track_dir.name}", leave=False):
                result = process_mix(
                    mix_path,
                    mix_spec,
                    track,
                    model,
                    augmenter,
                    global_conf
                )
                
                # Write result immediately
                f.write(json.dumps(result) + '\n')
                f.flush()
                
                if result.get("prediction") == "error":
                    error_mixes += 1
                else:
                    processed_mixes += 1
    
    # Summary
    logger.info("\n" + "="*60)
    logger.info("SUMMARY")
    logger.info("="*60)
    logger.info(f"Total tracks: {len(track_dirs)}")
    logger.info(f"Total mixes: {total_mixes}")
    logger.info(f"Processed: {processed_mixes}")
    logger.info(f"Errors: {error_mixes}")
    logger.info(f"\n📁 Results saved to: {output_file}")


if __name__ == "__main__":
    main()
