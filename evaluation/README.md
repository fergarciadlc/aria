# ARIA Evaluation Scripts

This directory contains scripts for evaluating the ARIA dataset using various deepfake detection models.

## aria_deepfake_inference.py

Runs the Deezer deepfake detector on ARIA dataset mixes, which contain combinations of real and AI-generated stems.

### Dependencies

This script requires the Deezer deepfake detector repository:
- **Path**: `/Users/fernando/dev/projects/deezer/deepfake-detector`
- **Repository**: Contains the pre-trained model weights and loader utilities

Required Python packages:
```bash
pip install tensorflow>=2.10
pip install torchaudio soxr
pip install loguru tqdm
```

### Usage

```bash
# Process all mixes from all tracks
python -m evaluation.aria_deepfake_inference

# Process mixes from first 10 tracks only (recommended for testing)
python -m evaluation.aria_deepfake_inference --limit 10

# Process only encodec_3 codec
python -m evaluation.aria_deepfake_inference --codec encodec_3

# Custom ARIA dataset path
python -m evaluation.aria_deepfake_inference --aria-path /path/to/aria

# Custom output file
python -m evaluation.aria_deepfake_inference --output my_results.jsonl

# Use GPU 0
python -m evaluation.aria_deepfake_inference --gpu 0

# Skip tracks that already have results (for resuming)
python -m evaluation.aria_deepfake_inference --skip-existing
```

### Output Format

Results are saved as JSONL (one JSON object per line). Each line contains:

```json
{
  "file": "mix_S-0-1.wav",
  "path": "/Users/fernando/datasets/aria/track_001/mixes/encodec_3/U2/mix_S-0-1.wav",
  "track_id": "track_001",
  "num_stems": 4,
  "stem_names": ["vocals", "drums", "bass", "other"],
  "num_ai_stems": 2,
  "ai_stem_indices": [0, 1],
  "ai_codec": "encodec_3",
  "real_probability": 0.342,
  "real_probability_std": 0.045,
  "real_probability_min": 0.298,
  "real_probability_max": 0.401,
  "real_probability_per_slice": [0.342, 0.356, 0.298, 0.401, 0.334],
  "deepfake_probability": 0.658,
  "prediction": "deepfake",
  "confidence": 0.658,
  "num_slices": 5,
  "encoder_distribution": {
    "real": 0.123,
    "encodec24": 0.089,
    "encodec3": 0.567,
    "encodec6": 0.134,
    "griffin256": 0.012,
    "griffin512": 0.008,
    "lac14": 0.023,
    "lac2": 0.015,
    "lac7": 0.018,
    "musika": 0.011
  },
  "most_likely_encoder": "encodec3"
}
```

### Key Features

1. **Metadata tracking**: Each result includes detailed information about:
   - Which stems are AI-generated vs real
   - What codec was used for AI stems
   - Track ID and stem names from ARIA dataset

2. **Multiple slices**: Each mix is split into up to 5 temporal slices for robust detection

3. **Encoder identification**: The model also predicts which AI codec was likely used

4. **Track limiting**: Use `--limit N` to process only the first N tracks (useful for testing)

5. **Codec filtering**: Use `--codec` to process only mixes from a specific codec

### Example Workflow

```bash
# Quick test on 5 tracks
python -m evaluation.aria_deepfake_inference --limit 5 --output test_run.jsonl

# Full evaluation of encodec_3 mixes
python -m evaluation.aria_deepfake_inference --codec encodec_3 --output encodec3_results.jsonl

# Process all mixes from all tracks (can take hours)
python -m evaluation.aria_deepfake_inference --output full_results.jsonl --gpu 0
```

### Notes

- The script expects the ARIA dataset at `/Users/fernando/datasets/aria` by default
- Mixes must be generated first using `scripts/generate_mixes.py`
- Results are written incrementally (safe to interrupt and resume)
- The Deezer model uses the "interpretable" config with pre-trained weights by default
