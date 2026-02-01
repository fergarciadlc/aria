# ARIA Dataset

**Authentic vs Reconstructed Isolated Audio**

ARIA is a dataset and Python library for evaluating audio models on the task of distinguishing real (authentic) from AI-generated (reconstructed) audio stems and their mixtures.

The name ARIA comes from the musical term meaning a solo vocal piece, reflecting the dataset's focus on isolated sources.

---

## Overview

ARIA provides:
- **240 tracks** with multiple isolated stems (vocals, drums, bass, etc.)
- **Real and AI-reconstructed** versions of each stem
- **On-the-fly mix generation** - create any combination of real/AI stems
- **PyTorch Dataset** for training deepfake detectors
- **Validation tools** to verify dataset integrity

### Research Applications

- Audio forgery and authenticity detection
- Hybrid real/AI mixture analysis
- Source separation evaluation
- Generative audio benchmarking
- Training discriminators and classifiers

---

## Installation

```bash
# Clone the repository
git clone https://github.com/your-org/aria.git
cd aria

# Install the library
pip install -e .

# Or with PyTorch support
pip install -e ".[torch]"

# Or with all dependencies (including dataset creation tools)
pip install -e ".[all]"
```

### Requirements

- Python 3.10+
- Core: `numpy`, `soundfile`, `scipy`
- PyTorch integration: `torch>=2.0`
- Dataset creation: `torchaudio`, `encodec`, `moisesdb`

---

## Quick Start

> 📓 **See also:** [examples/quickstart.ipynb](examples/quickstart.ipynb) for an interactive tutorial.

### Loading the Dataset

```python
from aria import ARIADataset

# Load the dataset
dataset = ARIADataset("/path/to/aria")

print(f"Loaded {len(dataset)} tracks")  # 240 tracks

# Access a track
track = dataset[0]
print(track.display_name)  # "Artist - Song"
print(track.stem_names)    # ['vocals', 'drums', 'bass', 'guitar']
```

### Working with Tracks

```python
# Get track by index or ID
track = dataset[0]
track = dataset["track-uuid-here"]

# Search by name
track = dataset.get_by_name("Artist Name")

# Filter tracks
rock_tracks = dataset.filter(genre="rock")
four_stem_tracks = dataset.filter(min_stems=4, max_stems=4)

# Dataset statistics
stats = dataset.stats()
print(stats)
# {'num_tracks': 240, 'total_stems': 1371, 'genres': {...}, ...}
```

### Loading Audio

```python
from aria import load_audio
from aria.track import get_stem_path, get_available_codecs

# Check available AI codecs
codecs = get_available_codecs(track)  # ['encodec_3', 'encodec_6', ...]

# Load a real stem
path = get_stem_path(track, "vocals", source="real")
audio, sr = load_audio(path)  # (2, samples), 48000

# Load an AI-generated stem
path = get_stem_path(track, "vocals", source="ai_generated", codec="encodec_3")
audio, sr = load_audio(path)
```

### Creating Mixes On-the-Fly

```python
from aria import MixSpec, compute_mix

# Define which stems should be AI-generated
spec = MixSpec(
    ai_indices=(0, 2),      # Stems 0 and 2 are AI
    codec="encodec_3"
)

# Compute the mix (loads stems and sums them)
audio = compute_mix(track, spec)  # (2, samples)

# Mix properties
print(spec.filename)      # "mix_S-0-2"
print(spec.u_folder)      # "U2"
print(spec.num_ai_stems)  # 2
```

### Generate All Mix Combinations

```python
from aria import generate_all_mix_specs

# Generate all 2^n combinations for a track
all_specs = generate_all_mix_specs(
    num_stems=track.num_stems,
    codec="encodec_3"
)

print(f"Generated {len(all_specs)} mix specs")  # 16 for 4 stems
```

---

## PyTorch Integration

ARIA provides a ready-to-use PyTorch Dataset for training:

```python
from aria.torch import ARIATorchDataset, aria_collate_fn
from torch.utils.data import DataLoader

# Create dataset
torch_dataset = ARIATorchDataset(
    path="/path/to/aria",
    codec="encodec_3",
    segment_duration=5.0,      # Random 5-second crops
    label_mode="binary",       # 0=all_real, 1=has_any_ai
    mix_sampling="all",        # All 2^n combinations
)

# Create DataLoader with custom collate function
loader = DataLoader(
    torch_dataset,
    batch_size=32,
    shuffle=True,
    num_workers=4,
    collate_fn=aria_collate_fn,  # Required for variable-length metadata
)

# Training loop
for batch in loader:
    audio = batch["audio"]        # (B, 2, T) - stereo audio
    labels = batch["label"]       # (B,) - 0 or 1
    track_ids = batch["track_id"]      # List of strings
    ai_indices = batch["ai_indices"]   # List of tuples
```

> **Note:** The `aria_collate_fn` is required because metadata fields like `ai_indices` have variable lengths across samples (e.g., empty tuple for all-real mixes vs. multiple indices for AI mixes).

### Label Modes

```python
# Binary: 0 = all real, 1 = any AI stem
ARIATorchDataset(..., label_mode="binary")

# Count: number of AI stems (0, 1, 2, ...)
ARIATorchDataset(..., label_mode="count")

# Ratio: fraction of AI stems (0.0 to 1.0)
ARIATorchDataset(..., label_mode="ratio")
```

### Mix Sampling Strategies

```python
# All combinations (2^n per track)
ARIATorchDataset(..., mix_sampling="all")

# Only extremes: U0 (all real) and Un (all AI)
ARIATorchDataset(..., mix_sampling="binary")

# Balanced across U-levels
ARIATorchDataset(..., mix_sampling="balanced")
```

### Class Weights for Imbalanced Data

```python
weights = torch_dataset.get_class_weights()
criterion = nn.CrossEntropyLoss(weight=weights)
```

---

## Dataset Validation

Verify dataset integrity before training:

```python
from aria import check_dataset_exists, validate_dataset, get_dataset_summary

# Quick existence check
if check_dataset_exists("/path/to/aria"):
    print("Dataset found!")

# Get summary without loading everything
summary = get_dataset_summary("/path/to/aria")
print(summary)
# {'exists': True, 'num_tracks': 240, 'codecs': ['encodec_3']}

# Full validation (checks all stems exist)
result = validate_dataset(
    "/path/to/aria",
    check_real_stems=True,
    check_ai_stems=True,
    codecs=["encodec_3"],
)

if result.valid:
    print("✓ Dataset is complete!")
else:
    for error in result.errors:
        print(f"✗ {error}")
```

---

## Dataset Structure

Each track is stored in its own directory:

```
aria/
├── {track-uuid}/
│   ├── metadata.json           # Track metadata
│   ├── stems/
│   │   ├── real/               # Authentic stems
│   │   │   ├── vocals.wav
│   │   │   ├── drums.wav
│   │   │   ├── bass.wav
│   │   │   └── guitar.wav
│   │   └── ai_generated/       # AI-reconstructed stems
│   │       ├── encodec_3/      # Encodec @ 3 kbps
│   │       │   ├── vocals.wav
│   │       │   └── ...
│   │       ├── encodec_6/      # Encodec @ 6 kbps
│   │       └── encodec_24/     # Encodec @ 24 kbps
│   │
│   └── mixes/                  # Pre-generated mixes (OPTIONAL)
│       └── encodec_3/
│           ├── U0/             # 0 AI stems (all real)
│           │   └── mix_S-empty.wav
│           ├── U1/             # 1 AI stem
│           │   ├── mix_S-0.wav
│           │   ├── mix_S-1.wav
│           │   └── ...
│           ├── U2/             # 2 AI stems
│           │   ├── mix_S-0-1.wav
│           │   └── ...
│           └── U4/             # All AI stems
│               └── mix_S-0-1-2-3.wav
└── ...
```

### Pre-generated vs On-the-fly Mixes

The `mixes/` folder is **optional**. The library supports two modes:

| Mode | Description | Use Case |
|------|-------------|----------|
| **On-the-fly** | Mixes computed when needed by summing stems | Default, saves storage (~6-12x) |
| **Pre-generated** | Mixes stored on disk | Faster loading, reproducibility |

```python
from aria.mix import get_or_compute_mix

# Automatically uses pre-generated if available, otherwise computes
audio = get_or_compute_mix(track, mix_spec, prefer_precomputed=True)

# Always compute fresh (ignore pre-generated)
audio = compute_mix(track, mix_spec)
```

**Storage comparison** (240 tracks, 4 stems avg):
- Stems only: ~85 GB
- With all pre-generated mixes: ~400+ GB

### Metadata Format

```json
{
  "id": "uuid-string",
  "artist": "Artist Name",
  "name": "Track Title",
  "genre": "rock",
  "stems": ["vocals", "drums", "bass", "guitar"],
  "number_of_stems": 4,
  "provider": "moisesdb"
}
```

### Audio Specifications

| Property | Value |
|----------|-------|
| Sample Rate | 48,000 Hz |
| Channels | Stereo (2) |
| Bit Depth | 16-bit / 24-bit |
| Format | WAV |

---

## Mix Naming Convention

Mixes are named by which stem indices are AI-generated:

| Mix Name | Description |
|----------|-------------|
| `mix_S-empty.wav` | All stems real (U0) |
| `mix_S-0.wav` | Stem 0 is AI (U1) |
| `mix_S-1-3.wav` | Stems 1 and 3 are AI (U2) |
| `mix_S-0-1-2-3.wav` | All stems AI (U4) |

For a track with **n** stems:
- Total mixes: **2ⁿ**
- Mixes in folder Uk: **C(n,k)** (binomial coefficient)

---

## API Reference

### Core Classes

| Class | Description |
|-------|-------------|
| `ARIADataset` | Main entry point for loading and iterating tracks |
| `Track` | Single track with metadata and stem paths |
| `MixSpec` | Specification for a mix (which stems are AI) |
| `Config` | Configuration options |

### Core Functions

| Function | Description |
|----------|-------------|
| `load_audio(path)` | Load audio file, returns (audio, sr) |
| `save_audio(path, audio, sr)` | Save audio to file |
| `compute_mix(track, spec)` | Compute mix on-the-fly |
| `generate_all_mix_specs(n, codec)` | Generate all 2ⁿ mix specs |
| `validate_dataset(path)` | Validate dataset integrity |
| `check_dataset_exists(path)` | Quick existence check |

### PyTorch

| Class | Description |
|-------|-------------|
| `ARIATorchDataset` | PyTorch Dataset with audio loading and labels |

---

## CLI Tools

ARIA includes command-line tools for dataset management:

```bash
# Show dataset information
python -m aria info /path/to/aria --detailed

# Validate dataset integrity  
python -m aria validate /path/to/aria

# Package for distribution (creates subset with specified format)
python -m aria package --source /full/aria --output /releases/tiny --tier tiny

# List available tiers
python -m aria tiers
```

---

## Dataset Creation

For dataset maintainers - creating ARIA from MoisesDB:

```bash
# 1. Prepare real stems from MoisesDB
python scripts/prepare_moisesdb.py \
    --moisesdb-path /path/to/moisesdb \
    --aria-path /path/to/aria \
    --sample-rate 48000

# 2. Generate AI-reconstructed stems
python scripts/generate_ai_stems.py \
    --aria-path /path/to/aria \
    --bandwidths 3 6 24

# 3. (Optional) Pre-generate mixes
python scripts/generate_mixes.py \
    --aria-path /path/to/aria \
    --model encodec_3
```

See [scripts/README.md](scripts/README.md) for generation scripts and [docs/packaging.md](docs/packaging.md) for creating distribution packages.

---

## Citation

If you use ARIA in your research, please cite:

```bibtex
@dataset{aria2024,
  title={ARIA: Authentic vs Reconstructed Isolated Audio Dataset},
  author={...},
  year={2024},
  url={https://github.com/your-org/aria}
}
```

---

## License

MIT License - see [LICENSE](LICENSE) for details.

---

## Acknowledgments

ARIA is built on stems from [MoisesDB](https://github.com/moises-ai/moises-db) and uses [Encodec](https://github.com/facebookresearch/encodec) for audio reconstruction.
