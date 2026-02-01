# Dataset Packaging Guide

This guide explains how to create distribution variants of the ARIA dataset for Zenodo or other repositories.

## Overview

The packaging tool creates subsets of the full ARIA dataset with configurable:
- **Size** — Number of tracks (tiny, small, medium, full)
- **Format** — Audio format (WAV, FLAC, MP3)
- **Content** — Stems only or stems + pre-generated mixes

---

## Quick Start

```bash
# List available tiers
python -m aria tiers

# Package a tiny version (10 tracks, FLAC)
python -m aria package \
    --source /path/to/full/aria \
    --output /releases/aria-tiny \
    --tier tiny
```

---

## CLI Reference

### `python -m aria package`

Create a distribution package from the full dataset.

```
Options:
  --source, -s PATH      Source dataset path (required)
  --output, -o PATH      Output path (required)
  --tier, -t TIER        Predefined tier: tiny, small, medium, full
  --num-tracks, -n N     Custom number of tracks (alternative to --tier)
  --format, -f FORMAT    Audio format: wav, flac, mp3 (default: flac)
  --include-mixes, -m    Pre-generate all mixes
  --codecs, -c CODECS    Comma-separated codecs (e.g., encodec_3,encodec_6)
  --seed N               Random seed for track selection (default: 42)
```

### `python -m aria tiers`

List available predefined tiers:

```
tiny      10 tracks - Quick testing
small     50 tracks - Development
medium   100 tracks - Experiments  
full     all tracks - Complete dataset
```

### `python -m aria info PATH`

Show dataset information:

```bash
python -m aria info /path/to/aria --detailed
```

### `python -m aria validate PATH`

Validate dataset integrity:

```bash
python -m aria validate /path/to/aria --codecs encodec_3
```

---

## Predefined Tiers

| Tier | Tracks | Est. Size (FLAC) | Use Case |
|------|--------|------------------|----------|
| `tiny` | 10 | ~2-3 GB | CI/CD, quick tests |
| `small` | 50 | ~15 GB | Development |
| `medium` | 100 | ~30 GB | Experiments |
| `full` | 240 | ~85 GB | Production training |

---

## Examples

### Basic Packaging

```bash
# Tiny FLAC version (default format)
python -m aria package -s /full/aria -o /out/aria-tiny --tier tiny

# Small WAV version
python -m aria package -s /full/aria -o /out/aria-small-wav --tier small -f wav

# Custom: 75 tracks in FLAC
python -m aria package -s /full/aria -o /out/aria-75 -n 75 -f flac
```

### With Pre-generated Mixes

```bash
# Full dataset with all mixes pre-generated
python -m aria package \
    -s /full/aria \
    -o /out/aria-full-mixes \
    --tier full \
    --include-mixes \
    --codecs encodec_3
```

⚠️ **Warning**: Pre-generating mixes significantly increases storage:
- Stems only: ~85 GB
- With mixes: ~400+ GB

### Selecting Specific Codecs

```bash
# Only include encodec_3 (exclude encodec_6, encodec_24)
python -m aria package -s /full/aria -o /out/aria-tiny -t tiny -c encodec_3

# Include multiple codecs
python -m aria package -s /full/aria -o /out/aria-multi -t small -c encodec_3,encodec_6
```

---

## Python API

You can also use the packaging functions directly:

```python
from aria import package_dataset, PackageConfig

# Using a predefined tier
stats = package_dataset(
    source_path="/path/to/full/aria",
    output_path="/releases/aria-tiny",
    tier="tiny",
)

# Using custom configuration
config = PackageConfig(
    num_tracks=50,
    audio_format="flac",
    include_mixes=False,
    codecs=["encodec_3"],
    seed=42,
)

stats = package_dataset(
    source_path="/path/to/full/aria",
    output_path="/releases/custom",
    config=config,
)

print(f"Packaged {stats.tracks_processed} tracks")
print(f"Copied {stats.stems_copied} stems")
```

---

## Output Structure

The packaged dataset maintains the same structure as the source:

```
aria-tiny/
├── package_info.json       # Package metadata
├── {track-uuid}/
│   ├── metadata.json
│   └── stems/
│       ├── real/
│       │   ├── vocals.flac
│       │   └── ...
│       └── ai_generated/
│           └── encodec_3/
│               ├── vocals.flac
│               └── ...
└── ...
```

The `package_info.json` contains:

```json
{
  "source": "/path/to/full/aria",
  "num_tracks": 10,
  "audio_format": "flac",
  "include_mixes": false,
  "codecs": null,
  "tier": "tiny"
}
```

---

## Zenodo Upload Workflow

Recommended workflow for creating Zenodo releases:

```bash
# 1. Create all tiers
python -m aria package -s /full/aria -o /releases/aria-tiny   -t tiny   -f flac
python -m aria package -s /full/aria -o /releases/aria-small  -t small  -f flac  
python -m aria package -s /full/aria -o /releases/aria-full   -t full   -f flac

# 2. Validate each package
python -m aria validate /releases/aria-tiny
python -m aria validate /releases/aria-small
python -m aria validate /releases/aria-full

# 3. Create archives
cd /releases
tar -czvf aria-tiny-v1.0.tar.gz aria-tiny/
tar -czvf aria-small-v1.0.tar.gz aria-small/
tar -czvf aria-full-v1.0.tar.gz aria-full/

# 4. Upload to Zenodo (separate DOI per tier)
```

---

## Track Selection

Tracks are selected deterministically based on:
1. Sorted alphabetically by `display_name` (Artist - Song)
2. First N tracks selected

The `--seed` option is reserved for future random selection but currently unused (deterministic selection ensures reproducibility).

---

## Format Comparison

| Format | Compression | Quality | Notes |
|--------|-------------|---------|-------|
| WAV | None | Lossless | Largest files |
| FLAC | ~50% | Lossless | **Recommended** |
| MP3 | ~90% | Lossy | Not yet fully supported |

⚠️ **Note**: MP3 support requires additional dependencies (pydub or ffmpeg). Currently falls back to FLAC.

---

## Troubleshooting

### "Source dataset not found"

Ensure the source path points to a valid ARIA dataset with track directories containing `metadata.json`.

### Slow packaging

Packaging involves reading and re-encoding audio files. For large datasets:
- Use FLAC format (faster than WAV due to smaller file sizes)
- Package without mixes (`--include-mixes` is slow)

### Validation errors after packaging

Run validation to identify issues:

```bash
python -m aria validate /packaged/aria --codecs encodec_3
```

Common issues:
- Missing stems (check source dataset)
- Audio encoding errors (check disk space)


