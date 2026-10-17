"""
Canonical ARIA dataset specification.

Single source of truth for the values shared by the library and the
dataset-creation scripts. Change them here, not in individual scripts.
"""

# Sample rate of real stems, AI stems and mixes on disk.
SAMPLE_RATE = 44100

# MP3 bitrate used for AI stems and pre-computed mixes.
MP3_BITRATE_KBPS = 256

# Encodec bandwidths (kbps) generated for the dataset. Codec folders are
# named f"encodec_{bandwidth}".
ENCODEC_BANDWIDTHS = (3, 6, 24)
DEFAULT_CODEC = "encodec_3"

# Recognised audio extensions, in lookup priority order.
AUDIO_EXTENSIONS = (".wav", ".flac", ".mp3")

# Format of pre-computed mixes written by scripts/generate_mixes.py.
MIX_FORMAT = "mp3"
