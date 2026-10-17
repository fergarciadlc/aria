"""
ARIA - Authentic vs Reconstructed Isolated Audio

A Python library for loading, exploring, and training on the ARIA dataset.
"""

from aria.audio import load_audio, save_audio
from aria.dataset import ARIADataset
from aria.exceptions import (
    ARIAError,
    AudioLoadError,
    DatasetNotFoundError,
    MissingStemError,
    TrackNotFoundError,
)
from aria.mix import (
    MixSpec,
    compute_mix,
    find_mix_path,
    generate_all_mix_specs,
    get_mix_path,
    get_or_compute_mix,
)
from aria.packaging import (
    PackageConfig,
    list_tiers,
    package_dataset,
)
from aria.splits import ManifestRecord, SplitManifest
from aria.track import Track, get_available_codecs, get_stem_path, load_track
from aria.validate import (
    check_dataset_exists,
    get_dataset_summary,
    validate_dataset,
    validate_track,
)

__version__ = "0.1.0"

__all__ = [
    # Main classes
    "ARIADataset",
    "Track",
    "MixSpec",
    "SplitManifest",
    "ManifestRecord",
    # Functions
    "load_track",
    "get_stem_path",
    "get_available_codecs",
    "compute_mix",
    "generate_all_mix_specs",
    "get_mix_path",
    "find_mix_path",
    "get_or_compute_mix",
    "load_audio",
    "save_audio",
    # Validation
    "check_dataset_exists",
    "validate_dataset",
    "validate_track",
    "get_dataset_summary",
    # Packaging
    "package_dataset",
    "PackageConfig",
    "list_tiers",
    # Exceptions
    "ARIAError",
    "DatasetNotFoundError",
    "TrackNotFoundError",
    "MissingStemError",
    "AudioLoadError",
]
